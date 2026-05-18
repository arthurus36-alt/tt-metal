// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "lltt.h"

using namespace ckernel;

// local function declarations
template <bool is_32bit, bool is_fp32 = false>
inline void transpose_dest_configure_addrmod();
template <bool transpose_of_faces, bool is_32bit, bool is_fp32 = false>
inline void transpose_dest_configure_mop();

// Notes on these template parameters:
// 1. <transpose_of_faces=false, is_32bit=false>: not supported.
// 2. <transpose_of_faces=false, is_32bit=true>: 4x 16x16 face transpose; can be combined with _llk_unpack_A_ with transpose_of_faces=true.
// 3. <transpose_of_faces=true, is_32bit=false>: the default case (full 32x32 tile transpose, non-32-bit).
// 4. <transpose_of_faces=true, is_32bit=true>: full 32x32 tile transpose for 32-bit.
//
// is_fp32 selects between two 32-bit MOPs:
//   - is_fp32 = true : MOVD2B / TRNSPSRCB / MOVB2D loop. Stays in the B
//     register, so all 32 bits of a Float32 mantissa survive the transfer.
//   - is_fp32 = false: MOVB2A / MOVA2D round-trip through the A register.
//     Required for Int32/UInt32 because it preserves the lower 16 bits across
//     the transpose, but is unsuitable for Float32 since A is read in tf32
//     format (lower mantissa bits would be dropped).
template <bool transpose_of_faces = true, bool is_32bit = false, bool is_fp32 = false>
inline void _llk_math_transpose_dest_(const std::uint32_t dst_index)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    math::reset_counters(p_setrwc::SET_ABD_F);

    // Wait condition SRCA_VLD is required as MOVB2A doesn't automatically wait
    // for SrcA[MatrixUnit.SrcABank].AllowedClient == SrcClient::MatrixUnit.
    // Wait condition SRCB_VLD is required as MOVD2B doesn't automatically wait
    // for SrcB[MatrixUnit.SrcBBank].AllowedClient == SrcClient::MatrixUnit.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);

    if constexpr (is_32bit)
    {
        // We need to disable the zero flag so that we don't lose bits when doing D2B/B2D
        // The data loss would happen if the bits that are mapped to the "exponent" field are 0
        // which would cause the "mantissa" bits to be flushed to 0 too.
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);

        if constexpr (transpose_of_faces)
        {
            // Full 32x32 tile transpose: 4 face transposes plus 8 middle-face row swaps.
            //   Float32  : 12 iterations, SFPU row swaps enabled via zmask 0xff0.
            //   Int32/UI : 3 iterations, row swaps folded into the MOP via zmask 0b101.
            if constexpr (is_fp32)
            {
                ckernel_unpack_template::run(12, 0xff0);
            }
            else
            {
                ckernel_unpack_template::run(3, 0b101);
            }
        }
        else
        {
            // 4x 16x16 face transposes only (no middle-face row swaps); same for both MOPs.
            ckernel_unpack_template::run(4, 0);
        }

        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0);
    }
    else
    {
        ckernel_unpack_template::run(2, 2);
    }

    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

template <bool is_32bit, bool is_fp32>
inline void transpose_dest_configure_addrmod()
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 16},
    }
        .set(ADDR_MOD_0);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_1);

    // ADDR_MOD_2 dest stride differs by MOP:
    //   - Float32 (is_32bit && is_fp32): stride of 2 rows for the final
    //     MOVB2D pair used by the transpose_of_faces=false path.
    //   - Int32 / UInt32 (is_32bit && !is_fp32) and 16-bit: stride of -16
    //     rows (10-bit two's complement) for the MOVA2D / MOVB2D step.
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = (is_32bit && is_fp32) ? 2 : (0x3ff & -16)},
    }
        .set(ADDR_MOD_2);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 32},
    }
        .set(ADDR_MOD_3);
}

template <bool transpose_of_faces, bool is_32bit, bool is_fp32>
inline void transpose_dest_configure_mop()
{
    if constexpr (is_32bit && is_fp32)
    {
        // Float32 32-bit transpose MOP.
        //
        // Records 16 MOVD2B / MOVB2D instructions that move data between DEST
        // and the B register only. The A register is never used as a transfer
        // slot, so the full Float32 mantissa is preserved across the MOP.
        //
        // The recorded body is identical for transpose_of_faces=true|false;
        // only the SFPU "middle-face row swap" macros are programmed for the
        // transpose_of_faces=true case, and the companion run(...) call in
        // _llk_math_transpose_dest_ controls the iteration count and zmask.
        lltt::record(16, 16);

#pragma GCC unroll 2
        for (int dest_32b_lo = 0; dest_32b_lo < 2; ++dest_32b_lo)
        {
            TTI_MOVD2B(dest_32b_lo, 16, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 0);
            TTI_MOVD2B(dest_32b_lo, 20, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 4);
            TTI_MOVD2B(dest_32b_lo, 24, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 8);
            TTI_MOVD2B(dest_32b_lo, 28, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 12);
            TTI_MOVB2D(dest_32b_lo, 16, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(dest_32b_lo, 20, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 4);
            TTI_MOVB2D(dest_32b_lo, 24, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 8);
            TTI_MOVB2D(dest_32b_lo, 28, dest_32b_lo == 1 ? ADDR_MOD_0 : ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 12);
        }

        std::uint32_t macro0 = TT_OP_SFPNOP;
        std::uint32_t macro1 = TT_OP_SFPNOP;

        if constexpr (transpose_of_faces)
        {
            TTI_SFPMOV(0, 0, 12, 0);

            TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (0x80 | (1 << 3) | 3) << 8);
            TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (0x40 | 4) << 0);
            TTI_SFPCONFIG(0, 4, 0);

            TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (0x40 | (1 << 3) | 3) << 8);
            TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, 0);
            TTI_SFPCONFIG(0, 5, 0);

            TTI_SFPCONFIG(0x330, 8, 1);

            macro0 = TT_OP_SFPLOADMACRO((0 << 2) | 1, 4, ADDR_MOD_1, 0x3ff & -48);
            macro1 = TT_OP_SFPLOADMACRO((1 << 2) | 0, 4, ADDR_MOD_2, 0x3ff & -32);
        }

        std::uint32_t movd2b_hi        = lltt::replay_insn(16, 4);
        std::uint32_t movb2d_hi_d2b_lo = lltt::replay_insn(20, 8);
        std::uint32_t movb2d_lo        = lltt::replay_insn(28, 4);
        std::uint32_t transpose        = TT_OP_TRNSPSRCB;

        ckernel_unpack_template tmp(
            true,
            true,
            movd2b_hi,
            transpose,
            movb2d_hi_d2b_lo,
            transpose,
            /* skip A */ macro0,
            /* B */ movb2d_lo,
            /* skip B */ macro1);
        tmp.program();
    }
    else if constexpr (is_32bit)
    {
        // Record instructions for 32-bit face transpose.
        // MOVB2D/MOVA2D(dest_32b_lo=0) can destroy the lo16 physical slot in DEST as a side effect.
        // To preserve lo16: save hi16 to A before handling lo16.
        if constexpr (transpose_of_faces)
        {
            lltt::record(0, 32);
            // [0..3] B@16 -> A@00 (4)
            TTI_MOVB2A(0, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 16);
            TTI_MOVB2A(4, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 20);
            TTI_MOVB2A(8, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 24);
            TTI_MOVB2A(12, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 28);
            // [4..7] B@16 -> A@16 (4) dst += 16
            TTI_MOVB2A(16, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 16);
            TTI_MOVB2A(20, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 20);
            TTI_MOVB2A(24, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 24);
            TTI_MOVB2A(28, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, 28);
            // [8..11] FHI -> B@16 (4)
            TTI_MOVD2B(0, 16, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 0);
            TTI_MOVD2B(0, 20, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 4);
            TTI_MOVD2B(0, 24, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 8);
            TTI_MOVD2B(0, 28, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 12);
            // [12] transpose B@16 (1)
            TTI_TRNSPSRCB;
            // [13..16] B@16 -> A@32 (4)
            TTI_MOVB2A(32, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 16);
            TTI_MOVB2A(36, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 20);
            TTI_MOVB2A(40, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 24);
            TTI_MOVB2A(44, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 28);
            // [17..20] FLO -> B@16 (4)
            TTI_MOVD2B(1, 16, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 0);
            TTI_MOVD2B(1, 20, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 4);
            TTI_MOVD2B(1, 24, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 8);
            TTI_MOVD2B(1, 28, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 12);
            // [21] transpose B@16 (1)
            TTI_TRNSPSRCB;
            // [22..23] A@00 -> FHI (2) dst -= 16
            TTI_MOVA2D(0, 0, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 0);
            TTI_MOVA2D(0, 8, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 8);
            // [24..25] A@32 -> FHI (2)
            TTI_MOVA2D(0, 32, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 0);
            TTI_MOVA2D(0, 40, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 8);
            // [26..29] B@16 -> FLO (4) dst += 16
            TTI_MOVB2D(1, 16, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(1, 20, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 4);
            TTI_MOVB2D(1, 24, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 8);
            TTI_MOVB2D(1, 28, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 12);
            // [30..31] A@16 -> FLO (2) dst += 16
            TTI_MOVA2D(1, 16, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 0);
            TTI_MOVA2D(1, 24, ADDR_MOD_0, p_mova2d::MOV_8_ROWS, 8);

            // The flow for face 2/3 is as follows:
            // MOP1: (replay offset 8)
            // F1HI -> B@16 (4)
            // TRANSPOSE    (1)
            // MOP2: (replay offset 0)
            // B@16 -> A@00 (4)
            // MOP3: (replay offset 17)
            // F1LO -> B@16 (4)
            // TRANSPOSE    (1)
            // MOP4: (replay offset 4)
            // B@16 -> A@16 (4) dst += 16
            // MOP5: (replay offset 8)
            // F2HI -> B@16 (4)
            // TRANSPOSE    (1)
            // B@16 -> A@32 (4)
            // F2LO -> B@16 (4)
            // TRANSPOSE    (1)
            // A@00 -> F2HI (2) dst -= 16
            // A@32 -> F1HI (2)
            // B@16 -> F1LO (4) dst += 16
            // A@16 -> F2LO (2) dst += 16

            const auto mop1 = lltt::replay_insn(8, 5);
            const auto mop2 = lltt::replay_insn(0, 4);
            const auto mop3 = lltt::replay_insn(17, 5);
            const auto mop4 = lltt::replay_insn(4, 4);
            const auto mop5 = lltt::replay_insn(8, 24);

            // The flow for face 1/4 is as follows:
            // MOPS1: (replay offset 8)
            // FHI -> B@16  (4)
            // TRANSPOSE    (1)
            // B@16 -> A@32 (4)
            // FLO -> B@16  (4)
            // TRANSPOSE    (1)
            // MOPS2: (replay offset 24)
            // A@32 -> FHI (2)
            // B@16 -> FLO (4) dst += 16
            const auto skip1 = lltt::replay_insn(8, 14);
            const auto skip2 = lltt::replay_insn(24, 6);

            ckernel_unpack_template tmp(
                true,
                true,
                mop1,
                mop2,
                mop3,
                mop4,
                /* skip A */ skip1,
                /* B */
                mop5,
                /* skip B */ skip2);
            tmp.program();
        }
        else
        {
            lltt::record(16, 16);
            // [0..3] hi16 reads: DEST -> B[16..28]
            TTI_MOVD2B(0, 16, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 0);
            TTI_MOVD2B(0, 20, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 4);
            TTI_MOVD2B(0, 24, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 8);
            TTI_MOVD2B(0, 28, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 12);
            // [4] transpose B[16..28]
            TTI_TRNSPSRCB;
            // [5..8] store B[16..28] -> A[16..28]
            TTI_MOVB2A(16, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 16);
            TTI_MOVB2A(20, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 20);
            TTI_MOVB2A(24, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 24);
            TTI_MOVB2A(28, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 28);
            // [9..12] lo16 reads: DEST → B[16..28]
            TTI_MOVD2B(1, 16, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 0);
            TTI_MOVD2B(1, 20, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 4);
            TTI_MOVD2B(1, 24, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 8);
            TTI_MOVD2B(1, 28, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 12);
            // [13] transpose B[16..28]
            TTI_TRNSPSRCB;
            // [14..15] hi16 store: A[16..28] → DEST (leaves lo16 in undefined state)
            TTI_MOVA2D(0, 16, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 0);
            TTI_MOVA2D(0, 24, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 8);

            // The flow is as follows:
            // MOP1: (replay offset 16):
            // HI -> B@16   (4)
            // TRANSPOSE    (1)
            // B@16 -> A@16 (4)
            // LO -> B@16   (4)
            // TRANSPOSE    (1)
            // A@16 -> HI   (2)
            // MOP2/3/4/5:
            // B@16 -> LO   (4) dst += 16
            const auto dhbta_dlbt_adh = lltt::replay_insn(16, 16);
            const auto movb2d_lo_1    = TT_OP_MOVB2D(1, 16, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 0);
            const auto movb2d_lo_2    = TT_OP_MOVB2D(1, 20, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 4);
            const auto movb2d_lo_3    = TT_OP_MOVB2D(1, 24, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 8);
            const auto movb2d_lo_4    = TT_OP_MOVB2D(1, 28, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 12);

            // MOP config:
            // - zmask 0-bits: 32b 16x16 face transpose.
            // - zmask 1-bits: NOP.
            ckernel_unpack_template tmp(
                true, true, dhbta_dlbt_adh, movb2d_lo_1, movb2d_lo_2, movb2d_lo_3, /* skip A */ TT_OP_SFPNOP, /* B */ movb2d_lo_4, /* skip B */ TT_OP_SFPNOP);
            tmp.program();
        }
    }
    else
    {
        lltt::record(16, 15);

        // ABCD
        TTI_MOVB2A(0, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 16);
        TTI_MOVB2A(4, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 20);
        TTI_MOVB2A(8, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 24);
        TTI_MOVB2A(12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, 28); // dst += 16

        // EFGHIJKLM
        TTI_MOVD2B(0, 16, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 0);
        TTI_MOVD2B(0, 20, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 4);
        TTI_MOVD2B(0, 24, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 8);
        TTI_MOVD2B(0, 28, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 12);
        TTI_TRNSPSRCB;
        TTI_MOVB2D(0, 16, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 0);
        TTI_MOVB2D(0, 20, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 4);
        TTI_MOVB2D(0, 24, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 8);
        TTI_MOVB2D(0, 28, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 12); // dst += 16

        // NO
        // Note: the 0x3ff mask ensures the negative offsets are 10 bits.
        TTI_MOVA2D(0, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0x3ff & (0 - 32)); // dst -= 16
        TTI_MOVA2D(0, 8, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0x3ff & (8 - 16)); // dst -= 16

        // The next 7 instructions expand as follows:
        // Face 0: 4x MOVD2B, TRNSPSRCB, 4x MOVB2D, dst += 16 (EFGHIJKLM)
        // Face 1: 4x MOVD2B, TRNSPSRCB, 4x MOVB2A, dst += 16 (EFGHI, ABCD..)
        // Face 2: 4x MOVD2B (dst -= 16), TRNSPSRCB, 4x MOVB2D, dst += 32 (..EFG, P, IJKL, Q)
        // Face 3: 4x MOVD2B, TRNSPSRCB, 4x MOVB2D, dst += 16 (EFGHIJKLM..)
        // Face 1: 2x MOVA2D (2x dst -= 16) (..NO)

        std::uint32_t EFGHIJKLM   = lltt::replay_insn(20, 9);
        std::uint32_t EFGHI       = lltt::replay_insn(20, 5);
        std::uint32_t ABCDEFG     = lltt::replay_insn(16, 7);
        std::uint32_t P           = TT_OP_MOVD2B(0, 28, ADDR_MOD_2, p_movd2b::MOV_4_ROWS, 12); // dst -= 16
        std::uint32_t IJKL        = lltt::replay_insn(24, 4);
        std::uint32_t Q           = TT_OP_MOVB2D(0, 28, ADDR_MOD_3, p_movb2d::MOV_4_ROWS, 12); // dst += 32
        std::uint32_t EFGHIJKLMNO = lltt::replay_insn(20, 11);

        // The following MOP config simply runs the above 7 instructions in order (when executed with zmask 0b10):
        ckernel_unpack_template tmp(true, true, EFGHIJKLM, EFGHI, ABCDEFG, P, /* skip A */ Q, /* B */ IJKL, /* skip B */ EFGHIJKLMNO);
        tmp.program();
    }
}

template <bool transpose_of_faces = true, bool is_32bit = false, bool is_fp32 = false>
inline void _llk_math_transpose_dest_init_()
{
    transpose_dest_configure_addrmod<is_32bit, is_fp32>();
    transpose_dest_configure_mop<transpose_of_faces, is_32bit, is_fp32>();

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
}

inline void _llk_math_transpose_dest_uninit_()
{
    // No state to restore - all states are transient or default
}
