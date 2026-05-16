// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// BH Fast-Untilize Pack — T5-B (tt-metal#42048 + #42049).
//
// Read math DEST layout (4 tiles, ct=4 FP16) emitted by `_llk_math_fast_tilize_*`:
//   Dst rows   0..63:  t0.F0 | t0.F1 | t1.F0 | t1.F1
//   Dst rows  64..127: t2.F0 | t2.F1 | t3.F0 | t3.F1
//   Dst rows 128..191: t0.F2 | t0.F3 | t1.F2 | t1.F3
//   Dst rows 192..255: t2.F2 | t2.F3 | t3.F2 | t3.F3
//
// Each PACR: ALL_INTF_ACTIVE + STRIDED_MODE → 4 interfaces read 4 face-rows
// at Dst rows R, R+16, R+32, R+48 (the 16-row interface stride), concatenated
// into 64 contiguous L1 datums.
//
// PACR sequence (64 PACRs total per block_ct_dim=4 call):
//   Phase 1 (top 16 strip rows):
//     PACR pair p=0..15: pack_row=p (block 0) + pack_row=p+64 (block 1)
//   Phase 2 (bottom 16 strip rows):
//     PACR pair p=0..15: pack_row=p+128 (block 2) + pack_row=p+192 (block 3)
//
// L1 writes are contiguous via PACR's internal datum counter — no per-PACR
// L1_Dest_addr cfg update needed. Single L1 cfg slot (SEC0_REG1).
//
// ADC strides (units: bytes; src_addr = Σ ch0.X*x + Y*y + Z*z + W*w divided
// by datum size to give datum offset; pack_row = datum_offset / FACE_C_DIM):
//   x_stride = bpe                  (per-datum)
//   y_stride = FACE_C_DIM * bpe     (1 face-row = 16 datums per y+=1)
//   z_stride = 64 * y_stride        (1 block of 64 face-rows per z+=1)
//   w_stride = 2 * z_stride         (1 phase = 128 face-rows per w+=1)
//
// AddrMod scheme:
//   ADDR_MOD_0: z_src.incr=1                       (post-PACR_pair_0: go to second block)
//   ADDR_MOD_1: y_src.incr=1, z_src.clr=1          (post-PACR_pair_1: advance row, reset block)
//
// Two MOP runs per call (phase 1 and phase 2). Between them: INCADCZW W+=1
// (jump to bottom-half blocks) + SETADCXY reset y/z.
//
// DOMAIN: block_ct_dim=4, num_faces=4, FP16 / bf16 output (matches
// fast_tilize math output size). Other shapes fall back to legacy
// `_llk_pack_untilize_` (T2+T3 wins still apply).

#pragma once

#include <cstdint>

#include "llk_pack.h"

namespace ckernel
{

inline void _llk_pack_fast_untilize_configure_addrmod_()
{
    // ADDR_MOD_0: after PACR reading block A, advance z to read block A+1.
    addr_mod_pack_t {.z_src = {.incr = 1}}.set(ADDR_MOD_0);

    // ADDR_MOD_1: after PACR reading block A+1, advance to next row in block A
    // (y+=1) and reset z to 0 (back to block A).
    addr_mod_pack_t {.y_src = {.incr = 1}, .z_src = {.clr = 1}}.set(ADDR_MOD_1);
}

// MOP body: 16 outer iterations × 2 inner PACRs.
// Inner PACR pair: PACR(AM0) writes 64 datums from block A, then PACR(AM1, Last=1)
// writes 64 datums from block A+1 and closes the row.
//
// MOP runs once per phase (top/bottom). 2 phases × 16 × 2 = 64 PACRs total.
inline void _llk_pack_fast_untilize_mop_config_()
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 16; // face_r_dim rows per phase
    constexpr std::uint32_t MOP_INNER_LOOP = 2;  // 2 PACRs per row pair (block A + block A+1)

    ckernel_template tmp(
        MOP_OUTER_LOOP,
        MOP_INNER_LOOP,
        // loop_op0: PACR reading current block, then advance via ADDR_MOD_0 (z+=1)
        TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_STRIDED_MODE,
            ADDR_MOD_0,
            p_pacr::ADDR_CNT_CTXT_0,
            0,
            p_pacr::ALL_INTF_ACTIVE,
            0,
            0,
            p_pacr::NO_CTXT_CTRL,
            0,
            0),
        // loop_op1: PACR reading next block, advance via ADDR_MOD_1 (y+=1, z=0)
        TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_STRIDED_MODE,
            ADDR_MOD_1,
            p_pacr::ADDR_CNT_CTXT_0,
            0,
            p_pacr::ALL_INTF_ACTIVE,
            0,
            0,
            p_pacr::NO_CTXT_CTRL,
            0,
            0));

    // Last inner of last outer = last PACR of the phase. Mark Last=1 to flush.
    std::uint32_t last_op = TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_STRIDED_MODE,
        ADDR_MOD_1,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        p_pacr::ALL_INTF_ACTIVE,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        1);
    tmp.set_last_outer_loop_instr(last_op);

    tmp.program();
}

template <DstSync Dst, bool is_fp32_dest_acc_en = false, std::uint32_t block_ct_dim = 4>
inline void _llk_pack_fast_untilize_init_(const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t num_faces = 4)
{
    static_assert(block_ct_dim == 4, "T5-B fast untilize only supports block_ct_dim=4 (matches fast_tilize math output)");

    if constexpr (is_fp32_dest_acc_en)
    {
        // Mirror fast_tilize init: reconfig pack_src to bf16-compat + Read_32b=0
        // for stride-16 stepping through DEST.
        constexpr std::uint32_t compat_src = ckernel::to_underlying(DataFormat::Float16_b);
        const std::uint32_t tile_size      = SCALE_DATUM_SIZE(pack_dst_format, TILE_C_DIM * TILE_R_DIM);
        reconfig_packer_data_format<is_fp32_dest_acc_en>(compat_src, pack_dst_format, tile_size, FACE_R_DIM, TILE_C_DIM, num_faces, /*partial_face=*/false);
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
        cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(0);
    }

    TTI_SETDMAREG(0, 0x000, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
    TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
    select_packer_dest_registers<Dst>();

    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);

    // Strides for our row/block/phase advance scheme.
    const std::uint32_t effective_src = is_fp32_dest_acc_en ? ckernel::to_underlying(DataFormat::Float16_b) : pack_src_format;
    const std::uint32_t x_stride      = (effective_src & 0x3) == ckernel::to_underlying(DataFormat::Float32)   ? 4
                                        : (effective_src & 0x3) == ckernel::to_underlying(DataFormat::Float16) ? 2
                                                                                                               : 1;
    // y_stride: 1 face-row of 16 datums per y+=1
    const std::uint32_t y_stride = FACE_C_DIM * x_stride;
    // z_stride: 64 face-rows per z+=1 (one block: 4 face-tile-groups of 16 rows)
    const std::uint32_t z_stride = 64 * FACE_C_DIM * x_stride;
    // w_stride: 2 blocks per w+=1 (one phase = 128 face-rows = top vs bottom half)
    const std::uint32_t w_stride = 128 * FACE_C_DIM * x_stride;

    TT_SETDMAREG(0, LOWER_HALFWORD(y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT), 0, LO_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT), 0, HI_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, LOWER_HALFWORD(z_stride << PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT), 0, LO_16(p_gpr_pack::TMP1));
    TT_SETDMAREG(0, UPPER_HALFWORD(w_stride << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT), 0, HI_16(p_gpr_pack::TMP1));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);

    _llk_pack_fast_untilize_configure_addrmod_();
    _llk_pack_fast_untilize_mop_config_();
}

// One call processes one block of block_ct_dim=4 tiles.
// Output: 4 tiles' worth of RM strip starting at `address` (in 16B units).
template <std::uint32_t block_ct_dim = 4>
inline void _llk_pack_fast_untilize_block_(const std::uint32_t address, [[maybe_unused]] const std::uint32_t num_faces = 4)
{
    static_assert(block_ct_dim == 4, "T5-B fast untilize only supports block_ct_dim=4");

    program_packer_destination(address);

    // Reset all ADC counters to 0.
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0011);

    // Phase 1: top 16 strip rows (blocks 0 + 1 at Dst rows 0..127).
    ckernel_template::run();

    // Phase boundary: jump to bottom half via w+=1.
    // After phase 1: y=16, z=0, w=0 → need y=0, z=0, w=1.
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011); // reset y
    TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 1);         // w+=1 → jump 128 rows

    // Phase 2: bottom 16 strip rows (blocks 2 + 3 at Dst rows 128..255).
    ckernel_template::run();
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_fast_untilize_uninit_(const std::uint32_t pack_dst_format, const std::uint32_t pack_src_format = (std::uint32_t)DataFormat::Float16_b)
{
    if constexpr (is_fp32_dest_acc_en)
    {
        const std::uint32_t tile_size = SCALE_DATUM_SIZE(pack_dst_format, TILE_C_DIM * TILE_R_DIM);
        reconfig_packer_data_format<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, tile_size, FACE_R_DIM, TILE_C_DIM, 4, /*partial_face=*/false);
    }
    set_packer_strides<PackMode::Default>(pack_src_format, TILE_C_DIM);
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
    _llk_pack_init_<PackMode::Default, false, false>(FACE_R_DIM, TILE_C_DIM, 4, 1);
}

} // namespace ckernel
