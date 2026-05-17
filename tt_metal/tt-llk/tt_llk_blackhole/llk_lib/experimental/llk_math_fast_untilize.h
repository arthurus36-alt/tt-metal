// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// BH Fast-Untilize Math - T5 MVP.
//
// Consumes 2/3/4 fast_untilize unpack tiles (four SrcA dvalids per tile) and
// writes a Dst layout matching llk_pack_fast_untilize.h:
//
//   Dst rows   0..63:  t0.F0 | t0.F1 | t1.F0 | t1.F1
//   Dst rows  64..127: t2.F0 | t2.F1 | t3.F0 | t3.F1
//   Dst rows 128..191: t0.F2 | t0.F3 | t1.F2 | t1.F3
//   Dst rows 192..255: t2.F2 | t2.F3 | t3.F2 | t3.F3
//
// Each Dst row is one 16-datum face row. The packer then uses
// ALL_INTF_ACTIVE + DST_ACCESS_STRIDED_MODE to read rows R, R+16, R+32, R+48
// and emit a contiguous 64-datum chunk.

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "cmath_common.h"
#include "llk_math_common.h"

namespace ckernel
{

inline void _llk_math_fast_untilize_configure_addrmod_()
{
    // Explicit MOVA2D source/destination row immediates drive all placement.
    // Keep RWCs unchanged so each face copy starts from SrcA rows 0/8 and
    // writes to the immediate Dst row selected below.
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_4);
}

template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_fast_untilize_init_([[maybe_unused]] const std::uint32_t unpack_dst_format)
{
    // Same Dst read remap required by pack_untilize/fast_tilize paths so packer
    // STRIDED_MODE sees a 16-row stride through Dst.
    _llk_math_reconfig_remap_(true);

    if constexpr (is_fp32_dest_acc_en)
    {
        // Keep this MVP aligned with fast_tilize: MOVA2D does not safely write
        // the 32-bit Dst view on BH, so use the 16-bit Dst data path.
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
    }

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    _llk_math_fast_untilize_configure_addrmod_();
}

inline void _llk_math_fast_untilize_copy_face_(const std::uint32_t dst_row)
{
    TTI_MOVA2D(0, 0, ADDR_MOD_4, p_mova2d::MOV_8_ROWS, dst_row);
    TTI_MOVA2D(0, 8, ADDR_MOD_4, p_mova2d::MOV_8_ROWS, dst_row + 8);

    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_AB);
}

inline void _llk_math_fast_untilize_copy_tile_(const std::uint32_t tile_index)
{
    const std::uint32_t top_row    = tile_index * 32;
    const std::uint32_t bottom_row = 128 + tile_index * 32;

    // The unpacker presents each tile as F2, F3, F0, F1.
    _llk_math_fast_untilize_copy_face_(bottom_row);
    _llk_math_fast_untilize_copy_face_(bottom_row + 16);
    _llk_math_fast_untilize_copy_face_(top_row);
    _llk_math_fast_untilize_copy_face_(top_row + 16);
}

template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_fast_untilize_block_(
    const std::uint32_t dst_index,
    [[maybe_unused]] const std::uint32_t unpack_dst_format,
    [[maybe_unused]] const std::uint32_t block_ct_dim = 4,
    [[maybe_unused]] const std::uint32_t num_faces    = 4)
{
    static_assert(!is_fp32_dest_acc_en, "T5 fast-untilize MVP only supports 16-bit Dst");
    LLK_ASSERT(block_ct_dim >= 2 && block_ct_dim <= 4, "T5 fast-untilize supports block_ct_dim 2, 3, or 4");
    LLK_ASSERT(num_faces == 4, "T5 fast-untilize MVP only supports four-face tiles");

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);

    _llk_math_fast_untilize_copy_tile_(0);
    _llk_math_fast_untilize_copy_tile_(1);
    if (block_ct_dim >= 3)
    {
        _llk_math_fast_untilize_copy_tile_(2);
    }
    if (block_ct_dim >= 4)
    {
        _llk_math_fast_untilize_copy_tile_(3);
    }

    math::clear_dst_reg_addr();
}

template <bool is_fp32_dest_acc_en>
inline void _llk_math_fast_untilize_uninit_([[maybe_unused]] const std::uint32_t unpack_dst_format)
{
    if constexpr (is_fp32_dest_acc_en)
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
    }

    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 0},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_2);
}

} // namespace ckernel
