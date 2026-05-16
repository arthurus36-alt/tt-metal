// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// BH Fast-Untilize Pack — WIP skeleton (T5-B, tt-metal#42048 + #42049).
//
// Goal: produce row-major L1 strip output from a swizzled DEST layout
// emitted by `_llk_math_fast_tilize_*` (no new math LLK needed).
//
// Math DEST layout (4 tiles, ct=4 FP16, mirrors fast_tilize):
//   Dst rows   0..63:   t0.F0 || t0.F1 || t1.F0 || t1.F1 (4×16 rows)
//   Dst rows  64..127:  t2.F0 || t2.F1 || t3.F0 || t3.F1
//   Dst rows 128..191:  t0.F2 || t0.F3 || t1.F2 || t1.F3
//   Dst rows 192..255:  t2.F2 || t2.F3 || t3.F2 || t3.F3
//
// Pack output (RM strip, 32 rows × 128 datums = 4096 datums = 4 tile-sizes):
//   L1 row 0 cols 0..63: PACR_0 (pack_row=0,  4-intf STRIDED) reads
//                        t0.F0[0], t0.F1[0], t1.F0[0], t1.F1[0] → concat 64 datums
//   L1 row 0 cols 64..127: PACR_1 (pack_row=64) reads
//                        t2.F0[0], t2.F1[0], t3.F0[0], t3.F1[0] → concat 64 datums
//   L1 row 1 cols 0..63: PACR_2 (pack_row=1) ... etc.
//
// Total: 64 PACRs per 4-tile call. L1 written naturally contiguous via
// PACR's internal datum advance — no per-tile L1_Dest_addr update needed.
//
// DOMAIN RESTRICTION: block_ct_dim=4 only (matches fast_tilize math output).
// Wider blocks would need either multiple calls or a larger math output.

#pragma once

#include <cstdint>

#include "llk_pack.h"

namespace ckernel
{

// AddrMod plan (provisional — needs silicon validation):
//   ADDR_MOD_0: y_src.incr=1            (advance Dst row within a 16-row block)
//   ADDR_MOD_1: y_src={clr=1,cr=1}      (close top-half: reset row, ready for bottom-half)
//   ADDR_MOD_2: TBD — face-pair boundary (F0/F1 group → F2/F3 group, jump z)
//   ADDR_MOD_3: TBD — block boundary (top-half → bottom-half quadrants)
inline void _llk_pack_fast_untilize_configure_addrmod_()
{
    addr_mod_pack_t {.y_src = {.incr = 1}}.set(ADDR_MOD_0);
    addr_mod_pack_t {.y_src = {.clr = 1, .cr = 1}, .z_src = {.clr = 1}}.set(ADDR_MOD_1);
    // ADDR_MOD_2/3 placeholders — populate when MOP body is written.
    addr_mod_pack_t {.y_src = {.incr = 1}}.set(ADDR_MOD_2);
    addr_mod_pack_t {.y_src = {.incr = 1}}.set(ADDR_MOD_3);
}

// MOP body placeholder. Final structure (to validate on silicon):
//   outer = face_r_dim (16 rows per face-row group)
//   inner = 2 (top-half + bottom-half-of-row PACR pair)
//   each PACR: ALL_INTF_ACTIVE + DST_ACCESS_STRIDED_MODE
//   between outer iters: nothing (Dst row advance via AddrMod, L1 via PACR internal)
//   between top-half and bottom-half phases: jump pack_row by 128 (Dst rows 0..127 → 128..255)
//
// This skeleton just builds; functional verification in T5.4.
template <std::uint32_t block_ct_dim>
inline void _llk_pack_fast_untilize_mop_config_()
{
    static_assert(block_ct_dim == 4, "T5-B fast untilize only supports block_ct_dim=4 (matches fast_tilize math output)");
    // TODO: write MOP body.
}

template <DstSync Dst, bool is_fp32_dest_acc_en = false, std::uint32_t block_ct_dim = 4>
inline void _llk_pack_fast_untilize_init_(const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t num_faces = 4)
{
    static_assert(block_ct_dim == 4, "T5-B fast untilize only supports block_ct_dim=4");

    if constexpr (is_fp32_dest_acc_en)
    {
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

    // Strides — same as fast_tilize for now (Dst-side input strides). Verify on silicon.
    const std::uint32_t effective_src = is_fp32_dest_acc_en ? ckernel::to_underlying(DataFormat::Float16_b) : pack_src_format;
    const std::uint32_t x_stride      = (effective_src & 0x3) == ckernel::to_underlying(DataFormat::Float32)   ? 4
                                        : (effective_src & 0x3) == ckernel::to_underlying(DataFormat::Float16) ? 2
                                                                                                               : 1;
    std::uint32_t y_stride            = 64 * FACE_C_DIM * x_stride;
    std::uint32_t z_stride            = FACE_C_DIM * x_stride;
    std::uint32_t w_stride            = 2 * FACE_C_DIM * x_stride;

    TT_SETDMAREG(0, LOWER_HALFWORD(y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT), 0, LO_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT), 0, HI_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, LOWER_HALFWORD(z_stride << PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT), 0, LO_16(p_gpr_pack::TMP1));
    TT_SETDMAREG(0, UPPER_HALFWORD(w_stride << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT), 0, HI_16(p_gpr_pack::TMP1));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);

    _llk_pack_fast_untilize_configure_addrmod_();
    _llk_pack_fast_untilize_mop_config_<block_ct_dim>();
}

template <std::uint32_t block_ct_dim = 4>
inline void _llk_pack_fast_untilize_block_(const std::uint32_t tile_index, const std::uint32_t address, const std::uint32_t num_faces = 4)
{
    static_assert(block_ct_dim == 4, "T5-B fast untilize only supports block_ct_dim=4");
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    program_packer_destination(address);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    // ckernel::ckernel_template::run();  // TODO: once MOP body is implemented
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
