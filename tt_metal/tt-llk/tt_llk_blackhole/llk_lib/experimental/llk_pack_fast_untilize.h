// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// BH Fast-Untilize Pack - T5-B (tt-metal#42048 + #42049).
//
// Read math DEST layout (4 tiles, ct=4; 16-bit or native fp32 DEST) emitted by
// `_llk_math_fast_untilize_*`:
//   Dst rows   0..63:  t0.F0 | t0.F1 | t1.F0 | t1.F1
//   Dst rows  64..127: t2.F0 | t2.F1 | t3.F0 | t3.F1
//   Dst rows 128..191: t0.F2 | t0.F3 | t1.F2 | t1.F3
//   Dst rows 192..255: t2.F2 | t2.F3 | t3.F2 | t3.F3
//
// Each PACR: ALL_INTF_ACTIVE + STRIDED_MODE makes 4 interfaces read 4 face-rows
// at Dst rows R, R+16, R+32, R+48 (the 16-row interface stride), concatenated
// into 64 contiguous L1 datums.
//
// Contiguous PACR sequence (32/64 PACRs total per unit_dim=2/3|4 call):
//   Phase 1 emits top strip rows with Last=0.
//   Phase 2 emits bottom strip rows with Last=1, closing one contiguous L1 stream.
//
// L1 writes are contiguous via PACR's internal datum counter; no per-PACR
// L1_Dest_addr cfg update needed. Single L1 cfg slot (SEC0_REG1). This is only
// valid when the 4-tile chunk is the full row.
//
// Wider rows use a row-strided stream: each chunk row closes with Last=1, then
// CFGSHIFTMASK advances L1_Dest_addr by the full output row stride. The direct
// row sequence is kept as a fallback while the MOP/replay variant is re-tested.
//
// ADC strides (units: bytes; src_addr = sum(ch0.X*x + Y*y + Z*z + W*w) divided
// by datum size to give datum offset; pack_row = datum_offset / FACE_C_DIM):
//   x_stride = bpe                  (per-datum)
//   y_stride = FACE_C_DIM * bpe     (1 face-row = 16 datums per y+=1)
//   z_stride = 64 * y_stride        (1 block of 64 face-rows per z+=1)
//
// AddrMod scheme:
//   ADDR_MOD_0: z_src.incr=1                       (post-PACR_pair_0: go to second block)
//   ADDR_MOD_1: y_src.incr=1, z_src.clr=1          (post-PACR_pair_1: advance row, reset block)
//
// The contiguous path uses two MOP runs per call (phase 1 and phase 2).
// L1_Dest_addr is programmed once at the base address; phase 1 intentionally
// leaves the pack stream open so phase 2 naturally continues at row 16. The
// phase source half is selected by reprogramming the active pack DEST target
// offset: active_half + 128 for top rows, then active_half + 0 for bottom rows.
//
// DOMAIN: unit_dim=2/3/4, num_faces=4, SyncHalf, fp16/fp32 output from supported
// fast-untilize math layouts. Other shapes fall back to legacy `_llk_pack_untilize_`
// (T2+T3 wins still apply).

#pragma once

#include <cstdint>

#include "llk_pack.h"

#ifndef FAST_UNTILIZE_STRIDED_MOP_REPLAY
#define FAST_UNTILIZE_STRIDED_MOP_REPLAY 1
#endif

namespace ckernel
{

constexpr std::uint32_t FAST_UNTILIZE_ROW_ADVANCE_REPLAY_OFFSET = ckernel::packer::replay_buf_offset;
constexpr std::uint32_t FAST_UNTILIZE_ROW_ADVANCE_REPLAY_LEN    = 2;

inline void _llk_pack_fast_untilize_configure_addrmod_()
{
    // ADDR_MOD_0: after PACR reading block A, advance z to read block A+1.
    addr_mod_pack_t {.z_src = {.incr = 1}}.set(ADDR_MOD_0);

    // ADDR_MOD_1: after PACR reading block A+1, advance to next row in block A
    // (y+=1) and reset z to 0 (back to block A).
    addr_mod_pack_t {.y_src = {.incr = 1}, .z_src = {.clr = 1}}.set(ADDR_MOD_1);
}

// MOP body: 16 outer iterations x 2 inner PACRs.
// Inner PACR pair: PACR(AM0) writes 64 datums from block A, then PACR(AM1)
// writes 64 datums from block A+1.
//
// MOP runs once per phase (top/bottom). 2 phases x 16 x 2 = 64 PACRs total.
inline std::uint32_t _llk_pack_fast_untilize_row_pacr_(
    const std::uint32_t addr_mod, const std::uint32_t read_intf_sel, const std::uint32_t last, const std::uint32_t concat = 0)
{
    return TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_STRIDED_MODE,
        addr_mod,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        read_intf_sel,
        0,
        concat,
        p_pacr::NO_CTXT_CTRL,
        0,
        last);
}

inline void _llk_pack_fast_untilize_mop_config_(const std::uint32_t unit_dim = 4, const bool last = true)
{
    LLK_ASSERT(unit_dim >= 2 && unit_dim <= 4, "fast_untilize pack supports unit_dim 2, 3, or 4");

    constexpr std::uint32_t MOP_OUTER_LOOP = 16; // face_r_dim rows per phase
    constexpr std::uint32_t MOP_INNER_LOOP = 1;  // one PACR pair per strip row

    if (unit_dim == 2)
    {
        ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, _llk_pack_fast_untilize_row_pacr_(ADDR_MOD_1, p_pacr::ALL_INTF_ACTIVE, 0));
        tmp.set_last_outer_loop_instr(_llk_pack_fast_untilize_row_pacr_(ADDR_MOD_1, p_pacr::ALL_INTF_ACTIVE, last ? 1 : 0));
        tmp.program();
    }
    else
    {
        const std::uint32_t tail_intf = unit_dim == 3 ? p_pacr::TWO_INTFS_ACTIVE : p_pacr::ALL_INTF_ACTIVE;
        ckernel_template tmp(
            MOP_OUTER_LOOP,
            MOP_INNER_LOOP,
            _llk_pack_fast_untilize_row_pacr_(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0),
            _llk_pack_fast_untilize_row_pacr_(ADDR_MOD_1, tail_intf, 0));
        tmp.set_last_outer_loop_instr(_llk_pack_fast_untilize_row_pacr_(ADDR_MOD_1, tail_intf, last ? 1 : 0));
        tmp.program();
    }
}

inline void _llk_pack_fast_untilize_load_row_advance_replay_()
{
    load_replay_buf(
        FAST_UNTILIZE_ROW_ADVANCE_REPLAY_OFFSET,
        FAST_UNTILIZE_ROW_ADVANCE_REPLAY_LEN,
        []
        {
            TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
            TTI_NOP;
        });
}

inline void _llk_pack_fast_untilize_strided_mop_config_(const std::uint32_t unit_dim)
{
    LLK_ASSERT(unit_dim >= 2 && unit_dim <= 4, "fast_untilize strided pack supports unit_dim 2, 3, or 4");

    constexpr std::uint32_t MOP_OUTER_LOOP = 16;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    if (unit_dim == 2)
    {
        ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, _llk_pack_fast_untilize_row_pacr_(ADDR_MOD_1, p_pacr::ALL_INTF_ACTIVE, 1));
        tmp.set_end_op(lltt::replay_insn(FAST_UNTILIZE_ROW_ADVANCE_REPLAY_OFFSET, FAST_UNTILIZE_ROW_ADVANCE_REPLAY_LEN));
        tmp.program();
    }
    else
    {
        const std::uint32_t tail_intf = unit_dim == 3 ? p_pacr::TWO_INTFS_ACTIVE : p_pacr::ALL_INTF_ACTIVE;
        ckernel_template tmp(
            MOP_OUTER_LOOP,
            MOP_INNER_LOOP,
            _llk_pack_fast_untilize_row_pacr_(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0, 1),
            _llk_pack_fast_untilize_row_pacr_(ADDR_MOD_1, tail_intf, 1));
        tmp.set_end_op(lltt::replay_insn(FAST_UNTILIZE_ROW_ADVANCE_REPLAY_OFFSET, FAST_UNTILIZE_ROW_ADVANCE_REPLAY_LEN));
        tmp.program();
    }
}

inline void _llk_pack_fast_untilize_strided_direct_row_(const std::uint32_t unit_dim)
{
    if (unit_dim == 2)
    {
        TTI_PACR(
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
    }
    else if (unit_dim == 3)
    {
        TTI_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_STRIDED_MODE,
            ADDR_MOD_0,
            p_pacr::ADDR_CNT_CTXT_0,
            0,
            p_pacr::ALL_INTF_ACTIVE,
            0,
            1,
            p_pacr::NO_CTXT_CTRL,
            0,
            0);
        TTI_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_STRIDED_MODE,
            ADDR_MOD_1,
            p_pacr::ADDR_CNT_CTXT_0,
            0,
            p_pacr::TWO_INTFS_ACTIVE,
            0,
            0,
            p_pacr::NO_CTXT_CTRL,
            0,
            1);
    }
    else
    {
        TTI_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_STRIDED_MODE,
            ADDR_MOD_0,
            p_pacr::ADDR_CNT_CTXT_0,
            0,
            p_pacr::ALL_INTF_ACTIVE,
            0,
            1,
            p_pacr::NO_CTXT_CTRL,
            0,
            0);
        TTI_PACR(
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
    }

    TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
    TTI_NOP;
}

template <DstSync Dst, bool is_fp32_dest_acc_en = false, std::uint32_t block_ct_dim = 4, std::uint32_t full_ct_dim = block_ct_dim>
inline void _llk_pack_fast_untilize_init_(const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t num_faces = 4)
{
    static_assert(block_ct_dim >= 2 && block_ct_dim <= 4, "T5-B fast untilize supports block_ct_dim 2, 3, or 4");
    LLK_ASSERT(num_faces == 4, "fast_untilize pack only supports four-face tiles");

    TTI_SETDMAREG(0, 0x000, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
    TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
    select_packer_dest_registers<Dst>();

    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);

    // Strides for our row/block/phase advance scheme.
    const std::uint32_t x_stride = (pack_src_format & 0x3) == ckernel::to_underlying(DataFormat::Float32)   ? 4
                                   : (pack_src_format & 0x3) == ckernel::to_underlying(DataFormat::Float16) ? 2
                                                                                                            : 1;
    // y_stride: 1 face-row of 16 datums per y+=1
    const std::uint32_t y_stride = FACE_C_DIM * x_stride;
    // z_stride: 64 face-rows per z+=1 (one block: 4 face-tile-groups of 16 rows)
    const std::uint32_t z_stride = 64 * FACE_C_DIM * x_stride;
    // w_stride: retained for consistency with the pack address generator setup.
    const std::uint32_t w_stride = 128 * FACE_C_DIM * x_stride;

    TT_SETDMAREG(0, LOWER_HALFWORD(y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT), 0, LO_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT), 0, HI_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, LOWER_HALFWORD(z_stride << PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT), 0, LO_16(p_gpr_pack::TMP1));
    TT_SETDMAREG(0, UPPER_HALFWORD(w_stride << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT), 0, HI_16(p_gpr_pack::TMP1));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);

    _llk_pack_fast_untilize_configure_addrmod_();
    if constexpr (full_ct_dim > block_ct_dim)
    {
        const std::uint32_t output_row_stride = SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * TILE_C_DIM);
        TT_SETDMAREG(0, LOWER_HALFWORD(output_row_stride / 16), 0, LO_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));
        TT_SETDMAREG(0, UPPER_HALFWORD(output_row_stride / 16), 0, HI_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR_OFFSET, 0, SCRATCH_SEC2_val_ADDR32);
        TTI_NOP;
        _llk_pack_fast_untilize_load_row_advance_replay_();
    }
}

template <DstSync Dst, std::uint32_t phase_offset>
inline void _llk_pack_fast_untilize_select_phase_()
{
    if constexpr (Dst == DstSync::SyncFull)
    {
        TTI_SETDMAREG(0, phase_offset, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
    }
    else
    {
        static_assert(Dst == DstSync::SyncHalf);
        TTI_SETDMAREG(0, phase_offset, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
        TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + phase_offset, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
    }

    select_packer_dest_registers<Dst>();
}

// One call processes one block of block_ct_dim=4 tiles.
// Output: 4 tiles' worth of RM strip starting at `address` (in 16B units).
template <std::uint32_t block_ct_dim = 4, DstSync Dst = DstSync::SyncHalf>
inline void _llk_pack_fast_untilize_block_(const std::uint32_t address, const std::uint32_t unit_dim = block_ct_dim, const std::uint32_t num_faces = 4)
{
    static_assert(block_ct_dim >= 2 && block_ct_dim <= 4, "T5-B fast untilize supports block_ct_dim 2, 3, or 4");
    LLK_ASSERT(unit_dim >= 2 && unit_dim <= block_ct_dim, "fast_untilize pack unit_dim must be in [2, block_ct_dim]");
    LLK_ASSERT(num_faces == 4, "fast_untilize pack only supports four-face tiles");

    program_packer_destination(address);

    // Phase 1 emits top strip rows and keeps the pack stream open.
    _llk_pack_fast_untilize_select_phase_<Dst, 128>();
    _llk_pack_fast_untilize_mop_config_(unit_dim, false);
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    ckernel_template::run();

    // Phase 2 emits bottom strip rows and closes the stream.
    ckernel::mop_sync();
    _llk_pack_fast_untilize_select_phase_<Dst, 0>();
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    _llk_pack_fast_untilize_mop_config_(unit_dim, true);
    ckernel_template::run();
}

// One call processes one block_ct_dim=4 chunk inside a wider row.
// Output address points at this chunk's row-0 column in the row-major tensor.
template <std::uint32_t block_ct_dim = 4, std::uint32_t full_ct_dim = block_ct_dim, DstSync Dst = DstSync::SyncHalf>
inline void _llk_pack_fast_untilize_block_strided_(
    const std::uint32_t address, const std::uint32_t unit_dim, [[maybe_unused]] std::uint32_t& prev_unit_dim, const std::uint32_t num_faces = 4)
{
    static_assert(block_ct_dim >= 2 && block_ct_dim <= 4, "T5-B fast untilize strided path supports block_ct_dim 2, 3, or 4");
    static_assert(full_ct_dim > block_ct_dim, "Use the contiguous fast_untilize block when the chunk is the full row");
    LLK_ASSERT(unit_dim >= 2 && unit_dim <= block_ct_dim, "fast_untilize pack unit_dim must be in [2, block_ct_dim]");
    LLK_ASSERT(num_faces == 4, "fast_untilize pack only supports four-face tiles");

#if FAST_UNTILIZE_STRIDED_MOP_REPLAY
    if (unit_dim != prev_unit_dim)
    {
        _llk_pack_fast_untilize_strided_mop_config_(unit_dim);
        prev_unit_dim = unit_dim;
    }
#endif

    program_packer_destination(address);

#if FAST_UNTILIZE_STRIDED_MOP_REPLAY
    _llk_pack_fast_untilize_select_phase_<Dst, 128>();
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    ckernel_template::run();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);

    // After 16 row-stride end-ops the destination is already at output row 16.
    _llk_pack_fast_untilize_select_phase_<Dst, 0>();
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    ckernel_template::run();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
#else
    // Phase 1 emits rows 0..15. Each row is closed and L1_Dest_addr is advanced
    // by the full output row stride from scratch.
    _llk_pack_fast_untilize_select_phase_<Dst, 128>();
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    for (std::uint32_t row = 0; row < FACE_R_DIM; row++)
    {
        _llk_pack_fast_untilize_strided_direct_row_(unit_dim);
    }
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);

    // After 16 row-stride end-ops the destination is already at output row 16.
    _llk_pack_fast_untilize_select_phase_<Dst, 0>();
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    for (std::uint32_t row = 0; row < FACE_R_DIM; row++)
    {
        _llk_pack_fast_untilize_strided_direct_row_(unit_dim);
    }
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
#endif
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_fast_untilize_uninit_(
    [[maybe_unused]] const std::uint32_t pack_dst_format, const std::uint32_t pack_src_format = (std::uint32_t)DataFormat::Float16_b)
{
    set_packer_strides<PackMode::Default>(pack_src_format, TILE_C_DIM);
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
    _llk_pack_init_<PackMode::Default, false, false>(FACE_R_DIM, TILE_C_DIM, 4, 1);
}

} // namespace ckernel
