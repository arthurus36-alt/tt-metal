// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// BH Fast-Untilize Unpack - T5-B experimental path.
//
// This is the standard unpack_A face stream specialized for fast_untilize:
// emit only real SrcA dvalids and skip the generic unpack_A zero SrcB sideband.
// Math can then clear only SrcA, reducing dvalid traffic for the focused
// fast-untilize path. unit_dim=1 is used for compressed BFP tile streams where
// each tile must be addressed explicitly to skip exponent sections.

#pragma once

#include <cstdint>

#include "llk_unpack_A.h"

namespace ckernel
{

template <bool is_fp32_dest_acc_en = false>
inline void _llk_unpack_fast_untilize_mop_config_(const std::uint32_t unit_dim = 4)
{
    LLK_ASSERT(unit_dim >= 1 && unit_dim <= 4, "fast_untilize unpack supports unit_dim 1, 2, 3, or 4");

    static constexpr std::uint32_t unpack_srca            = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb_set_dvalid = TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);

    const std::uint32_t outerloop     = unit_dim * 4;
    constexpr std::uint32_t innerloop = 1;

    if constexpr (is_fp32_dest_acc_en)
    {
        // Native fp32 DEST math uses ELWADD as a SrcA->DEST copy, which needs
        // a valid zero SrcB face alongside each real SrcA face.
        ckernel_template tmp(outerloop, innerloop, unpack_srca, unpack_srcb_set_dvalid);
        tmp.program();
    }
    else
    {
        ckernel_template tmp(outerloop, innerloop, unpack_srca);
        tmp.program();
    }
}

template <bool is_fp32_dest_acc_en = false>
inline void _llk_unpack_fast_untilize_init_(const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format, const std::uint32_t init_unit_dim = 4)
{
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(0, 0, FACE_R_DIM, 4, unpack_src_format, unpack_dst_format);
    _llk_unpack_fast_untilize_mop_config_<is_fp32_dest_acc_en>(init_unit_dim);
}

template <bool is_fp32_dest_acc_en = false>
inline void _llk_unpack_fast_untilize_reinit_unit_dim_(const std::uint32_t unit_dim)
{
    _llk_unpack_fast_untilize_mop_config_<is_fp32_dest_acc_en>(unit_dim);
}

inline void _llk_unpack_fast_untilize_block_(const std::uint32_t address, [[maybe_unused]] const std::uint32_t unit_dim = 4)
{
    LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");
    LLK_ASSERT(unit_dim >= 1 && unit_dim <= 4, "fast_untilize unpack supports unit_dim 1, 2, 3, or 4");

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);

    const std::uint32_t upk0_reg = (unp_cfg_context == 0) ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32;
    cfg[upk0_reg]                = address;

    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    ckernel::ckernel_template::run();

    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}

inline void _llk_unpack_fast_untilize_bfp_block_(const std::uint32_t address, const std::uint32_t tile_stride_16B, const std::uint32_t unit_dim)
{
    LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");
    LLK_ASSERT(is_valid_L1_address(address + tile_stride_16B * (unit_dim - 1)), "L1 address must be in valid L1 memory region");
    LLK_ASSERT(tile_stride_16B > 0, "fast_untilize BFP tile stride must be greater than zero");
    LLK_ASSERT(unit_dim >= 1 && unit_dim <= 4, "fast_untilize unpack supports unit_dim 1, 2, 3, or 4");

    if (unit_dim == 1)
    {
        _llk_unpack_fast_untilize_block_(address, 1);
        return;
    }

    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);

    const bool use_context_0     = unp_cfg_context == 0;
    const std::uint32_t upk0_reg = use_context_0 ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32;
    cfg[upk0_reg]                = address;
    cfg[SCRATCH_SEC0_val_ADDR32] = tile_stride_16B;

    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    for (std::uint32_t tile = 0; tile < unit_dim; tile++)
    {
        TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
        ckernel::ckernel_template::run();

        if (tile + 1 < unit_dim)
        {
            if (use_context_0)
            {
                TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_address_ADDR32);
            }
            else
            {
                TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
            }
            TTI_NOP;
        }
    }

    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}

inline void _llk_unpack_fast_untilize_uninit_()
{
    _llk_unpack_A_uninit_<BroadcastType::NONE>(FACE_R_DIM);
}

} // namespace ckernel
