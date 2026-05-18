// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/llk_unpack_fast_untilize.h"
#ifndef ENV_LLK_INFRA
#include "llk_unpack_common_api.h"
#endif

/*************************************************************************
 * LLK UNPACK FAST UNTILIZE (BH)
 *************************************************************************/

template <bool is_fp32_dest_acc_en>
inline void llk_unpack_fast_untilize_init_with_formats(
    const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format, const std::uint32_t init_unit_dim) {
    ckernel::_llk_unpack_fast_untilize_init_<is_fp32_dest_acc_en>(unpack_src_format, unpack_dst_format, init_unit_dim);
}

template <bool is_fp32_dest_acc_en>
inline void llk_unpack_fast_untilize_reinit_unit_dim(const std::uint32_t unit_dim) {
    ckernel::_llk_unpack_fast_untilize_reinit_unit_dim_<is_fp32_dest_acc_en>(unit_dim);
}

inline void llk_unpack_fast_untilize_block_at_address(const std::uint32_t address, const std::uint32_t unit_dim = 4) {
    ckernel::_llk_unpack_fast_untilize_block_(address, unit_dim);
}

inline void llk_unpack_fast_untilize_bfp_block_at_address(
    const std::uint32_t address, const std::uint32_t tile_stride_16B, const std::uint32_t unit_dim) {
    ckernel::_llk_unpack_fast_untilize_bfp_block_(address, tile_stride_16B, unit_dim);
}

#ifndef ENV_LLK_INFRA
template <bool is_fp32_dest_acc_en>
inline void llk_unpack_fast_untilize_init(const std::uint32_t operand, const std::uint32_t init_unit_dim) {
    const std::uint32_t operand_id = get_operand_id(operand);
    llk_unpack_fast_untilize_init_with_formats<is_fp32_dest_acc_en>(
        unpack_src_format[operand_id], unpack_dst_format[operand_id], init_unit_dim);
}

inline void llk_unpack_fast_untilize_block(
    const std::uint32_t operand, const std::uint32_t tile_index, const std::uint32_t unit_dim = 4) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t address = get_local_cb_interface(operand_id).fifo_rd_ptr +
                                  get_local_cb_interface(operand_id).fifo_page_size * tile_index - 1;
    llk_unpack_fast_untilize_block_at_address(address, unit_dim);
}

inline void llk_unpack_fast_untilize_bfp_block(
    const std::uint32_t operand, const std::uint32_t tile_index, const std::uint32_t unit_dim) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t address = get_local_cb_interface(operand_id).fifo_rd_ptr +
                                  get_local_cb_interface(operand_id).fifo_page_size * tile_index - 1;
    llk_unpack_fast_untilize_bfp_block_at_address(address, get_local_cb_interface(operand_id).fifo_page_size, unit_dim);
}
#endif

inline void llk_unpack_fast_untilize_uninit() { ckernel::_llk_unpack_fast_untilize_uninit_(); }
