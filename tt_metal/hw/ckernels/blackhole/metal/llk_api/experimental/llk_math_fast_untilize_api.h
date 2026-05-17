// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/llk_math_fast_untilize.h"
#ifndef ENV_LLK_INFRA
#include "llk_math_common_api.h"
#endif

/*************************************************************************
 * LLK MATH FAST UNTILIZE (BH)
 *************************************************************************/

template <bool is_fp32_dest_acc_en>
inline void llk_math_fast_untilize_init_with_format(const std::uint32_t unpack_dst_format) {
    ckernel::_llk_math_fast_untilize_init_<is_fp32_dest_acc_en>(unpack_dst_format);
}

template <bool is_fp32_dest_acc_en>
inline void llk_math_fast_untilize_block_with_format(
    const std::uint32_t dst_index,
    const std::uint32_t unpack_dst_format,
    const std::uint32_t block_ct_dim,
    const std::uint32_t num_faces = 4) {
    ckernel::_llk_math_fast_untilize_block_<is_fp32_dest_acc_en>(dst_index, unpack_dst_format, block_ct_dim, num_faces);
}

template <bool is_fp32_dest_acc_en>
inline void llk_math_fast_untilize_uninit_with_format(const std::uint32_t unpack_dst_format) {
    ckernel::_llk_math_fast_untilize_uninit_<is_fp32_dest_acc_en>(unpack_dst_format);
}

#ifndef ENV_LLK_INFRA
template <bool is_fp32_dest_acc_en>
inline void llk_math_fast_untilize_init(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    llk_math_fast_untilize_init_with_format<is_fp32_dest_acc_en>(unpack_dst_format[operand_id]);
}

template <bool is_fp32_dest_acc_en>
inline void llk_math_fast_untilize_block(
    const std::uint32_t dst_index,
    const std::uint32_t operand,
    const std::uint32_t block_ct_dim,
    const std::uint32_t num_faces = 4) {
    const std::uint32_t operand_id = get_operand_id(operand);
    llk_math_fast_untilize_block_with_format<is_fp32_dest_acc_en>(
        dst_index, unpack_dst_format[operand_id], block_ct_dim, num_faces);
}

template <bool is_fp32_dest_acc_en>
inline void llk_math_fast_untilize_uninit(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    llk_math_fast_untilize_uninit_with_format<is_fp32_dest_acc_en>(unpack_dst_format[operand_id]);
}
#endif
