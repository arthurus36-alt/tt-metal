// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/llk_pack_fast_untilize.h"
#ifndef ENV_LLK_INFRA
#include "llk_pack_common_api.h"
#endif

/*************************************************************************
 * LLK PACK FAST UNTILIZE (BH)
 *************************************************************************/

template <
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    std::uint32_t block_ct_dim = 4,
    std::uint32_t full_ct_dim = block_ct_dim>
inline void llk_pack_fast_untilize_init_with_formats(
    const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t num_faces = 4) {
    ckernel::_llk_pack_fast_untilize_init_<Dst, is_fp32_dest_acc_en, block_ct_dim, full_ct_dim>(
        pack_src_format, pack_dst_format, num_faces);
}

#ifndef ENV_LLK_INFRA
template <
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    std::uint32_t block_ct_dim = 4,
    std::uint32_t full_ct_dim = block_ct_dim>
inline void llk_pack_fast_untilize_init(const std::uint32_t output) {
    const std::uint32_t output_id = get_output_id(output);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    llk_pack_fast_untilize_init_with_formats<Dst, is_fp32_dest_acc_en, block_ct_dim, full_ct_dim>(
        pack_src_format[output_id], pack_dst_format[output_id], num_faces);
}
#endif

template <std::uint32_t block_ct_dim = 4, DstSync Dst = DstSync::SyncHalf>
inline void llk_pack_fast_untilize_block_at_address(
    const std::uint32_t address,
    const std::uint32_t unit_dim,
    std::uint32_t& prev_unit_dim,
    const std::uint32_t num_faces = 4) {
    ckernel::_llk_pack_fast_untilize_block_<block_ct_dim, Dst>(address, unit_dim, prev_unit_dim, num_faces);
}

template <std::uint32_t block_ct_dim = 4, std::uint32_t full_ct_dim = block_ct_dim, DstSync Dst = DstSync::SyncHalf>
inline void llk_pack_fast_untilize_block_strided_at_address(
    const std::uint32_t address,
    const std::uint32_t unit_dim,
    std::uint32_t& prev_unit_dim,
    const std::uint32_t num_faces = 4) {
    ckernel::_llk_pack_fast_untilize_block_strided_<block_ct_dim, full_ct_dim, Dst>(
        address, unit_dim, prev_unit_dim, num_faces);
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void llk_pack_fast_untilize_uninit_with_formats(
    const std::uint32_t pack_dst_format,
    const std::uint32_t pack_src_format = static_cast<std::uint32_t>(DataFormat::Float16_b)) {
    ckernel::_llk_pack_fast_untilize_uninit_<Dst, is_fp32_dest_acc_en>(pack_dst_format, pack_src_format);
}

#ifndef ENV_LLK_INFRA
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void llk_pack_fast_untilize_uninit(const std::uint32_t output) {
    const std::uint32_t output_id = get_output_id(output);
    llk_pack_fast_untilize_uninit_with_formats<Dst, is_fp32_dest_acc_en>(
        pack_dst_format[output_id], pack_src_format[output_id]);
}
#endif
