// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "hostdevcommon/kernel_structs.h"
#include "llk_math_common.h"
#include "llk_math_binary_api.h"
#include "llk_math_unary_datacopy_api.h"

// CB ids:
//   cb_in0      : tilized input A (consumed by untilize)
//   cb_in1      : input B for the eltwise binary
//   cb_intermed : untilized A (produced by pack from untilize, consumed by binary unpack)
//   cb_out      : final eltwise-binary output
constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
constexpr uint32_t cb_intermed = tt::CBIndex::c_24;
constexpr uint32_t cb_out = tt::CBIndex::c_16;

void kernel_main() {
    uint32_t per_core_num_blocks = get_compile_time_arg_val(0);
    uint32_t per_core_block_r_tiles = get_compile_time_arg_val(1);
    uint32_t per_core_block_c_tiles = get_compile_time_arg_val(2);

    llk_math_pack_sync_init<DST_ACCUM_MODE>();
    llk_math_hw_configure<DST_ACCUM_MODE>(cb_in0, cb_in1);
    for (uint32_t block = 0; block < per_core_num_blocks; block++) {
        for (uint32_t r = 0; r < per_core_block_r_tiles; r++) {
            // Untilize
            llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE>(cb_in0);
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_math_wait_for_dest_available();
                llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE>(
                    0 /*dst_index*/, cb_in0);
                llk_math_dest_section_done<DST_ACCUM_MODE>();
            }

            llk_math_eltwise_binary_init<ELWADD, BroadcastType::NONE, MathFidelity::LoFi>(cb_intermed, cb_in1);
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_math_wait_for_dest_available();
                llk_math_eltwise_binary<
                    ELWADD,
                    BroadcastType::NONE,
                    DST_ACCUM_MODE,
                    MATH_FIDELITY,
                    EltwiseBinaryReuseDestType::NONE>(0);
                llk_math_dest_section_done<DST_ACCUM_MODE>();
            }
        }
    }
}
