// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "hostdevcommon/kernel_structs.h"
#include "llk_pack_common.h"
#include "llk_pack.h"

// CB ids:
//   cb_in0      : tilized input A (unused on the pack side)
//   cb_in1      : input B (unused on the pack side)
//   cb_intermed : untilized A produced by this pack stage (phase 1)
//   cb_out      : final eltwise-binary output produced by this pack stage (phase 2)
constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
constexpr uint32_t cb_intermed = tt::CBIndex::c_24;
constexpr uint32_t cb_out = tt::CBIndex::c_16;

void kernel_main() {
    uint32_t per_core_num_blocks = get_compile_time_arg_val(0);
    uint32_t per_core_block_r_tiles = get_compile_time_arg_val(1);
    uint32_t per_core_block_c_tiles = get_compile_time_arg_val(2);
    llk_pack_init();
    llk_pack_hw_configure<DST_ACCUM_MODE>(cb_out);
    llk_pack_dest_init<DST_ACCUM_MODE, PackMode::Default>();

    for (uint32_t block = 0; block < per_core_num_blocks; block++) {
        for (uint32_t r = 0; r < per_core_block_r_tiles; r++) {
            llk_wait_for_free_tiles<false, false, false>(cb_intermed, per_core_block_c_tiles);
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_packer_wait_for_math_done();
                llk_pack<DST_ACCUM_MODE, false, PackMode::Default>(0, cb_intermed);
                llk_pack_dest_section_done<DST_ACCUM_MODE>();
            }
            llk_push_tiles<false, false>(cb_intermed, per_core_block_c_tiles);

            llk_wait_for_free_tiles<false, false, false>(cb_out, per_core_block_c_tiles);
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_packer_wait_for_math_done();
                llk_pack<DST_ACCUM_MODE, false, PackMode::Default>(0, cb_out);
                llk_pack_dest_section_done<DST_ACCUM_MODE>();
            }
            llk_push_tiles<false, false>(cb_out, per_core_block_c_tiles);
        }
    }
}
