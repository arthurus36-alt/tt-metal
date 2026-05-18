// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "hostdevcommon/kernel_structs.h"
#include "llk_unpack_common_api.h"
#include "llk_unpack_AB_api.h"
#include "llk_unpack_untilize_api.h"

// CB ids:
//   cb_in0      : tilized input A (consumed by untilize)
//   cb_in1      : input B for the eltwise binary
//   cb_intermed : untilized A (produced by pack from untilize, consumed by binary unpack)
//   cb_out      : final eltwise-binary output (unused on the unpack side)
constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
constexpr uint32_t cb_intermed = tt::CBIndex::c_24;
constexpr uint32_t cb_out = tt::CBIndex::c_16;

void kernel_main() {
    uint32_t per_core_num_blocks = get_compile_time_arg_val(0);
    uint32_t per_core_block_r_tiles = get_compile_time_arg_val(1);
    uint32_t per_core_block_c_tiles = get_compile_time_arg_val(2);

    llk_unpack_hw_configure<DST_ACCUM_MODE>(cb_in0, cb_in1);

    // llk_unpack_untilize_init(cb_in0);
    for (uint32_t block = 0U; block < per_core_num_blocks; ++block) {
        for (uint32_t r = 0; r < per_core_block_r_tiles; r++) {
            llk_unpack_untilize_init(cb_in0);
            llk_wait_tiles(cb_in0, per_core_block_c_tiles);
            llk_unpack_untilize_<true>(cb_in0, per_core_block_c_tiles);
            llk_unpack_untilize_<false>(cb_in0, per_core_block_c_tiles);
#ifdef ARCH_BLACKHOLE
            llk_unpack_untilize_uninit(cb_in0);
#else
            llk_unpack_untilize_uninit();
#endif
            llk_pop_tiles(cb_in0, per_core_block_c_tiles);
            llk_pop_tiles(cb_in1, per_core_block_c_tiles);

            llk_unpack_AB_init<BroadcastType::NONE>();
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_wait_tiles(cb_intermed, 1);
                llk_wait_tiles(cb_in1, 1);
                llk_unpack_AB(cb_intermed, cb_in1, 0, 0);
                llk_pop_tiles(cb_intermed, 1);
                llk_pop_tiles(cb_in1, 1);
            }
        }
    }
}
