// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#include "llk_math_transpose_dest_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#endif

namespace ckernel {

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for transpose_wh_dest to be
 * executed correctly.
 */
// is_fp32 = true selects the LLK transpose-dest MOP variant that keeps the
// full Float32 mantissa in DEST; is_fp32 = false selects the variant that
// preserves the lower 16 bits across the transpose (Int32 / UInt32).
template <bool is_32bit = false, bool is_fp32 = false>
ALWI void transpose_wh_dest_init_short() {
    MATH((llk_math_transpose_dest_init<true, is_32bit, is_fp32>()));
}

// clang-format off
/**
 * Performs a 32x32 in place transpose operation *B[w,h] = A[h,w]* on a tile in the DST register at idst.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                             | Type     | Valid Range                                    | Required |
 * |----------------|---------------------------------------------------------|----------|------------------------------------------------|----------|
 * | idst           | The index of the tile in DST REG to transpose           | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
 // clang-format on
template <bool is_32bit = false, bool is_fp32 = false>
ALWI void transpose_wh_dest(uint32_t idst) {
    UNPACK((llk_unpack_set_srcb_dummy_valid()));
    MATH((llk_math_transpose_dest<true, is_32bit, is_fp32>(idst)));
}

}  // namespace ckernel
