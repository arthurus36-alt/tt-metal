// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "llk_math_transpose_dest.h"

// is_fp32 selects between two 32-bit MOPs: a Float32-safe B-only loop and
// the Int32/UInt32 MOP that round-trips through the A register to preserve
// the lower 16 bits. See _llk_math_transpose_dest_ for details.
template <bool transpose_of_faces = true, bool is_32bit = false, bool is_fp32 = false>
inline void llk_math_transpose_dest(uint dst_index) {
    LLK_ASSERT((dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "");

    _llk_math_transpose_dest_<DST_ACCUM_MODE, transpose_of_faces, is_32bit, is_fp32>(dst_index);
}

template <bool transpose_of_faces = true, bool is_32bit = false, bool is_fp32 = false>
inline void llk_math_transpose_dest_init() {
    _llk_math_transpose_dest_init_<transpose_of_faces, is_32bit, is_fp32>();
}
