// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// BH fast-untilize bring-up test (T5-B / tt-metal#42048 + #42049).
// Fast-untilize unpack loads 2/3/4-tile chunks, custom math lays each chunk out for
// 4-interface row-major readout, and fast_untilize pack writes the RM strip.
//
// Current shapes: row-decomposed unit_dim={4,2,3}; ct=1 is left to the
// integrated legacy fallback path. Hardcoded constraints: num_faces=4,
// FP16 / bf16, SyncHalf.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr std::uint32_t FAST_UNTILIZE_MAX_UNIT_DIM = 4;
constexpr std::uint32_t MAX_UNITS_PER_ROW          = 16;

static_assert(PERF_RUN_TYPE != PerfRunType::L1_CONGESTION, "L1 congestion mode is not supported for fast_untilize MVP");
static_assert(BLOCK_CT_DIM == FULL_CT_DIM, "fast_untilize_test expects one full tile row per kernel instance");
static_assert(FULL_CT_DIM >= 2 && FULL_CT_DIM <= 8, "fast_untilize_test supports ct=2..8; ct=1 uses legacy fallback");

inline std::uint32_t decompose_row(const std::uint32_t ct_dim, std::uint32_t unit_dims[MAX_UNITS_PER_ROW])
{
    std::uint32_t n4  = ct_dim / 4;
    std::uint32_t rem = ct_dim % 4;
    std::uint32_t idx = 0;

    if (rem == 1 && n4 > 0)
    {
        n4--;
        rem = 5;
    }

    for (std::uint32_t i = 0; i < n4; i++)
    {
        unit_dims[idx++] = 4;
    }

    if (rem == 2)
    {
        unit_dims[idx++] = 2;
    }
    else if (rem == 3)
    {
        unit_dims[idx++] = 3;
    }
    else if (rem == 5)
    {
        unit_dims[idx++] = 2;
        unit_dims[idx++] = 3;
    }

    return idx;
}

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_fast_untilize.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;
#endif
    std::uint32_t unit_dims[MAX_UNITS_PER_ROW];
    const std::uint32_t units_per_row = decompose_row(FULL_CT_DIM, unit_dims);

    {
        ZONE_SCOPED("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);
        ckernel::_llk_unpack_fast_untilize_init_(formats.unpack_A_src, formats.unpack_A_dst, unit_dims[0]);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_loop_set_valid<true, false>(LOOP_FACTOR * FULL_RT_DIM * FULL_CT_DIM * 4);
            PROFILER_SYNC();
            return;
        }

        std::uint32_t prev_unit_dim = unit_dims[0];
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (std::uint32_t rt = 0; rt < FULL_RT_DIM; rt++)
            {
                std::uint32_t chunk_col = 0;
                for (std::uint32_t u = 0; u < units_per_row; u++)
                {
                    const std::uint32_t unit_dim = unit_dims[u];
                    if (unit_dim != prev_unit_dim)
                    {
                        ckernel::_llk_unpack_fast_untilize_reinit_unit_dim_(unit_dim);
                        prev_unit_dim = unit_dim;
                    }
                    ckernel::_llk_unpack_fast_untilize_block_(L1_ADDRESS(buffer_A[rt * FULL_CT_DIM + chunk_col]), unit_dim);
                    chunk_col += unit_dim;
                }
            }
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("UNINIT")
        ckernel::_llk_unpack_fast_untilize_uninit_();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_fast_untilize.h"
#include "llk_math_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    std::uint32_t unit_dims[MAX_UNITS_PER_ROW];
    const std::uint32_t units_per_row = decompose_row(FULL_CT_DIM, unit_dims);

    {
        ZONE_SCOPED("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        ckernel::_llk_math_fast_untilize_init_<is_fp32_dest_acc_en>(formats.math);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            _perf_math_loop_clear_valid<true, false>(LOOP_FACTOR * FULL_RT_DIM * FULL_CT_DIM * 4);
            PROFILER_SYNC();
            return;
        }

        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (std::uint32_t rt = 0; rt < FULL_RT_DIM; rt++)
            {
                for (std::uint32_t u = 0; u < units_per_row; u++)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    }
                    ckernel::_llk_math_fast_untilize_block_<is_fp32_dest_acc_en>(0, formats.math, unit_dims[u]);
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("UNINIT")
        ckernel::_llk_math_fast_untilize_uninit_<is_fp32_dest_acc_en>(formats.math);
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "experimental/llk_pack_fast_untilize.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_Res       = params.buffer_Res;
#endif
    std::uint32_t unit_dims[MAX_UNITS_PER_ROW];
    const std::uint32_t units_per_row = decompose_row(FULL_CT_DIM, unit_dims);

#ifdef LLK_PROFILER
    constexpr std::uint32_t NUM_GUARD = 0;
#else
    const std::uint32_t NUM_GUARD = params.NUM_GUARD_TILES;
#endif

    const std::uint32_t total_tiles = FULL_RT_DIM * FULL_CT_DIM;
    const std::uint32_t tile_bytes  = GET_L1_HEADERLESS_TILE_SIZE(formats.pack_dst) << 4;

    if (NUM_GUARD > 0)
    {
        for (std::uint32_t g = 0; g < NUM_GUARD; g++)
        {
            volatile std::uint16_t* guard = reinterpret_cast<volatile std::uint16_t*>(buffer_Res[total_tiles + g]);
            for (std::uint32_t i = 0; i < tile_bytes / 2; i++)
            {
                guard[i] = 0xACAF;
            }
        }
    }

    {
        ZONE_SCOPED("INIT")
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(
            formats.pack_src, formats.pack_dst, SCALE_DATUM_SIZE(formats.pack_dst, TILE_C_DIM * TILE_R_DIM));
        _llk_pack_fast_untilize_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, FAST_UNTILIZE_MAX_UNIT_DIM, FULL_CT_DIM>(formats.pack_src, formats.pack_dst);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t rt = 0; rt < FULL_RT_DIM; rt++)
                {
                    const std::uint32_t tile_row_offset_16B = SCALE_DATUM_SIZE(formats.pack_dst, rt * FULL_CT_DIM * TILE_R_DIM * TILE_C_DIM) / 16;
                    const std::uint32_t tile_row_address    = L1_ADDRESS(buffer_Res[0]) + tile_row_offset_16B;
                    std::uint32_t chunk_col                 = 0;
                    for (std::uint32_t u = 0; u < units_per_row; u++)
                    {
                        const std::uint32_t unit_dim         = unit_dims[u];
                        const std::uint32_t chunk_offset_16B = SCALE_DATUM_SIZE(formats.pack_dst, chunk_col * TILE_C_DIM) / 16;
                        const std::uint32_t chunk_address    = tile_row_address + chunk_offset_16B;
                        if constexpr (FULL_CT_DIM <= FAST_UNTILIZE_MAX_UNIT_DIM)
                        {
                            _llk_pack_fast_untilize_block_<FAST_UNTILIZE_MAX_UNIT_DIM, DstSync::SyncHalf>(chunk_address, unit_dim);
                        }
                        else
                        {
                            _llk_pack_fast_untilize_block_strided_<FAST_UNTILIZE_MAX_UNIT_DIM, FULL_CT_DIM, DstSync::SyncHalf>(chunk_address, unit_dim);
                        }
                        chunk_col += unit_dim;
                    }
                }
            }
            PROFILER_SYNC();
            return;
        }

        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (std::uint32_t rt = 0; rt < FULL_RT_DIM; rt++)
            {
                const std::uint32_t tile_row_offset_16B = SCALE_DATUM_SIZE(formats.pack_dst, rt * FULL_CT_DIM * TILE_R_DIM * TILE_C_DIM) / 16;
                const std::uint32_t tile_row_address    = L1_ADDRESS(buffer_Res[0]) + tile_row_offset_16B;
                std::uint32_t chunk_col                 = 0;
                for (std::uint32_t u = 0; u < units_per_row; u++)
                {
                    const std::uint32_t unit_dim         = unit_dims[u];
                    const std::uint32_t chunk_offset_16B = SCALE_DATUM_SIZE(formats.pack_dst, chunk_col * TILE_C_DIM) / 16;
                    const std::uint32_t chunk_address    = tile_row_address + chunk_offset_16B;

                    _llk_packer_wait_for_math_done_();
                    if constexpr (FULL_CT_DIM <= FAST_UNTILIZE_MAX_UNIT_DIM)
                    {
                        _llk_pack_fast_untilize_block_<FAST_UNTILIZE_MAX_UNIT_DIM, DstSync::SyncHalf>(chunk_address, unit_dim);
                    }
                    else
                    {
                        _llk_pack_fast_untilize_block_strided_<FAST_UNTILIZE_MAX_UNIT_DIM, FULL_CT_DIM, DstSync::SyncHalf>(chunk_address, unit_dim);
                    }
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                    chunk_col += unit_dim;
                }
            }
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("UNINIT")
        _llk_pack_fast_untilize_uninit_<DstSync::SyncHalf, is_fp32_dest_acc_en>(formats.pack_dst, formats.pack_src);
    }

    if (NUM_GUARD > 1)
    {
        volatile std::uint16_t* result_tile = reinterpret_cast<volatile std::uint16_t*>(buffer_Res[total_tiles + NUM_GUARD - 1]);
        for (std::uint32_t i = 0; i < tile_bytes / 2; i++)
        {
            result_tile[i] = 0;
        }
        result_tile[0] = 0x4680;

        for (std::uint32_t g = 0; g < NUM_GUARD - 1; g++)
        {
            volatile std::uint16_t* guard = reinterpret_cast<volatile std::uint16_t*>(buffer_Res[total_tiles + g]);
            std::uint32_t corrupted       = 0;
            for (std::uint32_t i = 0; i < tile_bytes / 2; i++)
            {
                if (guard[i] != 0xACAF)
                {
                    corrupted++;
                }
            }
            result_tile[g + 1] = static_cast<std::uint16_t>(corrupted);
        }
    }
}

#endif
