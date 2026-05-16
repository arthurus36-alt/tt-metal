// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// BH fast-untilize MVP test (T5-B / tt-metal#42048 + #42049).
// Reuses fast_tilize unpack + math; new fast_untilize pack writes RM strip.
//
// Hardcoded shape: block_ct_dim=4, num_faces=4, FP16 / bf16.
// First-pass goal: silicon-validate the pack MOP. Golden comparison TBD.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"
#include "profiler.h"

#ifndef PERF_RUN_TYPE
#define PERF_RUN_TYPE PerfRunType::L1_TO_L1
#endif
#include "perf.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr std::uint32_t FAST_UNTILIZE_BLOCK_CT_DIM = 4;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_fast_tilize.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;
#endif

    {
        ZONE_SCOPED("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);
        _llk_unpack_fast_tilize_init_(formats.unpack_A_dst, FAST_UNTILIZE_BLOCK_CT_DIM, FAST_UNTILIZE_BLOCK_CT_DIM);
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            _llk_unpack_fast_tilize_block_(L1_ADDRESS(buffer_A[0]), 0, formats.unpack_A_src, FAST_UNTILIZE_BLOCK_CT_DIM, 4);
        }
    }
    {
        ZONE_SCOPED("UNINIT")
        _llk_unpack_fast_tilize_uninit_<is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_fast_tilize.h"
#include "llk_math_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    constexpr std::uint32_t unit_dim = FAST_UNTILIZE_BLOCK_CT_DIM;

    {
        ZONE_SCOPED("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_fast_tilize_init_<is_fp32_dest_acc_en>(formats.math);
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
            _llk_math_fast_tilize_block_<is_fp32_dest_acc_en>(0, formats.math, unit_dim);
            _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        }
    }
    {
        ZONE_SCOPED("UNINIT")
        _llk_math_fast_tilize_uninit_<is_fp32_dest_acc_en>(formats.math);
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

    {
        ZONE_SCOPED("INIT")
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(
            formats.pack_src, formats.pack_dst, SCALE_DATUM_SIZE(formats.pack_dst, TILE_C_DIM * TILE_R_DIM));
        _llk_pack_fast_untilize_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, FAST_UNTILIZE_BLOCK_CT_DIM>(
            formats.pack_src, formats.pack_dst);
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            _llk_packer_wait_for_math_done_();
            _llk_pack_fast_untilize_block_<FAST_UNTILIZE_BLOCK_CT_DIM>(L1_ADDRESS(buffer_Res[0]));
            _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        }
    }
    {
        ZONE_SCOPED("UNINIT")
        _llk_pack_fast_untilize_uninit_<DstSync::SyncHalf, is_fp32_dest_acc_en>(formats.pack_dst, formats.pack_src);
    }
}

#endif
