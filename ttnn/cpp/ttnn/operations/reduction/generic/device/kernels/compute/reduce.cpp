// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 compute kernel for the multi-core reduction primitive (no negation).
//
// Migration notes:
//   - Compile-time arguments are bound by name (`args::Ht`, `args::Wt`, `args::NC`,
//     `args::post_mul_scaler_bits`).
//   - `Ht` is a compile-time argument. When `split_work_to_cores` produces two work
//     groups with different per-core row counts, the host emits two KernelSpecs of
//     this same source — one per group — with the per-group `Ht` value bound as a
//     CTA, placed in two WorkUnitSpecs (one per group). This preserves compile-time
//     loop unrolling on `Ht`.
//   - DataflowBuffers are bound by name (`dfb::input`, `dfb::scaler`, `dfb::output`)
//     and passed *as objects* (not raw ids) to compute_kernel_lib::reduce. The
//     helper templates on the buffer type, so the same kernel source compiles for
//     Gen1 (where DFB id == underlying CB id) and Gen2 (real DFB hardware).
//     No `#ifdef ARCH_QUASAR` is required.

#include <cstdint>

#include "experimental/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

#ifdef REDUCE_POST_MUL
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

void kernel_main() {
    // Compile-time arguments shared by every worker core in this KernelSpec's group.
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t NC = get_arg(args::NC);

    // Typed dataflow-buffer wrappers. The compute_kernel_lib::reduce() helper
    // is templated on the buffer type and works uniformly across Gen1/Gen2.
    experimental::DataflowBuffer dfb_input(dfb::input);
    experimental::DataflowBuffer dfb_scaler(dfb::scaler);
    experimental::DataflowBuffer dfb_output(dfb::output);

    compute_kernel_hw_startup(dfb_input.get_id(), dfb_scaler.get_id(), dfb_output.get_id());

    compute_kernel_lib::reduce<
        REDUCE_OP,
        REDUCE_DIM,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        dfb_input,
        dfb_scaler,
        dfb_output,
        compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC),
        compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
        compute_kernel_lib::NoAccumulation{},
#ifdef REDUCE_POST_MUL
        // GMPOOL only respects the scaler's exponent for MAX/MIN, so the host requests
        // reduction with scaler=1.0 and we apply the user scalar via mul_unary_tile
        // (SFPU) on each output DEST register.
        [](uint32_t dst_idx) {
            constexpr uint32_t post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
            binop_with_scalar_tile_init();
            mul_unary_tile(dst_idx, post_mul_scaler_bits);
        }
#else
        compute_kernel_lib::NoOp{}
#endif
    );
}
