// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;

// Compute geometry parameters and fill runtime_args for the given input tensor.
// Returns the input CB page size (aligned) and sets aligned_output_bytes.
uint32_t compute_geometry(
    const Tensor& input,
    const Tensor& out_num_indices,
    const Tensor& out_indices,
    std::vector<uint32_t>& runtime_args,
    uint32_t& aligned_output_bytes) {
    const bool is_tile = (input.layout() == Layout::TILE);
    const auto& lshape = input.logical_shape();
    const auto& pshape = input.padded_shape();
    aligned_output_bytes = out_indices.buffer()->aligned_page_size();

    uint32_t input_page_size;

    if (!is_tile) {
        // Use the buffer's physical page size so TensorAccessor gets the correct page size for
        // all layouts (interleaved, HEIGHT/WIDTH/BLOCK sharded). For HEIGHT_SHARDED this equals
        // last_dim * elem_size; for WIDTH/BLOCK_SHARDED it equals shard_width * elem_size.
        const uint32_t phys_page_bytes = input.buffer()->page_size();
        const uint32_t elements_per_page = phys_page_bytes / input.element_size();
        // Use num_dev_pages() directly: avoids recomputing from volume/page_size and correctly
        // handles any rounding the allocator may apply.
        const uint32_t num_pages = input.buffer()->num_dev_pages();
        // Use the buffer's allocator-aligned page size for the NOC transfer and interleaved
        // TensorAccessor stride, matching the buffer's actual per-page footprint in DRAM/L1.
        const uint32_t aligned_page_size = input.buffer()->aligned_page_size();
        input_page_size = aligned_page_size;

        // For WIDTH/BLOCK_SHARDED, TensorAccessor visits pages bank-by-bank, which differs from
        // logical row-major order. The kernel needs these three values to reconstruct flat_start.
        const uint32_t logical_last_dim = lshape[3];
        const uint32_t grid_w =
            (elements_per_page < logical_last_dim) ? (logical_last_dim / elements_per_page) : 1;
        uint32_t pages_per_bank = 1;
        const auto mem_layout = input.memory_config().memory_layout();
        if (mem_layout == TensorMemoryLayout::WIDTH_SHARDED || mem_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            pages_per_bank = input.memory_config().shard_spec().value().shape[0];
        }

        runtime_args = {
            input.buffer()->address(),
            out_num_indices.buffer()->address(),
            out_indices.buffer()->address(),
            aligned_output_bytes,
            num_pages,
            elements_per_page,
            aligned_page_size,
            logical_last_dim,
            pages_per_bank,
            grid_w};
    } else {
        const uint32_t tile_page_size = input.buffer()->aligned_page_size();
        input_page_size = tile_page_size;

        const uint32_t B = lshape[0];
        const uint32_t N = lshape[1];
        const uint32_t logical_H = lshape[2];
        const uint32_t logical_C = lshape[3];
        const uint32_t num_tile_rows = pshape[2] / TILE_H;
        const uint32_t num_tile_cols = pshape[3] / TILE_W;

        runtime_args = {
            input.buffer()->address(),
            out_num_indices.buffer()->address(),
            out_indices.buffer()->address(),
            aligned_output_bytes,
            B,
            N,
            logical_H,
            logical_C,
            num_tile_rows,
            num_tile_cols,
            tile_page_size};
    }

    return input_page_size;
}

}  // namespace

NonZeroIndicesProgramFactory::cached_program_t NonZeroIndicesProgramFactory::create(
    const NonzeroParams& /*operation_attributes*/, const NonzeroInputs& tensor_args, NonzeroResult& output_tensors) {
    const auto& input = tensor_args.input;
    const auto& out_num_indices = std::get<0>(output_tensors);
    const auto& out_indices = std::get<1>(output_tensors);

    tt::tt_metal::Program program{};

    const bool is_tile = (input.layout() == Layout::TILE);

    std::vector<uint32_t> runtime_args;
    uint32_t aligned_output_bytes = 0;
    const uint32_t input_page_size =
        compute_geometry(input, out_num_indices, out_indices, runtime_args, aligned_output_bytes);

    CoreCoord core = {0, 0};

    constexpr uint32_t input_cb_index = 0;
    constexpr uint32_t output_cb_index_0 = 1;
    constexpr uint32_t output_cb_index_1 = 2;

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(DataType::UINT32);

    // Input CB: single-buffered (barrier is issued after each read, no overlap benefit from double-buffering)
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(input_page_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    // Output CB 0: count tensor (32 bytes fixed)
    tt::tt_metal::CircularBufferConfig cb_dst0_config =
        tt::tt_metal::CircularBufferConfig(2 * 32, {{output_cb_index_0, output_cb_data_format}})
            .set_page_size(output_cb_index_0, 32);
    tt::tt_metal::CreateCircularBuffer(program, core, cb_dst0_config);

    // Output CB 1: indices tensor, one page holding all indices (worst case: all elements non-zero)
    tt::tt_metal::CircularBufferConfig cb_dst1_config =
        tt::tt_metal::CircularBufferConfig(2 * aligned_output_bytes, {{output_cb_index_1, output_cb_data_format}})
            .set_page_size(output_cb_index_1, aligned_output_bytes);
    tt::tt_metal::CreateCircularBuffer(program, core, cb_dst1_config);

    std::map<std::string, std::string> defines;
    defines["NUM_BYTES"] = std::to_string(input.element_size());
    if (is_tile) {
        defines["INPUT_IS_TILE"] = "1";
    }

    std::vector<uint32_t> compile_time_args = {
        static_cast<uint32_t>(input_cb_index),
        static_cast<uint32_t>(output_cb_index_0),
        static_cast<uint32_t>(output_cb_index_1),
    };
    TensorAccessorArgs(*input.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*out_num_indices.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*out_indices.buffer()).append_to(compile_time_args);

    auto kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/non_zero_indices/device/kernels/dataflow/"
        "non_zero_indices_sc_reader.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(compile_time_args, defines));

    tt::tt_metal::SetRuntimeArgs(program, kernel_id, core, runtime_args);

    return cached_program_t{std::move(program), {kernel_id, core, input_page_size}};
}

void NonZeroIndicesProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const NonzeroParams& /*operation_attributes*/,
    const NonzeroInputs& tensor_args,
    NonzeroResult& output_tensors) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;

    const auto& input = tensor_args.input;
    const auto& out_num_indices = std::get<0>(output_tensors);
    const auto& out_indices = std::get<1>(output_tensors);

    std::vector<uint32_t> new_runtime_args;
    uint32_t aligned_output_bytes = 0;
    compute_geometry(input, out_num_indices, out_indices, new_runtime_args, aligned_output_bytes);

    auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, shared_vars.kernel_id, shared_vars.core);
    TT_FATAL(
        runtime_args.size() == new_runtime_args.size(),
        "Runtime args size mismatch in non_zero_indices override: expected {}, got {}",
        runtime_args.size(),
        new_runtime_args.size());
    for (size_t i = 0; i < new_runtime_args.size(); ++i) {
        runtime_args[i] = new_runtime_args[i];
    }
}

}  // namespace ttnn::prim
