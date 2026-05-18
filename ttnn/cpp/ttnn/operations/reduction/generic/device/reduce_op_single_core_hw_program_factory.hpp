// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"
#include "reduce_op_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// ReduceSingleCoreHwProgramFactory: full HW-axis reduction on a single core.
// Built via the Metal 2.0 host API (ProgramSpec / DataflowBufferSpec / KernelSpec /
// WorkUnitSpec). Satisfies ProgramSpecFactoryConcept; the framework adapter handles
// Program construction (MakeProgramFromSpec) and cache-hit dispatch (UpdateTensorArgs).
struct ReduceSingleCoreHwProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const ReduceParams& operation_attributes, const Tensor& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
