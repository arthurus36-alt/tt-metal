// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"
#include "reduce_op_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// ReduceMultiCoreWProgramFactory: width-axis reduction across multiple cores.
// Built via the Metal 2.0 host API. Satisfies ProgramSpecFactoryConcept; the
// framework adapter handles Program construction and cache-hit dispatch.
struct ReduceMultiCoreWProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const ReduceParams& operation_attributes, const Tensor& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
