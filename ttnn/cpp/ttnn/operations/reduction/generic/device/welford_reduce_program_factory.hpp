// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"
#include "welford_reduce_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// WelfordReduceProgramFactory: variance / std-dev reduction (Welford's online algorithm)
// across one of W, H, HW. Migrated to the Metal 2.0 host API.
struct WelfordReduceProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const WelfordReduceParams& operation_attributes, const Tensor& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
