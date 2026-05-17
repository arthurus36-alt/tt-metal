# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("ct", [9, 10])
def test_untilize_multicore_wide_rows_multiple_blocks_per_core(device, ct):
    grid = device.compute_with_storage_grid_size()
    height = (grid.x * grid.y + 1) * 32
    width = ct * 32

    torch_input = torch.rand((height, width)).bfloat16().float()
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    output = ttnn.untilize(ttnn_input, use_multicore=True)

    assert_equal(torch_input, ttnn.to_torch(output))
