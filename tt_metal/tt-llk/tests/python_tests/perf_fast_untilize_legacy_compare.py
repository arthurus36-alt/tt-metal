# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import PerfRunType
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    TILE_COUNT,
    generate_input_dim,
)

from conftest import skip_for_quasar, skip_for_wormhole


def legacy_block_ct_dim(ct_dim):
    for candidate in range(min(ct_dim, 8), 0, -1):
        if ct_dim % candidate == 0:
            return candidate
    return 1


@pytest.mark.perf
@skip_for_wormhole
@skip_for_quasar
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    rt_dim=[1, 2, 4],
    ct_dim=[2, 3, 4, 5, 6, 7, 8],
    loop_factor=[1, 4, 16],
)
def test_perf_fast_untilize_legacy_compare(
    perf_report, formats, rt_dim, ct_dim, loop_factor
):
    tile_count = rt_dim * ct_dim
    dimensions = (rt_dim * 32, ct_dim * 32)
    block_ct_dim = legacy_block_ct_dim(ct_dim)

    configuration = PerfConfig(
        "sources/pack_untilize_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.PACK_ISOLATE,
        ],
        templates=[generate_input_dim(dimensions, dimensions, block_ct_dim)],
        runtimes=[
            TILE_COUNT(tile_count),
            LOOP_FACTOR(loop_factor),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        compile_time_formats=True,
    )

    configuration.run(perf_report)
