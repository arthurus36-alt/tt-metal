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
    NUM_FACES,
    TILE_COUNT,
    generate_input_dim,
)

from conftest import skip_for_quasar, skip_for_wormhole


@pytest.mark.perf
@skip_for_wormhole
@skip_for_quasar
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    rt_dim=[1, 2, 4],
    ct_dim=[2, 3, 4, 5, 6, 7, 8],
    loop_factor=[1, 4, 16],
)
def test_perf_fast_untilize(perf_report, formats, rt_dim, ct_dim, loop_factor):
    tile_count = rt_dim * ct_dim
    dimensions = (rt_dim * 32, ct_dim * 32)

    configuration = PerfConfig(
        "sources/fast_untilize_test.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
        templates=[generate_input_dim(dimensions, dimensions)],
        runtimes=[
            TILE_COUNT(tile_count),
            LOOP_FACTOR(loop_factor),
            NUM_FACES(4),
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
