# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Fast-untilize MVP test (T5-B / tt-metal#42048 + #42049).

Pipeline: unpack (fast_tilize unpack) → math (fast_tilize math) → pack
(NEW fast_untilize pack). Output: row-major strip.

Hardcoded: block_ct_dim=4, num_faces=4, FP16/bf16.
First-pass goal: silicon-validate the pack MOP. Functional PCC check.
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import UntilizeGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.utils import passed_test

TILE_R = 32
TILE_C = 32


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    dest_acc=[DestAccumulation.No],
    dimensions=[(1, 4)],  # ct=4, rt=1 — only supported shape for T5 MVP
)
def test_fast_untilize(formats, dest_acc, dimensions):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")

    input_height_tiles, input_width_tiles = dimensions
    assert input_width_tiles == 4, "T5-B MVP only supports ct=4"

    input_dimensions = [input_height_tiles * TILE_R, input_width_tiles * TILE_C]
    tile_count = input_height_tiles * input_width_tiles

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(UntilizeGolden)
    golden_tensor = generate_golden(src_A, formats.output_format, input_dimensions)

    configuration = TestConfig(
        "sources/fast_untilize_test.cpp",
        formats,
        templates=[],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(tile_count),
            LOOP_FACTOR(1),
            NUM_FACES(4),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_count,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(golden_tensor), (
        f"Result length {len(res_from_L1)} != golden length {len(golden_tensor)}"
    )

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "fast_untilize output does not match UntilizeGolden"
