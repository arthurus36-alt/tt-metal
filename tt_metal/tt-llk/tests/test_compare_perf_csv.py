# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import csv
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("compare_perf_csv.py")
FIELDNAMES = [
    "formats.input_A",
    "formats.input_B",
    "formats.output",
    "unpack_to_dest",
    "dest_acc",
    "full_rt_dim",
    "full_ct_dim",
    "block_ct_dim",
    "block_rt_dim",
    "tile_cnt",
    "loop_factor",
    "marker",
    "mean(L1_TO_L1)",
]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def perf_row(**overrides):
    row = {
        "formats.input_A": "Float16_b",
        "formats.input_B": "Float16_b",
        "formats.output": "Float16_b",
        "unpack_to_dest": "False",
        "dest_acc": "DestAccumulation.No",
        "full_rt_dim": "1",
        "full_ct_dim": "4",
        "block_ct_dim": "4",
        "block_rt_dim": "1",
        "tile_cnt": "4",
        "loop_factor": "1",
        "marker": "KERNEL",
        "mean(L1_TO_L1)": "100",
    }
    row.update(overrides)
    return row


def run_compare(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_marker_summary_counts_filtered_variants(tmp_path):
    baseline = tmp_path / "baseline.post.csv"
    candidate = tmp_path / "candidate.post.csv"
    write_csv(
        baseline,
        [
            perf_row(marker="KERNEL"),
            perf_row(marker="TILE_LOOP", **{"mean(L1_TO_L1)": "50"}),
        ],
    )
    write_csv(
        candidate,
        [
            perf_row(marker="KERNEL"),
            perf_row(marker="TILE_LOOP", **{"mean(L1_TO_L1)": "49"}),
        ],
    )

    result = run_compare(baseline, candidate, "--marker", "TILE_LOOP")

    assert result.returncode == 0
    assert "n=1 variants" in result.stdout


def test_ignore_cols_allows_apples_to_apples_join(tmp_path):
    baseline = tmp_path / "baseline.post.csv"
    candidate = tmp_path / "candidate.post.csv"
    write_csv(baseline, [perf_row(block_ct_dim="1")])
    write_csv(candidate, [perf_row(block_ct_dim="4")])

    result = run_compare(baseline, candidate, "--ignore-cols", "block_ct_dim")

    assert result.returncode == 0
    assert "n=1 variants" in result.stdout


def test_missing_warnings_respect_marker_filter(tmp_path):
    baseline = tmp_path / "baseline.post.csv"
    candidate = tmp_path / "candidate.post.csv"
    write_csv(baseline, [perf_row(marker="KERNEL")])
    write_csv(
        candidate,
        [
            perf_row(marker="KERNEL"),
            perf_row(marker="UNINIT", **{"mean(L1_TO_L1)": ""}),
        ],
    )

    result = run_compare(baseline, candidate, "--marker", "KERNEL")

    assert result.returncode == 0
    assert "[warn]" not in result.stdout
    assert "n=1 variants" in result.stdout


def test_missing_key_columns_return_friendly_error(tmp_path):
    baseline = tmp_path / "baseline.post.csv"
    candidate = tmp_path / "candidate.post.csv"
    write_csv(baseline, [perf_row()])
    with open(candidate, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[c for c in FIELDNAMES if c != "tile_cnt"]
        )
        writer.writeheader()
        writer.writerow({k: v for k, v in perf_row().items() if k != "tile_cnt"})

    result = run_compare(baseline, candidate)

    assert result.returncode == 2
    assert "missing required columns" in result.stderr
    assert "tile_cnt" in result.stderr
