#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compare two .post.csv perf reports produced by the LLK perf harness.

Joins on (format, dim, marker) keys and reports cyc/tile delta and %change for
every mean(<run_type>) column. Returns nonzero exit if any variant regresses
beyond the gate.

Usage:
    python compare_perf_csv.py BASELINE.post.csv CANDIDATE.post.csv [--gate 2]
    python compare_perf_csv.py BASELINE.post.csv CANDIDATE.post.csv --ignore-cols block_ct_dim
"""

import argparse
import csv
import sys
from collections import defaultdict

KEY_COLS = (
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
)

DETAIL_LABEL_COLS = {
    "formats.input_A",
    "formats.output",
    "full_rt_dim",
    "full_ct_dim",
    "block_ct_dim",
}


def format_key(key_cols, key):
    return "/".join(f"{k}={v}" for k, v in zip(key_cols, key))


def detail_label(key_cols, key):
    return "/".join(f"{k}={v}" for k, v in zip(key_cols, key) if k in DETAIL_LABEL_COLS)


def load(path, key_cols):
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        required_cols = set(key_cols) | {"marker"}
        missing = [c for c in sorted(required_cols) if c not in fieldnames]
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")

        for line_number, row in enumerate(reader, start=2):
            key = tuple(row[c] for c in key_cols)
            if key in rows:
                raise ValueError(
                    f"{path}: duplicate key after applying ignored columns "
                    f"at line {line_number}: {format_key(key_cols, key)}"
                )
            rows[key] = row
    return rows


def mean_columns(row):
    return [c for c in row if c.startswith("mean(") and c.endswith(")")]


def fmt_pct(p):
    sign = "+" if p > 0 else ""
    return f"{sign}{p:6.2f}%"


def run_col_name(col):
    return col.replace("mean(", "").rstrip(")")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument(
        "--gate", type=float, default=2.0, help="regression % gate (default 2)"
    )
    ap.add_argument(
        "--marker",
        default="KERNEL",
        help="filter to a single marker for summary (default KERNEL)",
    )
    ap.add_argument(
        "--ignore-cols",
        default="",
        help="comma-separated join-key columns to ignore, e.g. block_ct_dim",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="print every variant, not just regressions/wins",
    )
    args = ap.parse_args()

    ignored_cols = {c.strip() for c in args.ignore_cols.split(",") if c.strip()}
    unknown_ignored_cols = ignored_cols - set(KEY_COLS)
    if unknown_ignored_cols:
        ap.error(
            "unknown --ignore-cols entries: " + ", ".join(sorted(unknown_ignored_cols))
        )
    key_cols = tuple(c for c in KEY_COLS if c not in ignored_cols)

    base = load(args.baseline, key_cols)
    cand = load(args.candidate, key_cols)

    only_base = {k for k in set(base) - set(cand) if base[k]["marker"] == args.marker}
    only_cand = {k for k in set(cand) - set(base) if cand[k]["marker"] == args.marker}
    common = set(base) & set(cand)

    if only_base:
        print(f"[warn] {len(only_base)} variants in baseline missing from candidate")
    if only_cand:
        print(f"[warn] {len(only_cand)} variants in candidate missing from baseline")

    sample_row = next(iter(cand.values()), None) or next(iter(base.values()), None)
    if sample_row is None:
        raise ValueError("both CSVs are empty")
    run_cols = mean_columns(sample_row)

    # Summary stats per run_col
    regress = defaultdict(list)
    wins = defaultdict(list)
    deltas = defaultdict(list)

    rows_out = []
    marker_keys = []
    for key in sorted(common):
        b, c = base[key], cand[key]
        if b["marker"] != args.marker:
            continue
        marker_keys.append(key)
        for col in run_cols:
            bv = b.get(col, "")
            cv = c.get(col, "")
            if not bv or not cv:
                continue
            bv, cv = float(bv), float(cv)
            if bv == 0:
                continue
            pct = (cv - bv) / bv * 100.0
            deltas[col].append(pct)
            if pct > args.gate:
                regress[col].append((key, bv, cv, pct))
            elif pct < -args.gate:
                wins[col].append((key, bv, cv, pct))
            rows_out.append((key, col, bv, cv, pct))

    # Print summary
    print(
        f"\n=== Summary (marker={args.marker}, gate=±{args.gate}%, n={len(marker_keys)} variants) ==="
    )
    header = f"{'run_type':<24} {'min%':>8} {'mean%':>8} {'max%':>8} {'#regr':>6} {'#win':>6}"
    print(header)
    print("-" * len(header))
    for col in run_cols:
        d = deltas[col]
        if not d:
            continue
        col_name = run_col_name(col)
        print(
            f"{col_name:<24} "
            f"{min(d):>7.2f}% {sum(d)/len(d):>7.2f}% {max(d):>7.2f}% "
            f"{len(regress[col]):>6} {len(wins[col]):>6}"
        )

    # Detail regressions
    any_regress = False
    for col, items in regress.items():
        if not items:
            continue
        any_regress = True
        col_name = run_col_name(col)
        print(f"\n--- Regressions in {col_name} (>{args.gate}%) ---")
        for key, bv, cv, pct in sorted(items, key=lambda x: -x[3])[:20]:
            label = detail_label(key_cols, key)
            print(f"  {fmt_pct(pct)}  base={bv:8.0f}  cand={cv:8.0f}  {label}")

    # Top wins (informational)
    for col, items in wins.items():
        if not items:
            continue
        col_name = run_col_name(col)
        print(f"\n--- Top wins in {col_name} (<-{args.gate}%) ---")
        for key, bv, cv, pct in sorted(items, key=lambda x: x[3])[:10]:
            label = detail_label(key_cols, key)
            print(f"  {fmt_pct(pct)}  base={bv:8.0f}  cand={cv:8.0f}  {label}")

    if args.verbose:
        print(f"\n--- All variants ({len(rows_out)} rows) ---")
        for key, col, bv, cv, pct in rows_out:
            label = format_key(key_cols, key)
            col_name = run_col_name(col)
            print(
                f"  {col_name:<24} {fmt_pct(pct)}  base={bv:8.0f}  cand={cv:8.0f}  {label}"
            )

    return 1 if any_regress else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(2)
