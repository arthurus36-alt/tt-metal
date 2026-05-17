# BH Fast Untilize TT-Metal Rollout Plan

**Owner:** pavlejosipovic
**Branch:** `pjosipovic/bh-untilize-t5`
**Status:** TT-Metal rollout started. Experimental LLKs and public LLK wrappers are committed; compute API helpers and the first automatic TTNN helper gate are under validation.

## Goal

Promote the BH fast-untilize path from the tt-llk experimental harness into TT-Metal in the same style as BH fast-tilize:

- Keep TTNN callsites using `compute_kernel_lib::untilize<W, input_cb, output_cb>(num_blocks)`.
- Hide fast/legacy selection inside `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl`.
- Preserve the existing legacy `pack_untilize` path for unsupported cases and as the fallback.
- Do not auto-route raw `pack_untilize_dest`: fast-untilize requires a custom unpack stream and dirty DEST layout owned by the full helper pipeline.

## Current Experimental Contract

Validated fast path:

- BH only.
- Tile shape: 32x32, four faces.
- Row decomposition: `unit_dim={4,2,3}`; `ct=1` remains legacy fallback.
- Input formats:
  - `Float16_b -> Float16_b`
  - `Float32 -> Float32`
  - `Bfp8_b/Bfp4_b -> Float16_b`
  - `Bfp8_b/Bfp4_b -> Float32`
- DEST accumulation:
  - 16-bit DEST for supported FP16 outputs.
  - Native fp32 DEST for FP32 outputs and `dest_acc=Yes` FP16 outputs.
- Pack paths:
  - Contiguous path when the unit is the full row.
  - Row-strided MOP/replay path for wider rows.
  - Direct row-strided fallback remains behind `FAST_UNTILIZE_STRIDED_MOP_REPLAY=0`.

Validation harness now covers:

- `DstSync::SyncHalf` and `DstSync::SyncFull`.
- Representative `ct > 8` rows, starting with `ct={9,12,16}`.

## Integration Shape

### 1. Add BH LLK API wrappers

Mirror fast-tilize wrappers under:

- `tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_unpack_fast_untilize_api.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_math_fast_untilize_api.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_pack_fast_untilize_api.h`

The wrappers should expose:

- `llk_unpack_fast_untilize_init/reinit/block/uninit`
- `llk_math_fast_untilize_init/block/uninit`
- `llk_pack_fast_untilize_init/block/block_strided/uninit`

They should pass `DST_SYNC_MODE` and `DST_ACCUM_MODE` through rather than baking in SyncHalf.

### 2. Add compute API helpers

In `tt_metal/hw/inc/api/compute/pack_untilize.h`, add BH-only helpers analogous to `fast_tilize_*`:

- `fast_untilize_init(input_cb, full_dim, output_cb)`
- `fast_untilize_block(input_cb, full_dim, output_cb, input_tile_index = 0, output_tile_index = 0)`
- `fast_untilize_uninit(input_cb, output_cb, full_dim)`

These helpers own the fast unpack/math/pack pipeline. They should fall back to existing `pack_untilize_*` for `full_dim == 1`.

### 3. Hide the gate in `untilize_helpers`

Add helper predicates in `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl`:

- `has_32x32_tiles<input_cb>()`
- `has_32x32_tiles<output_cb>()`
- `has_supported_fast_untilize_format<input_cb, output_cb>()`
- `can_use_fast_untilize<block_width_tiles, input_cb, output_cb>()`

Final gate:

```cpp
constexpr bool use_fast =
    can_use_fast_untilize<block_width_tiles, input_cb, output_cb>();
```

Initial predicate:

```cpp
return is_blackhole &&
       block_width_tiles >= 2 &&
       has_32x32_tiles<input_cb>() &&
       has_32x32_tiles<output_cb>() &&
       has_supported_fast_untilize_format<input_cb, output_cb>();
```

Production auto-selection starts with exact `Float16_b` output cases only. The helper APIs still carry the native fp32 DEST path, but `Float32` automatic selection remains gated off until the input path is lossless enough for TTNN's exact `untilize` contract.

Do not add a permanent `block_width_tiles <= 8` gate. The helper should use the row-decomposition path for wider rows. If a temporary bring-up limit is needed, make it explicit and remove it once `ct>8` validation passes.

Do not add a permanent SyncHalf-only gate. The wrappers and test harness should support both `DST_SYNC_MODE` values.

Do not add an output block-format gate beyond the existing untilize assertion. Row-major output cannot be a block format by construction.

### 4. Keep legacy fallback behavior

Fallback to the existing path for:

- Non-BH architectures.
- `ct == 1`.
- Non-32x32 tiles.
- Unsupported input/output formats.
- Raw `pack_untilize_dest` usage.
- Any future shape that trips a targeted safety gate during bring-up.

## Validation Gates

### LLK harness

Required before integration:

- `test_fast_untilize.py`
  - row-id and random stimuli.
  - `DstSync::{SyncHalf, SyncFull}`.
  - `ct={2..8,9,12,16}` and `rt={1,2,4}`.
  - supported format/dest matrix.
- `test_fast_untilize_overflow_guard`
  - `L1_TO_L1` and `PACK_ISOLATE`.
  - `DstSync::{SyncHalf, SyncFull}`.
  - same dimension and format matrix.
- `perf_fast_untilize.py`
  - include `DEST_SYNC`.
  - compare Half vs Full for performance sanity.
- `perf_fast_untilize_legacy_compare.py`
  - apples-to-apples fast vs legacy after production helper wiring.

### TT-Metal helper tests

Add a focused production-helper test that calls:

```cpp
compute_kernel_lib::untilize<W, input_cb, output_cb>(num_blocks);
```

Cover:

- Fast-selected cases: `W={2,4,8,9,12,16}`.
- Legacy fallback: `W=1`.
- SyncHalf and SyncFull.
- fp16 DEST and fp32 DEST.
- Guard/sentinel output bounds.

### Perf gate

Keep the existing rule:

- No correctness regressions.
- No fast-vs-saved regressions >2% on the fast path.
- Fast-vs-legacy should win full-pipeline `L1_TO_L1` on every gated production case.
- If `PACK_ISOLATE` has a small non-win below 2%, it is acceptable only when full-pipeline still wins and the shape remains faster than legacy.

## Rollout Sequence

1. Expand and pass the experimental harness for `DEST_SYNC` and `ct>8`. Done in this branch.
2. Add LLK API wrappers and port `fast_untilize_test.cpp` to use them. Done in `4e7c84b79ea`.
3. Add `fast_untilize_*` compute API helpers. In progress.
4. Add `can_use_fast_untilize` in `untilize_helpers.inl`, but keep production helper tests focused. In progress, with the initial production gate limited to exact fp16-output cases.
5. Enable automatic selection inside `compute_kernel_lib::untilize`.
6. Run full LLK accuracy/perf plus production helper tests.
7. Run targeted TTNN operation smoke tests that use `untilize_helpers.hpp`.
8. Keep the direct row-strided fallback available until the production path has CI soak.

## Open Risks

- SyncFull phase selection has not been silicon-validated yet.
- Wider rows stress repeated row-strided chunks; guard tests must remain in the gate.
- Production helper waits/push/pop behavior differs from the standalone experimental harness, so helper-level tests are mandatory.
- Float32 input still follows the current SrcA/SrcB route and may narrow through TF32 before math; this is native fp32 DEST support, not a lossless fp32 input pipeline.
