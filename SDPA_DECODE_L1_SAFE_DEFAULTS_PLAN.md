# SDPA Decode L1-Safe Defaults Plan

## Context

Issue: `ttnn.transformer.paged_scaled_dot_product_attention_decode` can overflow per-core L1 when callers omit blocking controls or pass large default-ish schedules for wide-head paged decode shapes such as Gemma global attention.

Local repro shape:

- `b=1`
- `nh=8`
- `nkv=1`
- `s=1024`
- `d=512`
- `block_size=32`
- `kv_dtype=ttnn.bfloat8_b`
- `q_dtype=ttnn.bfloat16`
- grid `(8, 8)`

Observed failures:

- Explicit helper config with `k_chunk_size=512`, all 64 cores per head: static CB region grows to `2071008 B`, max L1 is `1499136 B`.
- True `program_config=None`: static CB region grows to `1985024 B`, max L1 is `1499136 B`.

Manual conservative config that passes locally:

```python
ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
    q_chunk_size=32,
    k_chunk_size=32,
    exp_approx_mode=False,
    max_cores_per_head_batch=16,
)
```

The implemented no-config default is intentionally more pessimistic than this
manual passing config: it uses `max_cores_per_head_batch=1` to minimize the
decode reduction scratch footprint by default.

## Prefill Reference Behavior

The non-decode SDPA path is conservative by default:

- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp` defaults `q_chunk_size=32` and `k_chunk_size=32` when no `program_config` is supplied.
- It uses lightweight masks and streaming output CB sizing where possible.
- It picks matmul subblocks with `detail::determine_largest_subblock_size`, but this controls DST/register blocking, not L1 planning.

Decode does not currently mirror that behavior:

- Paged decode defaults `k_chunk_size=0`, which enables dynamic chunking in kernels.
- When no `program_config` is supplied, `max_cores_per_head_batch` effectively becomes all available cores.
- Decode's L1-heavy term is:

```cpp
intermed_output_tiles = (out_tiles + 2 * PNHt) * (num_cores_per_head - 1)
```

That term grows directly with `num_cores_per_head`.

## Goals

1. Preserve all explicit user-provided `SDPAProgramConfig` behavior.
2. Make omitted paged decode blocking conservative by default.
3. Avoid host-side L1 footprint estimation in this change.
4. Keep the default behavior deterministic and included in the program cache key.
5. Add regression coverage for both the repro and existing small/head_dim=128 decode cases.

## Non-Goals

- Do not change kernel math or output numerics.
- Do not redesign decode parallelization.
- Do not optimize for peak performance in this change; prefer correctness and compile-time L1 safety.
- Do not silently override explicit user blocking values that fail L1 validation.

## Implementation Plan

### 1. Add A Fixed Default Config

Add the defaulting in the paged decode wrapper:

```text
ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp
```

The wrapper should materialize an `SDPAProgramConfig` in
`paged_scaled_dot_product_attention_decode` when `program_config` is absent.

Materializing the default before calling `prim::sdpa_decode` keeps the chosen
grid/chunk/core cap in the program cache key through the primitive's
`operation_attributes.program_config`, while keeping the behavior scoped to
`paged_scaled_dot_product_attention_decode`.

### 2. Mirror Prefill Conservative Chunking

For paged decode defaults:

- Set `q_chunk_size=32`.
- Set `k_chunk_size=32`.
- Set `compute_with_storage_grid_size=device->compute_with_storage_grid_size()`.
- Leave `exp_approx_mode=std::nullopt` unless there is a strong reason to force `false`.
- Set `max_cores_per_head_batch=1`.
- Preserve existing explicit `program_config.k_chunk_size > 0` validation.

This matches the prefill default chunking model and fixes paged non-causal validation, which currently requires a positive `k_chunk_size`. The core cap is deliberately more conservative than prefill because decode's reduction scratch grows with `num_cores_per_head`.

### 3. Use A Very Pessimistic Core Cap

Do not add a host-side L1 estimator. The default no-config path should use:

```cpp
max_cores_per_head_batch = 1;
```

This is the most conservative setting available through the existing program config and removes the L1-heavy reduction scratch term for default decode schedules. If this still does not fit for a future shape, let existing program validation fail with the normal precise L1 error.

### 4. Respect Explicit Configs

Rules:

- If `program_config` is `None` in `paged_scaled_dot_product_attention_decode`, materialize the conservative default.
- If `program_config` is present, do not override it.
- If `program_config` is present with `k_chunk_size=0`, keep current semantics unless the team decides partial config should also be fixed.

Optional follow-up:

- Treat `k_chunk_size=0` inside a provided config as "auto" and fill only that field, but this changes explicit-config behavior and should be separate.

### 5. Keep The Implementation Simple

Do not duplicate factory CB formulas in the wrapper. The wrapper should only
materialize fixed defaults; detailed L1 validation remains in the existing
program validation path.

### 6. Update Nightly Regression Test

Add the Gemma-style no-config repro to the existing nightly decode test file:

```text
tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_decode.py
```

Use the shared paged decode runner so the regression follows the same paging,
masking, PCC, seed, and device-grid skip behavior as the rest of the nightly
SDPA decode tests. The regression should call
`paged_scaled_dot_product_attention_decode` with `program_config=None`.

## Test Plan

### Unit/Repro Tests

Run the new nightly repro:

```bash
pytest -svv tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_decode.py::test_sdpa_decode_paged_attention_default_config_regression
```

Expected after implementation:

- Default/no-config case passes for the Gemma-style paged decode shape.
- Explicit unsafe configs are still covered as a targeted manual check because the implementation does not override supplied `program_config` values.

### Existing SDPA Decode Tests

Run paged decode and standard decode unit coverage:

```bash
pytest -svv tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py
pytest -svv tests/ttnn/unit_tests/operations/sdpa/test_paged_sdpa_decode_flexible_geometry.py
pytest -svv tests/ttnn/unit_tests/operations/sdpa/test_sdpa_prefill.py
```

`test_sdpa_prefill.py` is included because the plan intentionally mirrors prefill defaults and should not regress shared config assumptions.

### Targeted Manual Checks

Run direct calls for the repro shape:

1. `program_config=None`, causal, `cur_pos=1023`: should pass.
2. `program_config=None`, non-causal full sequence: should pass.
3. Explicit `k_chunk_size=512`, `max_cores_per_head_batch=64`: should fail with the existing L1 error.
4. Explicit conservative config `k_chunk_size=32`, `max_cores_per_head_batch=1`: should pass.

### Device Matrix

Minimum:

- Wormhole B0 single device, because the repro is confirmed locally.

Preferred:

- Blackhole P150, matching issue #44311.
- Any harvested Wormhole configuration where `compute_with_storage_grid_size()` is smaller than `(8, 8)`, to verify skip behavior.

### Regression Signals

Watch for:

- Changed output PCC in existing decode tests.
- Program-cache collisions from default configs not entering the hash.
- Unexpected performance drops for explicit configs.
- Non-causal paged decode requiring a positive `k_chunk_size`.
- Shapes with `head_dim=128` still using enough cores for existing performance-sensitive tests.

### Logging/Debugging

Add a `log_debug(tt::LogOp, ...)` line when the default resolver materializes a config:

```text
Paged SDPA decode default config: grid={}, q_chunk_size={}, k_chunk_size={}, max_cores_per_head_batch={}
```

This should stay debug-only.

## Acceptance Criteria

1. `program_config=None` no longer L1-overflows for the local Gemma-like paged decode repro.
2. Explicit unsafe configs are still honored and fail normally.
3. Existing decode/paged decode tests pass.
4. The chosen default config is included in the operation hash.
5. The implementation is a small fixed-default wrapper change with no host-side L1 estimator.
