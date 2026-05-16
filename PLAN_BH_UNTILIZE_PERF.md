# PLAN: BH `pack_untilize` / `unpack_untilize` Performance

**Owner:** pavlejosipovic
**Started:** 2026-05-16
**Tracking:** tt-metal#42047 (epic), tt-llk#916 (analysis), tt-llk#391 (Conv driver)
**Driver:** DeepSeek prefill un/tilize bottleneck; Conv bfp8 throughput

---

## 1. Goal and success metrics

Close the **~130% pack_untilize gap** between BH and WH (per tt-llk#916). Concretely:

| Metric (FP16→FP16, 8×8 block, L1_TO_L1 KERNEL) | Current BH | WH baseline | Target |
|---|---|---|---|
| Cycles / tile | **126.5** | ~54 | ≤ 70 (50% gap closed) |
| Cycles / tile, asymptote | 126 | 54 | ≤ 60 |
| Bfp8→FP16, ct=5/7 cliff | 460 | n/a | ≤ 180 (host fix) |

Stretch: approach the 8 cyc/tile speed-of-light (BH unpack/pack BW = 256 B/cyc × 2048 B/tile FP16). Not realistic in one project — the model `32 + 32·PACR + 6·32/block_ct_dim` says the achievable floor with full 4-interface usage is ~24 cyc/tile asymptotic. **24 cyc/tile is the real target.**

Coverage: same plan applies to `unpack_untilize` (smaller absolute win but same playbook).

---

## 2. Current state (measured 2026-05-16 on p100a, branch `main` @ a8aa13392b0)

Per-tile cycles, L1_TO_L1 KERNEL mean over 8×8 sweep:

| Format | cyc/tile (asymptote at 8×8) | vs SoL |
|---|---|---|
| FP16 → FP16 (pack) | 126.5 | 16× |
| Bfp8_b → FP16 (pack) | 171.5 (good ct); 460 (ct=5/7) | 21× / 57× |
| Float16 → Float16 (unpack) | 126.5 (KERNEL); 49 (TILE_LOOP isolated) | 6× isolated |

Full CSV: `tt_metal/tt-llk/perf_data/perf_{pack,unpack}_untilize/*.post.csv`.

---

## 3. HW capability gaps (current BH `llk_pack_untilize.h`)

Confirmed via ISA docs (`/localdev/pjosipovic/refs/tt-isa-documentation/`) and code inspection. Each gap maps to one or more epic sub-issues.

| # | Gap | Where | Maps to |
|---|---|---|---|
| G1 | Uses 2 of 4 packer interfaces (`TWO_INTFS_ACTIVE`) | `llk_pack_untilize.h:51` | #42048 |
| G2 | Only programs **one** L1_Dest_addr cfg slot (`THCON_SEC0_REG1`) — `program_packer_untilized_destination` is an empty stub | `cpack_common.h:713-716` (commented out) | #42048 |
| G3 | Per-row `ADDDMAREG → STALLWAIT(THCON) → WRCFG → NOP` (~6 cyc + stall × 32 rows / block) | `llk_pack_untilize.h:100-104` | **#42050** |
| G4 | Explicit `SETADC` / `INCADC*` / `ADDRCRZW` counter manipulation per row | `llk_pack_untilize.h:91, 107, 222-224, 231-235` | #42051 |
| G5 | Ch1 (output) counters not used for L1 address; everything routed through cfg writes | `llk_pack_untilize.h:97-104` | #42052 |
| G6 | Math thread datacopy emits clean tile layout in Dst; pack must transpose via strided mode | `llk_pack_untilize_perf.cpp`, math thread; `_llk_math_eltwise_unary_datacopy_` | #42049 (prereq for #42048) |
| G7 | Single `CFG_STATE_ID` — no bank ping-pong, per-row `STALLWAIT(C12)` mandatory | global LLK state | (post-epic) |
| G8 | Host harness caps `max_block_dim=4` for Bfp8_b input → ct=5/7 cliff | `perf_pack_untilize.py:62` | **G8 (host-side)** |

**Validation infrastructure gap (V1):** ttsim is functional-only, no cycle accuracy. Silicon p100a is the only cycle-truth source. Plan must include automated diff vs baseline CSV after every change.

---

## 4. Validation plan

### 4.1 Silicon (cycle truth)

Existing harness already gives per-format/dim sweep with `PerfConfig` (KERNEL / TILE_LOOP / INIT zones, multi-run-type: L1_TO_L1 / PACK_ISOLATE / L1_CONGESTION). Use it.

Workflow per change:
1. Run `perf_pack_untilize.py` + `perf_unpack_untilize.py` via `.claude/scripts/run_test.sh` (already validated, ~95s combined).
2. Diff `perf_data/perf_pack_untilize/perf_pack_untilize.post.csv` against `main` baseline using a small Python diff script (see Task 0.2).
3. Reject any regression > 2% on existing passing variants.

### 4.2 Functional regression (craq-sim — local ttsim implementation)

**Use the craq-sim build at `/localdev/pjosipovic/refs/craq-sim/`, NOT upstream `ttsim` from pip.** craq-sim is the local fork/workbench that gets parity fixes before they harden upstream; it's the implementation that backs `TT_METAL_SIMULATOR` for this work.

craq-sim has no cycle model but is bit-exact functional. Use it as a **correctness gate before silicon** — catches PCC regressions and `UnimplementedFunctionality` if a new MOP touches an unmodeled PACR sub-mode. Per `tt_metal/tt-llk/tests/TTSIM.md`:

```bash
# Build craq-sim (BH target)
cd /localdev/pjosipovic/refs/craq-sim/src
../make.py :build                                # produces _out/release_bh/libttsim.so

# Stage
mkdir -p $HOME/sim
cp _out/release_bh/libttsim.so $HOME/sim/libttsim_bh.so
cp /localdev/pjosipovic/tt-metal/tt_metal/soc_descriptors/blackhole_140_arch.yaml $HOME/sim/soc_descriptor.yaml

# Run LLK tests under craq-sim (not silicon)
export TT_METAL_HOME=/localdev/pjosipovic/tt-metal
export TT_METAL_SIMULATOR=$HOME/sim/libttsim_bh.so
export TT_METAL_DISABLE_SFPLOADMACRO=1           # craq-sim ISA gap
cd $TT_METAL_HOME/tt_metal/tt-llk/tests/python_tests
pytest --run-simulator test_pack_untilize.py test_unpack_untilize.py
```

Curated runners: `/localdev/pjosipovic/refs/craq-sim/tests/llk-smoke.sh bh` and `/localdev/pjosipovic/refs/craq-sim/scripts/llk-weekly.sh bh`.

### 4.3 ISA reference

`/localdev/pjosipovic/refs/tt-isa-documentation/`:
- BH instructions: `CFGSHIFTMASK.md`, `RMWCIB.md`, `WRCFG.md`, `STALLWAIT.md`, `ConfigurationUnit.md`, `BackendConfiguration.md`
- Packer arch (BH lacks dedicated docs — WH applies, same arch): `WormholeB0/.../Packers/{README,InputAddressGenerator,OutputAddressGenerator}.md`, `PACR.md`, `ADCs.md`

### 4.4 craq-sim — deep debugging

For hard-to-reproduce silicon bugs, single-step in craq-sim source (it's the same binary used in 4.2, just with debug build + breakpoints in the C++ model).

```bash
cd /localdev/pjosipovic/refs/craq-sim/src
../make.py _out/debug_bh/libttsim.so             # debug build
```

PACR functional model at `craq-sim/src/tensix.cpp:2452` (`TENSIX_EXECUTE_PACR`). UNPACR at `:3340`. PACR_SETREG at `:2386`. Some PACR sub-modes (`PACR_STRIDE`, `PACR0_FACE`, `PACR0_ROW`) throw `UnimplementedFunctionality` (`tensix.cpp:6926-6941`) — useful guardrail before silicon. If T5's 4-interface MOP hits one of these, file an upstream craq-sim issue + work around.

**Important:** Do not rely on `ttsim` pip package — it lags craq-sim. Always build from `/localdev/pjosipovic/refs/craq-sim/`.

---

## 5. Task list (priority order)

Order is by **expected-win ÷ risk**. Each task lists: prereqs, files, sketch, expected win, validation, rollback.

---

### Task 0 — Baseline & infra (1 day)

**0.1 Capture clean baseline** ✅
- Baseline CSVs at `/localdev/pjosipovic/perf_baselines/main_a8aa13392b0/` (commit a8aa13392b0). Not committed — `perf_data/` is gitignored and baselines are large.
- Pull WH baseline from tt-llk#916 attachment (TODO).

**0.2 Add `compare_perf_csv.py` script** ✅
- Location: `tt_metal/tt-llk/tests/compare_perf_csv.py`.
- Inputs: baseline.post.csv, candidate.post.csv. Outputs: per-variant cyc/tile delta, %change, pass/fail at ±2% gate.
- Sanity-tested (self-diff = 0%).

**0.3 Set up working branch** ✅
- Branch `pjosipovic/bh-untilize-perf` off `main` @ a8aa13392b0.

**Expected win:** 0%. Enables every subsequent step.

**Rollback:** N/A.

---

### Task 1 — Investigate Bfp8 `max_block_dim` cap (DROPPED — not over-conservative)

**Original gap (G8):** suspected harness-side cap that could be lifted.

**Finding (2026-05-16):** The cap is a real HW constraint, not a harness artifact. Flows from `is_format_combination_outlier` in `helpers/data_format_inference.py:106`:
- 8-bit exponent input (Float16_b, Bfp8_b) → Float16 output, dest_acc=No is an HW outlier — *cannot convert directly without storing as 32-bit intermediate in dest.*
- Dest silently widens to 32-bit → `max_tiles_in_dest = 8 // 2 = 4`.
- For dest_acc=Yes, same constraint applies via the standard fp32-dest path.

**Why ct=5/7 cliff persists:** divisors of 5 ≤ 4 = {1}; divisors of 7 ≤ 4 = {1}. The LLK has `static_assert(full_ct_dim % block_ct_dim == 0)` (`llk_pack_untilize.h:147`). To eliminate the cliff would require either:
- Relaxing the LLK to support non-uniform block sizes (e.g. ct=5 = [4,1] blocks) — significant LLK change, separate work.
- Caller-side split: caller issues two pack_untilize calls (4+1). This is a tt-metal op-level change, not LLK.

**Decision:** Drop T1 from Path A. Document the ct=5/7 cliff as a known structural limit. Revisit as a separate workstream if Conv/DeepSeek hit it on production shapes.

---

### Task 2 — `CFGSHIFTMASK` for per-row L1 address (#42050) (3 days)

**Gap:** G3.
**Files:**
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_untilize.h:93-107` (replay buf + `set_end_ops`)
- Optionally `cpack_common.h` if helper needed.

**Precedent to mirror exactly:** `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_tilize.h:285,292` — the *only* current CFGSHIFTMASK call site in BH `llk_lib`.

```cpp
// Pattern from llk_unpack_tilize.h:285
TTI_CFGSHIFTMASK(/*MaskMode=*/1, /*AluMode=*/0b011 /*+=*/,
                 /*MaskWidth=*/32-1, /*RotateAmt=*/0,
                 /*ScratchIndex=*/0b11,
                 THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
```

**Sketch:**
1. Pre-load row-stride into a SCRATCH_SEC slot during `_llk_pack_untilize_init_` (one-time cost — `TT_SETDMAREG` + `WRCFG` to scratch).
2. Replace the 4-instruction replay buf body:
   ```cpp
   // Before:
   TTI_ADDDMAREG(0, OUTPUT_ADDR, OUTPUT_ADDR, OUTPUT_ADDR_OFFSET);
   TTI_STALLWAIT(STALL_CFG, THCON);
   TTI_WRCFG(OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
   TTI_NOP;
   ```
   with a single `TTI_CFGSHIFTMASK` doing `cfg += SCRATCH`. (Saves ADDDMAREG and removes the STALLWAIT+NOP.)
3. Cost model: 2-cycle non-pipelined `CFGSHIFTMASK` vs 1+stall+2+1 = ≥4 cyc + STALLWAIT (≥5 cyc) → saves ≥3 cyc/row × 32 rows / 8-tile block = **96 cyc / block, or ~12 cyc/tile** at 8×8.

**Gotchas (from ISA):**
- CFGSHIFTMASK is **2-cycle, non-pipelined**. If two consecutive rows both fire CFGSHIFTMASK back-to-back, latency dominates pipelined WRCFG. So measure both — if pack's other work between rows fills the bubble, win is real; if not, win shrinks.
- The instruction immediately after CFGSHIFTMASK must not consume the just-written cfg (ISA pipeline rule). Insert one cheap instruction (e.g. INCADCXY for next row's Y counter) or rely on the next being a config op.
- Mask width is 5 bits — ensure row-stride bits fit.

**Expected win:** **10–20%** on FP16→FP16, similar on bfp8. Asymptote 126 → ~100 cyc/tile.

**Validation:**
1. ttsim functional PASS first.
2. Silicon: full perf_pack_untilize.py sweep. Diff vs baseline; require no functional regression, ≥10% perf win on at least 50% of variants.
3. Special attention to PACK_ISOLATE and L1_CONGESTION — make sure the savings aren't getting eaten by changed cross-thread sync.

**Rollback:** Revert one file. Self-contained.

**Related closed PR to study:** tt-llk PR #1219 (`filip/unpack_AB_mmul_CFGSHIFTMASK`) — same technique applied to unpack matmul. Closed without merge — read the discussion for pitfalls. PR #1543 (Quasar unpack_tilize_block) merged with similar pattern.

---

### Task 3 — `AddrMod`-driven dst+L1 advance (#42051) (4 days)

**Gap:** G4.
**Files:**
- `llk_pack_untilize.h:20-26` (`_llk_pack_untilize_configure_addrmod_`)
- `llk_pack_untilize.h:64-133` (MOP config)

**Sketch:**
Today the MOP uses `ADDR_MOD_0` with `.y_src.incr=0, .clr=0` and relies on **explicit** `INCADCZW`, `INCADCXY`, `ADDRCRZW`, `SETADCZW`, `SETADCXY` instructions interleaved with PACR. The hardware AddrMod registers (`ThreadConfig.ADDR_MOD_PACK_SEC[0..3]`) support per-PACR Y/Z auto-advance + carry-and-reset (CR) modes — see WH `ADCs.md`.

Move the per-row Y increment into the AddrMod of the PACR itself; eliminate the explicit `INCADCXY` end-op and the `ADDRCRZW` start-op. Use two AddrMod slots: `ADDR_MOD_0` for "advance W to next tile in row", `ADDR_MOD_1` for "advance Y to next row, reset W via CR".

**Precedent:** tt-llk PR #1543 (`_llk_unpack_tilize_block_` for Quasar) uses exactly this pattern — read the merged diff.

**Expected win:** ~5-10% on top of Task 2. Removes 2 instructions per row × 32 rows = 64 instructions per 8-tile block.

**Validation:** Same as Task 2. Particular care to functional correctness — AddrMod misprogramming silently produces wrong addresses.

**Rollback:** Revert.

---

### Task 4 — Ch1 (output side) counters (#42052) — ATTEMPTED, REVERTED (silicon evidence: BH has 256B mask)

**Status (2026-05-16):** Implemented and tested on silicon. Reverted; functional regression on small strides.

**What was tried:**
1. `_llk_pack_untilize_configure_addrmod_`: added `.y_dst.incr=1` to `ADDR_MOD_1` so the row-closing PACR auto-advances ch1.Y post-PACR.
2. `_llk_pack_untilize_init_`: programmed `PCK0_ADDR_CTRL_XY_REG_1_Ystride = output_addr_offset / 16`. Removed the SCRATCH_SEC2 setup (no CFGSHIFTMASK needed).
3. `_llk_pack_untilize_mop_config_`: removed the `load_replay_buf` and the `set_end_op(replay)` entirely. End-ops default to NOP.
4. `_llk_pack_untilize_`: SETADCXY BitMask `0b0011` → `0b1111` to reset ch1.X and ch1.Y at start of each pack call. Kept the between-face-passes ch0.Y-only reset so ch1.Y continues 16 → 32 across top/bottom passes.

**Findings (silicon p100a):**
- perf_pack_untilize.py: 832 passed (kernel doesn't hang) — but **perf harness doesn't validate output correctness** (`helpers/perf.py` measures cycles only).
- test_zzz_pack_untilize.py: **FAILED at the very first variant** (`Float16_b→Float16_b, [64,64]`, ct=2 → 128 B/row → 8 in 16B units → masks to 0). Output rows all wrote to the same base address. Golden-tensor mismatch.

**Conclusion:** BH applies the same `(YZW_Addr & ~0xf)` 16B-unit mask documented in the WH ISA — confirmed empirically. Effective per-row stride floor is **256 bytes**. For our perf sweep:
- ct=1..3 (FP16: 64..192 B/row) — broken
- ct=4 (FP16: 256 B/row) — works
- ct=5..7 (320..448 B/row, not multiples of 256) — broken (truncated to 256)
- ct=8 (512 B/row, multiple of 256) — works

T4 as a drop-in replacement for CFGSHIFTMASK is **not viable** because most shapes hit the mask. Two ways forward (both out of Path A scope):

1. **Conditional T4:** gate ch1.Y path behind `output_addr_offset % 256 == 0`; fall back to T2 CFGSHIFTMASK for small strides. Adds two MOP variants. Win only on ct=4, ct=8.
2. **Extend craq-sim** to model packer ch1.Y/Z/W → confirm the `~0xf` mask in software model → then revisit T4 design with full understanding.

**Reverted to T3 state.** Diff: `git show 0a16daf4b4b` (T3 commit unchanged). Functional test_zzz_pack_untilize.py passes (156 / 100 skip).

**For the record:** the mask hypothesis is now confirmed (was speculation before). Update `[[bh_pack_untilize_perf]]` memory with this finding.

### Task 4 (deferred) — original sketch retained below for reference

**Original Gap:** G5.

**Gap:** G5.
**Files:** `llk_pack_untilize.h` (MOP), `cpack_common.h` (addr ctrl programming).

**Sketch:**
Output address generator (ch1) has its own Y/Z/W counters with strides `PCK0_ADDR_CTRL_XY_REG_1_Ystride` / `PCK0_ADDR_CTRL_ZW_REG_1_*`. Currently only ch0 (input/dst) is driven through ADC instructions; ch1 (output L1) is driven through cfg writes (`L1_Dest_addr` update each row).

Configure ch1 strides at init so that `INCADCXY/ZW` auto-advances the L1 destination address **per PACR**, removing the need to write the cfg register every row entirely.

**Expected win:** Synergistic with #42050. If both ch1 counters and CFGSHIFTMASK are used, the cfg-write path becomes "set once at init, never again per row" — saves the remaining 2-cycle CFGSHIFTMASK per row.

**Validation:** Same workflow. This is harder to verify — ch1 stride misprogramming makes L1 writes land at wrong addresses, producing zero data corruption silently if mask happens to align. ttsim catches PCC mismatch, use it before silicon.

**Rollback:** Revert.

---

### T5 design correction (2026-05-16): BH packer model is radically different from WH

**Findings after deeper craq-sim source + LLK code survey (branch `pjosipovic/bh-untilize-t5`):**

1. BH has **1 packer with 4 read interfaces** (`cpack_common.h:24` `NUM_PACKERS = 1`). WH has 4 independent packers. The "4-way parallel" mental model lifted from WH is **wrong for BH**.

2. The constant `p_pacr::ALL_INTF_ACTIVE = 0b0000` on BH enables all 4 read interfaces, multiplying the per-PACR datum count by 4. Per `craq-sim/src/tensix.cpp:2675-2685`:
   ```c
   if (!read_intf_sel) { count *= 4; }
   else { count *= __builtin_popcount(read_intf_sel); }
   constexpr uint32_t n_packers = 1;
   ```
3. **Only 1 L1_Dest_addr cfg slot (SEC0_REG1) is consumed on BH.** craq-sim gates reads of SEC0_REG8/SEC1_REG1/SEC1_REG8 behind `#if TT_VERSION == 0` (WH only). Every silicon-validated BH ALL_INTF_ACTIVE callsite (`llk_pack.h:252`, `llk_pack_untilize.h:58` dense path, `experimental/llk_pack_fast_tilize.h`, `experimental/llk_pack_block.h`) programs exactly one L1 cfg slot via `program_packer_destination`.

4. **L1 output of one PACR is a single contiguous byte run** (`craq-sim/src/tensix.cpp:3183-3201`). The 4 interfaces *concatenate* their Dst reads into one L1 write — there's no L1-side fan-out. STRIDED_MODE only affects which 4 Dst row groups feed the interfaces (`pack_row + 16*(i/ROW_SIZE)`); it does not split the L1 destination.

5. **Implication for pack_untilize:** ALL_INTF_ACTIVE on the current Dst layout would read 4 face-rows of the **same** tile (Dst rows R, R+16, R+32, R+48 — the 4 quadrants of one tile) and concatenate them into one L1 byte run. This is the WRONG layout for an L1-row-major strip — adjacent L1 bytes should hold face-rows of *different tiles*, not different faces of the same tile.

**Two viable T5 paths (both invasive):**

**T5-A: Generalize the existing `dense` mode.** `dense=true` already uses ALL_INTF_ACTIVE on BH (`llk_pack_untilize.h:51-58`) for the num_faces=2 case where two 16x32 tiles laid out in Dst look like one 4-face tile to the packer. Extending to num_faces=4 requires the math thread to lay out 4 tiles in Dst such that "Dst rows R, R+16, R+32, R+48" hold face-row R of tiles 0, 1, 2, 3 respectively. **The current standard A2D datacopy does NOT produce this layout** — tile k normally occupies Dst rows 64·k..64·k+63 (one tile = 64 contiguous Dst rows). For T5-A, math must place tile k's face-row R at Dst row R+16·k.

**T5-B: Backport fast_tilize's pack MOP wholesale.** fast_tilize already does this Dst layout via `_llk_math_fast_tilize_init_` (`addr_mod_t{.dest.incr=16}` + MOV_8_ROWS). T5 would copy fast_tilize's math + repurpose its pack MOP with a different per-PACR L1 advance pattern (untilize output is row-major across multiple tile columns, not contiguous per-tile).

**Both paths require #42049 (math Dst rearrangement) to land FIRST.** The original plan called #42049 "may not be needed" — that was wrong. #42049 is the load-bearing piece; without it, no 4-interface fast path is possible on BH.

**Restriction:** fast_tilize's swizzle layout fits 4 tiles per 64 Dst rows (= 1 standard tile slot). So T5 fast path is naturally constrained to `block_ct_dim = 4` (or `≤4` with padding). For workloads needing block_ct_dim=8, would need either 2 sequential fast-path calls or 2-tile-slot Dst layout.

**Decision (2026-05-16):** T5 is significantly bigger than the original 8-day estimate. The math co-design (#42049) is now confirmed mandatory, and adds substantial complexity (new template path on math LLK, MOVA2D 32-bit Dst quirk, DEST remap concerns). **Re-scoping T5 to a longer multi-PR effort:** start with #42049 math implementation in isolation (silicon-validated against fast_tilize as oracle), then build pack-side T5-B atop it.

**T5.X probe (2026-05-16, on `pjosipovic/bh-untilize-t5`):** Tested whether a 1-line swap (`TWO_INTFS_ACTIVE` → `ALL_INTF_ACTIVE` in pack_untilize MOP) would deliver per-PACR throughput gain — bypassing the full math+pack rewrite to see if HW count multiplier alone helps. Result on silicon:

| Run-type | mean delta vs T3 | max regr |
|--|--|--|
| L1_TO_L1 | +0.91% | +25.2% (Bfp8→FP32) |
| PACK_ISOLATE | +1.03% | +29.2% (Bfp8→FP32) |
| L1_CONGESTION[PACK] | +1.07% | +30.5% |
| L1_CONGESTION[UNPACK] | +0.02% | (unchanged, expected) |

**Conclusion:** Per-PACR cycle cost scales with output datum count — emitting 64 datums (4 intfs) costs ~2× the cycles of emitting 32 (2 intfs). No net throughput multiplier from the mask alone. Bfp8→FP32 paths regress because format-conversion cost compounds.

**Implication:** The fast_tilize-style win comes from **MOP structure** (16 PACRs/tile filling one contiguous L1 region with the right Dst layout), not from the mask. T5-B requires the full math + pack co-design — no shortcut exists.

**T5.3 implementation status (2026-05-16, branch `pjosipovic/bh-untilize-t5`):**

Files added:
- `tt_llk_blackhole/llk_lib/experimental/llk_pack_fast_untilize.h` (pack MOP)
- `tests/sources/fast_untilize_test.cpp` (test source — fast_tilize unpack+math + new pack)
- `tests/python_tests/test_fast_untilize.py` (ct=4 FP16 only)

Status: **compiles clean, silicon runs without hang, PCC fails first attempt**.
Output data is present but at wrong positions — indicates ADC strides / AddrMod chain don't match the Dst layout fast_tilize math produces.

Hypotheses for next iteration:
1. **PACR output is fixed at 64 contiguous L1 datums.** With fast_tilize pack MOP (16 PACRs/tile × 4 tiles = 64 PACRs), output is "4 contiguous tiles in tile-format". For RM strip, the per-PACR Dst reads must be CHOSEN such that the concatenated byte stream IS the RM strip ordering. Current MOP doesn't do this — it tries to swap blocks via z+=1/clr but that may not produce RM-row-major byte order.
2. **Dst layout assumption (4 tiles vertically stacked at 16-row intervals) may not match what fast_tilize math actually produces.** Per fast_tilize comments, layout is 8-data + 8-gap pattern per tile = 128 Dst rows per tile, NOT 16. Re-derive correct stride values.
3. **Need a simpler test case first** — block_ct_dim=1 (single tile untilize via 4 interfaces) to isolate strides+addrmod without the multi-tile complications.

Recommended next steps when work resumes:
- Cross-reference fast_tilize's per-PACR data flow against ttsim trace
- Add Dst-introspection (print first N Dst rows post-math) to verify layout
- Implement block_ct_dim=1 fallback path FIRST, validate PCC, then scale up to 4

### Task 5 — `dirty tile layout` in Dst + 4 packer interfaces (#42048 + #42049) (originally 8 days, now larger)

**Gap:** G1, G2, G6.

**Existing precedent on BH silicon: `fast_tilize`** (in `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/`). It already implements three of the four pieces Task 5 needs and runs on silicon. **Read these first before writing code:**
- `experimental/llk_pack_fast_tilize.h` — `ALL_INTF_ACTIVE` PACRs with `DST_ACCESS_STRIDED_MODE` (lines 35-87, `EMIT_FACE_PACRS`); 4 AddrMod slots (lines 27-33); tile-granularity address advance in MOP `set_end_ops` (lines 101-104) — not per-row
- `experimental/llk_math_fast_tilize.h` — math-side dirty Dst layout via `addr_mod_t{.dest.incr=16}` + `MOV_8_ROWS` (lines 52-62, comments 38-41); template for #42049
- `experimental/llk_unpack_fast_tilize.h` — companion unpack-side patterns

**fast_tilize measured perf (silicon, BH p100a, FP16→FP16, rt=1, ct=8):**
- TILE_LOOP L1_TO_L1: **33.9 cyc/tile**
- PACK_ISOLATE: **27.6 cyc/tile**

This is a silicon proof that 4-intf + dirty Dst can push pack to ~3-4× faster than current pack_untilize (126 cyc/tile asymptote). Matches the stretch target of ~24 cyc/tile.

**What fast_tilize does NOT solve** (the genuinely new work in Task 5): it writes output to a **single contiguous tile**, so it programs only one `L1_Dest_addr` (`THCON_SEC0_REG1`). Pack untilize is the inverse direction — tile → L1-row-major, where 4 face quadrants must land at 4 **non-contiguous** L1 row groups. So we still need:
- The empty `program_packer_untilized_destination` stub at `cpack_common.h:711-716` filled in.
- Per-row cfg traffic to **four** `L1_Dest_addr` slots (`THCON_SEC0_REG1/REG8` + `SEC1_REG1/REG8`).
- Per-packer DEST_OFFSET differentiation, since each packer interface reads a different Dst quadrant.

**Files:**
- `cpack_common.h:711-716` — implement `program_packer_untilized_destination` (currently empty stub). Multi-cfg-slot programming is the unique-to-untilize piece.
- `llk_pack_untilize.h` — copy fast_tilize's MOP skeleton (`ALL_INTF_ACTIVE`, AddrMod-driven advance, tile-granularity end_ops); adapt addressing for the inverse direction (4 L1_Dest_addr slots stepping by face-row stride, not single tile base).
- `llk_math_eltwise_unary_datacopy_` — emit dirty tile layout in Dst when called from the pack_untilize variant. **This is #42049 and the prereq.** Copy `addr_mod_t{.dest.incr=16}` + `MOV_8_ROWS` pattern from `llk_math_fast_tilize.h:52-62`.

**Sketch:**

1. **#42049 first — math-side dirty layout.** Re-order faces in Dst so each packer interface reads from a fixed Dst offset. Reuse `llk_math_fast_tilize.h:52-62` pattern (`addr_mod_t{.dest.incr=16}` + MOVA2D 8-row groups). New template flag on `_llk_math_eltwise_unary_datacopy_` (e.g. `DestLayout::DirtyForUntilize`), default off — never affect non-untilize datacopy.
   - **Constraint inherited from fast_tilize:** MOVA2D doesn't handle 32-bit Dst rows correctly on BH (HW quirk, see `llk_math_fast_tilize.h:42-44`). The dirty-layout path forces Dst to 16b — same restriction applies to our untilize dirty-mode. fp32_dest_acc paths cannot use Task 5 optimization; keep them on the standard path.
   - Add test in `tests/python_tests/test_dense_pack_untilize.py` covering full 4-face dirty layout (extend the `dense` test infra).

2. **#42048 with 4-slot cfg setup.**
   - **Step 2a:** Implement `program_packer_untilized_destination` to write all four `THCON_SEC{0,1}_REG{1,8}_L1_Dest_addr` slots at init, each pointing to its assigned face-row offset (e.g. slot 0 → rows 0-7, slot 1 → rows 8-15, slot 2 → rows 16-23, slot 3 → rows 24-31 of the untilized output).
   - **Step 2b:** Switch MOP `PACK_INTF_SEL` from `TWO_INTFS_ACTIVE` to `ALL_INTF_ACTIVE` (mirror fast_tilize). Each PACR now writes 4 face-rows in parallel.
   - **Step 2c:** Per-tile (NOT per-row) end_op to bump all four L1 addresses. Mirror fast_tilize's `set_end_ops` pattern from `llk_pack_fast_tilize.h:101-104, 129`, but with 4× the cfg traffic (4 slots to update vs 1). Options for the 4-slot update, ordered by expected cost:
     - **4× `RMWCIB`** (1-cycle, IPC=1, pipelined) = 4 cyc + drain ≈ 6 cyc per tile
     - **4× `WRCFG`** (2-cyc, pipelined) = ~8 cyc per tile
     - **4× `CFGSHIFTMASK`** (2-cyc, non-pipelined) = 8 cyc per tile (worst, due to back-to-back stall)
     - Hybrid: use Task 2's CFGSHIFTMASK pattern for the common case, fall back to WRCFG for boundary
   - **Step 2d:** Use AddrMod scheme from fast_tilize (lines 27-33) for face boundary vs in-face advance.

3. **Cycle model:** Per tt-llk#916 model, 2→4 interfaces halves the PACR term: `32 + 32·PACR + 6·32/block_ct_dim` → `32 + 16·PACR + …` ≈ **~96 cyc/tile** with PACR=1. Combined with Task 2-4 wins (Task 2 saves ~12 cyc/tile, Task 3 ~5-10, Task 4 synergistic), target ~30 cyc/tile — consistent with fast_tilize's measured 27.6 cyc/tile PACK_ISOLATE.

**Risks (informed by fast_tilize's history):**
- **fast_tilize has an open proposed revert (tt-metal#44122) that has NOT landed** — claims sanity testsuite break + UB. Original PR #43577 merged 2026-05-12 and is still on main. The revert PR sits unmerged on `origin/nsidwell/sanitize`. Treat the UB concern as live — investigate and address before Task 5 ships. fast_tilize lives in `experimental/` deliberately from day one (part of #43577 itself, not a later move). Same approach applies here: gate behind a template flag (`DestLayout::DirtyForUntilize`), keep the legacy path intact, add A/B knob during bring-up.
- **MOVA2D 32-bit Dst quirk** forces 16b Dst — restricts Task 5 to ≤16-bit dest accum.
- **TileDescriptor HW bug** workaround required (per `llk_unpack_fast_tilize.h:90-92`) — same stall may apply on untilize side.
- **Width-1 fallback** — fast_tilize falls back to slow path for unit width 1 (`fast_tilize_bh_test.cpp:36-39, 112-136`). Mirror this gating.
- **DEST remap hoist concern** — `_llk_math_fast_tilize_init_` calls `_llk_math_reconfig_remap_(true)` and notes "Kernels calling init inside a loop pay tensix_sync per iteration" (open: tt-metal#17132, tt-llk#989). Either fix the hoist or document for callers.
- **craq-sim PACR stubs** — some PACR sub-modes throw `UnimplementedFunctionality` (`craq-sim/src/tensix.cpp:6926-6941`); silicon-only validation may be required for parts of Task 5.

**Validation:**
1. craq-sim PCC for extended dense_pack_untilize_test (4-face dirty layout). Expect some PACR sub-mode hits → file craq-sim upstream issue + work around.
2. Silicon: full perf_pack_untilize.py sweep. Required: ≥30% perf win on FP16→FP16 8×8; no functional regression; if any variant regresses >5%, add per-variant gating.
3. DeepSeek-V3 b1 integration smoke (Task 7) before merging.

**Rollback:** Gate entire new path behind `DestLayout::DirtyForUntilize` template flag + `PACK_UNTILIZE_LEGACY=1` env knob during bring-up. Three core files: `cpack_common.h`, `llk_pack_untilize.h`, `llk_math_eltwise_unary_datacopy_` (math LLK). Self-revert is one PR.

---

### Task 6 — Apply playbook to `unpack_untilize` — ATTEMPTED in session, DEFERRED

**Status (2026-05-16):** Two attempts; both reverted. Needs own branch + design.

**Attempt 1 — CFGSHIFTMASK port mirroring `llk_unpack_tilize.h:285,292`:**
- Replaced 6-instr replay buf (`DMANOP + 2*UNPACR + ADDDMAREG + STALLWAIT + ADDRCRZW`) with split-context pattern (UNPACR + CFGSHIFTMASK for cntx0 / UNPACR + CFGSHIFTMASK for cntx1)
- Problem: `unpack_untilize` MOP uses `ckernel_unpack_template(unpackB=true, halo=false)` with `A_instr = B_instr = replay(full buf)` and `skipA/skipB = WRCFG to cntx0/cntx1 offset cfg`. `unpack_tilize` uses `unpackB=false` with split halves and `skipA/skipB = 0`. Template wiring is fundamentally different — can't drop-in the unpack_tilize pattern.
- Also missed: SCRATCH preload at init, retaining ADDRCRZW for Z reset.

**Attempt 2 — drop the leading `TTI_DMANOP` (comment said it's needed for prior WRCFG retire):**
- Removed line 29 DMANOP. Replay buf shrunk to 5 instructions.
- Silicon **failed PCC** on first variant of `test_unpack_untilize.py`. DMANOP is genuinely required for cfg→unpacker propagation timing.
- Reverted.

**Why deferred (out of this branch):**
- Real T6 needs to redesign the replay buf + skipA/skipB to mirror unpack_tilize's CFGSHIFTMASK pattern AND handle the cross-context offset advance AND keep the Z-counter reset behavior.
- Per-row teardown (MULDMAREG + STALLWAIT + WRCFG) has STALLWAIT that may be ordering vs prior skipA WRCFG — can't be naively removed without race.
- Standalone unpack_untilize KERNEL is **555 cyc/tile** at 8×8 (vs pack 101 after T2+T3), so the win is real (~5x slower than pack), but unrelated to the pack_untilize end-to-end pipeline (which uses simple unpack_A, not unpack_untilize). T6 is its own workstream.

**Recommended next step:** new branch `pjosipovic/bh-unpack-untilize-perf` off main. Start with design notes mapping the `ckernel_unpack_template` slots (A/B vs skipA/skipB) to the CFGSHIFTMASK-direct pattern. Likely 3-5 days.

### Task 6 (original sketch retained below) — to be revisited

**Gap:** G3-like overhead in unpack side. From perf data, unpack is **less** of a bottleneck (49 cyc/tile TILE_LOOP isolated vs pack's higher overhead in pipeline), but same techniques apply.

**Files:** `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_untilize.h`.

**Targets:**
- DMANOP at MOP entry (line 29) — comment says "WRCFG needs additional cycle". With CFGSHIFTMASK, maybe removable.
- Two-pass top/bottom-faces (`_llk_unpack_untilize_pass_<true>` + `<false>` in test). Investigate single-pass via context bank.
- Per-row teardown (MULDMAREG + STALLWAIT + WRCFG + INCADCXY).

**Expected win:** 5-10% on full pipeline.

---

### Task 7 — DeepSeek-V3 b1 integration smoke (1 day, after Task 5)

**Why:** The only production callsite of `dense=true` pack_untilize is `models/demos/deepseek_v3_b1/kernel_includes/.../sdpa.h:534` and `unified_kernels/sdpa_reduce_worker.hpp:252`. They use `custom_pack_untilize.h` (a DeepSeek-private variant). Make sure the new BH `llk_pack_untilize` is wired in correctly and gives the expected end-to-end win on prefill.

**Files:** `models/demos/deepseek_v3_b1/...` — read-only verification, not modifying DeepSeek kernels (they are owned by another team).

**Validation:** Run DeepSeek SDPA prefill perf test on BH p100a, compare before/after this work.

---

### Task 8 (deferred) — CFG bank ping-pong

Pre-stage next iter's config in inactive bank while current iter is packing. Eliminates per-row STALLWAIT(C12). **Don't do this until 1-5 land** — orthogonal and the wins shrink after the other fixes.

---

## 6. Per-task tracking

| Task | Issue | Status | Est | Expected win |
|---|---|---|---|---|
| T0 Baseline + infra | — | **done** (c1e7b2d7d1f) | 1d | 0% |
| T1 Bfp8 max_block_dim | local | **dropped** | 0.5d | — (HW limit, not harness) |
| T2 CFGSHIFTMASK pack | #42050 | **done** (76452552f02) | 3d | -26.7% L1_TO_L1 mean (max -47.5%) |
| T3 AddrMod | #42051 | **done** (0a16daf4b4b) | 4d | -5.2% on top of T2 (cumulative -30.2%) |
| T4 Ch1 counters | #42052 | **attempted, reverted** | 3d | BH applies WH-style 16B-mask → broken for <256B strides |
| T5 4-intf + dirty dest | #42048 + #42049 | **WIP — compiles+runs, PCC fails first attempt** (branch pjosipovic/bh-untilize-t5) | 12-15d | 30-50% (structural, requires math co-design) |
| T6 Unpack | (no issue yet) | **attempted, deferred** | 3d | 5-10% (separate branch) |
| T7 DeepSeek integ smoke | — | not started | 1d | verify |
| T8 Bank ping-pong | (no issue yet) | deferred | — | 2-5% |

**Total estimate:** ~24 dev-days to land T0-T7. Target ~50% pack_untilize gap closed (126→63 cyc/tile FP16→FP16 8×8). Stretch with T5 fully optimized: ~24 cyc/tile.

---

## 7. Open questions

1. **Is `program_packer_untilized_destination` stub (cpack_common.h:711-716) intentionally empty or did someone start it and abandon?** Check git blame.
2. **Why was tt-llk PR #1219 (CFGSHIFTMASK on unpack matmul) closed without merge?** Read review discussion before T2.
3. **Does craq-sim cycle-collapsed PACR model still let us catch correctness bugs in 4-interface mode?** It models `pack_fmt_conv_mode` cases but `PACR_STRIDE`/`PACR0_FACE`/`PACR0_ROW` are stubbed. May need silicon for full correctness validation in T5.
4. **Does the BH packer have any undocumented 4-interface coupling we'll only learn about by writing the code?** Plan extra debugging time in T5.
5. **Can fast_tilize's `_llk_math_reconfig_remap_(true)` hoist (tt-metal#17132, tt-llk#989) be addressed at the same time?** Both fast_tilize and Task 5's dirty-Dst datacopy share this concern — if init is called inside a loop, tensix_sync cost per iteration kills perf wins. Worth checking whether DeepSeek's call pattern triggers it.
6. **Should Task 5's dirty-Dst datacopy live in `experimental/` alongside fast_tilize, or in `llk_lib/` proper?** fast_tilize's revert history (#43577 → #44122) suggests `experimental/` for the bring-up phase. Promote to `llk_lib/` after DeepSeek smoke (Task 7) passes.

---

## 8. Reference index

### Code (current BH)
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_untilize.h` — main target
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_untilize.h` — T6 target
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h` — pack init/state
- `tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h:711-716` — stub to implement
- `tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_instr_params.h:153-173` — PACK_INTF_SEL / CTXT constants
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_tilize.h:285,292` — only existing CFGSHIFTMASK precedent
- `tt_metal/hw/inc/api/compute/pack_untilize.h:74,226` — public API (`dense` template parameter)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_untilize_api.h:66,103` — metal wrapper

### fast_tilize precedent for Task 5 (BH `experimental/`)
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_pack_fast_tilize.h`
  - `:27-33` AddrMod scheme (4 slots — face vs in-face advance)
  - `:35-87` `EMIT_FACE_PACRS` — 4-intf ALL_INTF_ACTIVE + DST_ACCESS_STRIDED_MODE
  - `:92-104` MOP body with tile-granularity address advance
  - `:111-131` MOP outerloop=unit_dim, replay_buf body
  - `:169-171` `SETADCXX(FACE_C_DIM-1)` init
  - `:182-192` Y/W/Z stride programming
  - `:220-239` row-begin / row-chunk reset hoist
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_math_fast_tilize.h`
  - `:30-35` `_llk_math_reconfig_remap_` (hoist concern: tt-metal#17132, tt-llk#989)
  - `:38-41` dirty Dst layout comment
  - `:42-44` 16-bit Dst forcing for MOVA2D HW quirk
  - `:52-62` `addr_mod_t{.dest.incr=16}` + `MOV_8_ROWS` pattern (template for #42049)
  - `:76-80` `set_dst_write_addr<…, SrcRegs>` mailbox deadlock caveat
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_unpack_fast_tilize.h`
  - `:63-77` fp32/tf32 → bf16 downgrade (Dst16b constraint)
  - `:90-92` TileDescriptor HW bug workaround stall
- Tests + perf: `tests/sources/fast_tilize_bh_test.cpp`, `tests/python_tests/perf_fast_tilize_full.py`, `perf_data/perf_fast_tilize_full/perf_fast_tilize_full.post.csv`
- PR history: tt-metal PR #43577 merged 2026-05-12 (still on main). PR #44122 (proposed revert) is **OPEN, NOT merged** — claims sanity break + UB. Worth reading the diff and CI history to understand what they think is UB before Task 5 inherits the same patterns. **Gate behind template flag from day one** so we can A/B without a destructive revert.

### Code (WH reference)
- `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack_untilize.h` — WH 4-packer impl, lines 96-98 (single PACR over 4 packers), 116-123 (per-packer cfg writes)

### Tests
- `tt_metal/tt-llk/tests/python_tests/perf_pack_untilize.py` + `sources/pack_untilize_perf.cpp`
- `tt_metal/tt-llk/tests/python_tests/perf_unpack_untilize.py` + `sources/unpack_untilize_perf.cpp`
- `tt_metal/tt-llk/tests/python_tests/test_dense_pack_untilize.py`
- `tt_metal/tt-llk/tests/TTSIM.md` — ttsim setup
- `tt_metal/tt-llk/perf_data/` — historical baselines

### GitHub
- **Epic:** tt-metal#42047
- **Sub-issues:** tt-metal#42048 (4 interfaces), #42049 (dirty dest), #42050 (CFGSHIFTMASK), #42051 (AddrMod), #42052 (ch1 counters)
- **Analysis:** tt-llk#916 (perf model + WH/BH gap), tt-llk#61 (BH packer design notes), tt-llk#391 (bfp8 Conv driver)
- **Prior PRs:** tt-llk#1219 (closed — CFGSHIFTMASK unpack matmul; revival candidate), tt-llk#1316/#38140 (`dense` mode merged), tt-llk#1543 (Quasar AddrMod unpack tilize block — merged template), tt-llk#1362 (open — SDPA custom pack; integration target)
- **Related:** tt-metal#21205, #32515, #17641; tt-llk#1009, #1282, #1283

### HW docs (local clones at `/localdev/pjosipovic/refs/`)
- `tt-isa-documentation/BlackholeA0/TensixTile/TensixCoprocessor/{CFGSHIFTMASK,WRCFG,RMWCIB,STALLWAIT,ConfigurationUnit,BackendConfiguration,Dst}.md`
- `tt-isa-documentation/WormholeB0/TensixTile/TensixCoprocessor/{PACR,ADCs}.md` and `Packers/{README,InputAddressGenerator,OutputAddressGenerator}.md` (same packer arch as BH)

### Simulator — use craq-sim (local) as the ttsim implementation
- `/localdev/pjosipovic/refs/craq-sim/` — local clone; **build from here, not upstream ttsim**
- `craq-sim/src/tensix.cpp:2452` — `TENSIX_EXECUTE_PACR` model
- `craq-sim/src/tensix.cpp:3340` — `TENSIX_EXECUTE_UNPACR` model
- `craq-sim/src/tensix.cpp:6926-6941` — PACR sub-mode unimplemented stubs (guardrail)
- `craq-sim/README.md` (Simulator Behavior Contract at lines 774-810 — functional-only contract)
- `craq-sim/tests/llk-smoke.sh`, `scripts/llk-weekly.sh` — LLK runners
- `tt_metal/tt-llk/tests/TTSIM.md` — tt-llk-side wrapper, matches craq-sim flow
- Functional-only: use for PCC regression, not cycle perf.

---

## 9. Next action

Start with **Task 0** (baseline + diff infra) followed by **Task 1** (host harness fix — cheap win). Then split:
- **Path A (safe perf wins):** Task 2 → Task 3 → Task 4. Lands incremental 20-30% improvement with low risk over ~10 days.
- **Path B (structural):** Task 5. Higher risk, biggest single win (30-50%).

Recommended: Path A first to bank wins, then Path B with confidence baseline. T6 is opportunistic; T7 is a gate before merging T5; T8 is post-epic.
