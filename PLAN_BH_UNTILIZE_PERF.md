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

### Task 4 — Ch1 (output side) counters (#42052) — ATTEMPTED, REVERTED (superseded)

**Status (2026-05-16):** Implemented and tested on silicon. Reverted; functional regression on small strides.

**Update (2026-05-18):** The later fast-untilize ch1 pass showed that the
stride field must be programmed in bytes, not 16B units. Treat the 2026-05-16
mask conclusion below as a historical failed-attempt diagnosis, not a confirmed
BH hardware limit.

**What was tried:**
1. `_llk_pack_untilize_configure_addrmod_`: added `.y_dst.incr=1` to `ADDR_MOD_1` so the row-closing PACR auto-advances ch1.Y post-PACR.
2. `_llk_pack_untilize_init_`: programmed `PCK0_ADDR_CTRL_XY_REG_1_Ystride = output_addr_offset / 16`. Removed the SCRATCH_SEC2 setup (no CFGSHIFTMASK needed).
3. `_llk_pack_untilize_mop_config_`: removed the `load_replay_buf` and the `set_end_op(replay)` entirely. End-ops default to NOP.
4. `_llk_pack_untilize_`: SETADCXY BitMask `0b0011` → `0b1111` to reset ch1.X and ch1.Y at start of each pack call. Kept the between-face-passes ch0.Y-only reset so ch1.Y continues 16 → 32 across top/bottom passes.

**Findings (silicon p100a):**
- perf_pack_untilize.py: 832 passed (kernel doesn't hang) — but **perf harness doesn't validate output correctness** (`helpers/perf.py` measures cycles only).
- test_zzz_pack_untilize.py: **FAILED at the very first variant** (`Float16_b→Float16_b, [64,64]`, ct=2 → 128 B/row → 8 in 16B units → masks to 0). Output rows all wrote to the same base address. Golden-tensor mismatch.

**Original conclusion (superseded):** BH appeared to apply the same
`(YZW_Addr & ~0xf)` 16B-unit mask documented in the WH ISA. The later
byte-stride correction invalidated this as a hardware-limit conclusion. At the
time, the interpreted perf-sweep impact was:
- ct=1..3 (FP16: 64..192 B/row) — broken
- ct=4 (FP16: 256 B/row) — works
- ct=5..7 (320..448 B/row, not multiples of 256) — broken (truncated to 256)
- ct=8 (512 B/row, multiple of 256) — works

T4 as a drop-in replacement for CFGSHIFTMASK is **not viable** because most shapes hit the mask. Two ways forward (both out of Path A scope):

1. **Conditional T4:** gate ch1.Y path behind `output_addr_offset % 256 == 0`; fall back to T2 CFGSHIFTMASK for small strides. Adds two MOP variants. Win only on ct=4, ct=8.
2. **Extend craq-sim** to model packer ch1.Y/Z/W → confirm the `~0xf` mask in software model → then revisit T4 design with full understanding.

**Reverted to T3 state.** Diff: `git show 0a16daf4b4b` (T3 commit unchanged). Functional test_zzz_pack_untilize.py passes (156 / 100 skip).

**For the record:** this mask hypothesis was later superseded by the 2026-05-18
byte-stride finding in the fast-untilize path.

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

**T5.3d session 2 attempts (2026-05-16 follow-up):**
- Tried craq-sim functional sim — `UnsupportedFunctionality: bar0 offset=0x1fc00530 size=8`. tt-umd writes BH TLB cfg as 8-byte, sim only accepts 4-byte. Affects ALL tests through ttsim — not specific to T5. Env compat issue, out of scope to fix here.
- Re-derived math layout from fast_tilize comments: math produces **8 data + 8 gap pattern**, not dense 4-tile stacking. My MOP assumed dense layout. PCC fails because PACR reads at strides (0, +16, +32, +48) hit gap rows for half the iterations.
- gdb on craq-sim possible but blocked by same tt-umd compat issue (the test never reaches PACR execution).

Definitive blockers for T5 silicon iteration:
1. **Math layout assumption is wrong.** Need to either (a) use 8-data-+8-gap-aware MOP (skip gaps via doubled y-advance), or (b) write a NEW math LLK for pack_untilize that produces dense Dst layout.
2. **No fast functional iteration tool.** craq-sim broken in our env; silicon round-trip is 75s/iter; gdb-on-sim blocked.

Realistic continuation path:
- **Option X:** Write a "math layout dump" test that just runs math then prints Dst contents (via SFPU). One-time investment, unblocks debugging.
- **Option Y:** Skip math co-design entirely; use option (a) — adjust MOP to use y_stride=2*32=64 to skip gap rows. Still uses fast_tilize math, no new math LLK needed.

Option Y is the cheaper next attempt.

**Deeper analysis (2026-05-16 session 3 followup):**

Re-derived fast_tilize's actual Dst layout vs what 4-interface STRIDED PACR can produce for RM-strip output:
- fast_tilize math produces 8-data-rows + 8-gap-rows per group, 8 groups × 16 rows = 128 Dst rows per dvalid (4 dvalids = 512 rows = 1 half-bank)
- 4-intf STRIDED reads at (0, +16, +32, +48): reads 4 data groups simultaneously, 1 row each → 64 datums concatenated
- For TILE format L1, these 64 datums correctly assemble as 4 face-rows of 1 face
- For RM strip format L1, would need DIFFERENT Dst layout where the 4 reads give consecutive RM rows of 4 different tiles

**Verdict:** fast_tilize math's natural Dst layout CANNOT be re-purposed for fast_untilize via cleverer pack-side stride/AddrMod. Real T5 requires NEW math LLK producing an interleaved Dst layout specifically designed for RM-strip readout. This is the full #42049 scope.

Realistic T5 = 12-15 days (matches updated estimate). Math LLK design is the load-bearing piece. The pack-side MOP from T5.3b is a reasonable scaffold but won't pass PCC against fast_tilize math.

**Other sim path not tried in session:** vanilla ttsim from https://github.com/tenstorrent/ttsim-private — may have the tt-umd 8-byte TLB compat fix that craq-sim lacks. Worth a future-session attempt to unblock functional debugging.

**ttsim-private attempt (2026-05-16 session 4):**

Cloned and built `ttsim-private`. Same `bar0: offset=0x1fc00530 size=8` error on BH path.

Applied two local patches to ttsim-private (saved at `/localdev/pjosipovic/ttsim-private-patches-bh-fix.diff`):
1. **`src/libttsim.cpp`:** BH TLB cfg `0x1FC00000..0x1FC009D4` writes accept size=4 OR size=8 (split 8B into 2× 4B). Tt-umd writes some BH TLB regs as 8-byte.
2. **`src/tensix.cpp` UNPACR:** When `SRCA_SET_SetOvrdWithAddr=true`, SrcA row wraps `& 0x3F` (mod 64) instead of `UndefinedBehavior`. Mirrors `tt-isa-documentation` PR #52.

After patches, sim runs but hits **`TENSIX TIMED OUT … BRISC firmware did not signal boot-ready within 1.0s`** — affects even basic `test_eltwise_unary_datacopy`, not specific to T5 or fast_tilize. So tt-umd↔ttsim compat is broken at a deeper layer than these two PCIe ranges. Likely needs tt-umd downgrade or further ttsim PCIe surface patches.

**Sim path remains blocked.** Next iteration needs:
- Hunt down the BRISC firmware load PCIe write that's failing silently OR
- Match tt-umd to the version ttsim-private was tested against (`pip show tt-umd` = 0.9.5.dev260424; ttsim-private CI uses an unknown specific version)

Saving the two patches as a starting checkpoint for future-session ttsim debug. T5 iteration via sim still possible after BRISC boot issue is resolved.

**Update after G5.1 debug:** local `ttsim-private` was patched far enough for focused fast-untilize bring-up:
- BH TLB cfg writes now accept the 8-byte tt-umd path.
- `SCRATCH_SEC2`, CFG reg 211 writes, and the relevant CFGSHIFTMASK scratch selector were modeled.
- A focused fast-untilize case runs and passes under ttsim, but it does **not** reproduce the silicon pack hang. Use ttsim for basic functional coverage, not for this class of MOP/replay liveness bug.

**T5.3e session 5 (2026-05-16): dedicated math layout + focused silicon PASS**

Implemented dedicated experimental unpack/math paths:
- Added `experimental/llk_unpack_fast_untilize.h`.
- Added `experimental/llk_math_fast_untilize.h`.
- Switched the test source from fast_tilize unpack/math to fast_untilize unpack plus the new fast_untilize math.
- The unpacker presents faces to math as `F2,F3,F0,F1`; math now remaps them into the row-major strip Dst layout required by the 4-interface packer.
- The physical Dst layout places `F0/F1` in rows `0..127` and `F2/F3` in rows `128..255`.
- First unpack-side experiment removed the standard `unpack_A` zero SrcB sideband. Accuracy still passed, but unpack perf did not move by itself. The real unpack win came from consuming the four contiguous input tiles as one 16-face block with one context/address setup.

Pack-side fixes:
- `ckernel_template` with two loop ops already emits both PACRs per inner iteration, so `MOP_INNER_LOOP` is `1`, not `2`.
- Phase sequencing on BH silicon is one contiguous L1 stream. Program `L1_Dest_addr` once at the base, emit top rows first with `Last=0`, then emit bottom rows with `Last=1`.
- Direct W/Z counter phase selection did not latch reliably in this MOP. The original passing path reprogrammed `DEST_TARGET_REG_CFG_PACK` through `select_packer_dest_registers<DstSync::SyncFull>()`: offset `128` for the top half, then offset `0` for the bottom half while the L1 stream remained open.
- Follow-up: phase selection is now `SyncHalf`-native. The pack path programs both lower/upper DEST offset GPRs for each phase (`active_half + 128`, then `active_half + 0`) and lets `select_packer_dest_registers<DstSync::SyncHalf>()` select the current half. `SyncFull` is no longer the bring-up crutch.
- Header-only LLK changes can reuse stale `/tmp/tt-llk-build` ELFs; validation was rerun after clearing that generated build directory.

Validation:
```bash
cd tt_metal/tt-llk/tests
rm -rf /tmp/tt-llk-build
COMPILED=1 RUN_TEST=0 FILE_NAME=test_fast_untilize.py QUIET=1 PARALLEL_JOBS=4 ../.claude/scripts/run_test.sh
COMPILED=0 RUN_TEST=1 FILE_NAME=test_fast_untilize.py QUIET=1 ../.claude/scripts/run_test.sh
```
Result: `test_fast_untilize.py` passes twice for the focused BH ct=4 Float16_b -> Float16_b MVP case.

Focused perf check for the same working case (`rt=1`, `ct=4`, `Float16_b -> Float16_b`, `loop_factor=1`):
- Added `tests/python_tests/perf_fast_untilize.py` for single-case L1_TO_L1 + per-thread isolate measurements.
- Fast path: TILE_LOOP L1_TO_L1 = `82.25 cyc/tile`, PACK_ISOLATE = `58.25 cyc/tile`.
- Existing `perf_pack_untilize.py` baseline, same parameter: TILE_LOOP L1_TO_L1 = `160.0 cyc/tile`, PACK_ISOLATE = `109.75 cyc/tile`.
- Narrow-case improvement: L1_TO_L1 `-48.59%` (`1.95x`), PACK_ISOLATE `-46.92%` (`1.88x`).
- This clears the one-case >=30% L1_TO_L1 target, but does not replace the full perf sweep / regression gate.

Per-thread fast-path isolate check (`perf_fast_untilize.py`, same shape):

| loop_factor | L1_TO_L1 | UNPACK_ISOLATE | MATH_ISOLATE | PACK_ISOLATE |
|---:|---:|---:|---:|---:|
| 1 | 82.25 | 44.25 | 77.50 | 58.25 |
| 4 | 48.00 | 23.25 | 31.5625 | 35.25 |
| 16 | 40.875 | 17.8125 | 20.078125 | 29.4375 |

Conclusion from the focused steady-state case: the 16-face block unpack removed the previous unpack bottleneck (`UNPACK_ISOLATE` `55.14 -> 17.81 cyc/tile`, `L1_TO_L1` `57.34 -> 40.875 cyc/tile` at loop_factor=16). Pack is now the largest isolate component (`29.44 cyc/tile`) and the remaining full-pipeline gap is synchronization/overlap plus pack.

**T5.4 generalization plan (steal what maps from BH fast_tilize)**

Current focused fast path contract:
- BH only.
- Source-level row loop for `rt>=1`; the LLK unit remains one tile row chunk.
- Row widths decompose into `unit_dim={4,2,3}`; `ct=1` remains legacy fallback for integration.
- Focused bring-up currently validates `ct=2..8`, `num_faces=4`.
- `Float16_b -> Float16_b`, `dest_acc=No`.
- Each unit's input tiles are contiguous in L1 and are consumed as a compact SrcA face stream.
- Whole-row units (`ct=2/3/4`) use the contiguous pack stream. Wider rows use the row-strided pack MOP per chunk.

Fast-tilize ideas worth reusing:
- Row-only kernel contract first; outer caller loops rows.
- `decompose_row(ct_dim)` into unit chunks, with width-1 fallback and `4+1 -> 2+3` style tail handling.
- Initialize for the first unit dimension and only reinit when the next unit changes.
- Split APIs into row/chunk lifecycle (`row_begin`, `row_chunk`, `row_end`) rather than one monolithic block.
- Keep experimental path gated and preserve legacy fallback from day one.
- Copy the accuracy/perf matrix shape from `test_fast_tilize_full.py` and `perf_fast_tilize_full.py`, but stage it instead of enabling the full matrix at once.

Important difference from fast_tilize:
- Fast_tilize writes tilized output, so chunks can be streamed as independent contiguous tile groups.
- Untilize writes row-major output. For `ct>4`, chunk 0 row 0 must be followed by chunk 1 row 0, not by chunk 0 row 1. Therefore a naive loop over contiguous 4-tile chunks produces chunk-major output, not row-major output.
- General `ct>4` requires pack-side row-strided L1 addressing or per-row destination updates. The MVP avoided this only because `ct=4` made the whole row equal one chunk.

Milestone G1 - cleanup current MVP into reusable primitives:
- Rename comments and APIs around the current 4-tile unit, not the one-off test.
- Unpack: keep one-context 16-face block for unit_dim=4.
- Math: keep a unit_dim=4 layout primitive that maps four contiguous tiles into the packer layout.
- Pack: split into `row_begin(base)`, `row_chunk_4(last=false/true)`, `row_end()` or equivalent so later `rt`/`ct` loops do not duplicate setup.
- Keep focused accuracy + perf tests green after API cleanup.

Milestone G2 - row-by-row `rt` handling, still `ct=4`:
- Do not build a multi-row primitive. Match fast_tilize: the fast LLK unit is row-only and the caller/source loops rows.
- This is the safest next test because each tile row is still exactly one 4-tile row-major strip.
- For row `r`, input base is `buffer_A[r * 4]`; output base is `buffer_Res[r * 4]`.
- Reuse the same unpack/math/pack unit once per row.
- Add accuracy for `(rt, ct) = (2,4), (4,4), (8,4)` only to validate caller row iteration and address arithmetic.
- Add perf for `rt=1/4/8, ct=4`; expect per-row setup amortization behavior to be visible, but do not treat this as a separate kernel capability.

G2 implementation checkpoint:
- Unpack/math/pack now loop over `FULL_RT_DIM` at the source level; the LLK unit remains row/chunk-scoped.
- Accuracy passes for `(rt, ct) = (1,4), (1,8), (2,4), (4,4), (2,8)`.
- Focused `ct=4, rt=4` perf passes and reaches `37.16 cyc/tile` L1_TO_L1 at loop_factor=16.

Milestone G3 - format gating:
- Keep `dest_acc=No` fast path. `fp32_dest_acc` remains fallback because MOVA2D/Dst32 is unsafe in this family of paths.
- First expand same-format 16-bit/BFP outputs that use 16-bit Dst view: `Float16_b -> Float16_b`, then `Float16_b -> Bfp8_b/Bfp4_b` if pack conversion behaves.
- Treat `Float32 -> Float16_b/Bfp8_b/Bfp4_b` like fast_tilize: unpack converts to bf16-compatible SrcA, precision expectations must be explicit.
- Leave `Float32 -> Float32`, Int formats, `num_faces != 4`, and width-1 on legacy path until proven.

Milestone G4 - `ct>4` design spike:
- Do not simply loop chunked MVP calls; that is the wrong output order.
- **Design-gate decision:** use a second fast-pack MOP that mirrors legacy `llk_pack_untilize` row-close semantics, not fast_tilize's contiguous tile stream.
  - Keep the existing contiguous MVP path for `full_ct_dim == block_ct_dim == 4`; it is the fastest case because the 4-tile chunk is the whole row.
  - For `full_ct_dim > block_ct_dim`, each chunk call starts at:
    `tile_row_base + chunk_col * TILE_C_DIM * bytes_per_datum / 16`.
  - Store one element-row stride in scratch:
    `element_row_stride_16B = full_ct_dim * TILE_C_DIM * bytes_per_datum / 16`.
  - In the strided MOP, each strip row emits the chunk with the final PACR's `Last=1`, then the end-op does
    `THCON_SEC0_REG1_L1_Dest_addr += element_row_stride_16B` via `CFGSHIFTMASK`.
  - Because `Last=1` closes the row stream, the increment is the **full row stride**, not `row_stride - chunk_width`. This is the same contract as legacy `llk_pack_untilize`, where the caller offsets the block column and the MOP advances by full rows.
  - Top phase starts at output row 0. After 16 row closes, the L1 destination is already at row 16 for the bottom phase; switch only the DEST source offset (`128 -> 0`) and rerun the same row-strided MOP.
  - For tile row `rt`, the caller/source-level loop sets:
    `tile_row_base = output_base + rt * TILE_R_DIM * element_row_stride_16B`.
  - This supersedes the older BH four-L1-slot sketch for the fast path. On BH, ALL_INTF_ACTIVE concatenates the four read interfaces into one L1 stream behind `THCON_SEC0_REG1`; the correct scatter dimension is per-row cfg update, not multiple L1 destination slots.
- Bring-up rule: `SyncHalf` is the default path. Phase selection must always be relative to the active DEST half (`active_half + 128`, then `active_half + 0`); do not use `SyncFull` as the generalization crutch.
- Validation target: `(rt=1, ct=8)` accuracy first with two 4-tile chunks, then perf vs `perf_pack_untilize.py`.

G4 implementation checkpoint:
- Added the row-strided fast-pack MOP and wired `ct=8` as two 4-tile `SyncHalf` chunks.
- Accuracy passes for `(rt, ct) = (1,4), (1,8), (2,4), (4,4), (2,8)` with deterministic row-id stimuli.
- Focused perf now covers `ct=4` loop factors `1/4/16` and `ct=8` loop factors `1/4`. `ct=8, loop_factor=16` currently times out in the combined perf bring-up and is explicitly skipped until isolated.

Milestone G5 - row-width units and tails:
- Target the same high-level row decomposition policy as fast_tilize: support units `4`, `3`, and `2`; fallback to regular for width `1`; rewrite `4+1` tails as `2+3`.
- For untilize, only enable that decomposition after row-strided pack output is correct. The output ordering constraint is stricter than fast_tilize.
- `ct % 4 == 0`: fast path all 4-tile units.
- `ct % 4 == 2`: implement a 2-tile unit; likely uses fewer active packer interfaces or a masked/partial PACR sequence.
- `ct % 4 == 3`: implement a 3-tile unit after 2-tile is proven; likely needs explicit partial output handling because ALL_INTF_ACTIVE naturally emits 4 face-rows.
- `ct == 1`: legacy fallback, matching fast_tilize's width-1 fallback.

G5 implementation checkpoint:
- Added row decomposition matching BH fast_tilize: `4` chunks, `2`/`3` tails, and `4+1 -> 2+3`.
- Unpack now reprograms its face-stream MOP for `unit_dim=2/3/4`. Important fix: the source-level loop tracks the persistent hardware MOP unit across `LOOP_FACTOR`; resetting the software tracker each loop made decomposed rows run the wrong first-unit MOP and hang in `UNPACK_ISOLATE`.
- Math now copies only the occupied tile bands into the fast-untilize dirty DEST layout.
- Pack now has contiguous and row-strided `unit_dim=2/3/4` MOPs. Unit 2 uses one all-interface PACR per strip row; unit 3 uses an all-interface PACR plus a two-interface tail PACR.
- Accuracy passes all 63 cases over `ct=2..8`, `rt=1/2/4`: deterministic row-ID, random-data, and pack-thread guard-sentinel coverage after the result buffer.
- Focused perf passes all 63 cases over `ct=2..8`, `rt=1/2/4`, `loop_factor=1/4/16`. Wide rows now run through the stable direct pack fallback for all covered `rt`.
- Latest TILE_LOOP steady-state highlights:

| rt | ct | loop_factor | L1_TO_L1 | UNPACK_ISOLATE | MATH_ISOLATE | PACK_ISOLATE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 16 | 67.28 | 27.44 | 23.63 | 52.81 |
| 1 | 3 | 16 | 50.58 | 18.69 | 21.02 | 41.52 |
| 1 | 4 | 16 | 38.28 | 17.91 | 19.81 | 31.11 |
| 1 | 5 | 16 | 81.11 | 33.25 | 19.16 | 72.33 |
| 1 | 6 | 16 | 67.05 | 27.58 | 18.62 | 59.95 |
| 1 | 7 | 16 | 63.58 | 23.75 | 18.26 | 56.93 |
| 1 | 8 | 16 | 53.19 | 17.19 | 17.99 | 47.73 |
| 4 | 4 | 16 | 37.15 | 16.49 | 17.18 | 30.32 |
| 4 | 5 | 16 | 80.34 | 32.71 | 17.15 | 70.39 |
| 4 | 6 | 16 | 66.17 | 27.21 | 16.94 | 58.52 |
| 4 | 7 | 16 | 63.08 | 23.36 | 16.81 | 56.33 |
| 4 | 8 | 16 | 52.76 | 16.30 | 16.71 | 46.97 |

Apples-to-apples legacy comparison (`perf_fast_untilize_legacy_compare.py`, same format/dims/loop factors, `L1_TO_L1` + `PACK_ISOLATE`) passes 63 cases. Fast path is faster than legacy on **all 63 L1_TO_L1 points**. PACK_ISOLATE is faster on 62/63 points; the only pack-isolate regression is cold/small `rt=1, ct=5, loop_factor=1` (`107.6` vs legacy `102.4`), while full L1_TO_L1 still wins (`126.4` vs legacy `152.2`).

Steady-state (`loop_factor=16`) L1_TO_L1 comparison:

| rt | ct | legacy | fast | delta |
|---:|---:|---:|---:|---:|
| 1 | 2 | 135.31 | 67.25 | -50.3% |
| 1 | 3 | 112.67 | 50.53 | -55.1% |
| 1 | 4 | 100.98 | 38.28 | -62.1% |
| 1 | 5 | 94.36 | 81.11 | -14.0% |
| 1 | 6 | 89.48 | 67.05 | -25.1% |
| 1 | 7 | 86.26 | 63.58 | -26.3% |
| 1 | 8 | 83.83 | 53.19 | -36.6% |
| 2 | 2 | 132.48 | 75.14 | -43.3% |
| 2 | 3 | 110.27 | 56.95 | -48.4% |
| 2 | 4 | 99.07 | 34.63 | -65.0% |
| 2 | 5 | 92.16 | 80.92 | -12.2% |
| 2 | 6 | 87.67 | 66.78 | -23.8% |
| 2 | 7 | 84.46 | 63.50 | -24.8% |
| 2 | 8 | 82.14 | 53.12 | -35.3% |
| 4 | 2 | 131.03 | 71.63 | -45.3% |
| 4 | 3 | 109.10 | 54.35 | -50.2% |
| 4 | 4 | 97.87 | 37.16 | -62.0% |
| 4 | 5 | 91.28 | 80.35 | -12.0% |
| 4 | 6 | 86.79 | 66.17 | -23.8% |
| 4 | 7 | 83.61 | 63.08 | -24.6% |
| 4 | 8 | 81.36 | 52.76 | -35.1% |

G5.1 wide-row long-loop debug:
- Isolated the original `ct>4`, `loop_factor>=12` timeout to the pack thread. `UNPACK_ISOLATE` and `MATH_ISOLATE` passed for `ct=5/8`, `loop_factor=12/16`; `PACK_ISOLATE` timed out and full `L1_TO_L1` wedged behind packer backpressure.
- Silicon boundary on the focused harness: `ct=5`, `loop_factor=8/9` passed; `ct=5`, `loop_factor=10` failed. That is 18 strided chunks passing and 20 strided chunks failing.
- tt-exalens was useful where ttsim was not. At timeout, unpack/math had completed, pack had not; `trisc2_pc` landed in `ckernel::mop_sync()` / `store_blocking`, while pack debug bus state showed the packer datapath idle (`packer_busy=0`, `rwc_tdma_pack_busy=0`, pack request FIFO empty). This points at a MOP/replay sync liveness problem, not an active packer data-path stall.
- Negative probes that did not move the failure: start/end/mid-phase `mop_sync` placement, final `TTI_STALLWAIT(PACK|THCON)`, row-close `Flush=1`, pack reinit at loop boundaries, closing only the final row of each phase, disabling the replay-buffer row-address `CFGSHIFTMASK`, `Concat=1` changes on non-final/partial rows, and direct phase wait rearrangements.
- Fix: replace the wide-row strided MOP/replay body with a direct per-row RISC sequence of PACR(s) plus `CFGSHIFTMASK`. This bypasses the silicon `mop_sync` failure mode and restores correctness for `ct=5/8`, `loop_factor=16`.
- Tradeoff: wide rows are now stable but pack-limited. `ct=5`, `loop_factor=16` is `81.11 cyc/tile` L1_TO_L1 and `72.33 cyc/tile` PACK_ISOLATE; `ct=8`, `loop_factor=16` is `53.19 cyc/tile` L1_TO_L1 and `47.73 cyc/tile` PACK_ISOLATE. The contiguous `ct<=4` path remains the performance target (`ct=4`, `loop_factor=16` is `38.28 cyc/tile` L1_TO_L1, `31.11 cyc/tile` PACK_ISOLATE).
- Working conclusion: integration should include the direct wide-row path under the same narrow BH/format/dest gate for `ct=5..8`, because the apples-to-apples silicon comparison now confirms full-pipeline L1_TO_L1 wins for every covered wide shape. It is not the final wide-row performance ceiling, but it is shippable as a faster fallback within this gate.

G5.2 wide-row MOP/replay retry:
- Stable direct fallback was committed first (`9c63da9c8a6`) so the experiment has a clean rollback point.
- Reintroduced a strided MOP/replay path with two changes from the failing version:
  - The row-address advance replay is preloaded once and used as the MOP end-op; the MOP is reprogrammed only when `unit_dim` changes, not between top/bottom phases or repeated same-width chunks.
  - Top and bottom phases reuse the same MOP; the path uses `STALLWAIT(CFG|PACK)` between phases instead of reprogramming the MOP and forcing another `mop_sync`.
- Direct per-row PACR remains in the header behind `FAST_UNTILIZE_STRIDED_MOP_REPLAY=0`.
- Validation passes:
  - `test_fast_untilize.py`: 63 passed.
  - `perf_fast_untilize.py`: 63 passed.
- Fresh apples-to-apples regular comparison after the MOP/replay retry:
  - `perf_fast_untilize.py`: 63 passed.
  - `perf_fast_untilize_legacy_compare.py`: 63 passed.
  - Fast path wins on all 63 `L1_TO_L1` points and all 63 `PACK_ISOLATE` points.
  - Average delta: `L1_TO_L1 -37.9%`, `PACK_ISOLATE -36.8%`.
  - Steady-state (`loop_factor=16`) average delta: `L1_TO_L1 -39.7%`, `PACK_ISOLATE -42.2%`.
- Latest steady-state wide-row TILE_LOOP with MOP/replay:

| rt | ct | L1_TO_L1 | PACK_ISOLATE |
|---:|---:|---:|---:|
| 1 | 5 | 76.76 | 72.76 |
| 1 | 6 | 64.22 | 60.64 |
| 1 | 7 | 58.34 | 54.41 |
| 1 | 8 | 51.11 | 47.56 |
| 2 | 5 | 75.69 | 71.02 |
| 2 | 6 | 62.93 | 59.18 |
| 2 | 7 | 57.33 | 54.38 |
| 2 | 8 | 50.39 | 47.04 |
| 4 | 5 | 75.06 | 70.72 |
| 4 | 6 | 62.29 | 58.92 |
| 4 | 7 | 56.81 | 53.47 |
| 4 | 8 | 49.81 | 46.78 |

- Compared with the direct fallback, this mainly helps wide full-pipeline cases: `rt=4, ct=5` improves `80.35 -> 75.06`, `ct=7` improves `63.08 -> 56.81`, and `ct=8` improves `52.76 -> 49.81`.
- Working conclusion: the silicon liveness issue was not inherent to row-strided MOP/replay; it was likely triggered by the earlier hot-path reprogramming / phase sequencing. Keep this MOP/replay path as the preferred wide-row path and keep the direct row path as the safety fallback.

Milestone G6 - integration into real `pack_untilize`:
- Add a template/runtime gate, e.g. `DestLayout::DirtyForUntilize` or `FAST_PACK_UNTILIZE_BH`, with legacy fallback.
- First select the fast MOP path when all constraints are met: BH, 16-bit Dst view, `dest_acc=No`, `num_faces=4`, safe output format, and `full_ct_dim>=2`. Use the contiguous MOP for `ct<=4` and the row-strided MOP/replay path for wider decomposed rows. Keep the direct row implementation as a fallback knob and keep `ct=1` on legacy.
- Keep `test_fast_untilize.py` as the bring-up oracle, then add/extend `test_zzz_pack_untilize.py` coverage for the integrated path.

Milestone G7 - validation gates:
- Accuracy: deterministic row-ID stimuli, random stimuli, guard tile overflow, `rt>1`, supported format matrix, and legacy fallback cases. Current bring-up covers row-ID + random + guard sentinels for `ct=2..8`, `rt=1/2/4`; format expansion remains.
- Perf: `perf_fast_untilize.py` per-thread isolates, selected apples-to-apples `perf_pack_untilize.py` comparisons, then full sweep.
- Integration: DeepSeek SDPA smoke only after full sweep has no correctness regressions and per-variant perf gating is in place.

### Task 5 — `dirty tile layout` in Dst + 4 packer interfaces (#42048 + #42049) (originally 8 days, now larger)

**Gap:** G1, G2, G6.

**Historical/superseded sketch:** the original multi-L1-slot untilize sketch below is kept for context, but the current BH design gate supersedes it. BH ALL_INTF_ACTIVE concatenates all read interfaces into one L1 stream behind `THCON_SEC0_REG1`; wide-row scatter is handled by row-address cfg updates, not four L1 destination slots.

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
| T4 Ch1 counters | #42052 | **attempted, reverted; later superseded** | 3d | Later fast-untilize pass showed ch1 output stride is byte-addressed |
| T5 4-intf + dirty dest | #42048 + #42049 | **bring-up accuracy + perf pass** (branch pjosipovic/bh-untilize-t5; ct=2..8, rt coverage staged) | 12-15d | 30-50% (structural, requires math co-design) |
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

### 2026-05-17 cleanup pass status

- Current cleanup commits on `pjosipovic/bh-untilize-t5`:
  - `b0b26099c98` Trim dead BH fast untilize pack init setup.
  - `0fbdca92919` Share fast untilize test sweep setup.
  - `abaea410520` Harden LLK perf CSV comparison.
  - `8906dd32197` Gate BH fast untilize row-stride scratch setup.
  - `654bc667d17` Use `pytest.fail` for fast untilize mismatches.
  - `3e71ade886c` Refresh fast untilize experimental wording.
  - `873e7de362a` Clean up fast untilize comments and golden helper.
  - `77d7011a9d9` Share unpack input quantization in golden generators.
  - `2eea5540289` Centralize fast untilize test constants.
  - `13f1c215329` Simplify perf comparison reporting helpers.
  - `5d78484c046` Name fast untilize face layout constants.
  - `f5c9ee980b9` Simplify fast untilize phase selection.
  - `9d4d34cfea6` Name fast untilize pack stride constants.
  - `4bf791d0774` Share fast untilize pack test loop.
  - `871f18c143c` Clarify fast untilize pack datum stride.
  - `9d5eb9f89b7` Drop unused perf CSV tile count plumbing.
- Current validation after the latest code/comment cleanup:
  - Accuracy: `python3 -m pytest -q tt_metal/tt-llk/tests/python_tests/test_fast_untilize.py` -> `567 passed in 72.64s`.
  - Perf: `python3 -m pytest -q tt_metal/tt-llk/tests/python_tests/perf_fast_untilize.py tt_metal/tt-llk/tests/python_tests/perf_fast_untilize_legacy_compare.py` -> `378 passed in 121.02s`.
  - Fast-vs-saved CSV gate: no `TILE_LOOP` regressions >2%; max deltas were `L1_TO_L1 +0.16%`, `UNPACK_ISOLATE +0.40%`, `MATH_ISOLATE +0.85%`, `PACK_ISOLATE +0.09%`.
  - Fast-vs-legacy apples-to-apples: `L1_TO_L1` wins `189/189`; `PACK_ISOLATE` wins `188/189` with max non-win `+0.98%`.
  - Latest focused C++ cleanup gate: two representative accuracy cases passed; three representative perf cases passed; CSV compare vs prior cleanup showed no >2% regressions, with only `MATH_ISOLATE` noise up to `+0.03%`.
  - Latest perf helper cleanup gate: `python_env/bin/python3 -m pytest -q tt_metal/tt-llk/tests/test_compare_perf_csv.py` -> `4 passed in 0.11s`; verbose CSV compare output remained semantically unchanged.
- Cleanup lesson: extracting the duplicated pack row/chunk loop in `fast_untilize_test.cpp` into inline helpers was functionally correct but regressed small `ct<=3` `PACK_ISOLATE` cases by up to ~23%. Keep that measured hot loop spelled out unless a future refactor proves codegen parity.

### 2026-05-17 fp32 DEST status

- Native fp32 DEST support is now implemented for the experimental BH fast-untilize path rather than using the fast-tilize compat shortcut.
- Unpack emits zero SrcB dvalids only when `is_fp32_dest_acc_en` so math can use `ELWADD` as the native SrcA + zero-SrcB -> DEST copy.
- Math keeps fp32 DEST enabled and uses an `ELWADD` copy path for fp32 DEST; the original `MOVA2D` path remains for 16-bit DEST.
- Pack no longer forces `Read_32b_data=0` or reconfigures source format to bf16 in fast-untilize init; it relies on normal pack configuration and uses the caller's `pack_src_format` for strides.
- Correctness status: `pytest -q --tb=short tt_metal/tt-llk/tests/python_tests/test_fast_untilize.py` passes with `2160 passed`, covering bf16/fp32/BFP8/BFP4 inputs, fp16/fp32 outputs, `dest_acc=No/Yes` where valid, `dest_sync=Half/Full`, `rt={1,2,4}`, `ct={2..9,12,16}`, row-id/random stimuli, and overflow guards.
- Caveat: Float32 input through this path still follows existing source-register format inference (`Float32 -> Tf32` before math unless a future unpack-to-dest design is added). This is native fp32 DEST, not a full-precision Float32 unpack-to-dest pipeline.
- Production integration caveat: a conv3d fused-kernel regression exposed that fast untilize must re-enter the normal math/pack sync contract and restore PACK format state when matmul/bias `pack_tile` traffic precedes untilize in the same kernel. A raw math-side `CLEARDVALID` experiment hung and should not be used as the production fix.
- Production validation: the fused conv3d repro `tests/ttnn/unit_tests/operations/conv/test_conv3d.py::test_conv3d_sweep_shapes[padding_mode=zeros-padding_011-groups_1-stride_111-kernel_333-W=9-H=10-T=8-C_out=64-C_in=12-B=1]` passes with PCC `0.9999901378033722`, and the prior fold/permutation hang repro passes under `scripts/run_safe_pytest.sh`.

### 2026-05-17 production-sync perf rerun

Command:
- `python_env/bin/python3 -m pytest -q --tb=short tt_metal/tt-llk/tests/python_tests/perf_fast_untilize.py tt_metal/tt-llk/tests/python_tests/perf_fast_untilize_legacy_compare.py` -> `2430 passed in 859.36s`.

Coverage:
- Fast path: `1620` variants, doubled across `DestSync.Half` and `DestSync.Full`.
- Legacy comparison: `810` variants.
- Formats: `Float16_b -> Float16_b`, `Float32 -> Float32`, `Bfp8_b -> Float16_b`, `Bfp8_b -> Float32`, `Bfp4_b -> Float16_b`, `Bfp4_b -> Float32`.

Production-sync fix perf impact:
- The saved exact one-case snapshot (`Float16_b -> Float16_b`, `dest_acc=No`, `rt=2`, `ct=9`, `loop_factor=16`, `DestSync.Half`) is unchanged: KERNEL `L1_TO_L1` `18976.5 -> 18976.0`, TILE_LOOP `L1_TO_L1` `63.8906 -> 63.8889`.
- Against the nearest broad saved run (`/tmp/fast_untilize_after_mopseq_full.post.csv`), current `DestSync.Half` common rows are noise-level: KERNEL `L1_TO_L1` mean `-0.01%`, max `+1.33%`; KERNEL `PACK_ISOLATE` mean `+0.04%`, max `+1.00%`; TILE_LOOP `L1_TO_L1` mean `-0.03%`, max `+2.68%`.

Current fast-vs-legacy summary (`KERNEL`, lower is better):

| dest sync | L1_TO_L1 wins | L1_TO_L1 mean delta | aggregate speedup | PACK_ISOLATE wins | PACK_ISOLATE mean delta | aggregate pack speedup |
|:---|---:|---:|---:|---:|---:|---:|
| Half | 788/810 | -30.28% | 1.624x | 803/810 | -35.30% | 1.682x |
| Full | 703/810 | -17.34% | 1.265x | 810/810 | -37.13% | 1.705x |

Current fast-vs-legacy steady per-tile summary (`TILE_LOOP`, lower is better):

| dest sync | L1_TO_L1 wins | L1_TO_L1 mean delta | aggregate speedup | PACK_ISOLATE wins | PACK_ISOLATE mean delta | aggregate pack speedup |
|:---|---:|---:|---:|---:|---:|---:|
| Half | 810/810 | -38.21% | 1.678x | 803/810 | -40.42% | 1.741x |
| Full | 764/810 | -24.21% | 1.381x | 806/810 | -41.86% | 1.789x |

Conclusion:
- The production sync/state fix does not create a measurable `DestSync.Half` perf regression on the common broad run.
- `DestSync.Half` remains the default/focus: all per-tile `L1_TO_L1` points beat legacy, and the small KERNEL-only regressions are cold/short-loop overhead.
- `DestSync.Full` is still functionally covered and pack-isolate is consistently faster, but some full-pipeline KERNEL regressions remain, concentrated in BFP4/BFP8 `-> Float16_b`, `dest_acc=No`, `ct=5`. Treat SyncFull follow-up as perf work, not cleanup.

### 2026-05-17 perf comparison with dest mode

- Expanded `perf_fast_untilize.py` and `perf_fast_untilize_legacy_compare.py` to include destination mode in the sweep:
  - `Float16_b -> Float16_b`, `dest_acc=No`
  - `Float16_b -> Float16_b`, `dest_acc=Yes`
  - `Float32 -> Float32`, `dest_acc=Yes`
- Full comparison command: `python3 -m pytest -q tt_metal/tt-llk/tests/python_tests/perf_fast_untilize.py tt_metal/tt-llk/tests/python_tests/perf_fast_untilize_legacy_compare.py`
- Latest result: `378 passed in 121.02s`.
- Fast path wins on all `189/189` `L1_TO_L1` points and `188/189` `PACK_ISOLATE` points. The remaining pack-isolate non-win is below the 2% gate (`max +0.98%`), while full `L1_TO_L1` still wins.
- Legacy baseline note: for `dest_acc=Yes`, SyncHalf has only four destination tiles, so the regular baseline caps `BLOCK_CT_DIM` at 4. That means `ct=5/7` use `block_ct=1`, `ct=6` uses `block_ct=3`, and `ct=8` uses `block_ct=4`.

Average delta over all `loop_factor={1,4,16}` points:

| format | dest_acc | points | L1_TO_L1 avg delta | PACK_ISOLATE avg delta |
|:---|:---:|---:|---:|---:|
| Float16_b | No | 63 | -38.0% | -36.8% |
| Float16_b | Yes | 63 | -50.5% | -51.6% |
| Float32 | Yes | 63 | -41.0% | -41.7% |

Steady-state (`loop_factor=16`) average delta:

| format | dest_acc | points | L1_TO_L1 avg delta | PACK_ISOLATE avg delta |
|:---|:---:|---:|---:|---:|
| Float16_b | No | 21 | -39.7% | -42.2% |
| Float16_b | Yes | 21 | -54.8% | -56.5% |
| Float32 | Yes | 21 | -45.3% | -46.5% |

Steady-state (`loop_factor=16`) `L1_TO_L1` comparison:

| format | dest_acc | rt | ct | legacy | fast | delta |
|:---|:---:|---:|---:|---:|---:|---:|
| Float16_b | No | 1 | 2 | 135.31 | 67.41 | -50.2% |
| Float16_b | No | 1 | 3 | 112.67 | 50.73 | -55.0% |
| Float16_b | No | 1 | 4 | 100.98 | 38.23 | -62.1% |
| Float16_b | No | 1 | 5 | 94.36 | 76.75 | -18.7% |
| Float16_b | No | 1 | 6 | 89.48 | 64.24 | -28.2% |
| Float16_b | No | 1 | 7 | 86.26 | 58.35 | -32.4% |
| Float16_b | No | 1 | 8 | 83.83 | 51.08 | -39.1% |
| Float16_b | No | 2 | 2 | 132.48 | 75.14 | -43.3% |
| Float16_b | No | 2 | 3 | 110.27 | 56.98 | -48.3% |
| Float16_b | No | 2 | 4 | 99.07 | 34.65 | -65.0% |
| Float16_b | No | 2 | 5 | 92.16 | 75.69 | -17.9% |
| Float16_b | No | 2 | 6 | 87.67 | 62.93 | -28.2% |
| Float16_b | No | 2 | 7 | 84.46 | 57.33 | -32.1% |
| Float16_b | No | 2 | 8 | 82.14 | 50.39 | -38.6% |
| Float16_b | No | 4 | 2 | 131.03 | 71.65 | -45.3% |
| Float16_b | No | 4 | 3 | 109.10 | 54.35 | -50.2% |
| Float16_b | No | 4 | 4 | 97.87 | 37.14 | -62.1% |
| Float16_b | No | 4 | 5 | 91.28 | 75.06 | -17.8% |
| Float16_b | No | 4 | 6 | 86.79 | 62.30 | -28.2% |
| Float16_b | No | 4 | 7 | 83.61 | 56.81 | -32.1% |
| Float16_b | No | 4 | 8 | 81.36 | 49.81 | -38.8% |
| Float16_b | Yes | 1 | 2 | 135.34 | 67.44 | -50.2% |
| Float16_b | Yes | 1 | 3 | 112.60 | 50.81 | -54.9% |
| Float16_b | Yes | 1 | 4 | 100.98 | 38.53 | -61.8% |
| Float16_b | Yes | 1 | 5 | 198.82 | 76.78 | -61.4% |
| Float16_b | Yes | 1 | 6 | 110.18 | 64.76 | -41.2% |
| Float16_b | Yes | 1 | 7 | 198.31 | 58.57 | -70.5% |
| Float16_b | Yes | 1 | 8 | 99.09 | 51.26 | -48.3% |
| Float16_b | Yes | 2 | 2 | 132.50 | 75.23 | -43.2% |
| Float16_b | Yes | 2 | 3 | 110.18 | 56.96 | -48.3% |
| Float16_b | Yes | 2 | 4 | 99.09 | 34.86 | -64.8% |
| Float16_b | Yes | 2 | 5 | 197.92 | 75.34 | -61.9% |
| Float16_b | Yes | 2 | 6 | 108.92 | 63.06 | -42.1% |
| Float16_b | Yes | 2 | 7 | 197.66 | 57.46 | -70.9% |
| Float16_b | Yes | 2 | 8 | 97.87 | 50.28 | -48.6% |
| Float16_b | Yes | 4 | 2 | 131.03 | 71.67 | -45.3% |
| Float16_b | Yes | 4 | 3 | 108.92 | 54.38 | -50.1% |
| Float16_b | Yes | 4 | 4 | 97.87 | 37.23 | -62.0% |
| Float16_b | Yes | 4 | 5 | 197.46 | 74.67 | -62.2% |
| Float16_b | Yes | 4 | 6 | 108.29 | 62.36 | -42.4% |
| Float16_b | Yes | 4 | 7 | 197.33 | 56.87 | -71.2% |
| Float16_b | Yes | 4 | 8 | 97.40 | 49.80 | -48.9% |
| Float32 | Yes | 1 | 2 | 137.75 | 69.75 | -49.4% |
| Float32 | Yes | 1 | 3 | 114.15 | 68.00 | -40.4% |
| Float32 | Yes | 1 | 4 | 102.13 | 51.89 | -49.2% |
| Float32 | Yes | 1 | 5 | 200.97 | 93.25 | -53.6% |
| Float32 | Yes | 1 | 6 | 111.53 | 78.71 | -29.4% |
| Float32 | Yes | 1 | 7 | 200.58 | 76.30 | -62.0% |
| Float32 | Yes | 1 | 8 | 99.95 | 67.10 | -32.9% |
| Float32 | Yes | 2 | 2 | 134.72 | 74.98 | -44.3% |
| Float32 | Yes | 2 | 3 | 111.53 | 67.55 | -39.4% |
| Float32 | Yes | 2 | 4 | 99.95 | 48.49 | -51.5% |
| Float32 | Yes | 2 | 5 | 200.22 | 91.86 | -54.1% |
| Float32 | Yes | 2 | 6 | 110.40 | 77.24 | -30.0% |
| Float32 | Yes | 2 | 7 | 200.02 | 75.08 | -62.5% |
| Float32 | Yes | 2 | 8 | 98.86 | 65.93 | -33.3% |
| Float32 | Yes | 4 | 2 | 133.59 | 71.55 | -46.4% |
| Float32 | Yes | 4 | 3 | 110.43 | 66.17 | -40.1% |
| Float32 | Yes | 4 | 4 | 98.86 | 48.55 | -50.9% |
| Float32 | Yes | 4 | 5 | 199.86 | 91.13 | -54.4% |
| Float32 | Yes | 4 | 6 | 109.67 | 76.46 | -30.3% |
| Float32 | Yes | 4 | 7 | 199.76 | 74.40 | -62.8% |
| Float32 | Yes | 4 | 8 | 98.30 | 65.35 | -33.5% |

### 2026-05-17 MOP sequencing perf pass

Treated hot pack-loop/MOP sequencing as perf work, not cleanup, and landed three narrow commits:
- `23364630382` Avoid redundant fast untilize phase sync.
- `1361e1917a1` Patch only the phase-close MOP word instead of rebuilding the full MOP.
- `521f4810477` Cache the contiguous MOP `unit_dim` and patch phase close when the width repeats.

Scope:
- Affects the contiguous `ct<=4` fast pack path only.
- Wide `ct>4` remains on the row-strided MOP/replay path and is intentionally unchanged.
- The direct row fallback remains available behind `FAST_UNTILIZE_STRIDED_MOP_REPLAY=0`.

Focused pre/post comparison (`/tmp/fast_untilize_mopseq_baseline.post.csv` -> `/tmp/fast_untilize_mopcache_broader_candidate.post.csv`, 5 hot variants, `loop_factor=16`):

| marker | run type | mean delta | max win | regressions >2% |
|:---|:---|---:|---:|---:|
| KERNEL | PACK_ISOLATE | -15.65% | -20.31% | 0 |
| KERNEL | L1_TO_L1 | -11.11% | -18.15% | 0 |
| TILE_LOOP | PACK_ISOLATE | -16.61% | -21.42% | 0 |
| TILE_LOOP | L1_TO_L1 | -11.72% | -19.11% | 0 |

The third cache step was the smallest win by itself: KERNEL PACK_ISOLATE mean `-1.30%`, L1_TO_L1 mean `-0.89%`, with the only >2% wins on `ct=3`. Still worth keeping because it is localized and did not trip the gate.

Validation after the final MOP sequencing commit:
- Accuracy: `python_env/bin/python3 -m pytest -q --tb=short tt_metal/tt-llk/tests/python_tests/test_fast_untilize.py` -> `567 passed in 72.71s`.
- Focused affected perf: `perf_fast_untilize.py -k '(ct_dim:2 or ct_dim:3 or ct_dim:4) and loop_factor:16'` -> `81 passed, 486 deselected in 85.61s`.
- Full fast + legacy perf gate: `python_env/bin/python3 -m pytest -q --tb=short tt_metal/tt-llk/tests/python_tests/perf_fast_untilize.py tt_metal/tt-llk/tests/python_tests/perf_fast_untilize_legacy_compare.py` -> `1134 passed in 363.02s`.
- Hooks passed for the touched LLK/test files during each commit.

Working conclusion:
- Yes, the MOP sequencing path had real headroom. Most of it came from avoiding the redundant phase sync and full MOP rebuild; the remaining cache optimization is incremental.
- Do not treat further hot pack-loop refactors as cleanup. Continue only as measured perf experiments with focused contiguous-path perf first, then the full fast-vs-legacy gate.

### High-risk MOP optimization backlog

These are explicitly perf experiments, not cleanup. Each item needs an isolated commit, a kill switch when practical, focused accuracy/perf first, and then the full fast-vs-legacy gate before it is allowed to stay.

| idea | likely target | potential win | main risk |
|:---|:---|:---|:---|
| Ch1 output-counter row advance for strided rows | `ct>4` wide rows | completed | Kept in the 2026-05-18 pass after correcting the ch1 output stride units to bytes. |
| Further compress row-strided `ct>4` MOP/replay | wide rows, especially `ct=5..8` | medium/high | Previous wide path had silicon-only `mop_sync` liveness failures. Long-loop pack-isolate must be the first gate. |
| Hoist or reuse destination programming across repeated same-width chunks | repeated chunks / repeated rows | medium | Stale `L1_Dest_addr` or phase state can produce plausible but wrong row-major output. |
| Replace row-close `CFGSHIFTMASK` with `RMWCIB` or manual cfg patching | strided pack row advance | medium | Config-pipeline ordering is touchy; simulator may not catch timing hazards. |
| Cache/reuse strided MOP across more loop/row boundaries | wide rows | low/medium | Stale `unit_dim`, row stride, or end-op state can break decomposed rows. |
| Padded 4-interface output for `unit_dim=2/3` with discarded lanes | narrow tails | medium | Extra-lane overflow/guard correctness is fragile and easy to miss without sentinels. |
| Fuse top/bottom phases into one MOP run | all fast pack paths | medium | DEST phase selection already proved fragile; wrong offset latching silently corrupts output or hangs. |
| Remove or relax phase `STALLWAIT`/`mop_sync` sequencing | all fast pack paths | medium | Highest chance of recreating silicon-only packer liveness failures. |
| CFG bank ping-pong for pack config/MOP state | broad pack config overhead | low/medium | Bigger LLK-state blast radius; touches shared config-bank assumptions. |

Preferred order if we take more perf risk:
1. Row-strided `ct>4` MOP/replay compression, starting with `ct=5/8`, `loop_factor>=16`, `PACK_ISOLATE`.
2. Destination-programming hoist for repeated same-width chunks, with guard-sentinel accuracy first.
3. Phase fusion or sync removal only after the above are exhausted.

Current next action: promotion from the experimental test path into the production untilize path. Keep the fast path gated by the existing BH/format/dest constraints, preserve legacy fallback from day one, and use the full `test_fast_untilize.py` plus `perf_fast_untilize.py`/`perf_fast_untilize_legacy_compare.py` matrix as the merge gate.

### 2026-05-17 ct=2 production re-enable

New conv3d repro:
- `tests/ttnn/unit_tests/operations/conv/test_conv3d.py::test_conv3d_sweep_shapes[padding_mode=zeros-padding_011-groups_1-stride_111-kernel_111-W=9-H=10-T=8-C_out=64-C_in=12-B=1]`

Findings:
- Fast path failed with bad PCC around `0.18-0.19`.
- Forcing legacy untilize passed with PCC `0.9999905603834628`.
- The no-bias variant also failed, so the issue is not bias math.
- The direct LLK `ct=2` path still passes in isolation; the failure is a fused production interaction after the K=1 regular-tilize/matmul producer.
- Restoring only PAC W after fast pack did not fix the repro. Restoring PAC X/Y/Z and W after the final fast-pack phase fixes it, matching regular untilize's post-pack counter discipline before the next LLK in a fused kernel.

Decision:
- Re-enable production fast-untilize for `ct>=2`.
- Keep the post-block PAC counter restore in the LLK pack path for both contiguous and row-strided fast untilize. The extra restore is a correctness boundary between fast untilize and subsequent regular pack users.

Validation after the fix:
- `kernel_111` repro passes with PCC `0.9999905603834628`.
- Prior `kernel_333` conv3d repro still passes with PCC `0.9999901378033722`.
- Focused LLK coverage: `test_fast_untilize.py -k 'Float16_b and Half'` -> `720 passed, 1440 deselected`, including ct=2, BFP inputs, random/row-id stimuli, and overflow guards.
- Exact Full-sync ct=2 smoke: 4 selected direct/guard cases passed.
- Focused perf (`Float16_b->Float16_b`, `rt=1`, `ct=2`, SyncHalf, `loop_factor=16`, dest_acc=No): current fast KERNEL `L1_TO_L1=2371.5`, `PACK_ISOLATE=1840.0`; prior fast baseline was `2339.0` / `1824.0` (`~+1-2%`); legacy is `4743.0` / `4400.0`, so the fast path keeps the large win.

### 2026-05-17 wide-row MOP sequencing autonomous pass

Goal: continue the high-risk MOP optimization backlog in a test-driven way, keeping only changes with focused correctness plus matched perf evidence.

Candidate list and outcomes:

| idea | target | outcome |
|:---|:---|:---|
| Shape-gated ch1 output counters | aligned wide-row output strides | Initially rejected. Superseded by the 2026-05-18 ch1 pass; the corruption was a byte-vs-16B stride programming bug. |
| `ct=5` decomposition `3+2` instead of `2+3` | odd wide rows | Rejected. Correctness passed, perf was effectively unchanged. |
| Multi-tile BFP unpack MOP | BFP inputs | Rejected. Correctness passed for focused cases, perf was unchanged. |
| Row-advance replay length 1 / remove replay NOP | strided row close | Rejected. Correctness failed; the NOP is required before the next PACR observes the shifted address. |
| Preconfigure strided MOP in init for `ct % 4 == 0` | `ct=8/12/16` | Rejected. Correctness passed, perf was neutral/slightly worse. |
| Hoist BFP format check out of unpack loop | BFP inputs | Rejected as non-perf cleanup. Correctness passed, perf was noise. |
| Direct row fallback instead of replay | wide rows | Rejected. Correctness passed but perf regressed (`L1_TO_L1` about `+6.9%`, `PACK_ISOLATE` about `+1.8%`). |
| Hoist pack output address calculation | strided chunks | Rejected. Correctness passed, perf was unchanged. |
| Remove `mop_sync()` from phase-close MOP patch | contiguous path | Rejected. Correctness failed for `ct=2/4`; phase-1 read bottom rows. |
| Add `unit_dim=1` tail support | `ct=5/9` tails | Rejected. Correctness passed, `ct=5` was marginal, `ct=9` regressed. |
| Remove phase-1 strided `STALLWAIT` | wide strided MOP/replay | Kept. Correctness passed and matched perf shows a consistent wide-row pack win. |
| Remove final strided `STALLWAIT` | wide strided MOP/replay restore boundary | Kept. Full correctness passed and matched perf shows another pack win with no KERNEL/TILE_LOOP regressions over 2%. |

Kept changes:
- In `_llk_pack_fast_untilize_block_strided_`, remove the `TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK)` between the top and bottom strided MOP phases.
- In the same strided MOP/replay path, remove the final `TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK)` before restoring standard pack counters.
- Rationale: phase 1 emits 16 row-close end-ops, so `L1_Dest_addr` is already advanced to output row 16 before phase 2. Phase selection, source-counter reset, and final counter restore configure subsequent PACRs and match the contiguous fast path's no-wait sequencing. The direct non-replay fallback keeps its explicit `STALLWAIT`s.

Matched perf gate:
- Baseline CSV: `/tmp/fast_untilize_phase1_stall_wide_allformats_baseline.post.csv`
- Phase-1-only candidate CSV: `/tmp/fast_untilize_no_phase1_stall_wide_allformats_candidate.post.csv`
- Final candidate CSV: `/tmp/fast_untilize_no_final_stall_wide_allformats_candidate.post.csv`
- Command: `python_env/bin/python3 -m pytest -q --tb=short tt_metal/tt-llk/tests/python_tests/perf_fast_untilize.py -k 'loop_factor:16 and (ct_dim:5 or ct_dim:6 or ct_dim:7 or ct_dim:8 or ct_dim:9 or ct_dim:12 or ct_dim:16)'`
- Baseline: `378 passed, 1242 deselected in 397.81s`.
- Phase-1-only candidate: `378 passed, 1242 deselected in 397.93s`.
- Final candidate: `378 passed, 1242 deselected in 399.47s`.

Matched perf result for final candidate vs original two-stall baseline, over 378 KERNEL variants (`ct=5/6/7/8/9/12/16`, `rt=1/2/4`, all supported formats, `SyncHalf`/`SyncFull`, dest 16/32 as applicable):

| marker | run type | min delta | mean delta | max delta | regressions >2% | wins >2% |
|:---|:---|---:|---:|---:|---:|---:|
| KERNEL | `L1_TO_L1` | `-12.62%` | `-8.17%` | `-0.03%` | 0 | 354 |
| KERNEL | `PACK_ISOLATE` | `-26.45%` | `-17.91%` | `-8.61%` | 0 | 378 |
| KERNEL | `UNPACK_ISOLATE` | `-0.16%` | `0.00%` | `+0.09%` | 0 | 0 |
| KERNEL | `MATH_ISOLATE` | `-0.08%` | `0.00%` | `+0.11%` | 0 | 0 |
| TILE_LOOP | `L1_TO_L1` | `-12.89%` | `-8.45%` | `-0.03%` | 0 | 354 |
| TILE_LOOP | `PACK_ISOLATE` | `-26.77%` | `-18.52%` | `-8.94%` | 0 | 378 |

Incremental effect of removing the final strided `STALLWAIT`, measured against the phase-1-only candidate:

| marker | run type | min delta | mean delta | max delta | regressions >2% | wins >2% |
|:---|:---|---:|---:|---:|---:|---:|
| KERNEL | `L1_TO_L1` | `-1.62%` | `-0.72%` | `+0.05%` | 0 | 0 |
| KERNEL | `PACK_ISOLATE` | `-16.86%` | `-8.96%` | `+0.07%` | 0 | 310 |
| TILE_LOOP | `L1_TO_L1` | `-1.64%` | `-0.75%` | `0.00%` | 0 | 0 |
| TILE_LOOP | `PACK_ISOLATE` | `-17.08%` | `-9.28%` | `+0.01%` | 0 | 311 |

Correctness gate:
- `python_env/bin/python3 -m pytest -q --tb=short tt_metal/tt-llk/tests/python_tests/test_fast_untilize.py`
- Result after both kept changes: `2160 passed in 404.68s`, including row-id/random stimuli, `SyncHalf`/`SyncFull`, supported format matrix, and overflow guard sentinels.

### 2026-05-18 ch1 output-counter row advance

Goal: replace the hot row-close `CFGSHIFTMASK` replay in the strided `ct>4`
path with pack output-counter ch1.Y advancement.

Finding:
- The earlier "256B stride floor" conclusion was wrong. The failure came from
  programming `PCK0_ADDR_CTRL_XY_REG_1_Ystride` in 16B units. On BH, the ch1
  output stride field expects **bytes**; using `/16` made `ct=8` fp16 row 0
  write like `[1, 2, 3, ...]` instead of the expected row-major interleave
  `[1, 17, 65, 81, ...]`.
- The ttsim model also needed output ch1 base/stride support and the same byte
  addressing rule. With that patch, the simulator is useful for fp16/fp32 row
  layout smoke tests. It still rejects BFP4 unpack format pairing, so BFP4
  coverage remains hardware-only for now.

Kept implementation:
- Add `FAST_UNTILIZE_STRIDED_CH1_ROW_ADVANCE`, default enabled.
- For strided `full_ct_dim > block_ct_dim` with MOP/replay enabled, program
  `PCK0_ADDR_CTRL_XY_REG_1_Ystride` to the full output row stride in bytes.
- Add `y_dst.incr=1` to `ADDR_MOD_1` and omit the per-row replay end-op from the
  strided MOP when ch1 row advance is active.
- Reset ch1.Y at the start of each strided block and when restoring pack
  counters, so repeated chunks start from the programmed base address.
- Clear ch1 output stride/base only in fast-untilize uninit instantiations that
  actually use the strided ch1 path. Production and the LLK harness pass
  `full_ct_dim` through pack uninit for that compile-time gate.

Validation:
- Smoke after the uninit gate: `ct=2` and `ct=8` fp16 row-id cases passed.
- Full LLK correctness:
  `python_env/bin/python3 -m pytest -q --tb=short -o log_cli=false tt_metal/tt-llk/tests/python_tests/test_fast_untilize.py`
  -> `2160 passed in 397.34s`.
- Wide-row perf gate:
  `python_env/bin/python3 -m pytest -q --tb=short -o log_cli=false tt_metal/tt-llk/tests/python_tests/perf_fast_untilize.py -k 'loop_factor:16 and (ct_dim:5 or ct_dim:6 or ct_dim:7 or ct_dim:8 or ct_dim:9 or ct_dim:12 or ct_dim:16)'`
  -> `378 passed, 1242 deselected in 397.81s`.

Matched perf result vs `/tmp/fast_untilize_no_final_stall_wide_allformats_candidate.post.csv`
over 378 KERNEL variants and 378 TILE_LOOP variants (`ct=5/6/7/8/9/12/16`,
`rt=1/2/4`, all supported formats, `SyncHalf`/`SyncFull`, dest 16/32 as
applicable):

| marker | run type | min delta | mean delta | max delta | regressions >2% |
|:---|:---|---:|---:|---:|---:|
| KERNEL | `L1_TO_L1` | `-8.17%` | `-2.07%` | `+0.05%` | 0 |
| KERNEL | `PACK_ISOLATE` | `-24.59%` | `-9.56%` | `+0.18%` | 0 |
| KERNEL | `UNPACK_ISOLATE` | `-0.09%` | `0.00%` | `+0.09%` | 0 |
| KERNEL | `MATH_ISOLATE` | `-0.26%` | `-0.01%` | `+0.23%` | 0 |
| TILE_LOOP | `L1_TO_L1` | `-8.40%` | `-2.21%` | `-0.01%` | 0 |
| TILE_LOOP | `PACK_ISOLATE` | `-25.54%` | `-10.02%` | `+0.02%` | 0 |

By `ct`, KERNEL `PACK_ISOLATE` mean deltas:

| `ct` | mean delta | min delta | max delta |
|---:|---:|---:|---:|
| 5 | `-12.86%` | `-24.59%` | `-2.14%` |
| 6 | `-11.93%` | `-24.46%` | `-0.07%` |
| 7 | `-10.32%` | `-22.02%` | `+0.10%` |
| 8 | `-6.70%` | `-16.27%` | `+0.18%` |
| 9 | `-11.25%` | `-23.77%` | `-0.01%` |
| 12 | `-6.88%` | `-16.48%` | `+0.09%` |
| 16 | `-6.96%` | `-16.60%` | `+0.04%` |

Conclusion:
- Keep ch1 row advance enabled by default for strided `ct>4`.
- The old CFGSHIFTMASK replay path remains behind
  `FAST_UNTILIZE_STRIDED_CH1_ROW_ADVANCE=0` as a fallback.
- Remaining wide-row pack headroom is now shape-specific; the generic row-close
  replay cost is mostly gone from the default path.

Next perf ideas, in preferred order:
1. Profile remaining neutral/low-win shapes (`ct=8/12/16`, dest-acc fp32) to see
   whether output-counter advancement is no longer the dominant cost.
2. Destination-programming hoist/reuse across repeated same-width chunks:
   attempted and kept only for rows with at least four equal chunks. See the
   2026-05-18 destination-reuse note below.
3. Revisit strided MOP compression only with ttsim counter visibility and a
   long-loop pack-isolate gate; prior MOP liveness issues were silicon-only.
4. Treat `unit_dim=1` tails as functionally possible but not profitable unless a
   different pack MOP can avoid the `ct=9` regression.

### 2026-05-18 strided destination reuse

Goal: reduce repeated `program_packer_destination(address)` overhead for wide
rows that decompose into same-width chunks. The experiment programs the row base
once, uses one `CFGSHIFTMASK` to advance `THCON_SEC0_REG1_L1_Dest_addr` between
chunks, and runs the strided pack MOP at the current destination address.

Finding:
- Enabling this for all exact multiples (`ct=8/12/16`) gave a repeatable
  steady-state full-pipeline win, but cold `loop_factor=1` PACK_ISOLATE had
  >2% regressions for `ct=8/12`.
- Combining the row-stride and chunk-stride init writes under one `STALLWAIT`
  reduced init overhead but did not fully remove those cold pack regressions.
- Safe default gate is therefore exact multiples with at least four chunks:
  `full_ct_dim % block_ct_dim == 0 && full_ct_dim >= 4 * block_ct_dim`.
  In the current matrix that enables `ct=16` and leaves `ct=8/12` on the
  existing per-chunk destination path.

Kept implementation:
- Add `FAST_UNTILIZE_STRIDED_DEST_REUSE`, default enabled, behind the ch1
  row-advance path.
- Add row-scoped pack helpers:
  `llk_pack_fast_untilize_strided_row_begin_at_address`,
  `llk_pack_fast_untilize_block_strided_current`, and
  `llk_pack_fast_untilize_advance_strided_row_address`.
- Program row stride and chunk stride together when the gated path is active;
  the old per-chunk destination path remains the fallback for `ct=8/12` and
  non-exact decompositions.

Validation:
- Full LLK correctness:
  `python_env/bin/python3 -m pytest -q --tb=short -o log_cli=false tt_metal/tt-llk/tests/python_tests/test_fast_untilize.py`
  -> `2160 passed in 399.75s`.
- Wide perf gate:
  `python_env/bin/python3 -m pytest -q --tb=short -o log_cli=false tt_metal/tt-llk/tests/python_tests/perf_fast_untilize.py -k 'loop_factor:16 and (ct_dim:5 or ct_dim:6 or ct_dim:7 or ct_dim:8 or ct_dim:9 or ct_dim:12 or ct_dim:16)'`
  -> `378 passed, 1242 deselected in 400.92s`.
- Focused cold/steady A/B against `FAST_UNTILIZE_STRIDED_DEST_REUSE=0` covered
  `ct=8/12/16`, `loop_factor=1/16`, all supported formats, `SyncHalf`/`SyncFull`,
  and dest 16/32 modes.

Matched perf result for the accepted gate:

| scope | marker | run type | min delta | mean delta | max delta | regressions >2% |
|:---|:---|:---|---:|---:|---:|---:|
| `ct=8/12/16`, `loop_factor=1` | KERNEL | `L1_TO_L1` | `-2.44%` | `-0.40%` | `+0.40%` | 0 |
| `ct=8/12/16`, `loop_factor=1` | KERNEL | `PACK_ISOLATE` | `-0.36%` | `+0.21%` | `+1.88%` | 0 |
| `ct=8/12/16`, `loop_factor=16` | KERNEL | `L1_TO_L1` | `-3.09%` | `-0.61%` | `+0.04%` | 0 |
| `ct=8/12/16`, `loop_factor=16` | KERNEL | `PACK_ISOLATE` | `-0.74%` | `-0.05%` | `+0.25%` | 0 |
| `ct=8/12/16`, `loop_factor=16` | TILE_LOOP | `L1_TO_L1` | `-3.13%` | `-0.62%` | `+0.03%` | 0 |
| `ct=8/12/16`, `loop_factor=16` | TILE_LOOP | `PACK_ISOLATE` | `-0.78%` | `-0.07%` | `+0.14%` | 0 |

By `ct`, KERNEL `loop_factor=16` deltas:

| `ct` | `L1_TO_L1` mean | `PACK_ISOLATE` mean | regressions >2% |
|---:|---:|---:|---:|
| 8 | `0.00%` | `0.00%` | 0 |
| 12 | `-0.00%` | `0.00%` | 0 |
| 16 | `-1.82%` | `-0.17%` | 0 |
