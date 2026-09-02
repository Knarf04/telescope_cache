# Telescoping Attention 
Goals: 
- [ ] 1B prototype in `fms` format to show TA outperforms SWA
- [ ] `hf` format implementation for larger-scaled granite


TODOs:
- [ ] Extract Davis' implementation ([training](https://github.com/daviswer/foundation-model-stack-sandbox/blob/telescoping-cache-flex/fms/modules/attention.py), [inference](https://github.com/daviswer/foundation-model-stack-sandbox/blob/telescoping-inference-hybrid/fms/modules/attention.py)) into modules
- [x] Implement sink token & gated attention ([hf pr](https://github.com/huggingface/transformers/pull/47179)) — `reference.apply_attention_sink`/`apply_output_gate`, `use_sinks`/`use_gate` in `fms/train.py`
- [x] Think about proper RoPE implementation for aggregated caches — post-summary RoPE (`position_mode="rope"`): a summary is a virtual token at its interval's right endpoint, rotated after aggregation (deliberately NOT the merge of individually-rotated keys); see `multilevel_attention_forward`'s positional-encoding contract and `test/test_position.py`

## File roles

| File | Role |
|---|---|
| `range_spec.py` | Frozen pure-integer range spec: `range_bounds`, `elem_mask`, `fwd_bounds`, `bwd_bounds`, `node_query_bounds`, packed offsets. Level-local coordinates only. |
| `reference.py` | **The implementation** (extract the kernel from here): summary weights, dyadic builder, `pack_levels`, `multilevel_attention_forward` → `(out, lse)` with optional positional encoding (`position_mode` none/rope/relative: post-summary RoPE at interval right endpoints + learned summary-bin relative bias), explicit two-pass `multilevel_attention_backward`. No oracles. |
| `test/test_forward.py` | Historical oracles (original scan plan, dense flattened-mask reference, loop POCs, analytic range oracles) and the six-configuration forward chain checking `reference.multilevel_attention_forward` — `out` and `lse` — against all of them, with coverage assertions (warm-up, eviction, odd lengths, partial tiles, `Dv≠Dk`, GQA). Defines `CASES`. |
| `test/test_backward.py` | Part 1: end-to-end autograd gradient oracle (original chain vs `reference.py`, incl. the `q/k → w →` summary-tree path, `detach_weights`, `dv` invariance). Part 2: `reference.multilevel_attention_backward` vs autograd at the packed boundary (saved-LSE reconstruction, softcap derivative, never-visible nodes, conservative-hull tile). Part 3: explicit Phase 2 + autograd Phase-1 VJP ≡ Part-1 oracle. The Phase-2 backward contract also takes an optional keyword-only `dlse` seed (gradient wrt the returned lse, for attention-sink support; `None` = legacy behavior, tested in `test_sink_gate.py`). |
| `test/test_range.py` | Exhaustive property tests of `range_spec` at small N against the analytic range oracles. |
| `test/test_shortconv.py` | Unit contract for the optional short causal depthwise K/V conv (`reference.short_conv` / `fms_template.ShortConv1d`, enabled in `fms/train.py` via `use_kv_short_conv`; `fms/inference.py` has no conv support yet): causality, hand-loop equivalence, FP32 residual-add ordering, K=1, channel independence, zero-init identity, K/V parameter independence, error paths, e2e gradients. Training-only; decode TODOs live on the modules. |
| `test/test_sink_gate.py` | Unit contract for the learned attention sink and SiLU output gate (`reference.apply_attention_sink` / `apply_output_gate`; enabled in `fms/train.py` via `use_sinks` / `use_gate`, absent from `fms/inference.py`): sink vs a literal augmented-softmax oracle in forward AND gradients (catches a dropped lse gradient path), zero-sink ≠ identity, SiLU gate exactness incl. GQA / `Dv≠Dk` / bf16, four-combo e2e composition with grads to `sinks`/`gate_proj`, error paths, and the explicit-backward composition `attention_sink_backward` → `multilevel_attention_backward(dlse=...)` vs autograd, with a negative control for the dropped-LSE-path failure mode. Post-ops over the frozen `(out, lse)` contract — the kernel suites are unaffected. |
| `test/test_position.py` | Positional-encoding contract (`position_mode` none/rope/relative): NoPE bitwise regression; post-summary RoPE vs dense causal RoPE oracles on L0-only schedules (full, windowed, empty-coarse-level); right-endpoint position units; analytic 0-based summary-bin distances vs the slow range oracle across three schedules, incl. the partition property and cross-level span independence; dense relative-bias oracle; gradient flow to `relative_weight`/`relative_proj`/q/k/x; tree position-independence; error paths (explicit backward raises for non-none modes). Training-only; decode + CuTeDSL kernel support are TODOs on the forward. |
| `cute/ranges.py` | Branch-free closed forms of `range_spec` (`+ - * >> min max` only, no data-dependent `if`), so one source runs on host ints, torch tensors and CuTeDSL Int32. Replaces `fwd_bounds`' block scan with an O(1) hull. |
| `cute/packing.py` | Block-aligned level-major K/V: each level padded (with **zeros**) up to a multiple of `tile_n`, so a level-local tile index is a compile-time offset from the level's block base and no K/V copy needs seqlen predication. |
| `cute/flash_fwd_telescope.py` | **The kernel.** FA4's `FlashAttentionForwardSm80` with the block schedule and the mask replaced; loads, GEMMs, online softmax and epilogue inherited. Introduces the *virtual block index* so FA4's contiguous-descending pipeline can drive a union of L+1 disjoint block ranges. |
| `cute/flash_bwd_telescope.py` | **The backward kernel.** FA4's `FlashAttentionBackwardSm80` with the same two regions replaced. One KV-owned pass (dK/dV in registers, dQ scattered into an fp32 accumulator) rather than the reference's two explicit passes; only the Pass-B direction of the range spec is needed, and no virtual index, because a KV tile's query range is a single contiguous interval. |
| `cute/interface.py` | Host entry point for both directions: builds cute tensors, instantiates kernels specialized to one schedule, compiles once per configuration. Forward: `telescope_attn_packed` (fast path) and `telescope_attn_func` (drop-in for the reference). Backward: `telescope_attn_bwd`, including the preprocess/postprocess plumbing FA4 leaves unfinished on Ampere. |
| `cute/mask_mod.py` | Baseline GPU path: the same semantics as a FlashAttention-4 `mask_mod` over the packed buffer. Correctness cross-check and the performance floor to beat -- it walks every key block, the kernel walks only the selected ones. |
| `test/test_cute_ranges.py` | `cute/ranges.py` vs `range_spec` exhaustively over every (query, level) in nine schedules, plus hull containment over five `BLOCK_M` values. |
| `test/test_cute_forward.py` | GPU forward: both CuTeDSL paths vs `reference.multilevel_attention_forward`, scored against the bf16 rounding floor so dtype error is not read as kernel error. |
| `test/test_cute_backward.py` | GPU backward: the kernel vs `reference.multilevel_attention_backward` at the packed boundary, fed the forward's saved LSE, scored against the bf16 floor. Also asserts the K/V padding receives exactly zero dK/dV. |
| `test/flex_baseline.py` | `fms/train.py`'s FlexAttention path isolated to attention alone. Imports `get_scan_plan` and `scan` from `fms/train.py` rather than reimplementing them; the cache build, dense mask, `create_block_mask` and `soft_cap` are copied verbatim from `MultiHeadAttention.forward`. |
| `test/test_flex_equivalence.py` | Anchors the equivalence chain to the code that actually trains: `train.py`'s flex path vs `reference.multilevel_attention_forward` vs the kernel. Checks the selection (mask row sums == `\|A(q)\|` from `range_spec`) separately from the values. |
| `test/bench_forward.py` | Forward benchmark: telescope kernel vs the `fms/train.py` flex baseline, FA4 `mask_mod`, sliding window of `cache_size`, dense causal, and a dense N x cache_size floor. |
| `test/analyze_tiles.py` | Host-side tile-occupancy model (no GPU): useful lanes vs lanes the MMA actually computes, per (tile_m, tile_n) and per level. Explains where the kernel's time goes and bounds what tuning can buy. |
| `test/sweep_tiles.py` | GPU sweep of (tile_m, tile_n, num_threads) validating that model. |

For kernel work: `reference.py` + `range_spec.py` define **what to implement**;
the test files define **what must still pass**. The oracle implementations in
`test_forward.py` are deliberately *not* part of the clean module.

Run everything (CPU, from `test/`):

```bash
python test_range.py && python test_forward.py --device cpu && python test_backward.py --device cpu && python test_shortconv.py --device cpu && python test_sink_gate.py --device cpu && python test_position.py --device cpu
```

Kernel work (needs a CUDA device and `flash_attn_4`; from `test/`):

```bash
python test_cute_ranges.py                 # CPU, no GPU needed
python test_cute_forward.py                # fwd kernel + mask_mod vs the reference
python test_cute_backward.py               # bwd kernel vs the reference
python test_flex_equivalence.py            # vs fms/train.py's flex_attention
python analyze_tiles.py                    # CPU: where the work goes
python bench_forward.py                    # timings
python sweep_tiles.py                      # tile tuning
```

Two things about FA4 on Ampere that shaped the design:

* **Sliding-window block skipping is not implemented on SM80.** `flash_fwd.py`
  computes `n_block_min` and then loops to block 0 anyway (`# TODO: local`
  is on the line after). Measured: at N=8192 the runtime is the same for
  windows of 127 and 4095 and for full causal. So FA4's own SWA is O(N^2)
  here; do not read the telescope-vs-SWA ratio as an algorithmic result.
* **The backward refuses Ampere and its arch-8 host path has never run** --
  `_flash_attn_bwd` asserts capability in {9,10,11,12}, and past the assert
  it dies on `UnboundLocalError: dQ_single_wg`. The *kernel* is fine; it
  reproduces a float32 torch reference on sm86 when driven directly, which
  is why `cute/interface.py` carries its own backward launcher.

`flash_attn_4` is the FlashAttention-4 CuTeDSL package. It is a pure-Python
wheel (all kernels are JIT-compiled through CuTeDSL, nothing is prebuilt per
arch) published on the flash-attention releases page, not on PyPI:

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/fa4-v4.0.0.beta28/flash_attn_4-4.0.0b28-py3-none-any.whl
```

Despite the name, FA4 is not Blackwell-only: `interface.py` dispatches
compute capability 8.x to a generic SM80 kernel (`flash_fwd.py`), which is
what the telescope kernel derives from and what runs here on sm86. The
Hopper/Blackwell-specific files (`flash_fwd_sm90.py`, `flash_fwd_sm100.py`)
are separate; porting the telescope schedule to them is future work.

## Using the kernels

```python
from telescope_cache.cute.interface import (
    BWD_N_BLOCK, telescope_attn_packed, telescope_attn_bwd,
)
from telescope_cache.cute.packing import pack_levels_aligned
from telescope_cache.cute.ranges import coarsest_lifetime_span
from telescope_cache.range_spec import RangeSpec
from telescope_cache.reference import build_dyadic_summaries

spec = RangeSpec.from_fmap(fmap, cache_size, N)
k_levels, v_levels, _ = build_dyadic_summaries(q, k, v, max(fmap))

# Pack ONCE, at the backward's block size. The forward may run at a finer
# tile_n inside the same buffer -- all it needs is that every level start on
# one of its block boundaries, and 64-aligned offsets are also 16-aligned.
packed = pack_levels_aligned(k_levels, v_levels, tile_n=BWD_N_BLOCK)

out, lse = telescope_attn_packed(q, packed, spec.activation_times, cache_size)
dq, dk, dv = telescope_attn_bwd(
    q, packed, out, lse, dout,
    spec.activation_times, cache_size,
    coarsest_span=coarsest_lifetime_span(spec),
)
```

`lse` is the forward's saved natural-log value and must be passed to the
backward unchanged, never recomputed -- the same contract `reference.py`
states. `dk`/`dv` come back in the padded packed layout; slice them per level
with `packed.pad_offsets`, and feed them as VJP seeds into the Phase-1 graph
exactly as `multilevel_attention_backward`'s docstring describes. The padding
rows are zero, because every padded column is masked.

Both directions are specialized at compile time to one
`(activation_times, cache_size, pad_offsets)` triple, so each distinct
schedule compiles its own kernel and they are cached per configuration.

## Flattened cache vs. dyadic tree

`fms/train.py` builds the telescoping cache by *replaying* a slot-shifting
process (`get_scan_plan` + `scan`) and hands FlexAttention a dense Boolean
mask. The tests under `test/` show the same attention can be expressed over a
**canonical dyadic summary tree** with contiguous per-level storage, which is
what a custom GPU kernel wants. The two hold identical values; they differ in
how entries are *addressed*. `reference.py` is the FlashAttention-shaped
implementation over the dyadic tree; `test/test_forward.py` carries the
equivalence chain from the original scan/mask semantics through to it.

### Worked example

`N = 8` tokens `t0…t7`, `cache_size = 6`, `fmap = {1: 2, 2: 3}` — a scaled-down
analogue of the real `{1: 64, 2: 72, 3: 80}`, chosen so the schedule is
dyadically aligned (the property `validate_dyadic_fmap` checks). Ruler levels
for steps 1..7 are `1,2,1,3,1,2,1`. Level 1 merges at slot 2, level 2 at
slot 3, level ≥ 3 shifts the whole cache and evicts slot 5.

Notation: `D` = dummy, `P` = pair summary (old level 2), `Q` = quad summary
(old level 3).

#### The replay (what `scan` simulates)

```
step  lvl  cache slots [0..5]                merge that happened
 0     -   [t0, D,  D,  D,  D,  D ]
 1     1   [t1, t0, P0, D,  D,  D ]          P0 = merge(D, D)      <- dummy
 2     2   [t2, t1, t0, Q0, D,  D ]          Q0 = merge(D, P0)     <- dummy
 3     1   [t3, t2, P1, Q0, D,  D ]          P1 = merge(t0, t1)    <- first real L1 (a[1]=3)
 4     3   [t4, t3, t2, P1, Q0, D ]          whole shift, D evicted
 5     1   [t5, t4, P2, P1, Q0, D ]          P2 = merge(t2, t3)
 6     2   [t6, t5, t4, Q1, Q0, D ]          Q1 = merge(P1, P2)    <- first real L2 (a[2]=6)
 7     1   [t7, t6, P3, Q1, Q0, D ]          P3 = merge(t4, t5)
```

`P0` and `Q0` exist only because the slots started out empty; they never
summarize tokens. The L1 pair `[t6, t7]` and the L2 quad `[t4..t7]` are
**never created** — by step 7 those tokens have not aged past slot 2.

#### Flattened cache (`train.py`) — 15 entries

All levels concatenated in creation order:

```
index:   0   1   2   3   4   5   6   7   8  |  9   10  11  12 | 13  14
entry:   D   t0  t1  t2  t3  t4  t5  t6  t7 |  P0  P1  P2  P3 | Q0  Q1
         '-- old level 1 (padded raw) ----'  '-- level 2 --'  '- level 3 -'
```

Merge plan (creation order, indices into the previous level):

```
plan[2] = [[0,0], [1,2], [3,4], [5,6]]   ->  P0, P1, P2, P3
plan[3] = [[0,0], [1,2]]                 ->  Q0, Q1
```

`P1 = [1,2]` means "level-1 entries 1 and 2" = `t0, t1`. Which tokens `P2`
covers can only be recovered by following the plan.

Query 7's slots are `[t7, t6, P3, Q1, Q0, D]`, so its mask row over the 15
columns is:

```
col:   0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
mask:  . . . . . . . 1 1 .  .  .  1  .  1      (Q0 and D zeroed by the validity flags)
```

`create_block_mask` must encode that row for every query: `[8, 15]` here,
`[N, ~2N]` in general.

#### Dyadic tree (`reference.py`) — 14 entries, three contiguous arrays

```
L0:  idx 0   1   2   3   4   5   6   7
     [t0, t1, t2, t3, t4, t5, t6, t7]

L1:  idx   0        1        2        3
     [ (t0,t1)  (t2,t3)  (t4,t5)  (t6,t7) ]        L1[j] = merge(L0[2j], L0[2j+1])

L2:  idx     0            1
     [ (t0..t3)     (t4..t7) ]                     L2[j] = merge(L1[2j], L1[2j+1])
```

Same merge rule, same values: `L1[0] == P1`, `L1[1] == P2`, `L1[2] == P3`,
`L2[0] == Q1` (verified by `check_summary_equivalence`). Differences: no
`P0`/`Q0`; `L1[3]` and `L2[1]` exist eagerly even though no query reads them;
and the index *is* the position — `L1[j]` covers tokens `[2j, 2j+1]`, no plan
needed. Storage is a wash (15 vs 14); the point is addressability.

Query 7's selection becomes one interval per level, computed in closed form
from `q = 7` and the activation times `a = [0, 3, 6]`
(`dyadic_ranges_for_query`):

```
L0: [6, 8)   -> t6, t7
L1: [2, 3)   -> (t4,t5)        = P3
L2: [0, 1)   -> (t0..t3)       = Q1
```

Same four entries as the mask row, expressed as three `(start, end)` pairs.
Query 5 for comparison: `L0=[4,6), L1=[0,2), L2=empty`, matching slots
`[t5, t4, P2, P1, Q0, D]`.

#### What the kernel sees

- **Flattened:** for each query block, walk a `[BLOCK_M, cache_len]` slice of
  a materialized mask to find which scattered columns to load, then gather
  them. The access pattern is data-dependent.
- **Tree:** for each query block and each level, load the tile range
  `[min(start), max(end))` — a contiguous, coalesced read shared by all
  `BLOCK_M` queries and all GQA heads — and mask each row with
  `start <= k < end` computed in registers. This is the sliding-window
  attention loop, repeated once per level, feeding one online softmax.

Same numbers in, same numbers out; the tree stores them where the hardware
can find them.
