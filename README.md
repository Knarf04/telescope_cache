# Telescoping Attention 
Goals: 
- [ ] 1B prototype in `fms` format to show TA outperforms SWA
- [ ] `hf` format implementation for larger-scaled granite


TODOs:
- [ ] Extract Davis' implementation ([training](https://github.com/daviswer/foundation-model-stack-sandbox/blob/telescoping-cache-flex/fms/modules/attention.py), [inference](https://github.com/daviswer/foundation-model-stack-sandbox/blob/telescoping-inference-hybrid/fms/modules/attention.py)) into modules
- [ ] Implement sink token & gated attention ([hf pr](https://github.com/huggingface/transformers/pull/47179))
- [ ] Think about proper RoPE implementation for aggregated caches

## File roles

| File | Role |
|---|---|
| `range_spec.py` | Frozen pure-integer range spec: `range_bounds`, `elem_mask`, `fwd_bounds`, `bwd_bounds`, `node_query_bounds`, packed offsets. Level-local coordinates only. |
| `reference.py` | **The implementation** (extract the kernel from here): summary weights, dyadic builder, `pack_levels`, `multilevel_attention_forward` → `(out, lse)`, explicit two-pass `multilevel_attention_backward`. No oracles. |
| `test/test_forward.py` | Historical oracles (original scan plan, dense flattened-mask reference, loop POCs, analytic range oracles) and the six-configuration forward chain checking `reference.multilevel_attention_forward` — `out` and `lse` — against all of them, with coverage assertions (warm-up, eviction, odd lengths, partial tiles, `Dv≠Dk`, GQA). Defines `CASES`. |
| `test/test_backward.py` | Part 1: end-to-end autograd gradient oracle (original chain vs `reference.py`, incl. the `q/k → w →` summary-tree path, `detach_weights`, `dv` invariance). Part 2: `reference.multilevel_attention_backward` vs autograd at the packed boundary (saved-LSE reconstruction, softcap derivative, never-visible nodes, conservative-hull tile). Part 3: explicit Phase 2 + autograd Phase-1 VJP ≡ Part-1 oracle. |
| `test/test_range.py` | Exhaustive property tests of `range_spec` at small N against the analytic range oracles. |

For kernel work: `reference.py` + `range_spec.py` define **what to implement**;
the test files define **what must still pass**. The oracle implementations in
`test_forward.py` are deliberately *not* part of the clean module.

Run everything (CPU, from `test/`):

```bash
python test_range.py && python test_forward.py --device cpu && python test_backward.py --device cpu
```

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
