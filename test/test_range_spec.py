"""
Exhaustive verification of telescope_cache/range_spec.py against the analytic
range oracles in test_tiled.py, at small N.

Properties checked per configuration and level (all half-open):

    act        spec.activation_times == validate_dyadic_fmap(fmap)
    range      range_bounds(q, l) == oracle R_l(q)      (scalar and block oracle)
    elem       elem_mask(q, k, l) <=> k in R_l(q)
    mono       k_lo(q), k_hi(q) non-decreasing over nonempty rows;
               min Q_l(k), max Q_l(k) non-decreasing over visible nodes
    fwd        fwd_bounds == hull over rows, == O(1) first/last-row form,
               contains every row; exactness measured
    inverse    Q_l(k) contiguous for every node; node_query_bounds == hull(Q)
               (never-visible nodes -> (0,0)); q in Q_l(k) <=> k in R_l(q)
    bwd        bwd_bounds == hull over nonempty Q_l(k) in the window;
               contains every Q_l(k) (required); exactness measured
    domain     out-of-domain inputs raise ValueError; empty windows -> (0,0)

Run:  python test_range_spec.py            (pure Python + torch for the block oracle)
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

from telescope_cache.range_spec import (  # noqa: E402
    EMPTY,
    RangeSpec,
    bwd_bounds,
    elem_mask,
    fwd_bounds,
    node_query_bounds,
    packed_bounds,
    range_bounds,
)
from test_tiled import (  # noqa: E402
    dyadic_ranges_for_query,
    dyadic_ranges_for_query_block,
    validate_dyadic_fmap,
)

FWD_WIDTHS = (1, 3, 16, 24)
BWD_WIDTHS = (1, 2, 32, 40)

CONFIGS = [
    # name, N, cache_size, fmap, exhaustive elem/inverse pairs?
    ("readme_tiny", 8, 6, {1: 2, 2: 3}, True),
    ("tiny_evict", 64, 12, {1: 2, 2: 3}, True),
    ("mid", 400, 40, {1: 8, 2: 12, 3: 16}, True),
    ("four_small", 300, 40, {1: 4, 2: 6, 3: 10, 4: 11}, True),
    ("baseline", 128, 512, {1: 64, 2: 72, 3: 80}, True),
    ("eviction", 1024, 160, {1: 64, 2: 72, 3: 80}, False),
]


def build_tables(N, fmap, cache_size, level, activation_times):
    """Oracle tables built once: R[q] = (lo, hi); Q[k] = sorted list of q."""
    R = [
        dyadic_ranges_for_query(q, fmap, cache_size, activation_times)[level]
        for q in range(N)
    ]
    n_nodes = N >> level
    Q = [[] for _ in range(n_nodes)]
    for q, (lo, hi) in enumerate(R):
        for k in range(lo, hi):
            Q[k].append(q)
    return R, Q


def non_decreasing(xs):
    return all(x <= y for x, y in zip(xs, xs[1:]))


def check_level(name, spec, N, fmap, cache_size, level, exhaustive, block):
    a = spec.activation_times
    R, Q = build_tables(N, fmap, cache_size, level, list(a))
    n_nodes = spec.level_len(level)
    assert n_nodes == len(Q), (name, level, n_nodes, len(Q))
    flags = {}

    # ---- range: spec == oracle (scalar and block); packed translation ----
    starts, ends = block
    off = spec.level_offsets()
    for q in range(N):
        got = range_bounds(spec, q, level)
        assert got == R[q], f"[{name}] L{level} q={q}: {got} != {R[q]}"
        assert got == (starts[q][level], ends[q][level])
        pb = packed_bounds(spec, q, level)
        if got == EMPTY:
            assert pb == EMPTY
        else:
            assert pb == (off[level] + got[0], off[level] + got[1])
            assert off[level] <= pb[0] < pb[1] <= off[level + 1]
    flags["range"] = True

    # ---- elem: elem_mask <=> k in R_l(q) ----
    if exhaustive:
        for q in range(N):
            lo, hi = R[q]
            for k in range(n_nodes):
                assert elem_mask(spec, q, k, level) == (lo <= k < hi), (
                    f"[{name}] L{level} elem_mask({q},{k})"
                )
    else:
        # Boundaries of every row + a strided sample of the interior.
        for q in range(N):
            lo, hi = R[q]
            ks = {lo - 1, lo, hi - 1, hi, 0, n_nodes - 1}
            ks.update(range(0, n_nodes, 37))
            for k in ks:
                if 0 <= k < n_nodes:
                    assert elem_mask(spec, q, k, level) == (lo <= k < hi)
    flags["elem"] = True

    # ---- mono: row bounds over nonempty rows; node hulls over visible nodes ----
    rows = [(lo, hi) for lo, hi in R if lo < hi]
    assert non_decreasing([lo for lo, _ in rows]), f"[{name}] L{level} k_lo"
    assert non_decreasing([hi for _, hi in rows]), f"[{name}] L{level} k_hi"
    vis_nodes = [k for k in range(n_nodes) if Q[k]]
    assert non_decreasing([Q[k][0] for k in vis_nodes]), f"[{name}] L{level} q_lo"
    assert non_decreasing([Q[k][-1] for k in vis_nodes]), f"[{name}] L{level} q_hi"
    flags["mono"] = True

    # ---- fwd: hull over rows; O(1) form; containment; exactness ----
    fwd_exact = True
    windows = [(0, N)]
    for w in FWD_WIDTHS:
        windows += [(s, min(s + w, N)) for s in range(0, N)]
    for q_lo, q_hi in windows:
        got = fwd_bounds(spec, q_lo, q_hi, level)
        nonempty = [R[q] for q in range(q_lo, q_hi) if R[q][0] < R[q][1]]
        expect = (
            (min(lo for lo, _ in nonempty), max(hi for _, hi in nonempty))
            if nonempty else EMPTY
        )
        assert got == expect, f"[{name}] L{level} fwd[{q_lo},{q_hi}) {got}!={expect}"
        if nonempty:
            assert got == (nonempty[0][0], nonempty[-1][1])  # O(1) form
            for lo, hi in nonempty:
                assert got[0] <= lo and hi <= got[1]
            attended = set()
            for lo, hi in nonempty:
                attended.update(range(lo, hi))
            if len(attended) != got[1] - got[0]:
                fwd_exact = False
    flags["fwd"] = True
    flags["fwd-exact"] = fwd_exact

    # ---- inverse: contiguity, node_query_bounds, second iff ----
    never_visible = 0
    for k in range(n_nodes):
        qs = Q[k]
        got = node_query_bounds(spec, k, level)
        if not qs:
            never_visible += 1
            assert got == EMPTY, f"[{name}] L{level} node {k} never visible: {got}"
            continue
        assert qs[-1] - qs[0] + 1 == len(qs), (
            f"[{name}] L{level} node {k}: inverse set not contiguous"
        )
        assert got == (qs[0], qs[-1] + 1), (
            f"[{name}] L{level} node {k}: {got} != {(qs[0], qs[-1] + 1)}"
        )
    if exhaustive:
        for k in range(n_nodes):
            qset = set(Q[k])
            for q in range(N):
                assert elem_mask(spec, q, k, level) == (q in qset)
    flags["inverse-contig"] = True

    # ---- bwd: hull over nonempty Q; containment; exactness ----
    bwd_exact = True
    windows = [(0, n_nodes)]
    for w in BWD_WIDTHS:
        windows += [(s, min(s + w, n_nodes)) for s in range(0, n_nodes)]
    for k_lo, k_hi in windows:
        got = bwd_bounds(spec, k_lo, k_hi, level)
        vis = [Q[k] for k in range(k_lo, k_hi) if Q[k]]
        expect = (
            (min(qs[0] for qs in vis), max(qs[-1] for qs in vis) + 1)
            if vis else EMPTY
        )
        assert got == expect, f"[{name}] L{level} bwd[{k_lo},{k_hi}) {got}!={expect}"
        for qs in vis:
            assert got[0] <= qs[0] and qs[-1] < got[1]
        if vis:
            for q in range(got[0], got[1]):
                lo, hi = R[q]
                if not max(lo, k_lo) < min(hi, k_hi):
                    bwd_exact = False
                    break
    flags["bwd-contain"] = True
    flags["bwd-exact"] = bwd_exact

    return flags, never_visible


def check_domain(spec):
    N, L = spec.seq_len, spec.coarsest

    def raises(fn):
        try:
            fn()
        except ValueError:
            return True
        return False

    assert raises(lambda: range_bounds(spec, N, 0))
    assert raises(lambda: range_bounds(spec, -1, 0))
    assert raises(lambda: range_bounds(spec, 0, L + 1))
    assert raises(lambda: elem_mask(spec, 0, spec.level_len(0), 0))
    assert raises(lambda: fwd_bounds(spec, 0, N + 1, 0))
    assert raises(lambda: fwd_bounds(spec, 2, 1, 0))
    assert raises(lambda: node_query_bounds(spec, spec.level_len(L), L))
    assert raises(lambda: bwd_bounds(spec, 0, spec.level_len(0) + 1, 0))
    assert fwd_bounds(spec, 3 % N, 3 % N, 0) == EMPTY
    assert bwd_bounds(spec, 0, 0, 0) == EMPTY


def run_config(name, N, cache_size, fmap, exhaustive):
    spec = RangeSpec.from_fmap(fmap, cache_size, N)

    # act: pure-int duplicate of the oracle's activation formula must agree.
    oracle_a = validate_dyadic_fmap(fmap)
    assert spec.activation_times == tuple(oracle_a), (name, spec.activation_times, oracle_a)

    # offsets: prefix sums of floor-halved lengths; total_len is their sum.
    lens = [N >> l for l in range(spec.num_levels)]
    expect_off = [0]
    for n in lens:
        expect_off.append(expect_off[-1] + n)
    assert spec.level_offsets() == tuple(expect_off), (name, spec.level_offsets())
    assert spec.total_len == sum(lens)

    starts, ends = dyadic_ranges_for_query_block(
        torch.arange(N), fmap, cache_size, oracle_a
    )
    block = (starts.tolist(), ends.tolist())
    block = (
        [[int(x) for x in row] for row in block[0]],
        [[int(x) for x in row] for row in block[1]],
    )

    all_flags = None
    never_visible = []
    inexact_bwd, inexact_fwd = [], []
    for level in range(spec.num_levels):
        flags, nv = check_level(
            name, spec, N, fmap, cache_size, level, exhaustive, block
        )
        never_visible.append(nv)
        if not flags["bwd-exact"]:
            inexact_bwd.append(level)
        if not flags["fwd-exact"]:
            inexact_fwd.append(level)
        all_flags = flags if all_flags is None else {
            k: all_flags[k] and flags[k] for k in flags
        }
    check_domain(spec)

    def tick(key):
        return "✓" if all_flags[key] else "✗"

    line = (
        f"{name:<12} N={N:<5} C={cache_size:<4} levels={spec.num_levels} "
        f"act✓ offsets✓ range{tick('range')} elem{tick('elem')} mono{tick('mono')} "
        f"fwd{tick('fwd')} inverse-contig{tick('inverse-contig')} "
        f"bwd-contain{tick('bwd-contain')} "
        f"fwd-exact{'✓' if not inexact_fwd else '✗ (conservative at L' + ','.join(map(str, inexact_fwd)) + ')'} "
        f"bwd-exact{'✓' if not inexact_bwd else '✗ (conservative at L' + ','.join(map(str, inexact_bwd)) + ')'} "
        f"never_visible={never_visible}"
        f"{'' if exhaustive else '  (elem/inverse pairs sampled)'}"
    )
    print(line)


def main():
    for cfg in CONFIGS:
        run_config(*cfg)
    print(f"\nALL RANGE-SPEC TESTS PASSED ({len(CONFIGS)} configs)")


if __name__ == "__main__":
    main()
