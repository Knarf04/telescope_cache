"""
The consolidated CPU correctness suite for telescoping attention.

Three implementations of the same semantics are compared pairwise:

    mini   telescope_cache/minimal_reference.py -- the naive semantic oracle
           (per-query visible-entry enumeration, ordinary softmax, literal
           sink logit, autograd-only backward)
    ref    telescope_cache/reference.py -- the kernel-shaped implementation
           (packed level-major buffers, tiled online softmax, LSE-sigmoid
           sink rescale, explicit two-pass Phase-2 backward)
    flex   telescope_cache/fms/train.py -- the FlexAttention training path
           (original slot-shifting scan plan, dense flat mask, compiled
           flex_attention; requires torch >= 2.5 + CUDA, so it SKIPs on
           machines without them)

Select the pair with --compare (default: mini ref):

    python test_reference.py --device cpu
    python test_reference.py --compare ref flex --device cuda
    python test_reference.py --compare mini flex [test names...]

Structure
---------
    Section 1  schedule: range_spec vs the analytic oracles + hand goldens
               (always runs -- both mini and ref consume range_bounds, so a
               range_spec bug is invisible to their comparison; the hull
               properties here are what the CuTeDSL kernels rely on)
    Section 2  forward semantics: pairwise comparison over a curated
               feature matrix (weights/conv/position/sink/gate/o_proj/
               softcap), summary-tree and attention-boundary checks
    Section 3  backward semantics: the packed-boundary Phase-2 contract
               (the future CuTeDSL backward interface), full-block
               gradients (mini autograd vs ref explicit composition; pure
               autograd for flex pairs), structural gradient invariants
    Section 4  contracts: mixed-precision (bf16/fp32-ordering), the fms
               ShortConv1d module, negative controls (test power), error
               paths (always runs)

Attention-boundary terminology (frozen): `out` is the POST-sink attention
output whenever the sink is enabled; `lse` is ALWAYS the pre-sink ordinary
attention LSE (float32, natural log). Enabling sinks changes `out`, never
`lse`.

Deliberately dropped from the historical suites (the migration that this
file consolidates): the original scan-plan machinery and its oracles
(get_structured_plan, build_plan_based_summaries, remap_plan_to_dyadic,
the plan/dyadic/ranges/online attention POCs, dense_reference_attention,
check_summary_equivalence, validate_analytic_ranges_against_plan), the
old-chain end-to-end gradient oracle, and the large-N CPU comparison
regimes (minimal is a per-query Python loop; the previously flex-validated
N=1024 config survives as a ref-flex case for the GPU box).
"""

import argparse
import functools
import os
import sys
from typing import Dict

import torch

# Make `telescope_cache` importable (namespace package).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

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
from telescope_cache.minimal_reference import (  # noqa: E402
    _check_summary_causality,
    _qk_summary_weights,
    _short_conv,
    _summary_tree,
    minimal_attention_block,
    minimal_multilevel_attention,
    minimal_rope_tables,
    minimal_visible_entries,
)
from telescope_cache.reference import (  # noqa: E402
    PackedKV,
    apply_attention_sink,
    apply_output_gate,
    attention_sink_backward,
    build_dyadic_summaries,
    compute_relative_states,
    multilevel_attention_backward,
    multilevel_attention_forward,
    pack_levels,
    relative_bin_distance,
    rope_tables,
    short_conv,
    summary_token_position,
)
from telescope_cache.fms.fms_template import ShortConv1d  # noqa: E402

RTOL = ATOL = 1e-5
GRAD_RTOL = GRAD_ATOL = 1e-4
GRAD_MIN_NORM = 1e-6
SOFTCAP = 20.0
# Flex runs through torch.compile on GPU; looser by construction. Tune on
# the GPU box if its numerics demand it.
FLEX_RTOL = FLEX_ATOL = 1e-3

# ---------------------------------------------------------------------------
# Geometries. Small ones with verified regimes (evict/odd_n evict from q=16,
# four_levels from q=48; max visible M never exceeds cache_size; every
# schedule satisfies the summary-causality invariant). `flex_big` is the
# config previously validated against fms/train.py's flex path -- too slow
# for minimal's per-query loop, so it is capped out of mini pairs.
# ---------------------------------------------------------------------------
GEOS = {
    "tiny": dict(B=2, N=8, Hq=4, Hkv=2, Dk=8, Dv=16,
                 fmap={1: 2, 2: 3}, cache=6, bm=3, bn=2),
    "evict": dict(B=1, N=64, Hq=4, Hkv=2, Dk=8, Dv=16,
                  fmap={1: 2, 2: 3}, cache=6, bm=16, bn=32),
    "evict_gqa3": dict(B=1, N=64, Hq=6, Hkv=2, Dk=8, Dv=8,
                       fmap={1: 2, 2: 3}, cache=6, bm=16, bn=32),
    "odd_n": dict(B=1, N=63, Hq=4, Hkv=2, Dk=8, Dv=16,
                  fmap={1: 2, 2: 3}, cache=6, bm=16, bn=32),
    "four_levels": dict(B=1, N=96, Hq=8, Hkv=4, Dk=16, Dv=16,
                        fmap={1: 4, 2: 6, 3: 8}, cache=12, bm=16, bn=32),
    "flex_big": dict(B=1, N=1024, Hq=4, Hkv=2, Dk=32, Dv=64,
                     fmap={1: 64, 2: 72, 3: 80}, cache=160, bm=16, bn=32),
}

_EMB, _KC, _DREL = 32, 4, 6

# ---------------------------------------------------------------------------
# Backend capabilities, declared once and consulted by the case runners
# (a future `cute` backend adds one row here, not scattered ifs).
# `boundary`: exposes the frozen (out, lse) attention boundary.
# ---------------------------------------------------------------------------
BACKEND_CAPS = {
    "mini": dict(position={"none", "rope", "relative"},
                 softcap={SOFTCAP, None}, boundary=True, max_n=256),
    "ref": dict(position={"none", "rope", "relative"},
                softcap={SOFTCAP, None}, boundary=True, max_n=None),
    "flex": dict(position={"none"},
                 softcap={SOFTCAP}, boundary=False, max_n=None),
}


def _unsupported(backend, case, g):
    """None if `backend` can run `case` on geometry `g`, else the reason."""
    caps = BACKEND_CAPS[backend]
    if case["position"] not in caps["position"]:
        return f"{backend}: position_mode={case['position']}"
    if case["softcap"] not in caps["softcap"]:
        return f"{backend}: softcap={case['softcap']}"
    if caps["max_n"] is not None and g["N"] > caps["max_n"]:
        return f"{backend}: N={g['N']} too large"
    return None


def pair_skip_reason(pair, case, g):
    for b in pair:
        why = _unsupported(b, case, g)
        if why:
            return why
    return None


# ---------------------------------------------------------------------------
# Forward feature matrix: curated toggle rows x geometries, not a Cartesian
# product. Every row: weights / conv / position / sink / gate / o_proj /
# softcap. Rows outside a pair's capabilities are skipped with a reason.
# ---------------------------------------------------------------------------
def _case(name, geos, *, weights="qk", conv=False, position="none",
          sink=False, gate=False, o_proj=False, softcap=SOFTCAP):
    return dict(name=name, geos=geos, weights=weights, conv=conv,
                position=position, sink=sink, gate=gate, o_proj=o_proj,
                softcap=softcap)


FWD_CASES = [
    _case("plain", ["tiny", "evict", "evict_gqa3", "odd_n", "four_levels"]),
    _case("linear", ["tiny", "odd_n"], weights="linear"),
    _case("conv", ["tiny", "four_levels"], conv=True),
    _case("rope", ["tiny", "evict"], position="rope"),
    _case("relative", ["tiny", "evict", "four_levels"], position="relative"),
    _case("sink", ["tiny", "evict"], sink=True),
    _case("gate", ["tiny"], gate=True),
    _case("nocap", ["tiny"], softcap=None),
    _case("full_none", ["tiny", "evict"], weights="linear", conv=True,
          sink=True, gate=True, o_proj=True),
    _case("full_rope", ["tiny"], conv=True, position="rope",
          sink=True, gate=True, o_proj=True),
    _case("full_relative", ["tiny"], conv=True, position="relative",
          sink=True, gate=True, o_proj=True),
    _case("flex_big", ["flex_big"]),
]

# Full-stack gradient cases (Section 3B). All toggles on; weights/position
# vary. mini<->ref uses the explicit composition; other pairs use autograd.
GRAD_CASES = [
    ("none", "linear", "tiny", 210),
    ("rope", "qk", "tiny", 214),
    ("relative", "qk", "tiny", 218),
    ("none", "qk", "evict", 222),
]


def report(name, got, want, rtol=RTOL, atol=ATOL):
    try:
        torch.testing.assert_close(got, want, rtol=rtol, atol=atol)
    except AssertionError as exc:
        raise AssertionError(f"{name}: {exc}") from None


def expect_error(err, label, fn):
    try:
        fn()
    except err:
        return
    raise AssertionError(f"{label}: no {err.__name__} raised")


def seeded_randn(shape, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=g).to(device)


def check_lse_contract(name, lse, B, N, Hq):
    if tuple(lse.shape) != (B, N, Hq):
        raise AssertionError(f"{name}: lse shape {tuple(lse.shape)}")
    if lse.dtype != torch.float32:
        raise AssertionError(f"{name}: lse dtype {lse.dtype} != float32")
    if not bool(torch.isfinite(lse).all()):
        raise AssertionError(f"{name}: non-finite lse entries")


# ===========================================================================
# Analytic schedule oracles (moved verbatim from the historical suite).
# These are the independent ground truth for range_spec.py: closed-form
# activation times and per-query ranges derived from the original
# ruler/slot-update policy, with no dependency on range_spec itself.
# ===========================================================================

def compute_dyadic_activation_times(fmap: Dict[int, int]):
    """
    For a dyadically aligned fmap, return a[l], the first query position at
    which canonical dyadic node 0 at level l becomes visible.

    Canonical levels:
        L0 span 1
        L1 span 2
        L2 span 4
        ...

    For the fmap {1:64, 2:72, 3:80}, this gives a = [0, 65, 82, 116].
    These constants are induced by the original ruler/slot-update policy.
    (range_spec.activation_times_from_fmap deliberately duplicates this
    formula; test_schedule_properties asserts the two agree.)
    """
    if not fmap:
        return [0]

    L = max(fmap)
    expected = list(range(1, L + 1))
    if sorted(fmap) != expected:
        raise ValueError(
            f"fmap keys must be consecutive 1..L; got {sorted(fmap)}"
        )

    a = [0] * (L + 1)
    a[0] = 0
    a[1] = fmap[1] + 1

    for level in range(2, L + 1):
        a[level] = (
            a[level - 1]
            + (2 ** (level - 1)) * (fmap[level] - fmap[level - 1])
            + 2 ** (level - 2)
        )

    return a


def validate_dyadic_fmap(fmap: Dict[int, int]):
    """
    Check that the slot schedule is compatible with the canonical aligned
    dyadic tree: for canonical level l, node 0 must enter on the correct
    ruler phase, a[l] mod 2^l == 2^(l-1) for l >= 1. Returns a as a list.
    """
    a = compute_dyadic_activation_times(fmap)

    for level in range(1, len(a)):
        expected_phase = 2 ** (level - 1)
        actual_phase = a[level] % (2 ** level)
        if actual_phase != expected_phase:
            raise ValueError(
                f"fmap is not compatible with the canonical dyadic tree at "
                f"level {level}: activation a[{level}]={a[level]} has phase "
                f"{actual_phase} mod {2 ** level}, expected {expected_phase}."
            )

    return a


def dyadic_ranges_for_query(
    q_index: int,
    fmap: Dict[int, int],
    cache_size: int,
    activation_times=None,
):
    """
    Analytically compute the contiguous cache range used from every dyadic
    level for one query. Returns half-open local-index ranges
    [(start, end)] for L0, L1, ..., LL.

    A canonical node j at level l first becomes visible at
    q = a[l] + 2^l * j; for every non-coarsest level it remains present
    until its pair is merged into the next level, so the live nodes at
    each level form one contiguous interval. The coarsest level uses all
    remaining logical cache slots and evicts the oldest summaries.
    """
    if activation_times is None:
        activation_times = validate_dyadic_fmap(fmap)

    L = len(activation_times) - 1
    ranges = []
    used_slots = 0

    for level in range(L):
        span = 2 ** level
        a = activation_times[level]

        if q_index < a:
            start = end = 0
        else:
            newest = (q_index - a) // span

            next_a = activation_times[level + 1]
            if q_index < next_a:
                oldest = 0
            else:
                parent_span = 2 * span
                newest_parent = (q_index - next_a) // parent_span
                # Every completed parent removes its two children from
                # this finer level.
                oldest = 2 * (newest_parent + 1)

            if oldest > newest:
                start = end = 0
            else:
                start = oldest
                end = newest + 1

        ranges.append((start, end))
        used_slots += end - start

    level = L
    span = 2 ** level
    a = activation_times[level]
    remaining = cache_size - used_slots

    if remaining < 0:
        raise RuntimeError(
            f"Fine levels already require {used_slots} slots, larger than "
            f"cache_size={cache_size}."
        )

    if q_index < a or remaining == 0:
        start = end = 0
    else:
        newest = (q_index - a) // span
        count = min(remaining, newest + 1)
        start = newest - count + 1
        end = newest + 1

    ranges.append((start, end))

    return ranges


def dyadic_ranges_for_query_block(
    q_indices: torch.Tensor,
    fmap: Dict[int, int],
    cache_size: int,
    activation_times=None,
):
    """
    Vectorized dyadic_ranges_for_query for one query block: returns
    (starts, ends), [M, num_levels] long tensors with the valid KV
    interval [starts[i,l], ends[i,l]) per query row and level.
    """
    if activation_times is None:
        activation_times = validate_dyadic_fmap(fmap)

    q_indices = q_indices.long()
    device = q_indices.device
    L = len(activation_times) - 1
    M = q_indices.numel()

    starts = torch.zeros(M, L + 1, dtype=torch.long, device=device)
    ends = torch.zeros_like(starts)
    used_slots = torch.zeros(M, dtype=torch.long, device=device)

    for level in range(L):
        span = 2 ** level
        a = activation_times[level]
        next_a = activation_times[level + 1]

        visible = q_indices >= a
        newest = torch.div(q_indices - a, span, rounding_mode="floor")

        parent_visible = q_indices >= next_a
        newest_parent = torch.div(
            q_indices - next_a, 2 * span, rounding_mode="floor"
        )

        oldest = torch.where(
            parent_visible,
            2 * (newest_parent + 1),
            torch.zeros_like(q_indices),
        )

        nonempty = visible & (oldest <= newest)
        start = torch.where(nonempty, oldest, torch.zeros_like(oldest))
        end = torch.where(nonempty, newest + 1, torch.zeros_like(newest))

        starts[:, level] = start
        ends[:, level] = end
        used_slots += end - start

    level = L
    span = 2 ** level
    a = activation_times[level]
    remaining = cache_size - used_slots

    if (remaining < 0).any():
        raise RuntimeError(
            "Fine hierarchy levels exceed cache_size for at least one "
            "query in the block."
        )

    visible = (q_indices >= a) & (remaining > 0)
    newest = torch.div(q_indices - a, span, rounding_mode="floor")
    count = torch.minimum(remaining, newest + 1)
    start = newest - count + 1
    end = newest + 1

    starts[:, level] = torch.where(visible, start, torch.zeros_like(start))
    ends[:, level] = torch.where(visible, end, torch.zeros_like(end))

    return starts, ends


# Hand-derived from the README replay table for N=8, cache_size=6,
# fmap={1:2,2:3} -- the one oracle that is not "some other implementation
# of the same formula":
#
#   q=0: [t0]                    q=4: [t4,t3,t2,P1,Q0]
#   q=1: [t1,t0,P0(dummy)]       q=5: [t5,t4,P2,P1,Q0]
#   q=2: [t2,t1,t0,Q0(dummy)]    q=6: [t6,t5,t4,Q1,Q0]
#   q=3: [t3,t2,P1,Q0]           q=7: [t7,t6,P3,Q1,Q0,D]
#
# P1=merge(t0,t1)=L1[0], P2=L1[1], P3=merge(t4,t5)=L1[2],
# Q1=merge(P1,P2)=tokens 0..3=L2[0].
README_TINY_GOLDEN = {
    0: [(0, 1), (0, 0), (0, 0)],
    1: [(0, 2), (0, 0), (0, 0)],
    2: [(0, 3), (0, 0), (0, 0)],
    3: [(2, 4), (0, 1), (0, 0)],
    4: [(2, 5), (0, 1), (0, 0)],
    5: [(4, 6), (0, 2), (0, 0)],
    6: [(4, 7), (0, 0), (0, 1)],
    7: [(6, 8), (2, 3), (0, 1)],
}


# ===========================================================================
# Backends. Each maps (params, case, geometry) -> a dict with
#     final : the differentiable end-of-block output
#     out   : POST-sink attention output   (mini/ref only, boundary rows)
#     lse   : pre-sink attention LSE fp32  (mini/ref only, boundary rows)
# The boundary tensors are produced only for rows without gate/o_proj
# (where final == out); full-stack rows compare `final` alone.
#
# Parameter discipline: make_case_params draws ONE frozen base dict; every
# backend takes fresh leaves via clone_leaves. No backend draws randomness
# internally, so every comparison is implementation-only.
# ===========================================================================

def _spec(g):
    return RangeSpec.from_fmap(g["fmap"], g["cache"], g["N"])


def make_case_params(seed, case, g, device):
    B, N, Hq, Hkv, Dk, Dv = (g[s] for s in ("B", "N", "Hq", "Hkv", "Dk", "Dv"))
    emb, Kc, d_rel = _EMB, _KC, _DREL
    torch.manual_seed(seed)
    base = dict(
        x=torch.randn(B, N, emb, device=device),
        Wq=torch.randn(Hq * Dk, emb, device=device) / emb ** 0.5,
        Wk=torch.randn(Hkv * Dk, emb, device=device) / emb ** 0.5,
        Wv=torch.randn(Hkv * Dv, emb, device=device) / emb ** 0.5,
    )
    if case["conv"]:
        base["kcw"] = torch.randn(Hkv * Dk, Kc, device=device) / Kc ** 0.5
        base["vcw"] = torch.randn(Hkv * Dv, Kc, device=device) / Kc ** 0.5
    if case["weights"] == "linear":
        base["w_proj"] = torch.randn(Hkv, emb, device=device) / emb ** 0.5
    if case["position"] == "relative":
        max_bins = max(8, g["cache"])
        base["rel_w"] = torch.randn(Hq * d_rel, emb, device=device) / emb ** 0.5
        base["rel_proj"] = torch.randn(d_rel, max_bins, device=device) / d_rel ** 0.5
    if case["gate"]:
        base["gate_w"] = torch.randn(Hq * Dv, emb, device=device) / emb ** 0.5
    if case["o_proj"]:
        base["o_w"] = torch.randn(emb, Hq * Dv, device=device) / (Hq * Dv) ** 0.5
    if case["sink"]:
        # Center sinks on this configuration's lse so sigmoid(lse - s) ~ 0.5
        # and the sink path stays load-bearing. The probe only picks
        # constants; both backends then receive the same sinks.
        with torch.no_grad():
            probe = run_ref(base, case, g, device)
        torch.manual_seed(seed + 1)
        base["sinks"] = (probe["probe_lse"].float().mean(dim=(0, 1))
                         + 0.1 * torch.randn(Hq, device=device))
    return base


def clone_leaves(base):
    return {k: t.detach().clone().requires_grad_(True)
            for k, t in base.items()}


def _projections(t, g):
    B, N, Hq, Hkv, Dk, Dv = (g[s] for s in ("B", "N", "Hq", "Hkv", "Dk", "Dv"))
    x = t["x"]
    q4 = x.matmul(t["Wq"].t()).view(B, N, Hq, Dk)
    k4 = x.matmul(t["Wk"].t()).view(B, N, Hkv, Dk)
    v4 = x.matmul(t["Wv"].t()).view(B, N, Hkv, Dv)
    return q4, k4, v4


def run_ref(t, case, g, device, attn_no_grad=False):
    """reference.py chain. Boundary rows return (out, lse) with out
    post-sink and lse the raw forward lse (pre-sink, always). The chain
    internals (q4, packed, states, out_plain, lse_plain) are always
    returned for the explicit-composition gradient tests; attn_no_grad
    runs the attention itself outside the graph while keeping the Phase-1
    graph alive (the explicit-backward leaf discipline)."""
    B, N, Hq, Dv = g["B"], g["N"], g["Hq"], g["Dv"]
    L = len(g["fmap"])
    q4, k4, v4 = _projections(t, g)
    bd_kwargs = {}
    if case["conv"]:
        bd_kwargs.update(k_conv_weight=t["kcw"], v_conv_weight=t["vcw"])
    if case["weights"] == "linear":
        bd_kwargs.update(x=t["x"], w_proj=t["w_proj"])
    kl, vl, _ = build_dyadic_summaries(q4, k4, v4, L, **bd_kwargs)
    packed = pack_levels(kl, vl)
    states = None
    fw_kwargs = {}
    if case["position"] == "rope":
        cos, sin = rope_tables(N, g["Dk"], device=device)
        fw_kwargs = dict(rope_cos=cos, rope_sin=sin)
    elif case["position"] == "relative":
        states = compute_relative_states(t["x"], t["rel_w"], Hq)
        fw_kwargs = dict(relative_states=states,
                         relative_proj=t["rel_proj"])

    def _fwd():
        return multilevel_attention_forward(
            q4, packed, g["fmap"], g["cache"], block_m=g["bm"],
            block_n=g["bn"], softcap=case["softcap"],
            position_mode=case["position"], **fw_kwargs)

    if attn_no_grad:
        with torch.no_grad():
            out_plain, lse_plain = _fwd()
    else:
        out_plain, lse_plain = _fwd()
    check_lse_contract("ref", lse_plain, B, N, Hq)
    if "sinks" not in t:
        # Probe path for sink centering (case may declare sink before
        # sinks exist in the dict).
        if case["sink"]:
            return dict(probe_lse=lse_plain)
        out = out_plain
    else:
        out = apply_attention_sink(out_plain, lse_plain, t["sinks"])
    final = out
    if case["gate"]:
        final = apply_output_gate(final, t["x"], t["gate_w"])
    if case["o_proj"]:
        final = final.reshape(B, N, Hq * Dv).matmul(t["o_w"].t())
    boundary = not (case["gate"] or case["o_proj"])
    return dict(final=final,
                out=out if boundary else None,
                lse=lse_plain if boundary else None,
                out_plain=out_plain, lse_plain=lse_plain,
                q4=q4, packed=packed, states=states)


def run_mini(t, case, g, device):
    """minimal_reference.py block. Boundary rows go through
    minimal_multilevel_attention directly (its out is post-sink via the
    literal sink logit; its lse is pre-sink by contract)."""
    B, N, Hq, Dv = g["B"], g["N"], g["Hq"], g["Dv"]
    L = len(g["fmap"])
    q4, k4, v4 = _projections(t, g)
    cos = sin = None
    if case["position"] == "rope":
        cos, sin = minimal_rope_tables(N, g["Dk"], device=device)
    boundary = not (case["gate"] or case["o_proj"])
    if boundary:
        k_c, v_c = k4, v4
        if case["conv"]:
            Hkv, Dk = g["Hkv"], g["Dk"]
            k_c = _short_conv(k4.reshape(B, N, Hkv * Dk),
                              t["kcw"]).reshape(B, N, Hkv, Dk)
            v_c = _short_conv(v4.reshape(B, N, Hkv * Dv),
                              t["vcw"]).reshape(B, N, Hkv, Dv)
        if case["weights"] == "linear":
            w = t["x"].matmul(t["w_proj"].t()).unsqueeze(-1)
        else:
            w = _qk_summary_weights(q4, k_c)
        kl, vl, _ = _summary_tree(k_c, v_c, w, L)
        states = None
        if case["position"] == "relative":
            states = t["x"].matmul(t["rel_w"].t()).reshape(B, N, Hq, _DREL)
        out, lse = minimal_multilevel_attention(
            q4, kl, vl, g["fmap"], g["cache"], softcap=case["softcap"],
            position_mode=case["position"], rope_cos=cos, rope_sin=sin,
            relative_states=states, relative_proj=t.get("rel_proj"),
            sinks=t.get("sinks"))
        check_lse_contract("mini", lse, B, N, Hq)
        return dict(final=out, out=out, lse=lse)
    final = minimal_attention_block(
        t["x"], q4, k4, v4, g["fmap"], g["cache"], num_summary_levels=L,
        k_conv_weight=t.get("kcw"), v_conv_weight=t.get("vcw"),
        weight_mode=case["weights"], w_proj=t.get("w_proj"),
        position_mode=case["position"], rope_cos=cos, rope_sin=sin,
        relative_weight=t.get("rel_w"), relative_proj=t.get("rel_proj"),
        sinks=t.get("sinks"), gate_weight=t.get("gate_w"),
        output_weight=t.get("o_w"), softcap=case["softcap"])
    return dict(final=final, out=None, lse=None)


# --- flex backend ----------------------------------------------------------
# Availability boundary is STRICT: the broad exception-catch wraps ONLY the
# lazy import (plus the CUDA check). Once the import succeeds, any later
# exception in the adapter is a real failure and fails the suite.

_FLEX = {"probed": False, "mod": None, "err": None}


def flex_module():
    if not _FLEX["probed"]:
        _FLEX["probed"] = True
        if not torch.cuda.is_available():
            _FLEX["err"] = RuntimeError(
                "flex backend requires a CUDA device")
        else:
            try:
                import telescope_cache.fms.train as m
                _FLEX["mod"] = m
            except Exception as exc:  # torch < 2.5 lacks flex_attention
                _FLEX["err"] = exc
    return _FLEX["mod"], _FLEX["err"]


def _flex_conv(x, weight):
    """Inline ShortConv1d forward (FP32 conv + FP32 residual, cast once)."""
    N = x.shape[1]
    K = weight.shape[1]
    xf = x.float()
    y = torch.nn.functional.conv1d(
        xf.transpose(1, 2), weight.float().unsqueeze(1),
        padding=K - 1, groups=weight.shape[0],
    )[:, :, :N].transpose(1, 2)
    return (xf + y).to(x.dtype)


def _mask_index(mask, b, h, q_i, k_i):
    return mask[q_i.clamp(min=0, max=mask.size(0) - 1),
                k_i.clamp(min=0, max=mask.size(1) - 1)]


def run_flex(t, case, g, device):
    """fms/train.py's flex path, replicated verbatim from
    MultiHeadAttention.forward (train.py:323-433) with fmap/cache_size as
    parameters instead of the module's hardcoded config."""
    m, err = flex_module()
    assert m is not None, err  # caller checks availability first
    B, N, Hq, Hkv, Dk, Dv = (g[s] for s in ("B", "N", "Hq", "Hkv", "Dk", "Dv"))
    E = Hq // Hkv
    x = t["x"]
    q_out = x.matmul(t["Wq"].t())      # [B, N, Hq*Dk] flat, as in_proj
    k_out = x.matmul(t["Wk"].t())
    v_out = x.matmul(t["Wv"].t())
    if case["conv"]:
        k_out = _flex_conv(k_out, t["kcw"])
        v_out = _flex_conv(v_out, t["vcw"])
    queries = q_out.view(B, N, Hq, Dk)
    keys = k_out.view(B, N, Hkv, Dk)
    values = v_out.view(B, N, Hkv, Dv)

    with torch.no_grad():
        plan = m.get_scan_plan(keys, g["fmap"], g["cache"])

    q_g = queries.unflatten(2, (Hkv, E))  # b l h e d
    if case["weights"] == "linear":
        w = x.matmul(t["w_proj"].t()).unsqueeze(-1)
    else:
        w = q_g.div(Dk ** 0.5).matmul(
            keys.unsqueeze(-1)).squeeze(-1).logsumexp(-1, True)

    scan = m.MultiHeadAttention.scan  # pure function; touches no self
    keys_c = scan(None, keys, plan, w)      # [B, cache_len, Hkv, Dk]
    values_c = scan(None, values, plan, w)
    cache_len = keys_c.size(1)

    mask = torch.zeros(N, cache_len, device=device, dtype=torch.bool)
    with torch.no_grad():
        mask.scatter_(1, plan[-1], True)
        flags = torch.ones(1, N, device=device)
        flags = scan(None, flags, plan, flags)
        flags = flags[0].bool().logical_not()
        mask[:, flags] = False  # kill warm-up dummies

    if E != 1:
        keys_e = keys_c.transpose(1, 2).unsqueeze(2).expand(
            -1, -1, E, -1, -1).flatten(1, 2)
        values_e = values_c.transpose(1, 2).unsqueeze(2).expand(
            -1, -1, E, -1, -1).flatten(1, 2)
    else:
        keys_e = keys_c.transpose(1, 2)
        values_e = values_c.transpose(1, 2)
    q_t = queries.transpose(1, 2)  # b h n d

    block_mask = m.create_block_mask(
        functools.partial(_mask_index, mask), 1, 1, N, cache_len)

    def soft_cap(score, b, h, q_i, kv_i):
        return 20 * score.div(20).tanh()

    attention = functools.partial(
        m.flex_attention, block_mask=block_mask, score_mod=soft_cap)
    if case["sink"]:
        attn, lse = attention(q_t, keys_e, values_e, return_lse=True)
        sink_scale = torch.sigmoid(
            lse.float() - t["sinks"].float()[None, :, None])
        attn = attn * sink_scale.unsqueeze(-1).to(attn.dtype)
    else:
        attn = attention(q_t, keys_e, values_e)
    attn = attn.transpose(1, 2).reshape(B, N, Hq * Dv)
    if case["gate"]:
        attn = torch.nn.functional.silu(x.matmul(t["gate_w"].t())) * attn
    if case["o_proj"]:
        final = attn.matmul(t["o_w"].t())
    else:
        final = attn.view(B, N, Hq, Dv)
    return dict(final=final, out=None, lse=None)


BACKEND_RUN = {"mini": run_mini, "ref": run_ref, "flex": run_flex}


def pair_tols(pair):
    if "flex" in pair:
        return FLEX_RTOL, FLEX_ATOL
    return RTOL, ATOL


def pair_grad_tols(pair):
    if "flex" in pair:
        return FLEX_RTOL, FLEX_ATOL
    return GRAD_RTOL, GRAD_ATOL


def flex_gate(ctx):
    """If the pair needs flex and it is unavailable, print one clean SKIP
    block (with the captured import error) and return False."""
    if "flex" not in ctx["pair"]:
        return True
    m, err = flex_module()
    if m is None:
        print(f"    SKIP: flex backend unavailable "
              f"({type(err).__name__}: {err})")
        return False
    return True


# ===========================================================================
# Section 1 -- schedule: range_spec vs the analytic oracles. Always runs.
# ===========================================================================

FWD_WIDTHS = (1, 3, 16, 24)
BWD_WIDTHS = (1, 2, 32, 40)

SCHEDULE_CONFIGS = [
    # name, N, cache_size, fmap, exhaustive elem/inverse pairs?
    ("readme_tiny", 8, 6, {1: 2, 2: 3}, True),
    ("tiny_evict", 64, 12, {1: 2, 2: 3}, True),
    ("mid", 400, 40, {1: 8, 2: 12, 3: 16}, True),
    ("four_small", 300, 40, {1: 4, 2: 6, 3: 10, 4: 11}, True),
    ("baseline", 128, 512, {1: 64, 2: 72, 3: 80}, True),
    ("eviction", 1024, 160, {1: 64, 2: 72, 3: 80}, False),
]


def _build_tables(N, fmap, cache_size, level, activation_times):
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


def _non_decreasing(xs):
    return all(x <= y for x, y in zip(xs, xs[1:]))


def _check_schedule_level(name, spec, N, fmap, cache_size, level,
                          exhaustive, block):
    a = spec.activation_times
    R, Q = _build_tables(N, fmap, cache_size, level, list(a))
    n_nodes = spec.level_len(level)
    assert n_nodes == len(Q), (name, level, n_nodes, len(Q))

    # range: spec == oracle (scalar and block); packed translation.
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

    # elem: elem_mask <=> k in R_l(q).
    if exhaustive:
        for q in range(N):
            lo, hi = R[q]
            for k in range(n_nodes):
                assert elem_mask(spec, q, k, level) == (lo <= k < hi), (
                    f"[{name}] L{level} elem_mask({q},{k})"
                )
    else:
        for q in range(N):
            lo, hi = R[q]
            ks = {lo - 1, lo, hi - 1, hi, 0, n_nodes - 1}
            ks.update(range(0, n_nodes, 37))
            for k in ks:
                if 0 <= k < n_nodes:
                    assert elem_mask(spec, q, k, level) == (lo <= k < hi)

    # mono: row bounds over nonempty rows; node hulls over visible nodes.
    rows = [(lo, hi) for lo, hi in R if lo < hi]
    assert _non_decreasing([lo for lo, _ in rows]), f"[{name}] L{level} k_lo"
    assert _non_decreasing([hi for _, hi in rows]), f"[{name}] L{level} k_hi"
    vis_nodes = [k for k in range(n_nodes) if Q[k]]
    assert _non_decreasing([Q[k][0] for k in vis_nodes]), (
        f"[{name}] L{level} q_lo")
    assert _non_decreasing([Q[k][-1] for k in vis_nodes]), (
        f"[{name}] L{level} q_hi")

    # fwd: hull over rows; O(1) form; containment.
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
        assert got == expect, (
            f"[{name}] L{level} fwd[{q_lo},{q_hi}) {got}!={expect}")
        if nonempty:
            assert got == (nonempty[0][0], nonempty[-1][1])  # O(1) form
            for lo, hi in nonempty:
                assert got[0] <= lo and hi <= got[1]

    # inverse: contiguity, node_query_bounds, second iff.
    for k in range(n_nodes):
        qs = Q[k]
        got = node_query_bounds(spec, k, level)
        if not qs:
            assert got == EMPTY, (
                f"[{name}] L{level} node {k} never visible: {got}")
            continue
        assert qs[-1] - qs[0] + 1 == len(qs), (
            f"[{name}] L{level} node {k}: inverse set not contiguous")
        assert got == (qs[0], qs[-1] + 1), (
            f"[{name}] L{level} node {k}: {got} != {(qs[0], qs[-1] + 1)}")
    if exhaustive:
        for k in range(n_nodes):
            qset = set(Q[k])
            for q in range(N):
                assert elem_mask(spec, q, k, level) == (q in qset)

    # bwd: hull over nonempty Q; containment (required; exactness is not).
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
        assert got == expect, (
            f"[{name}] L{level} bwd[{k_lo},{k_hi}) {got}!={expect}")
        for qs in vis:
            assert got[0] <= qs[0] and qs[-1] < got[1]


def _check_schedule_domain(spec):
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


def test_schedule_properties(ctx):
    for name, N, cache_size, fmap, exhaustive in SCHEDULE_CONFIGS:
        spec = RangeSpec.from_fmap(fmap, cache_size, N)

        # act: pure-int duplicate of the oracle's activation formula.
        oracle_a = validate_dyadic_fmap(fmap)
        assert spec.activation_times == tuple(oracle_a), (
            name, spec.activation_times, oracle_a)

        # offsets: prefix sums of floor-halved lengths.
        lens = [N >> lv for lv in range(spec.num_levels)]
        expect_off = [0]
        for n in lens:
            expect_off.append(expect_off[-1] + n)
        assert spec.level_offsets() == tuple(expect_off), (
            name, spec.level_offsets())
        assert spec.total_len == sum(lens)

        starts, ends = dyadic_ranges_for_query_block(
            torch.arange(N), fmap, cache_size, oracle_a)
        block = (
            [[int(x) for x in row] for row in starts.tolist()],
            [[int(x) for x in row] for row in ends.tolist()],
        )
        for level in range(spec.num_levels):
            _check_schedule_level(
                name, spec, N, fmap, cache_size, level, exhaustive, block)
        _check_schedule_domain(spec)
    print(f"  schedule-properties     range/elem/mono/fwd/inverse/bwd/"
          f"domain vs analytic oracles ({len(SCHEDULE_CONFIGS)} configs) "
          f"PASS")


def test_visible_entries(ctx):
    for name in ("tiny", "evict", "odd_n", "four_levels"):
        g = GEOS[name]
        spec = _spec(g)
        for qi in range(g["N"]):
            entries = minimal_visible_entries(spec, qi)
            ranges = dyadic_ranges_for_query(qi, g["fmap"], g["cache"])
            oracle = sorted(
                [(lv, j) for lv, (lo, hi) in enumerate(ranges)
                 for j in range(lo, hi)],
                key=lambda e: e[1] << e[0])
            if entries != oracle:
                raise AssertionError(
                    f"{name} q={qi}: entries {entries} != oracle {oracle}")
            if entries[-1] != (0, qi):
                raise AssertionError(
                    f"{name} q={qi}: newest entry {entries[-1]} is not the "
                    f"query's own L0 token")
            starts = [j << lv for lv, j in entries]
            if any(a >= b for a, b in zip(starts, starts[1:])):
                raise AssertionError(
                    f"{name} q={qi}: interval starts not strictly "
                    f"increasing: {starts}")
    # Hand-derived README golden: an oracle that is not "another
    # implementation of the same formula".
    spec = RangeSpec.from_fmap({1: 2, 2: 3}, 6, 8)
    for qi, golden in README_TINY_GOLDEN.items():
        got = [range_bounds(spec, qi, lv) for lv in range(spec.num_levels)]
        if got != golden:
            raise AssertionError(
                f"readme_tiny golden q={qi}: {got} != {golden}")
    print("  visible-entries         chronological list == range oracle + "
          "README hand golden PASS")


def test_positional_schedule(ctx):
    # Right-endpoint summary position units.
    for (level, j), expected in [
        ((0, 0), 0), ((0, 5), 5), ((1, 0), 1), ((2, 1), 7),
        ((3, 5), 47), ((10, 3), 4095),
    ]:
        got = summary_token_position(level, j)
        if got != expected:
            raise AssertionError(
                f"position({level},{j}) = {got} != {expected}")
    for bad in [(-1, 0), (0, -1)]:
        expect_error(ValueError, f"position{bad}",
                     lambda bad=bad: summary_token_position(*bad))

    # Chronological-bin distances: partition property, causality (V1),
    # analytic rank == slow-oracle rank, cross-level adjacency delta = 1.
    bin_configs = [
        (8, {1: 2, 2: 3}, 6),
        (128, {1: 64, 2: 72, 3: 80}, 512),
        (200, {1: 32, 2: 40, 3: 48, 4: 56}, 96),
    ]
    cross_level = 0
    for N, fmap, cache in bin_configs:
        spec = RangeSpec.from_fmap(fmap, cache, N)
        for qi in range(N):
            ranges = dyadic_ranges_for_query(qi, fmap, cache)
            entries = sorted(
                [(lv, j, j << lv, (j + 1) << lv)
                 for lv, (lo, hi) in enumerate(ranges)
                 for j in range(lo, hi)],
                key=lambda e: e[2])
            M = len(entries)
            for a, b in zip(entries, entries[1:]):
                if a[3] != b[2]:
                    raise AssertionError(
                        f"N={N} q={qi}: intervals not contiguous/disjoint "
                        f"at {a} -> {b}")
                if a[0] != b[0]:
                    da = relative_bin_distance(spec, qi, a[0], a[1])
                    db = relative_bin_distance(spec, qi, b[0], b[1])
                    if da - db != 1:
                        raise AssertionError(
                            f"N={N} q={qi}: cross-level step {a}->{b} has "
                            f"distance delta {da - db} != 1")
                    cross_level += 1
            if entries[-1][3] != qi + 1:
                raise AssertionError(
                    f"N={N} q={qi}: coverage does not end at q")
            for rank, (level, j, _, _) in enumerate(entries):
                expected = (M - 1) - rank
                got = relative_bin_distance(spec, qi, level, j)
                if got != expected:
                    raise AssertionError(
                        f"N={N} q={qi} entry (L{level},{j}): distance "
                        f"{got} != {expected}")
                if summary_token_position(level, j) > qi:
                    raise AssertionError(
                        f"N={N} q={qi}: (L{level},{j}) reaches the future")
    assert cross_level > 0, "no cross-level adjacencies exercised"
    # Golden spot-check (readme_tiny, q=7): own token = 0.
    spec = RangeSpec.from_fmap({1: 2, 2: 3}, 6, 8)
    for (level, j), expected in [((2, 0), 3), ((1, 2), 2),
                                 ((0, 6), 1), ((0, 7), 0)]:
        assert relative_bin_distance(spec, 7, level, j) == expected, (level, j)
    print(f"  positional-schedule     position units + bin partition/"
          f"causality ({cross_level} cross-level steps) PASS")


# Regime pins: a geometry edit must not silently stop exercising a regime.
COVERAGE_PINS = {
    # first_evict, odd_levels, partial_m, partial_n, dv_ne_dk, expansion
    "tiny": dict(first_evict=None, odd_levels=[], partial_m=True,
                 partial_n=False, dv_ne_dk=True, expansion=2),
    "evict": dict(first_evict=16, odd_levels=[], partial_m=False,
                  partial_n=True, dv_ne_dk=True, expansion=2),
    "evict_gqa3": dict(first_evict=16, odd_levels=[], partial_m=False,
                       partial_n=True, dv_ne_dk=False, expansion=3),
    "odd_n": dict(first_evict=16, odd_levels=[0, 1, 2], partial_m=True,
                  partial_n=True, dv_ne_dk=True, expansion=2),
    "four_levels": dict(first_evict=48, odd_levels=[], partial_m=False,
                        partial_n=True, dv_ne_dk=False, expansion=2),
}


def test_coverage_pins(ctx):
    for name, expect in COVERAGE_PINS.items():
        g = GEOS[name]
        spec = _spec(g)
        starts, ends = dyadic_ranges_for_query_block(
            torch.arange(g["N"]), g["fmap"], g["cache"])
        evict_q = torch.nonzero(starts[:, -1] > 0).flatten()
        totals = (ends - starts).sum(dim=1)
        full_q = torch.nonzero(totals == g["cache"]).flatten()
        lens = [spec.level_len(lv) for lv in range(spec.num_levels)]
        actual = dict(
            first_evict=(int(evict_q[0]) if evict_q.numel() else None),
            odd_levels=[lv for lv, n in enumerate(lens) if n % 2 == 1],
            partial_m=g["N"] % g["bm"] != 0,
            partial_n=any(n % g["bn"] != 0 for n in lens),
            dv_ne_dk=g["Dv"] != g["Dk"],
            expansion=g["Hq"] // g["Hkv"],
        )
        if actual["first_evict"] is not None and not full_q.numel():
            raise AssertionError(
                f"{name}: eviction without a full cache is impossible")
        mismatch = {k: (expect[k], actual[k]) for k in expect
                    if expect[k] != actual[k]}
        if mismatch:
            raise AssertionError(
                f"{name}: coverage pins mismatch (expected, actual): "
                f"{mismatch}")
    print("  coverage-pins           eviction/odd/partial-tile/GQA regimes "
          "pinned per geometry PASS")


# ===========================================================================
# Section 2 -- forward semantics (pair-driven).
# ===========================================================================

def test_summary_tree(ctx):
    """Phase 1: mini's per-pair loop tree vs ref's strided builder,
    qk/linear/sconv on odd and deep geometries. mini<->ref only."""
    if set(ctx["pair"]) != {"mini", "ref"}:
        print("    skip summary-tree (mini<->ref only)")
        return
    device = ctx["device"]
    Kc, emb = 3, 16
    for i, name in enumerate(("tiny", "odd_n", "four_levels")):
        g = GEOS[name]
        L = len(g["fmap"])
        B, N, Hq, Hkv, Dk, Dv = (
            g[s] for s in ("B", "N", "Hq", "Hkv", "Dk", "Dv"))
        torch.manual_seed(10 + i)
        q = torch.randn(B, N, Hq, Dk, device=device)
        k = torch.randn(B, N, Hkv, Dk, device=device)
        v = torch.randn(B, N, Hkv, Dv, device=device)
        x = torch.randn(B, N, emb, device=device)
        w_proj = torch.randn(Hkv, emb, device=device) / emb ** 0.5
        kcw = torch.randn(Hkv * Dk, Kc, device=device) / Kc ** 0.5
        vcw = torch.randn(Hkv * Dv, Kc, device=device) / Kc ** 0.5

        cases = [
            ("qk", _summary_tree(k, v, _qk_summary_weights(q, k), L),
             build_dyadic_summaries(q, k, v, L)),
            ("lin", _summary_tree(k, v,
                                  x.matmul(w_proj.t()).unsqueeze(-1), L),
             build_dyadic_summaries(q, k, v, L, x=x, w_proj=w_proj)),
        ]
        kc = _short_conv(k.reshape(B, N, Hkv * Dk),
                         kcw).reshape(B, N, Hkv, Dk)
        vc = _short_conv(v.reshape(B, N, Hkv * Dv),
                         vcw).reshape(B, N, Hkv, Dv)
        report(f"{name} shortconv",
               _short_conv(k.reshape(B, N, Hkv * Dk), kcw),
               short_conv(k.reshape(B, N, Hkv * Dk), kcw))
        cases.append(
            ("sconv",
             _summary_tree(kc, vc, _qk_summary_weights(q, kc), L),
             build_dyadic_summaries(q, k, v, L, k_conv_weight=kcw,
                                    v_conv_weight=vcw)))
        for label, (kl_m, vl_m, wl_m), (kl_r, vl_r, wl_r) in cases:
            for lv in range(L + 1):
                report(f"{name} {label} k[{lv}]", kl_m[lv], kl_r[lv])
                report(f"{name} {label} v[{lv}]", vl_m[lv], vl_r[lv])
                report(f"{name} {label} w[{lv}]", wl_m[lv], wl_r[lv])
    print("  summary-tree            per-pair loop == strided builder "
          "(qk/linear/sconv, odd lengths) PASS")


def test_forward(ctx):
    """The forward feature matrix, pairwise. Boundary rows compare
    (out, lse); full-stack rows compare the final block output."""
    if not flex_gate(ctx):
        return
    device = ctx["device"]
    a_name, b_name = ctx["pair"]
    rtol, atol = pair_tols(ctx["pair"])
    ran = skipped = 0
    for ci, case in enumerate(FWD_CASES):
        for gi, geo_name in enumerate(case["geos"]):
            g = GEOS[geo_name]
            label = f"{case['name']}/{geo_name}"
            why = pair_skip_reason(ctx["pair"], case, g)
            if why:
                skipped += 1
                continue
            base = make_case_params(100 + 10 * ci + gi, case, g, device)
            with torch.no_grad():
                res_a = BACKEND_RUN[a_name](base, case, g, device)
                res_b = BACKEND_RUN[b_name](base, case, g, device)
            report(f"{label} final", res_a["final"], res_b["final"],
                   rtol=rtol, atol=atol)
            if res_a["lse"] is not None and res_b["lse"] is not None:
                report(f"{label} lse", res_a["lse"], res_b["lse"],
                       rtol=rtol, atol=atol)
            ran += 1

            # Feature-specific identity checks (mini<->ref rows only).
            if set(ctx["pair"]) != {"mini", "ref"}:
                continue
            if case["name"] == "rope" and geo_name == "tiny":
                cos_m, sin_m = minimal_rope_tables(
                    g["N"], g["Dk"], device=device)
                cos_r, sin_r = rope_tables(g["N"], g["Dk"], device=device)
                report("rope cos tables", cos_m, cos_r)
                report("rope sin tables", sin_m, sin_r)
                spec = _spec(g)
                if not any(
                    range_bounds(spec, g["N"] - 1, lv) != EMPTY
                    for lv in range(1, spec.num_levels)
                ):
                    raise AssertionError(
                        "rope geometry exposes no coarse summaries")
            if case["name"] == "sink":
                # Internal identity on mini's own tensors: literal sink
                # == out_plain * sigmoid(lse - s); lse is sink-invariant.
                plain = dict(case, sink=False)
                base_plain = {kk: t for kk, t in base.items()
                              if kk != "sinks"}
                with torch.no_grad():
                    mp = run_mini(base_plain, plain, g, device)
                    ms = run_mini(base, case, g, device)
                scale = torch.sigmoid(
                    mp["lse"] - base["sinks"][None, None, :])
                report(f"{label} sink identity", ms["out"],
                       mp["out"] * scale.unsqueeze(-1))
                if not torch.equal(ms["lse"], mp["lse"]):
                    raise AssertionError(
                        f"{label}: sinks changed the returned lse")
            if case["name"] == "relative":
                spec = _spec(g)
                for qi in (0, g["N"] // 2, g["N"] - 1):
                    entries = minimal_visible_entries(spec, qi)
                    M = len(entries)
                    for rank, (lv, j) in enumerate(entries):
                        want = M - 1 - rank
                        got = relative_bin_distance(spec, qi, lv, j)
                        if got != want:
                            raise AssertionError(
                                f"{label} q={qi} (L{lv},{j}): distance "
                                f"{got} != {want}")
    print(f"  forward                 {a_name} == {b_name} over the "
          f"feature matrix ({ran} runs, {skipped} skipped) PASS")


# ===========================================================================
# Section 3 -- backward semantics.
# ===========================================================================

def _phase2_case(device, mode, geo_name, softcap, seed, block_n=None,
                 with_sink=False):
    """
    One packed-boundary comparison: q and packed.k/v are direct leaves, so
    autograd yields dq/dk_packed/dv_packed with no summary-tree path, and
    the explicit multilevel_attention_backward must match them
    INDIVIDUALLY -- this is the CuTeDSL backward kernel's contract.
    Returns pieces for the structural checks.
    """
    g = GEOS[geo_name]
    B, N, Hq, Hkv, Dk, Dv = (g[s] for s in ("B", "N", "Hq", "Hkv", "Dk", "Dv"))
    bn = block_n if block_n is not None else g["bn"]
    spec = _spec(g)
    torch.manual_seed(seed)
    q = torch.randn(B, N, Hq, Dk, device=device)
    k = torch.randn(B, N, Hkv, Dk, device=device)
    v = torch.randn(B, N, Hkv, Dv, device=device)
    with torch.no_grad():
        kl, vl, _ = build_dyadic_summaries(q, k, v, spec.num_levels - 1)
        packed = pack_levels(kl, vl)

    q_leaf = q.detach().clone().requires_grad_(True)
    k_leaf = packed.k.detach().clone().requires_grad_(True)
    v_leaf = packed.v.detach().clone().requires_grad_(True)
    packed_leaf = PackedKV(k_leaf, v_leaf, packed.level_offsets)
    leaves = [q_leaf, k_leaf, v_leaf]

    fw_kwargs = {}
    states_leaf = proj_leaf = None
    if mode == "rope":
        cos, sin = rope_tables(N, Dk, device=device)
        fw_kwargs = dict(rope_cos=cos, rope_sin=sin)
    elif mode == "relative":
        states_leaf = torch.randn(
            B, N, Hq, _DREL, device=device).requires_grad_(True)
        proj_leaf = (torch.randn(_DREL, max(8, g["cache"]), device=device)
                     / _DREL ** 0.5).requires_grad_(True)
        fw_kwargs = dict(relative_states=states_leaf,
                         relative_proj=proj_leaf)
        leaves += [states_leaf, proj_leaf]

    out, lse = multilevel_attention_forward(
        q_leaf, packed_leaf, g["fmap"], g["cache"], block_m=g["bm"],
        block_n=bn, softcap=softcap, position_mode=mode, **fw_kwargs)

    sinks_leaf = None
    if with_sink:
        torch.manual_seed(seed + 3)
        with torch.no_grad():
            sinks_val = (lse.float().mean(dim=(0, 1))
                         + 0.1 * torch.randn(Hq, device=device))
        sinks_leaf = sinks_val.requires_grad_(True)
        final = apply_attention_sink(out, lse, sinks_leaf)
        leaves.append(sinks_leaf)
    else:
        final = out

    dout = seeded_randn(final.shape, seed + 1, device)
    auto = torch.autograd.grad(final, tuple(leaves), dout)

    # Explicit path on detached values.
    bw_kwargs = {}
    if mode == "rope":
        bw_kwargs = fw_kwargs
    elif mode == "relative":
        bw_kwargs = dict(relative_states=states_leaf.detach(),
                         relative_proj=proj_leaf.detach())
    packed_det = PackedKV(k_leaf.detach(), v_leaf.detach(),
                          packed.level_offsets)
    if with_sink:
        dout_pre, dlse, dsinks = attention_sink_backward(
            out.detach(), lse.detach(), sinks_leaf.detach(), dout)
    else:
        dout_pre, dlse, dsinks = dout, None, None
    dq, dk_p, dv_p, dpos, stats = multilevel_attention_backward(
        q_leaf.detach(), packed_det, out.detach(), lse.detach(), dout_pre,
        g["fmap"], g["cache"], block_m=g["bm"], block_n=bn,
        softcap=softcap, dlse=dlse, position_mode=mode, **bw_kwargs)

    tag = (f"{mode}/{geo_name}"
           + ("/nocap" if softcap is None else "")
           + (f"/bn{bn}" if block_n is not None else "")
           + ("/sink" if with_sink else ""))
    if mode == "relative":
        if dpos is None:
            raise AssertionError(f"{tag}: missing dposition")
        if (dpos.dtype != torch.float32
                or tuple(dpos.shape) != (B, N, Hq, proj_leaf.shape[1])):
            raise AssertionError(
                f"{tag}: dposition {tuple(dpos.shape)}/{dpos.dtype}")
    elif dpos is not None:
        raise AssertionError(
            f"{tag}: dposition must be None outside relative mode")

    report(f"dq/{tag}", dq, auto[0], rtol=GRAD_RTOL, atol=GRAD_ATOL)
    report(f"dk_packed/{tag}", dk_p, auto[1],
           rtol=GRAD_RTOL, atol=GRAD_ATOL)
    report(f"dv_packed/{tag}", dv_p, auto[2],
           rtol=GRAD_RTOL, atol=GRAD_ATOL)
    if mode == "relative":
        dstates = torch.einsum(
            "bnhr,dr->bnhd", dpos, proj_leaf.detach().float())
        dproj = torch.einsum(
            "bnhd,bnhr->dr", states_leaf.detach().float(), dpos)
        report(f"dstates/{tag}", dstates, auto[3],
               rtol=GRAD_RTOL, atol=GRAD_ATOL)
        report(f"dproj/{tag}", dproj, auto[4],
               rtol=GRAD_RTOL, atol=GRAD_ATOL)
    if with_sink:
        report(f"dsinks/{tag}", dsinks, auto[-1],
               rtol=GRAD_RTOL, atol=GRAD_ATOL)
        if auto[-1].norm().item() <= GRAD_MIN_NORM:
            raise AssertionError(f"{tag}: dsinks carries no gradient")
    return dict(spec=spec, dk_p=dk_p, dv_p=dv_p,
                dk_auto=auto[1], dv_auto=auto[2], stats=stats,
                out=out.detach(), lse=lse.detach(), dout=dout,
                q=q_leaf.detach(), packed=packed_det, g=g, bn=bn,
                dq=dq)


def test_phase2_backward(ctx):
    """3A: the packed-boundary Phase-2 contract (the future CuTeDSL
    backward interface). Runs whenever ref is in the pair; it is a
    ref-internal contract, not a pair comparison."""
    if "ref" not in ctx["pair"]:
        print("    skip phase2-backward (ref not in pair)")
        return
    device = ctx["device"]
    _phase2_case(device, "none", "tiny", SOFTCAP, 300)
    _phase2_case(device, "rope", "tiny", SOFTCAP, 304)
    _phase2_case(device, "relative", "tiny", SOFTCAP, 308)
    _phase2_case(device, "relative", "evict", SOFTCAP, 312)
    _phase2_case(device, "none", "tiny", None, 316)
    # BLOCK_N=3 makes the conservative L1 hull a real tile.
    _phase2_case(device, "none", "tiny", SOFTCAP, 320, block_n=3)
    # The sink-dlse seam: attention_sink_backward -> dout_pre + dlse ->
    # multilevel_attention_backward(dlse=...), at the packed boundary.
    _phase2_case(device, "rope", "tiny", SOFTCAP, 324, with_sink=True)
    _phase2_case(device, "relative", "tiny", SOFTCAP, 328, with_sink=True)
    print("  phase2-backward         explicit == autograd at the packed "
          "boundary (dq/dk/dv/dposition + sink-dlse seam) PASS")


def test_block_gradients(ctx):
    """3B: full-block gradients. mini<->ref pits mini's pure autograd
    against ref's explicit composition (Phase-2 backward + sink dlse +
    Phase-1 tree VJP + downstream gate/o_proj VJPs); other pairs compare
    pure autograd on the shared leaves."""
    if not flex_gate(ctx):
        return
    device = ctx["device"]
    pair = ctx["pair"]
    explicit = set(pair) == {"mini", "ref"}
    ran = skipped = 0
    for mode, weights, geo_name, seed in GRAD_CASES:
        case = _case("grad", [geo_name], weights=weights, conv=True,
                     position=mode, sink=True, gate=True, o_proj=True)
        g = GEOS[geo_name]
        name = f"{mode}/{weights}/{geo_name}"
        if pair_skip_reason(pair, case, g):
            skipped += 1
            continue
        base = make_case_params(seed, case, g, device)
        keys = list(base.keys())
        B, N, Hq, Dv = g["B"], g["N"], g["Hq"], g["Dv"]

        if not explicit:
            # Generic autograd-vs-autograd on the shared leaves.
            grads = {}
            for bname in pair:
                lv = clone_leaves(base)
                res = BACKEND_RUN[bname](lv, case, g, device)
                dfinal = seeded_randn(res["final"].shape, seed + 2, device)
                grads[bname] = dict(zip(keys, torch.autograd.grad(
                    res["final"], tuple(lv[kk] for kk in keys), dfinal)))
            rtol, atol = pair_grad_tols(pair)
            for kname in keys:
                a_g, b_g = grads[pair[0]][kname], grads[pair[1]][kname]
                if b_g.norm().item() <= GRAD_MIN_NORM:
                    raise AssertionError(f"d{kname}/{name}: dead gradient")
                report(f"d{kname}/{name}", a_g, b_g, rtol=rtol, atol=atol)
            ran += 1
            continue

        # --- mini autograd oracle -------------------------------------
        a = clone_leaves(base)
        out_min = run_mini(a, case, g, device)["final"]
        dfinal = seeded_randn(out_min.shape, seed + 2, device)
        oracle = dict(zip(keys, torch.autograd.grad(
            out_min, tuple(a[kk] for kk in keys), dfinal)))

        # --- ref explicit composition ---------------------------------
        e = clone_leaves(base)
        ce = run_ref(e, case, g, device, attn_no_grad=True)
        out, lse = ce["out_plain"], ce["lse_plain"]

        out_sink_leaf = apply_attention_sink(
            out, lse, e["sinks"].detach()).detach().requires_grad_(True)
        gated = apply_output_gate(out_sink_leaf, e["x"], e["gate_w"])
        final_e = gated.reshape(B, N, Hq * Dv).matmul(e["o_w"].t())
        report(f"{name} fwd match", out_min.detach(), final_e.detach())
        dout_sink, dx_gate, dgate_w, do_w = torch.autograd.grad(
            final_e, (out_sink_leaf, e["x"], e["gate_w"], e["o_w"]), dfinal)

        dout_pre, dlse, dsinks = attention_sink_backward(
            out, lse, e["sinks"].detach(), dout_sink)
        bw_kwargs = {}
        if mode == "rope":
            cos, sin = rope_tables(N, g["Dk"], device=device)
            bw_kwargs = dict(rope_cos=cos, rope_sin=sin)
        elif mode == "relative":
            bw_kwargs = dict(relative_states=ce["states"].detach(),
                             relative_proj=e["rel_proj"].detach())
        dq_attn, dk_p, dv_p, dpos, _ = multilevel_attention_backward(
            ce["q4"], ce["packed"], out, lse, dout_pre, g["fmap"],
            g["cache"], block_m=g["bm"], block_n=g["bn"], softcap=SOFTCAP,
            dlse=dlse, position_mode=mode, **bw_kwargs)

        up_outputs = [ce["packed"].k, ce["packed"].v, ce["q4"]]
        up_seeds = [dk_p, dv_p, dq_attn]
        up_keys = ["x", "Wq", "Wk", "Wv", "kcw", "vcw"]
        if weights == "linear":
            up_keys.append("w_proj")
        if mode == "relative":
            dstates = torch.einsum(
                "bnhr,dr->bnhd", dpos, e["rel_proj"].detach().float())
            up_outputs.append(ce["states"])
            up_seeds.append(dstates)
            up_keys.append("rel_w")
        up = dict(zip(up_keys, torch.autograd.grad(
            tuple(up_outputs), tuple(e[kk] for kk in up_keys),
            tuple(up_seeds), retain_graph=(mode == "rope"))))

        got = dict(up)
        got["x"] = up["x"] + dx_gate      # multi-path sum: upstream + gate
        got["gate_w"] = dgate_w
        got["o_w"] = do_w
        got["sinks"] = dsinks
        if mode == "relative":
            got["rel_proj"] = torch.einsum(
                "bnhd,bnhr->dr", ce["states"].detach().float(), dpos)

        for kname in keys:
            want, gg = oracle[kname], got[kname]
            if want.norm().item() <= GRAD_MIN_NORM:
                raise AssertionError(
                    f"d{kname}/{name}: minimal-oracle gradient is dead "
                    f"({want.norm().item():.3e})")
            report(f"d{kname}/{name}", gg, want,
                   rtol=GRAD_RTOL, atol=GRAD_ATOL)
        ran += 1

        # Negative controls (rope run): dropping exactly one composition
        # term must visibly fail -- these prove the test has power.
        if mode == "rope":
            err_gate = (up["x"] - oracle["x"]).abs().max().item()
            if err_gate <= GRAD_ATOL:
                raise AssertionError(
                    f"{name} negative control: dropping the gate path "
                    f"still matched dx (max err {err_gate:.3e})")
            dq_b, dk_b, dv_b, _, _ = multilevel_attention_backward(
                ce["q4"], ce["packed"], out, lse, dout_pre, g["fmap"],
                g["cache"], block_m=g["bm"], block_n=g["bn"],
                softcap=SOFTCAP, position_mode=mode, **bw_kwargs)
            dx_up_bad, = torch.autograd.grad(
                (ce["packed"].k, ce["packed"].v, ce["q4"]), (e["x"],),
                (dk_b, dv_b, dq_b))
            err_dlse = (dx_up_bad + dx_gate
                        - oracle["x"]).abs().max().item()
            if err_dlse <= GRAD_ATOL:
                raise AssertionError(
                    f"{name} negative control: dropping dlse still "
                    f"matched dx (max err {err_dlse:.3e})")
    kind = "explicit composition" if explicit else "pure autograd"
    print(f"  block-gradients         {pair[0]} vs {pair[1]} via {kind} "
          f"({ran} cases, {skipped} skipped) PASS")


def _never_visible_mask(spec, device):
    nv = torch.zeros(spec.total_len, dtype=torch.bool, device=device)
    off = spec.level_offsets()
    for lv in range(spec.num_levels):
        for kk in range(spec.level_len(lv)):
            if node_query_bounds(spec, kk, lv) == EMPTY:
                nv[off[lv] + kk] = True
    return nv


def test_grad_structure(ctx):
    """3C: ref-only structural gradient invariants."""
    if "ref" not in ctx["pair"]:
        print("    skip grad-structure (ref not in pair)")
        return
    device = ctx["device"]

    # Never-visible packed nodes get exactly zero gradient, in both the
    # explicit and the autograd path.
    nv_total = 0
    for geo_name, seed in (("evict", 340), ("four_levels", 344)):
        r = _phase2_case(device, "none", geo_name, SOFTCAP, seed)
        nv = _never_visible_mask(r["spec"], device)
        nv_total += int(nv.sum())
        if nv.any():
            for nm, t in [("dk_p", r["dk_p"]), ("dv_p", r["dv_p"]),
                          ("dk_auto", r["dk_auto"]),
                          ("dv_auto", r["dv_auto"])]:
                if not bool((t[:, nv] == 0).all().item()):
                    raise AssertionError(
                        f"{geo_name}/{nm}: nonzero gradient on "
                        f"never-visible nodes")

    # detach_weights is strictly gradient-only: forward bitwise unchanged;
    # dv exactly invariant (w never depends on v); in qk mode the w path
    # carries gradient into dq/dk, in linear mode dq/dk are invariant.
    g = GEOS["tiny"]
    B, N, Hq, Hkv, Dk, Dv = (g[s] for s in ("B", "N", "Hq", "Hkv", "Dk", "Dv"))
    L = len(g["fmap"])
    torch.manual_seed(350)
    q0 = torch.randn(B, N, Hq, Dk, device=device)
    k0 = torch.randn(B, N, Hkv, Dk, device=device)
    v0 = torch.randn(B, N, Hkv, Dv, device=device)
    x0 = torch.randn(B, N, _EMB, device=device)
    w_proj0 = torch.randn(Hkv, _EMB, device=device) / _EMB ** 0.5

    def run(detach, linear):
        q, k, v = (t.detach().clone().requires_grad_(True)
                   for t in (q0, k0, v0))
        kw = dict(x=x0, w_proj=w_proj0) if linear else {}
        kl, vl, _ = build_dyadic_summaries(
            q, k, v, L, detach_weights=detach, **kw)
        packed = pack_levels(kl, vl)
        out, _ = multilevel_attention_forward(
            q, packed, g["fmap"], g["cache"],
            block_m=g["bm"], block_n=g["bn"], softcap=SOFTCAP)
        dout = seeded_randn(out.shape, 351, device)
        return out, torch.autograd.grad(out, (q, k, v), dout)

    for linear in (False, True):
        out, (dq, dk, dv) = run(False, linear)
        out_d, (dq_d, dk_d, dv_d) = run(True, linear)
        label = "linear" if linear else "qk"
        report(f"detach fwd bitwise/{label}", out_d, out, rtol=0, atol=0)
        report(f"detach dv invariant/{label}", dv_d, dv, rtol=0, atol=1e-7)
        if linear:
            report("detach dq invariant/linear", dq_d, dq,
                   rtol=0, atol=1e-7)
            report("detach dk invariant/linear", dk_d, dk,
                   rtol=0, atol=1e-7)
        else:
            for nm, w_g in (("dq_w", dq - dq_d), ("dk_w", dk - dk_d)):
                if w_g.norm().item() <= GRAD_MIN_NORM:
                    raise AssertionError(
                        f"{nm}: the q/k -> w path carries no gradient")
    print(f"  grad-structure          never-visible zero-grad "
          f"({nv_total} nodes) + detach_weights invariants PASS")


# ===========================================================================
# Section 4 -- contracts (always runs; not pair comparisons).
# minimal_reference is an FP32-only oracle, so the mixed-precision
# contracts here can only be pinned directly, never as a comparison.
# ===========================================================================

def _conv_hand_oracle(x, w):
    """Independent double-loop short-conv contract, same FP32 ordering."""
    B, N, C = x.shape
    K = w.shape[1]
    xf, wf = x.float(), w.float()
    y = torch.zeros_like(xf)
    for t in range(N):
        for j in range(K):
            s = t - K + 1 + j
            if s >= 0:
                y[:, t] += wf[:, j] * xf[:, s]
    return (xf + y).to(x.dtype)


def _module_with_weight(w):
    m = ShortConv1d(w.shape[0], w.shape[1])
    with torch.no_grad():
        m.weight.copy_(w)
    return m


def test_mixed_precision(ctx):
    device = ctx["device"]

    # fp32-residual-then-cast ordering (bf16, bitwise, non-vacuous).
    torch.manual_seed(400)
    B, N, C, K = 4, 64, 32, 4
    x = (torch.randn(B, N, C, device=device) * 100.0).to(torch.bfloat16)
    w = torch.randn(C, K, device=device) / K ** 0.5
    xf = x.float()
    y = torch.nn.functional.conv1d(
        xf.transpose(1, 2), w.float().unsqueeze(1), padding=K - 1, groups=C
    )[:, :, :N].transpose(1, 2)
    fp32_order = (xf + y).to(torch.bfloat16)
    cast_order = x + y.to(torch.bfloat16)
    if torch.equal(fp32_order, cast_order):
        raise AssertionError("orderings agree everywhere; inputs vacuous")
    for name, out in [
        ("short_conv", short_conv(x, w)),
        ("ShortConv1d", _module_with_weight(w).to(device)(x)),
        ("_short_conv(mini)", _short_conv(x, w)),
    ]:
        if not torch.equal(out, fp32_order):
            raise AssertionError(f"{name} does not add the residual in FP32")
        if torch.equal(out, cast_order):
            raise AssertionError(f"{name} matches cast-then-add ordering")

    # Conv equivalence vs the hand oracle, fp32 + bf16, dtype preserved.
    for dtype in (torch.float32, torch.bfloat16):
        for N2, K2, C2 in ((5, 4, 3), (8, 2, 8)):
            torch.manual_seed(410 + N2 + K2)
            x2 = torch.randn(2, N2, C2, device=device).to(dtype)
            w2 = torch.randn(C2, K2, device=device) / K2 ** 0.5
            ref = _conv_hand_oracle(x2, w2)
            for name, out in [
                ("short_conv", short_conv(x2, w2)),
                ("ShortConv1d", _module_with_weight(w2).to(device)(x2)),
            ]:
                assert out.dtype == dtype, (name, out.dtype)
                torch.testing.assert_close(out, ref)

    # Sink: LSE-sigmoid rescale == literal augmented softmax, fp32 + bf16,
    # zero + random sinks (a dense unit oracle; the telescope-level check
    # lives in Section 2).
    torch.manual_seed(420)
    for dtype in (torch.float32, torch.bfloat16):
        for sinks in (torch.zeros(3, device=device),
                      torch.randn(3, device=device)):
            z = torch.randn(2, 3, 5, 7, device=device)
            v = torch.randn(2, 3, 7, 6, device=device).to(dtype)
            lse = torch.logsumexp(z.float(), dim=-1)
            out = torch.softmax(z.float(), dim=-1).matmul(
                v.float()).to(v.dtype)
            got = apply_attention_sink(
                out.permute(0, 2, 1, 3), lse.permute(0, 2, 1), sinks
            ).permute(0, 2, 1, 3)
            sink_col = sinks.float()[None, :, None, None].expand(2, 3, 5, 1)
            z_aug = torch.cat([z.float(), sink_col], dim=-1)
            v_aug = torch.cat(
                [v.float(), torch.zeros(2, 3, 1, 6, device=device)], dim=-2)
            want = torch.softmax(z_aug, dim=-1).matmul(v_aug).to(v.dtype)
            torch.testing.assert_close(got, want)

    # Gate: exact formula, fp32 + bf16, bitwise.
    torch.manual_seed(430)
    for dtype in (torch.float32, torch.bfloat16):
        out = torch.randn(2, 6, 4, 8, device=device).to(dtype)
        xg = torch.randn(2, 6, 24, device=device).to(dtype)
        wg = (torch.randn(32, 24, device=device) / 24 ** 0.5).to(dtype)
        want = torch.nn.functional.silu(
            xg.matmul(wg.t())).reshape(2, 6, 4, 8) * out
        report(f"gate bitwise/{dtype}", apply_output_gate(out, xg, wg),
               want, rtol=0, atol=0)
    print("  mixed-precision         fp32-ordering + bf16 conv/sink/gate "
          "contracts PASS")


def test_shortconv_module(ctx):
    device = ctx["device"]
    torch.manual_seed(440)
    B, N = 2, 11
    Hkv, Dk, Dv, K = 2, 8, 16, 4
    # Bitwise module == functional at train.py channel counts.
    for C in (Hkv * Dk, Hkv * Dv):
        x = torch.randn(B, N, C, device=device)
        w = torch.randn(C, K, device=device) / K ** 0.5
        m = _module_with_weight(w).to(device)
        if not torch.equal(m(x), short_conv(x, w)):
            raise AssertionError(
                f"ShortConv1d != short_conv for C={C} (drifted)")
    # Zero-init module is an exact identity; so is zero-weight functional.
    x = torch.randn(B, N, 6, device=device)
    if not torch.equal(ShortConv1d(6, K).to(device)(x), x):
        raise AssertionError("zero-init ShortConv1d is not an identity")
    if not torch.equal(short_conv(x, torch.zeros(6, K, device=device)), x):
        raise AssertionError("zero-weight short_conv is not an identity")
    # K/V parameter independence.
    k_sconv = ShortConv1d(Hkv * Dk, K)
    v_sconv = ShortConv1d(Hkv * Dv, K)
    assert k_sconv.weight.data_ptr() != v_sconv.weight.data_ptr()
    with torch.no_grad():
        k_sconv.weight.fill_(1.0)
    if not torch.equal(v_sconv.weight, torch.zeros(Hkv * Dv, K)):
        raise AssertionError("K/V conv modules share parameters")
    print("  shortconv-module        ShortConv1d bitwise + zero-init "
          "identity + K/V independence PASS")


def test_negative_controls(ctx):
    """Controls that establish test POWER: each drops or perturbs exactly
    one semantic ingredient and must visibly change the result."""
    device = ctx["device"]
    g = GEOS["tiny"]
    B, N, Hq, Hkv, Dk, Dv = (g[s] for s in ("B", "N", "Hq", "Hkv", "Dk", "Dv"))
    L = len(g["fmap"])
    torch.manual_seed(450)
    q = torch.randn(B, N, Hq, Dk, device=device)
    k = torch.randn(B, N, Hkv, Dk, device=device)
    v = torch.randn(B, N, Hkv, Dv, device=device)
    kl, vl, _ = build_dyadic_summaries(q, k, v, L)
    packed = pack_levels(kl, vl)

    def fwd(**kw):
        return multilevel_attention_forward(
            q, packed, g["fmap"], g["cache"],
            block_m=g["bm"], block_n=g["bn"], **kw)

    # Default forward == explicit position_mode="none", bitwise.
    out_a, lse_a = fwd()
    out_b, lse_b = fwd(position_mode="none")
    report("none bitwise out", out_b, out_a, rtol=0, atol=0)
    report("none bitwise lse", lse_b, lse_a, rtol=0, atol=0)

    # No position mode may mutate the packed buffers.
    k_snap, v_snap = packed.k.clone(), packed.v.clone()
    cos, sin = rope_tables(N, Dk, device=device)
    states = torch.randn(B, N, Hq, _DREL, device=device)
    proj = torch.randn(_DREL, 8, device=device)
    for kw in (dict(position_mode="none"),
               dict(position_mode="rope", rope_cos=cos, rope_sin=sin),
               dict(position_mode="relative", relative_states=states,
                    relative_proj=proj)):
        fwd(**kw)
        if not (torch.equal(packed.k, k_snap)
                and torch.equal(packed.v, v_snap)):
            raise AssertionError(
                f"{kw['position_mode']}: packed K/V mutated")

    # sinks=0 is NOT the identity; it is exactly out * sigmoid(lse).
    got = apply_attention_sink(out_a, lse_a, torch.zeros(Hq, device=device))
    if torch.equal(got, out_a):
        raise AssertionError("zero sink behaved as identity")
    report("sink zero", got,
           out_a * torch.sigmoid(lse_a.float()).unsqueeze(-1).to(out_a.dtype),
           rtol=0, atol=0)

    # SiLU-vs-sigmoid gate guard, plus sink -> gate composition order.
    torch.manual_seed(451)
    x = torch.randn(B, N, _EMB, device=device)
    gate_w = torch.randn(Hq * Dv, _EMB, device=device) / _EMB ** 0.5
    sinks = torch.randn(Hq, device=device)
    gated = apply_output_gate(out_a, x, gate_w)
    sig = torch.sigmoid(x.matmul(gate_w.t())).reshape(B, N, Hq, Dv) * out_a
    if torch.equal(gated, sig):
        raise AssertionError("gate matches sigmoid, not SiLU")
    combined = apply_output_gate(
        apply_attention_sink(out_a, lse_a, sinks), x, gate_w)
    inlined = (
        torch.nn.functional.silu(x.matmul(gate_w.t())).reshape(B, N, Hq, Dv)
        * (out_a * torch.sigmoid(
            lse_a.float() - sinks.float()[None, None, :]
        ).unsqueeze(-1).to(out_a.dtype))
    )
    report("composition order", combined, inlined, rtol=0, atol=0)

    # Dropped q->w tree path must fail; adding the tree VJP must succeed.
    q_l = q.detach().clone().requires_grad_(True)
    k_l = k.detach().clone().requires_grad_(True)
    v_l = v.detach().clone().requires_grad_(True)
    kl2, vl2, _ = build_dyadic_summaries(q_l, k_l, v_l, L)
    packed2 = pack_levels(kl2, vl2)
    with torch.no_grad():
        out2, lse2 = multilevel_attention_forward(
            q_l, packed2, g["fmap"], g["cache"],
            block_m=g["bm"], block_n=g["bn"])
    dout = seeded_randn(out2.shape, 452, device)
    dq_attn, dk_p, dv_p, _, _ = multilevel_attention_backward(
        q_l.detach(),
        PackedKV(packed2.k.detach(), packed2.v.detach(),
                 packed2.level_offsets),
        out2, lse2, dout, g["fmap"], g["cache"],
        block_m=g["bm"], block_n=g["bn"])
    dq_tree, dk_tot, dv_tot = torch.autograd.grad(
        (packed2.k, packed2.v), (q_l, k_l, v_l), (dk_p, dv_p))
    q_o = q.detach().clone().requires_grad_(True)
    k_o = k.detach().clone().requires_grad_(True)
    v_o = v.detach().clone().requires_grad_(True)
    kl3, vl3, _ = build_dyadic_summaries(q_o, k_o, v_o, L)
    out3, _ = multilevel_attention_forward(
        q_o, pack_levels(kl3, vl3), g["fmap"], g["cache"],
        block_m=g["bm"], block_n=g["bn"])
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        out3, (q_o, k_o, v_o), dout)
    report("tree composition dq", dq_attn + dq_tree, dq_ref,
           rtol=GRAD_RTOL, atol=GRAD_ATOL)
    report("tree composition dk", dk_tot, dk_ref,
           rtol=GRAD_RTOL, atol=GRAD_ATOL)
    report("tree composition dv", dv_tot, dv_ref,
           rtol=GRAD_RTOL, atol=GRAD_ATOL)
    err = (dq_attn - dq_ref).abs().max().item()
    if err <= GRAD_ATOL:
        raise AssertionError(
            f"negative control: dropping the q->w tree path still "
            f"matched dq (max err {err:.3e})")
    print("  negative-controls       none-bitwise / no-mutation / sink-0 / "
          "SiLU / order / dropped-tree PASS")


def test_error_paths(ctx):
    device = ctx["device"]
    g = GEOS["tiny"]
    B, N, Hq, Hkv, Dk, Dv = (g[s] for s in ("B", "N", "Hq", "Hkv", "Dk", "Dv"))
    L = len(g["fmap"])
    torch.manual_seed(460)
    q = torch.randn(B, N, Hq, Dk, device=device)
    k = torch.randn(B, N, Hkv, Dk, device=device)
    v = torch.randn(B, N, Hkv, Dv, device=device)
    x = torch.randn(B, N, _EMB, device=device)
    kl_r, vl_r, _ = build_dyadic_summaries(q, k, v, L)
    packed = pack_levels(kl_r, vl_r)
    spec = _spec(g)
    cos, sin = rope_tables(N, Dk, device=device)
    states = torch.randn(B, N, Hq, _DREL, device=device)
    proj = torch.randn(_DREL, 8, device=device)

    def fwd(**kw):
        return multilevel_attention_forward(
            q, packed, g["fmap"], g["cache"], block_m=3, block_n=2, **kw)

    ve = functools.partial(expect_error, ValueError)
    # reference forward positional-argument matrix.
    ve("unknown mode", lambda: fwd(position_mode="alibi"))
    ve("rope args with none",
       lambda: fwd(position_mode="none", rope_cos=cos, rope_sin=sin))
    ve("relative args with rope",
       lambda: fwd(position_mode="rope", rope_cos=cos, rope_sin=sin,
                   relative_states=states, relative_proj=proj))
    ve("rope args with relative",
       lambda: fwd(position_mode="relative", relative_states=states,
                   relative_proj=proj, rope_cos=cos, rope_sin=sin))
    ve("missing sin", lambda: fwd(position_mode="rope", rope_cos=cos))
    ve("short rope table",
       lambda: fwd(position_mode="rope",
                   rope_cos=cos[:N - 1], rope_sin=sin[:N - 1]))
    ve("mismatched cos/sin",
       lambda: fwd(position_mode="rope", rope_cos=cos, rope_sin=sin[:N - 1]))
    ve("wrong proj shape",
       lambda: fwd(position_mode="relative", relative_states=states,
                   relative_proj=torch.randn(5, 8, device=device)))
    ve("1-D proj",
       lambda: fwd(position_mode="relative", relative_states=states,
                   relative_proj=torch.randn(8, device=device)))
    ve("table too small",
       lambda: fwd(position_mode="relative", relative_states=states,
                   relative_proj=torch.randn(_DREL, 2, device=device)))
    q_odd = torch.randn(B, N, Hq, 7, device=device)
    ve("odd Dk", lambda: multilevel_attention_forward(
        q_odd, packed, g["fmap"], g["cache"], block_m=3, block_n=2,
        position_mode="rope", rope_cos=cos, rope_sin=sin))
    ve("invisible entry", lambda: relative_bin_distance(spec, 0, 0, 5))

    # reference backward mirrors the forward's matrix; plus the dlse seed.
    out, lse = fwd()

    def bwd(**kw):
        return multilevel_attention_backward(
            q, packed, out, lse, torch.randn_like(out), g["fmap"],
            g["cache"], block_m=3, block_n=2, **kw)

    ve("backward unknown mode", lambda: bwd(position_mode="alibi"))
    ve("backward rope args with none",
       lambda: bwd(position_mode="none", rope_cos=cos, rope_sin=sin))
    ve("backward missing sin", lambda: bwd(position_mode="rope",
                                           rope_cos=cos))
    ve("backward missing proj",
       lambda: bwd(position_mode="relative", relative_states=states))
    ve("backward table too small",
       lambda: bwd(position_mode="relative", relative_states=states,
                   relative_proj=torch.randn(_DREL, 2, device=device)))
    ve("dlse wrong shape", lambda: bwd(dlse=lse[:, :-1]))
    ve("dlse non-floating",
       lambda: bwd(dlse=torch.zeros_like(lse, dtype=torch.long)))

    # sink / gate contracts.
    sinks = torch.randn(Hq, device=device)
    ve("sinks wrong length",
       lambda: apply_attention_sink(out, lse,
                                    torch.randn(Hq + 1, device=device)))
    ve("sinks per KV head",
       lambda: apply_attention_sink(out, lse,
                                    torch.randn(Hkv, device=device)))
    ve("gate weight shape",
       lambda: apply_output_gate(out, x,
                                 torch.randn(Hq * Dv + 1, _EMB,
                                             device=device)))
    ve("sink-backward dout dtype",
       lambda: attention_sink_backward(
           out, lse, sinks, torch.randn_like(out).to(torch.bfloat16)))

    # conv contracts.
    kcw = torch.randn(Hkv * Dk, 3, device=device)
    ve("one-sided conv (ref)",
       lambda: build_dyadic_summaries(q, k, v, L, k_conv_weight=kcw))
    ve("conv wrong channels",
       lambda: short_conv(x, torch.randn(_EMB + 1, 3, device=device)))
    ve("conv 1-D weight",
       lambda: short_conv(x, torch.randn(3, device=device)))
    ve("conv zero-width",
       lambda: short_conv(x, torch.randn(_EMB, 0, device=device)))

    # minimal API mirrors.
    kl_m, vl_m, _ = _summary_tree(k, v, _qk_summary_weights(q, k), L)
    cos_m, sin_m = minimal_rope_tables(N, Dk, device=device)
    ve("mini rope args with none", lambda: minimal_multilevel_attention(
        q, kl_m, vl_m, g["fmap"], g["cache"],
        rope_cos=cos_m, rope_sin=sin_m))
    ve("mini rope without tables", lambda: minimal_multilevel_attention(
        q, kl_m, vl_m, g["fmap"], g["cache"], position_mode="rope"))
    ve("mini relative without args", lambda: minimal_multilevel_attention(
        q, kl_m, vl_m, g["fmap"], g["cache"], position_mode="relative"))
    ve("mini Hq % Hkv", lambda: minimal_multilevel_attention(
        q[:, :, :3], kl_m, vl_m, g["fmap"], g["cache"]))
    ve("mini sinks shape", lambda: minimal_multilevel_attention(
        q, kl_m, vl_m, g["fmap"], g["cache"],
        sinks=torch.zeros(Hq + 1, device=device)))
    ve("mini one-sided conv", lambda: minimal_attention_block(
        x, q, k, v, g["fmap"], g["cache"], num_summary_levels=L,
        k_conv_weight=kcw))
    ve("mini linear without w_proj", lambda: minimal_attention_block(
        x, q, k, v, g["fmap"], g["cache"], num_summary_levels=L,
        weight_mode="linear"))
    ve("mini w_proj with qk", lambda: minimal_attention_block(
        x, q, k, v, g["fmap"], g["cache"], num_summary_levels=L,
        w_proj=torch.randn(Hkv, _EMB, device=device)))
    # The causality guard itself (no aligned fmap can violate it).
    ve("summary causality", lambda: _check_summary_causality(
        RangeSpec(activation_times=(0, 0), cache_size=6, seq_len=8)))
    print("  error-paths             ValueError on misuse "
          "(ref fwd/bwd + sink/gate/conv + minimal API) PASS")


# ===========================================================================
# Harness.
# ===========================================================================

TESTS = [
    # Section 1 -- schedule (always runs).
    test_schedule_properties,
    test_visible_entries,
    test_positional_schedule,
    test_coverage_pins,
    # Section 2 -- forward semantics (pair-driven).
    test_summary_tree,
    test_forward,
    # Section 3 -- backward semantics.
    test_phase2_backward,
    test_block_gradients,
    test_grad_structure,
    # Section 4 -- contracts (always runs).
    test_mixed_precision,
    test_shortconv_module,
    test_negative_controls,
    test_error_paths,
]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Consolidated telescope-attention correctness suite: "
                    "pairwise implementation comparison (mini/ref/flex) + "
                    "schedule oracles + contracts."
    )
    parser.add_argument(
        "--device", default=None,
        help="cpu or cuda (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--compare", nargs=2, default=["mini", "ref"],
        choices=sorted(BACKEND_CAPS),
        metavar=("A", "B"),
        help="implementation pair to compare (default: mini ref); "
             "flex needs torch >= 2.5 + CUDA and SKIPs cleanly otherwise",
    )
    parser.add_argument(
        "tests", nargs="*",
        help=f"test names to run (default: all). "
             f"Available: {[t.__name__ for t in TESTS]}",
    )
    args = parser.parse_args(argv)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.compare[0] == args.compare[1]:
        raise SystemExit("--compare needs two distinct backends")

    names = {t.__name__ for t in TESTS}
    unknown = [t for t in args.tests if t not in names]
    if unknown:
        raise SystemExit(
            f"Unknown test(s) {unknown}; available: {sorted(names)}")
    selected = [t for t in TESTS if not args.tests or t.__name__ in args.tests]

    ctx = dict(device=device, pair=tuple(args.compare))
    print(f"=== telescope reference suite: device={device} "
          f"compare={ctx['pair'][0]}<->{ctx['pair'][1]} ===")
    for t in selected:
        t(ctx)
    print(f"\nALL REFERENCE-SUITE TESTS PASSED ({len(selected)} tests)")


if __name__ == "__main__":
    main()
