"""
Cross-checks between telescope_cache.minimal_reference (the naive semantic
oracle: per-query visible-entry enumeration, ordinary softmax, literal sink
logit + zero value row, autograd-only backward) and
telescope_cache.reference (the kernel-shaped implementation: packed
level-major buffers, tiled online softmax, LSE-sigmoid sink rescale,
explicit Phase-2 backward).

The value of these tests is structural independence: both sides start from
the same tensors but compute through deliberately different machinery
(chronological entry lists vs. tile hulls, arange(M-1..0) bin distances vs.
the analytic rank formula, a literal sink token vs. the sigmoid(lse - s)
identity), so agreement validates the semantics rather than a shared helper.
The final gradient test pits minimal's pure autograd against reference.py's
explicit-composition backward (Phase-2 backward + sink dlse seed + Phase-1
tree VJP), the same composition test_position.py validates against
reference-side autograd.
"""

import argparse
import os
import sys

import torch

# Make `telescope_cache` importable (namespace package).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, _HERE)

from telescope_cache.range_spec import RangeSpec, range_bounds  # noqa: E402
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
)
from test_forward import dyadic_ranges_for_query  # noqa: E402

RTOL = ATOL = 1e-5
GRAD_RTOL = GRAD_ATOL = 1e-4
GRAD_MIN_NORM = 1e-6
SOFTCAP = 20.0

# Small geometries with real coverage (verified against range_spec):
# evict/odd_n evict from q=16, four_levels from q=48; max M never exceeds
# cache_size; every schedule satisfies the summary-causality invariant.
# Deliberately no N=1024 case -- the per-query oracle loop is slow.
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
}


def report(name, got, want, rtol=RTOL, atol=ATOL):
    try:
        torch.testing.assert_close(got, want, rtol=rtol, atol=atol)
    except AssertionError as exc:
        raise AssertionError(f"{name}: {exc}") from None


def _spec(g):
    return RangeSpec.from_fmap(g["fmap"], g["cache"], g["N"])


def _make_qkv(g, device, seed):
    torch.manual_seed(seed)
    q = torch.randn(g["B"], g["N"], g["Hq"], g["Dk"], device=device)
    k = torch.randn(g["B"], g["N"], g["Hkv"], g["Dk"], device=device)
    v = torch.randn(g["B"], g["N"], g["Hkv"], g["Dv"], device=device)
    return q, k, v


def _minimal_forward(q, k, v, g, softcap=SOFTCAP, **kwargs):
    kl, vl, _ = _summary_tree(k, v, _qk_summary_weights(q, k), len(g["fmap"]))
    return minimal_multilevel_attention(
        q, kl, vl, g["fmap"], g["cache"], softcap=softcap, **kwargs)


def _ref_forward(q, k, v, g, softcap=SOFTCAP, **fw_kwargs):
    kl, vl, _ = build_dyadic_summaries(q, k, v, len(g["fmap"]))
    packed = pack_levels(kl, vl)
    return multilevel_attention_forward(
        q, packed, g["fmap"], g["cache"], block_m=g["bm"], block_n=g["bn"],
        softcap=softcap, **fw_kwargs)


# ---------------------------------------------------------------------------
# Full-block helpers (Tests 7 and 8): both sides consume the same base
# parameter dict; the minimal side never touches reference.py helpers
# (its RoPE tables come from minimal_rope_tables, its relative states from
# minimal_attention_block's own x @ rel_w^T path).
# ---------------------------------------------------------------------------

_EMB, _KC, _DREL = 32, 4, 6


def _block_params(g, device, seed, mode, weight_mode):
    B, N, Hq, Hkv, Dk, Dv = (g[s] for s in ("B", "N", "Hq", "Hkv", "Dk", "Dv"))
    emb, Kc, d_rel = _EMB, _KC, _DREL
    max_bins = max(8, g["cache"])
    torch.manual_seed(seed)
    base = dict(
        x=torch.randn(B, N, emb, device=device),
        Wq=torch.randn(Hq * Dk, emb, device=device) / emb ** 0.5,
        Wk=torch.randn(Hkv * Dk, emb, device=device) / emb ** 0.5,
        Wv=torch.randn(Hkv * Dv, emb, device=device) / emb ** 0.5,
        kcw=torch.randn(Hkv * Dk, Kc, device=device) / Kc ** 0.5,
        vcw=torch.randn(Hkv * Dv, Kc, device=device) / Kc ** 0.5,
        gate_w=torch.randn(Hq * Dv, emb, device=device) / emb ** 0.5,
        o_w=torch.randn(emb, Hq * Dv, device=device) / (Hq * Dv) ** 0.5,
    )
    if weight_mode == "linear":
        base["w_proj"] = torch.randn(Hkv, emb, device=device) / emb ** 0.5
    if mode == "relative":
        base["rel_w"] = torch.randn(Hq * d_rel, emb, device=device) / emb ** 0.5
        base["rel_proj"] = torch.randn(d_rel, max_bins, device=device) / d_rel ** 0.5
    return base


def _projections(t, g):
    B, N, Hq, Hkv, Dk, Dv = (g[s] for s in ("B", "N", "Hq", "Hkv", "Dk", "Dv"))
    x = t["x"]
    q4 = x.matmul(t["Wq"].t()).view(B, N, Hq, Dk)
    k4 = x.matmul(t["Wk"].t()).view(B, N, Hkv, Dk)
    v4 = x.matmul(t["Wv"].t()).view(B, N, Hkv, Dv)
    return q4, k4, v4


def _ref_chain(t, g, mode, weight_mode, cos, sin, attn_no_grad=False):
    """Reference-side block up to (out, lse); mirrors
    test_position._full_block_case's chain."""
    q4, k4, v4 = _projections(t, g)
    bd_kwargs = dict(k_conv_weight=t["kcw"], v_conv_weight=t["vcw"])
    if weight_mode == "linear":
        bd_kwargs.update(x=t["x"], w_proj=t["w_proj"])
    kl, vl, _ = build_dyadic_summaries(q4, k4, v4, len(g["fmap"]), **bd_kwargs)
    packed = pack_levels(kl, vl)
    states = None
    fw_kwargs = {}
    if mode == "rope":
        fw_kwargs = dict(rope_cos=cos, rope_sin=sin)
    elif mode == "relative":
        states = compute_relative_states(t["x"], t["rel_w"], g["Hq"])
        fw_kwargs = dict(relative_states=states, relative_proj=t["rel_proj"])
    if attn_no_grad:
        with torch.no_grad():
            out, lse = multilevel_attention_forward(
                q4, packed, g["fmap"], g["cache"], block_m=g["bm"],
                block_n=g["bn"], softcap=SOFTCAP, position_mode=mode,
                **fw_kwargs)
    else:
        out, lse = multilevel_attention_forward(
            q4, packed, g["fmap"], g["cache"], block_m=g["bm"],
            block_n=g["bn"], softcap=SOFTCAP, position_mode=mode, **fw_kwargs)
    return dict(q4=q4, packed=packed, states=states, out=out, lse=lse)


def _ref_final(t, ce, g):
    out_sink = apply_attention_sink(ce["out"], ce["lse"], t["sinks"])
    gated = apply_output_gate(out_sink, t["x"], t["gate_w"])
    return gated.reshape(g["B"], g["N"], g["Hq"] * g["Dv"]).matmul(t["o_w"].t())


def _minimal_final(t, g, mode, weight_mode, cos, sin):
    q4, k4, v4 = _projections(t, g)
    return minimal_attention_block(
        t["x"], q4, k4, v4, g["fmap"], g["cache"],
        num_summary_levels=len(g["fmap"]),
        k_conv_weight=t["kcw"], v_conv_weight=t["vcw"],
        weight_mode=weight_mode, w_proj=t.get("w_proj"),
        position_mode=mode, rope_cos=cos, rope_sin=sin,
        relative_weight=t.get("rel_w"), relative_proj=t.get("rel_proj"),
        sinks=t["sinks"], gate_weight=t["gate_w"], output_weight=t["o_w"],
        softcap=SOFTCAP)


def _centered_sinks(base, g, mode, weight_mode, cos, sin, seed):
    """Sinks centered on this configuration's lse so sigmoid(lse - s) ~ 0.5
    and the sink path stays load-bearing (test_sink_gate's trick)."""
    with torch.no_grad():
        probe = _ref_chain(base, g, mode, weight_mode, cos, sin,
                           attn_no_grad=True)
    torch.manual_seed(seed)
    return (probe["lse"].float().mean(dim=(0, 1))
            + 0.1 * torch.randn(g["Hq"], device=base["x"].device))


_BLOCK_CASES = [
    ("none", "linear", "tiny", 60),
    ("rope", "qk", "tiny", 64),
    ("relative", "qk", "tiny", 68),
    ("none", "qk", "evict", 72),
]


def _block_tables(g, mode, device):
    """(cos_min, sin_min, cos_ref, sin_ref) -- each side builds its own."""
    if mode != "rope":
        return None, None, None, None
    cos_m, sin_m = minimal_rope_tables(g["N"], g["Dk"], device=device)
    cos_r, sin_r = rope_tables(g["N"], g["Dk"], device=device)
    return cos_m, sin_m, cos_r, sin_r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_visible_entries(device):
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
    print("  visible-entries         chronological list == range oracle "
          "(4 geometries, every q) PASS")


def test_summary_tree(device):
    Kc, emb = 3, 16
    for i, name in enumerate(("tiny", "odd_n", "four_levels")):
        g = GEOS[name]
        L = len(g["fmap"])
        B, N, Hkv, Dk, Dv = g["B"], g["N"], g["Hkv"], g["Dk"], g["Dv"]
        q, k, v = _make_qkv(g, device, seed=10 + i)
        x = torch.randn(B, N, emb, device=device)
        w_proj = torch.randn(Hkv, emb, device=device) / emb ** 0.5
        kcw = torch.randn(Hkv * Dk, Kc, device=device) / Kc ** 0.5
        vcw = torch.randn(Hkv * Dv, Kc, device=device) / Kc ** 0.5

        # QK weights, no conv.
        kl_m, vl_m, wl_m = _summary_tree(k, v, _qk_summary_weights(q, k), L)
        kl_r, vl_r, wl_r = build_dyadic_summaries(q, k, v, L)
        for lv in range(L + 1):
            report(f"{name} qk k[{lv}]", kl_m[lv], kl_r[lv])
            report(f"{name} qk v[{lv}]", vl_m[lv], vl_r[lv])
            report(f"{name} qk w[{lv}]", wl_m[lv], wl_r[lv])

        # Linear weights, no conv.
        w_lin = x.matmul(w_proj.t()).unsqueeze(-1)
        kl_m, vl_m, wl_m = _summary_tree(k, v, w_lin, L)
        kl_r, vl_r, wl_r = build_dyadic_summaries(q, k, v, L, x=x,
                                                  w_proj=w_proj)
        for lv in range(L + 1):
            report(f"{name} lin k[{lv}]", kl_m[lv], kl_r[lv])
            report(f"{name} lin v[{lv}]", vl_m[lv], vl_r[lv])
            report(f"{name} lin w[{lv}]", wl_m[lv], wl_r[lv])

        # Short conv, then QK weights from the CONVOLVED K.
        report(f"{name} shortconv",
               _short_conv(k.reshape(B, N, Hkv * Dk), kcw),
               short_conv(k.reshape(B, N, Hkv * Dk), kcw))
        kc = _short_conv(k.reshape(B, N, Hkv * Dk), kcw).reshape(B, N, Hkv, Dk)
        vc = _short_conv(v.reshape(B, N, Hkv * Dv), vcw).reshape(B, N, Hkv, Dv)
        kl_m, vl_m, wl_m = _summary_tree(kc, vc, _qk_summary_weights(q, kc), L)
        kl_r, vl_r, wl_r = build_dyadic_summaries(
            q, k, v, L, k_conv_weight=kcw, v_conv_weight=vcw)
        for lv in range(L + 1):
            report(f"{name} sconv k[{lv}]", kl_m[lv], kl_r[lv])
            report(f"{name} sconv v[{lv}]", vl_m[lv], vl_r[lv])
            report(f"{name} sconv w[{lv}]", wl_m[lv], wl_r[lv])
    print("  summary-tree            per-pair loop == reference tree "
          "(qk/linear/sconv, odd lengths) PASS")


def test_forward_nope(device):
    for i, name in enumerate(GEOS):
        g = GEOS[name]
        q, k, v = _make_qkv(g, device, seed=20 + i)
        out_m, lse_m = _minimal_forward(q, k, v, g)
        out_r, lse_r = _ref_forward(q, k, v, g)
        report(f"{name} out", out_m, out_r)
        report(f"{name} lse", lse_m, lse_r)
    q, k, v = _make_qkv(GEOS["tiny"], device, seed=29)
    out_m, lse_m = _minimal_forward(q, k, v, GEOS["tiny"], softcap=None)
    out_r, lse_r = _ref_forward(q, k, v, GEOS["tiny"], softcap=None)
    report("tiny nocap out", out_m, out_r)
    report("tiny nocap lse", lse_m, lse_r)
    print("  forward-nope            ordinary softmax == tiled online "
          "softmax (5 geometries + softcap=None) PASS")


def test_forward_rope(device):
    for i, name in enumerate(("tiny", "evict")):
        g = GEOS[name]
        spec = _spec(g)
        # The claim "RoPE with actual coarse summaries" must not rot.
        lo_hi = [range_bounds(spec, g["N"] - 1, lv)
                 for lv in range(1, spec.num_levels)]
        if not any(hi > lo for lo, hi in lo_hi):
            raise AssertionError(
                f"{name}: no coarse summary visible at q=N-1; geometry "
                f"does not exercise post-summary RoPE")
        cos_m, sin_m = minimal_rope_tables(g["N"], g["Dk"], device=device)
        cos_r, sin_r = rope_tables(g["N"], g["Dk"], device=device)
        report(f"{name} cos table", cos_m, cos_r)
        report(f"{name} sin table", sin_m, sin_r)
        q, k, v = _make_qkv(g, device, seed=30 + i)
        out_m, lse_m = _minimal_forward(
            q, k, v, g, position_mode="rope", rope_cos=cos_m, rope_sin=sin_m)
        out_r, lse_r = _ref_forward(
            q, k, v, g, position_mode="rope", rope_cos=cos_m, rope_sin=sin_m)
        report(f"{name} rope out", out_m, out_r)
        report(f"{name} rope lse", lse_m, lse_r)
    print("  forward-rope            right-endpoint summary rotation == "
          "reference (tables cross-checked) PASS")


def test_forward_relative(device):
    for i, name in enumerate(("tiny", "evict", "four_levels")):
        g = GEOS[name]
        spec = _spec(g)
        q, k, v = _make_qkv(g, device, seed=40 + i)
        states = torch.randn(g["B"], g["N"], g["Hq"], _DREL, device=device)
        proj = torch.randn(_DREL, g["cache"], device=device) / _DREL ** 0.5
        out_m, lse_m = _minimal_forward(
            q, k, v, g, position_mode="relative",
            relative_states=states, relative_proj=proj)
        out_r, lse_r = _ref_forward(
            q, k, v, g, position_mode="relative",
            relative_states=states, relative_proj=proj)
        report(f"{name} rel out", out_m, out_r)
        report(f"{name} rel lse", lse_m, lse_r)
        # arange(M-1..0) over the sorted list == the analytic bin rank.
        for qi in (0, g["N"] // 2, g["N"] - 1):
            entries = minimal_visible_entries(spec, qi)
            M = len(entries)
            for rank, (lv, j) in enumerate(entries):
                want = M - 1 - rank
                got = relative_bin_distance(spec, qi, lv, j)
                if got != want:
                    raise AssertionError(
                        f"{name} q={qi} entry ({lv},{j}): distance {got} "
                        f"!= chronological {want}")
    # Too-short table must raise, never clamp (evict sees M=6 > 4 bins).
    g = GEOS["evict"]
    q, k, v = _make_qkv(g, device, seed=44)
    states = torch.randn(g["B"], g["N"], g["Hq"], _DREL, device=device)
    proj4 = torch.randn(_DREL, 4, device=device)
    try:
        _minimal_forward(q, k, v, g, position_mode="relative",
                         relative_states=states, relative_proj=proj4)
    except ValueError:
        pass
    else:
        raise AssertionError("M > max_relative_bins did not raise")
    print("  forward-relative        arange(M-1..0) bins == analytic "
          "tile-rank reference (3 geometries) PASS")


def test_sink(device):
    for i, name in enumerate(("tiny", "evict")):
        g = GEOS[name]
        q, k, v = _make_qkv(g, device, seed=50 + i)
        out_r, lse_r = _ref_forward(q, k, v, g)
        torch.manual_seed(55 + i)
        sinks = (lse_r.mean(dim=(0, 1))
                 + 0.1 * torch.randn(g["Hq"], device=device))
        out_m_sink, lse_m_sink = _minimal_forward(q, k, v, g, sinks=sinks)
        # Literal extra logit + zero value vs. the LSE-sigmoid rescale.
        report(f"{name} sink out", out_m_sink,
               apply_attention_sink(out_r, lse_r, sinks))
        # Internal identity on minimal's own tensors.
        out_m, lse_m = _minimal_forward(q, k, v, g)
        scale = torch.sigmoid(lse_m - sinks[None, None, :])
        report(f"{name} sink identity", out_m_sink,
               out_m * scale.unsqueeze(-1))
        # `lse` is the pre-sink LSE: enabling sinks never changes it.
        if not torch.equal(lse_m_sink, lse_m):
            raise AssertionError(f"{name}: sinks changed the returned lse")
    print("  sink                    literal sink token == "
          "sigmoid(lse - s) rescale PASS")


def test_gate_and_block_forward(device):
    # Gate-only unit check.
    g = GEOS["tiny"]
    q, k, v = _make_qkv(g, device, seed=59)
    x = torch.randn(g["B"], g["N"], _EMB, device=device)
    gate_w = torch.randn(g["Hq"] * g["Dv"], _EMB, device=device) / _EMB ** 0.5
    out_r, _ = _ref_forward(q, k, v, g)
    out_m = minimal_attention_block(
        x, q, k, v, g["fmap"], g["cache"],
        num_summary_levels=len(g["fmap"]), gate_weight=gate_w)
    report("gate-only", out_m, apply_output_gate(out_r, x, gate_w))

    # Full block: conv + weights + tree + position + sink + gate + o_proj.
    for mode, weight_mode, geo_name, seed in _BLOCK_CASES:
        g = GEOS[geo_name]
        cos_m, sin_m, cos_r, sin_r = _block_tables(g, mode, device)
        base = _block_params(g, device, seed, mode, weight_mode)
        base["sinks"] = _centered_sinks(
            base, g, mode, weight_mode, cos_r, sin_r, seed + 1)
        ce = _ref_chain(base, g, mode, weight_mode, cos_r, sin_r,
                        attn_no_grad=True)
        with torch.no_grad():
            want = _ref_final(base, ce, g)
            got = _minimal_final(base, g, mode, weight_mode, cos_m, sin_m)
        report(f"block {mode}/{weight_mode}/{geo_name}", got, want)
    print("  block-forward           minimal block == reference "
          "composition (gate unit + 4 mode/weight cases) PASS")


def test_block_gradients(device):
    """Minimal pure-autograd oracle vs. reference.py's explicit-composition
    backward (Phase-2 backward + sink dlse + Phase-1 tree VJP + downstream
    gate/o_proj VJPs), the composition of test_position._full_block_case."""
    cases = [
        ("none", "linear", "tiny", 210),
        ("rope", "qk", "tiny", 214),
        ("relative", "qk", "tiny", 218),
        ("none", "qk", "evict", 222),
    ]
    for mode, weight_mode, geo_name, seed in cases:
        g = GEOS[geo_name]
        B, N, Hq, Dv = g["B"], g["N"], g["Hq"], g["Dv"]
        name = f"{mode}/{weight_mode}/{geo_name}"
        cos_m, sin_m, cos_r, sin_r = _block_tables(g, mode, device)
        base = _block_params(g, device, seed, mode, weight_mode)
        base["sinks"] = _centered_sinks(
            base, g, mode, weight_mode, cos_r, sin_r, seed + 1)
        keys = list(base.keys())

        def leaves():
            return {kk: t.detach().clone().requires_grad_(True)
                    for kk, t in base.items()}

        # --- minimal side: one ordinary autograd graph -------------------
        a = leaves()
        out_min = _minimal_final(a, g, mode, weight_mode, cos_m, sin_m)
        gen = torch.Generator(device="cpu").manual_seed(seed + 2)
        dfinal = torch.randn(out_min.shape, generator=gen).to(device)
        oracle = dict(zip(keys, torch.autograd.grad(
            out_min, tuple(a[kk] for kk in keys), dfinal)))

        # --- reference side: explicit composition ------------------------
        e = leaves()
        ce = _ref_chain(e, g, mode, weight_mode, cos_r, sin_r,
                        attn_no_grad=True)   # Phase-1 graph alive
        out, lse = ce["out"], ce["lse"]

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
            bw_kwargs = dict(rope_cos=cos_r, rope_sin=sin_r)
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
        if weight_mode == "linear":
            up_keys.append("w_proj")
        if mode == "relative":
            dstates = torch.einsum(
                "bnhr,dr->bnhd", dpos, e["rel_proj"].detach().float())
            up_outputs.append(ce["states"])
            up_seeds.append(dstates)
            up_keys.append("rel_w")
        up = dict(zip(up_keys, torch.autograd.grad(
            tuple(up_outputs), tuple(e[kk] for kk in up_keys),
            tuple(up_seeds))))

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
                    f"({want.norm().item():.3e}) -- test has no power here")
            report(f"d{kname}/{name}", gg, want,
                   rtol=GRAD_RTOL, atol=GRAD_ATOL)
    print("  block-gradients         minimal autograd == explicit "
          "composition (4 mode/weight cases, all leaves) PASS")


def test_error_paths(device):
    def raises(label, fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"{label}: expected ValueError")

    g = GEOS["tiny"]
    q, k, v = _make_qkv(g, device, seed=90)
    kl, vl, _ = _summary_tree(k, v, _qk_summary_weights(q, k), len(g["fmap"]))
    cos, sin = minimal_rope_tables(g["N"], g["Dk"], device=device)
    x = torch.randn(g["B"], g["N"], _EMB, device=device)

    raises("rope args in none mode", lambda: minimal_multilevel_attention(
        q, kl, vl, g["fmap"], g["cache"], rope_cos=cos, rope_sin=sin))
    raises("rope mode without tables", lambda: minimal_multilevel_attention(
        q, kl, vl, g["fmap"], g["cache"], position_mode="rope"))
    raises("relative mode without args", lambda: minimal_multilevel_attention(
        q, kl, vl, g["fmap"], g["cache"], position_mode="relative"))
    raises("Hq not divisible by Hkv", lambda: minimal_multilevel_attention(
        q[:, :, :3], kl, vl, g["fmap"], g["cache"]))
    raises("sinks wrong shape", lambda: minimal_multilevel_attention(
        q, kl, vl, g["fmap"], g["cache"],
        sinks=torch.zeros(g["Hq"] + 1, device=device)))
    raises("one-sided conv weight", lambda: minimal_attention_block(
        x, q, k, v, g["fmap"], g["cache"],
        num_summary_levels=len(g["fmap"]),
        k_conv_weight=torch.randn(g["Hkv"] * g["Dk"], 3, device=device)))
    raises("linear mode without w_proj", lambda: minimal_attention_block(
        x, q, k, v, g["fmap"], g["cache"],
        num_summary_levels=len(g["fmap"]), weight_mode="linear"))
    raises("w_proj with qk mode", lambda: minimal_attention_block(
        x, q, k, v, g["fmap"], g["cache"],
        num_summary_levels=len(g["fmap"]),
        w_proj=torch.randn(g["Hkv"], _EMB, device=device)))
    # The causality guard itself (no aligned fmap can violate it, so the
    # bad schedule is built directly).
    raises("summary causality", lambda: _check_summary_causality(
        RangeSpec(activation_times=(0, 0), cache_size=6, seq_len=8)))
    print("  error-paths             ValueError on misuse PASS")


TESTS = [
    test_visible_entries,
    test_summary_tree,
    test_forward_nope,
    test_forward_rope,
    test_forward_relative,
    test_sink,
    test_gate_and_block_forward,
    test_block_gradients,
    test_error_paths,
]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Minimal semantic oracle vs. the kernel-shaped "
                    "reference (telescope_cache.minimal_reference vs. "
                    "telescope_cache.reference)."
    )
    parser.add_argument(
        "--device", default=None,
        help="cpu or cuda (default: cuda if available, else cpu)",
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

    names = {t.__name__ for t in TESTS}
    unknown = [t for t in args.tests if t not in names]
    if unknown:
        raise SystemExit(
            f"Unknown test(s) {unknown}; available: {sorted(names)}")
    selected = [t for t in TESTS if not args.tests or t.__name__ in args.tests]

    print(f"=== minimal reference vs. reference: device={device} ===")
    for t in selected:
        t(device)
    print(f"\nALL MINIMAL-REFERENCE TESTS PASSED ({len(selected)} tests)")


if __name__ == "__main__":
    main()
