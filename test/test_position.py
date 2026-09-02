"""
Unit contract for optional positional encoding in reference.py:

    position_mode="none"      bitwise-identical to the frozen contract
    position_mode="rope"      post-summary RoPE (Q + summarized K, right-
                              endpoint virtual-token positions)
    position_mode="relative"  learned query-conditioned bias in summary-bin
                              space (0-based chronological distance,
                              current bin = 0)

Structural anchors: the partition property (one query's visible entries
tile a contiguous span of the past disjointly across levels) asserted
against the slow range oracle, analytic bin distances vs that oracle,
dense RoPE and dense relative-bias oracles, gradient flow, the
position-independence of the summary tree, the explicit RoPE and
relative backwards vs autograd at the packed-K/V boundary (the relative
mode additionally returns the drel_logits positional-gradient seed,
composed to dstates/dproj in closed form), their composition with the
attention-sink dlse seed, the full Phase-2 + Phase-1 composition
(explicit seeds through the tree VJP == end-to-end autograd), and the
complete attention-block composition (projections -> K/V short conv ->
tree -> positional attention -> sink -> gate -> o_proj) vs pure autograd
for all trainable parameters. Training-only.
"""

import argparse
import os
import sys

import torch

# Make `telescope_cache` importable (namespace package).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, _HERE)

from telescope_cache.range_spec import RangeSpec  # noqa: E402
from telescope_cache.reference import (  # noqa: E402
    PackedKV,
    apply_attention_sink,
    apply_output_gate,
    apply_rope,
    attention_sink_backward,
    build_dyadic_summaries,
    compute_relative_states,
    multilevel_attention_backward,
    multilevel_attention_forward,
    pack_levels,
    relative_bin_distance,
    rope_tables,
    summary_token_position,
)
from test_forward import dyadic_ranges_for_query  # noqa: E402

FWD_RTOL = FWD_ATOL = 1e-5
# GRAD tolerances match test_backward.py's: two materially different
# reduction orders (autograd through the streaming forward vs. the explicit
# two-pass backward); do not tune per case.
GRAD_RTOL = GRAD_ATOL = 1e-4
GRAD_MIN_NORM = 1e-6
SOFTCAP = 20.0

# (N, fmap, cache_size) triples taken from existing suite schedules
# (guaranteed dyadically aligned).
BIN_CONFIGS = [
    (8, {1: 2, 2: 3}, 6),                       # readme_tiny
    (128, {1: 64, 2: 72, 3: 80}, 512),          # baseline
    (200, {1: 32, 2: 40, 3: 48, 4: 56}, 96),    # four_levels, shorter N
]


def _tiny_case(device, seed):
    torch.manual_seed(seed)
    B, N, Hq, Hkv, Dk, Dv = 2, 8, 4, 2, 8, 16
    fmap, cache_size = {1: 2, 2: 3}, 6
    q = torch.randn(B, N, Hq, Dk, device=device)
    k = torch.randn(B, N, Hkv, Dk, device=device)
    v = torch.randn(B, N, Hkv, Dv, device=device)
    spec = RangeSpec.from_fmap(fmap, cache_size, N)
    return q, k, v, spec, fmap, cache_size


def _build_and_pack(q, k, v, spec):
    kl, vl, _ = build_dyadic_summaries(q, k, v, spec.num_levels - 1)
    return kl, vl, pack_levels(kl, vl)


def visible_entries_chronological(fmap, cache_size, q_index):
    """Slow oracle: [(level, j, t_start, t_end)] oldest -> newest."""
    ranges = dyadic_ranges_for_query(q_index, fmap, cache_size)
    entries = [
        (level, j, j << level, (j + 1) << level)
        for level, (lo, hi) in enumerate(ranges)
        for j in range(lo, hi)
    ]
    return sorted(entries, key=lambda e: e[2])


def test_none_bitwise_regression(device):
    for cfg_i, (B, N, Hq, Hkv, Dk, Dv, fmap, cache, bm, bn) in enumerate([
        (2, 8, 4, 2, 8, 16, {1: 2, 2: 3}, 6, 3, 2),
        (1, 128, 8, 2, 32, 32, {1: 64, 2: 72, 3: 80}, 512, 16, 32),
    ]):
        torch.manual_seed(20 + cfg_i)
        q = torch.randn(B, N, Hq, Dk, device=device)
        k = torch.randn(B, N, Hkv, Dk, device=device)
        v = torch.randn(B, N, Hkv, Dv, device=device)
        spec = RangeSpec.from_fmap(fmap, cache, N)
        _, _, packed = _build_and_pack(q, k, v, spec)
        out_a, lse_a = multilevel_attention_forward(
            q, packed, fmap, cache, block_m=bm, block_n=bn)
        out_b, lse_b = multilevel_attention_forward(
            q, packed, fmap, cache, block_m=bm, block_n=bn,
            position_mode="none")
        torch.testing.assert_close(out_b, out_a, rtol=0, atol=0)
        torch.testing.assert_close(lse_b, lse_a, rtol=0, atol=0)
    print("  none-bitwise            default == position_mode='none' PASS")


def dense_rope_oracle(q, k, v, cos, sin, window):
    """Dense causal RoPE attention with softcap + optional sliding window."""
    B, N, Hq, Dk = q.shape
    Hkv = k.shape[2]
    E = Hq // Hkv
    pos = torch.arange(N, device=q.device)
    q_r = apply_rope(q, pos[None, :, None], cos, sin)
    k_r = apply_rope(k, pos[None, :, None], cos, sin)
    # GQA: each KV head serves its E consecutive query heads.
    k_e = k_r.unsqueeze(3).expand(B, N, Hkv, E, Dk).reshape(B, N, Hq, Dk)
    v_e = v.unsqueeze(3).expand(B, N, Hkv, E, v.shape[-1]).reshape(
        B, N, Hq, v.shape[-1])
    scores = torch.einsum("bqhd,bkhd->bhqk", q_r.float(), k_e.float())
    scores = scores / Dk ** 0.5
    scores = SOFTCAP * torch.tanh(scores / SOFTCAP)
    qi = torch.arange(N, device=q.device)[:, None]
    ki = torch.arange(N, device=q.device)[None, :]
    mask = (ki <= qi) & (ki > qi - window)
    scores = scores.masked_fill(~mask[None, None], -float("inf"))
    lse = torch.logsumexp(scores, dim=-1).permute(0, 2, 1)  # [B, N, Hq]
    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", p, v_e.float())
    return out, lse


def test_rope_l0_dense_equivalence(device):
    torch.manual_seed(30)
    B, N, Hq, Hkv, Dk, Dv = 2, 8, 4, 2, 8, 16
    q = torch.randn(B, N, Hq, Dk, device=device)
    k = torch.randn(B, N, Hkv, Dk, device=device)
    v = torch.randn(B, N, Hkv, Dv, device=device)
    cos, sin = rope_tables(N, Dk, device=device)

    outs = {}
    for label, fmap, cache, window in [
        ("full", {}, 8, 8),
        ("window", {}, 5, 5),
        ("empty-coarse", {1: 8}, 8, 8),
    ]:
        spec = RangeSpec.from_fmap(fmap, cache, N)
        _, _, packed = _build_and_pack(q, k, v, spec)
        out, lse = multilevel_attention_forward(
            q, packed, fmap, cache, block_m=3, block_n=2,
            position_mode="rope", rope_cos=cos, rope_sin=sin)
        ref_out, ref_lse = dense_rope_oracle(q, k, v, cos, sin, window)
        torch.testing.assert_close(out.float(), ref_out,
                                   rtol=FWD_RTOL, atol=FWD_ATOL)
        torch.testing.assert_close(lse, ref_lse,
                                   rtol=FWD_RTOL, atol=FWD_ATOL)
        outs[label] = (out, lse)
    # An empty coarse level must not disturb the rotation of L0.
    torch.testing.assert_close(outs["empty-coarse"][0], outs["full"][0],
                               rtol=0, atol=0)
    print("  rope-l0-dense           full/window/empty-coarse == oracle PASS")


def test_summary_token_position_units(device):
    for (level, j), expected in [
        ((0, 0), 0), ((0, 5), 5), ((1, 0), 1), ((2, 1), 7),
        ((3, 5), 47), ((10, 3), 4095),
    ]:
        got = summary_token_position(level, j)
        if got != expected:
            raise AssertionError(f"position({level},{j}) = {got} != {expected}")
    for bad in [(-1, 0), (0, -1)]:
        try:
            summary_token_position(*bad)
            raise AssertionError(f"no ValueError for {bad}")
        except ValueError:
            pass
    print("  position-units          right-endpoint formula PASS")


def test_rope_helpers_units(device):
    cos, sin = rope_tables(16, 8, device=device)
    assert cos.shape == sin.shape == (16, 4)
    assert cos.dtype == sin.dtype == torch.float32
    torch.manual_seed(40)
    x = torch.randn(3, 5, 8, device=device)
    # Position 0 is the identity rotation.
    pos0 = torch.zeros(3, 5, dtype=torch.long, device=device)
    torch.testing.assert_close(apply_rope(x, pos0, cos, sin), x)
    # Hand-written rotate-half at mixed positions.
    pos = torch.randint(0, 16, (3, 5), device=device)
    x1, x2 = x[..., :4], x[..., 4:]
    c, s = cos[pos], sin[pos]
    ref = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    torch.testing.assert_close(apply_rope(x, pos, cos, sin), ref)
    # Norm preservation (orthogonal rotation).
    torch.testing.assert_close(
        apply_rope(x, pos, cos, sin).norm(dim=-1), x.norm(dim=-1))
    # inverse=True is R(-theta): round-trip is the identity (FP32 x), and
    # pins the inverse independently of the attention backward test.
    x_round = apply_rope(
        apply_rope(x, pos, cos, sin), pos, cos, sin, inverse=True)
    torch.testing.assert_close(x_round, x)
    try:
        rope_tables(8, 7)
        raise AssertionError("no ValueError for odd dim")
    except ValueError:
        pass
    print("  rope-units              tables + rotate-half + norms PASS")


def test_relative_bins_vs_oracle(device):
    for N, fmap, cache in BIN_CONFIGS:
        spec = RangeSpec.from_fmap(fmap, cache, N)
        for qi in range(N):
            entries = visible_entries_chronological(fmap, cache, qi)
            M = len(entries)
            # Partition property: disjoint, contiguous, ends at qi.
            for a, b in zip(entries, entries[1:]):
                if a[3] != b[2]:
                    raise AssertionError(
                        f"N={N} q={qi}: intervals not contiguous/disjoint "
                        f"at {a} -> {b}")
            if entries[-1][3] != qi + 1:
                raise AssertionError(f"N={N} q={qi}: coverage does not end at q")
            for rank, (level, j, _, t_end) in enumerate(entries):
                expected = (M - 1) - rank
                got = relative_bin_distance(spec, qi, level, j)
                if got != expected:
                    raise AssertionError(
                        f"N={N} q={qi} entry (L{level},{j}): distance "
                        f"{got} != {expected}")
                # V1: visible summaries are entirely past-or-current.
                if summary_token_position(level, j) > qi:
                    raise AssertionError(
                        f"N={N} q={qi}: (L{level},{j}) reaches the future")
    # Golden spot-check (readme_tiny, q=7): own token = 0.
    spec = RangeSpec.from_fmap({1: 2, 2: 3}, 6, 8)
    golden = [((2, 0), 3), ((1, 2), 2), ((0, 6), 1), ((0, 7), 0)]
    for (level, j), expected in golden:
        assert relative_bin_distance(spec, 7, level, j) == expected, (level, j)
    print("  relative-bins           analytic == slow oracle (3 configs) PASS")


def test_relative_span_independence(device):
    checked = 0
    for N, fmap, cache in BIN_CONFIGS:
        spec = RangeSpec.from_fmap(fmap, cache, N)
        for qi in range(N):
            entries = visible_entries_chronological(fmap, cache, qi)
            for a, b in zip(entries, entries[1:]):
                if a[0] != b[0]:  # adjacent bins from different levels
                    da = relative_bin_distance(spec, qi, a[0], a[1])
                    db = relative_bin_distance(spec, qi, b[0], b[1])
                    if da - db != 1:
                        raise AssertionError(
                            f"N={N} q={qi}: cross-level step {a}->{b} has "
                            f"distance delta {da - db} != 1")
                    checked += 1
    assert checked > 0, "no cross-level adjacencies exercised"
    print(f"  span-independence       {checked} cross-level steps, delta=1 PASS")


def test_relative_dense_oracle(device):
    q, k, v, spec, fmap, cache = _tiny_case(device, seed=50)
    B, N, Hq, Dk = q.shape
    Hkv, Dv = k.shape[2], v.shape[-1]
    E = Hq // Hkv
    d_rel, max_bins = 6, 8
    torch.manual_seed(51)
    states = torch.randn(B, N, Hq, d_rel, device=device)
    proj = torch.randn(d_rel, max_bins, device=device) / d_rel ** 0.5

    kl, vl, packed = _build_and_pack(q, k, v, spec)
    out, lse = multilevel_attention_forward(
        q, packed, fmap, cache, block_m=3, block_n=2,
        position_mode="relative", relative_states=states, relative_proj=proj)

    rel_logits = states.float() @ proj.float()  # [B, N, Hq, bins]
    scale = 1.0 / Dk ** 0.5
    for b in range(B):
        for qi in range(N):
            entries = visible_entries_chronological(fmap, cache, qi)
            M = len(entries)
            for hq in range(Hq):
                hkv = hq // E
                scores = []
                vals = []
                for rank, (level, j, _, _) in enumerate(entries):
                    kv_k = kl[level][b, j, hkv].float()
                    kv_v = vl[level][b, j, hkv].float()
                    d = (M - 1) - rank
                    s = (q[b, qi, hq].float() @ kv_k) * scale
                    s = s + rel_logits[b, qi, hq, d]
                    s = SOFTCAP * torch.tanh(s / SOFTCAP)
                    scores.append(s)
                    vals.append(kv_v)
                scores = torch.stack(scores)
                vals = torch.stack(vals)
                ref_lse = torch.logsumexp(scores, dim=0)
                p = torch.softmax(scores, dim=0)
                ref_out = (p[:, None] * vals).sum(0)
                torch.testing.assert_close(
                    out[b, qi, hq].float(), ref_out,
                    rtol=FWD_RTOL, atol=FWD_ATOL)
                torch.testing.assert_close(
                    lse[b, qi, hq], ref_lse, rtol=FWD_RTOL, atol=FWD_ATOL)
    print("  relative-dense          bias-in-logit oracle == forward PASS")


def test_relative_gradient_flow(device):
    q0, k0, v0, spec, fmap, cache = _tiny_case(device, seed=60)
    B, N, Hq = q0.shape[0], q0.shape[1], q0.shape[2]
    emb_dim, d_rel, max_bins = 24, 6, 8
    torch.manual_seed(61)
    x = torch.randn(B, N, emb_dim, device=device, requires_grad=True)
    rw = (torch.randn(Hq * d_rel, emb_dim, device=device)
          / emb_dim ** 0.5).requires_grad_(True)
    proj = (torch.randn(d_rel, max_bins, device=device)
            / d_rel ** 0.5).requires_grad_(True)
    q, k, v = (t.detach().clone().requires_grad_(True)
               for t in (q0, k0, v0))
    states = compute_relative_states(x, rw, Hq)
    _, _, packed = _build_and_pack(q, k, v, spec)
    out, _ = multilevel_attention_forward(
        q, packed, fmap, cache, block_m=3, block_n=2,
        position_mode="relative", relative_states=states, relative_proj=proj)
    g = torch.Generator(device="cpu").manual_seed(62)
    dout = torch.randn(out.shape, generator=g).to(device)
    grads = torch.autograd.grad(out, (q, k, v, x, rw, proj), dout)
    for nm, grad in zip(("dq", "dk", "dv", "dx", "drel_w", "drel_proj"),
                        grads):
        if not bool(torch.isfinite(grad).all()):
            raise AssertionError(f"{nm}: non-finite")
        if grad.norm().item() <= GRAD_MIN_NORM:
            raise AssertionError(f"{nm}: no gradient")
    # RoPE mode: gradients reach q and k through the rotation.
    q, k, v = (t.detach().clone().requires_grad_(True)
               for t in (q0, k0, v0))
    cos, sin = rope_tables(N, q0.shape[-1], device=device)
    _, _, packed = _build_and_pack(q, k, v, spec)
    out, _ = multilevel_attention_forward(
        q, packed, fmap, cache, block_m=3, block_n=2,
        position_mode="rope", rope_cos=cos, rope_sin=sin)
    grads = torch.autograd.grad(out, (q, k, v), dout)
    for nm, grad in zip(("dq", "dk", "dv"), grads):
        if not bool(torch.isfinite(grad).all()) or grad.norm() <= GRAD_MIN_NORM:
            raise AssertionError(f"rope {nm}: missing/non-finite gradient")
    print("  gradient-flow           relative + rope grads finite/nonzero PASS")


def test_tree_mode_independence(device):
    q, k, v, spec, fmap, cache = _tiny_case(device, seed=70)
    N, Hq, Dk = q.shape[1], q.shape[2], q.shape[-1]
    _, _, packed = _build_and_pack(q, k, v, spec)
    k_snap, v_snap = packed.k.clone(), packed.v.clone()
    cos, sin = rope_tables(N, Dk, device=device)
    torch.manual_seed(71)
    states = torch.randn(*q.shape[:2], Hq, 6, device=device)
    proj = torch.randn(6, 8, device=device)
    for kwargs in (
        dict(position_mode="none"),
        dict(position_mode="rope", rope_cos=cos, rope_sin=sin),
        dict(position_mode="relative", relative_states=states,
             relative_proj=proj),
    ):
        multilevel_attention_forward(
            q, packed, fmap, cache, block_m=3, block_n=2, **kwargs)
        if not (torch.equal(packed.k, k_snap)
                and torch.equal(packed.v, v_snap)):
            raise AssertionError(
                f"{kwargs['position_mode']}: packed K/V mutated")
    print("  tree-independence       packed K/V untouched by all modes PASS")


def _rope_backward_case(device, name, B, N, Hq, Hkv, Dk, Dv, fmap, cache,
                        bm, bn, softcap, seed):
    """
    One packed-boundary comparison (the part2_phase2 pattern from
    test_backward.py): K/V leaves are the packed buffers themselves, so
    autograd yields dk_packed/dv_packed with no summary-tree path, and the
    explicit rope backward must match them.
    """
    torch.manual_seed(seed)
    q = torch.randn(B, N, Hq, Dk, device=device)
    k = torch.randn(B, N, Hkv, Dk, device=device)
    v = torch.randn(B, N, Hkv, Dv, device=device)
    spec = RangeSpec.from_fmap(fmap, cache, N)
    cos, sin = rope_tables(N, Dk, device=device)
    with torch.no_grad():
        _, _, packed = _build_and_pack(q, k, v, spec)

    q_leaf = q.detach().clone().requires_grad_(True)
    k_leaf = packed.k.detach().clone().requires_grad_(True)
    v_leaf = packed.v.detach().clone().requires_grad_(True)
    packed_leaf = PackedKV(k_leaf, v_leaf, packed.level_offsets)

    out, lse = multilevel_attention_forward(
        q_leaf, packed_leaf, fmap, cache, block_m=bm, block_n=bn,
        softcap=softcap, position_mode="rope", rope_cos=cos, rope_sin=sin)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    dout = torch.randn(out.shape, generator=g).to(device)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        out, (q_leaf, k_leaf, v_leaf), dout)

    packed_detached = PackedKV(
        k_leaf.detach(), v_leaf.detach(), packed.level_offsets)
    dq, dk_p, dv_p, dposition, _ = multilevel_attention_backward(
        q_leaf.detach(), packed_detached, out.detach(), lse.detach(), dout,
        fmap, cache, block_m=bm, block_n=bn, softcap=softcap,
        position_mode="rope", rope_cos=cos, rope_sin=sin)
    if dposition is not None:
        raise AssertionError(
            f"{name}: dposition must be None outside relative mode")

    for nm, got, ref in [(f"dq/{name}", dq, dq_ref),
                         (f"dk-packed/{name}", dk_p, dk_ref),
                         (f"dv-packed/{name}", dv_p, dv_ref)]:
        if got.shape != ref.shape or got.dtype != ref.dtype:
            raise AssertionError(
                f"{nm}: shape/dtype {tuple(got.shape)}/{got.dtype} != "
                f"{tuple(ref.shape)}/{ref.dtype}")
        if not bool(torch.isfinite(got).all()):
            raise AssertionError(f"{nm}: non-finite entries")
        torch.testing.assert_close(got, ref, rtol=GRAD_RTOL, atol=GRAD_ATOL)

    return (q_leaf, packed_detached, out, lse, dout, dq_ref, fmap, cache,
            bm, bn, softcap)


def test_rope_explicit_backward(device):
    # Tiny: misaligned blocks. Eviction: full cache, coarsest-level
    # eviction, partial M/N tiles, GQA, Dv != Dk (test_forward's
    # `eviction` geometry; the N=128/cache=512 baseline never evicts).
    tiny = dict(B=2, N=8, Hq=4, Hkv=2, Dk=8, Dv=16,
                fmap={1: 2, 2: 3}, cache=6, bm=3, bn=2)
    evic = dict(B=1, N=1024, Hq=4, Hkv=2, Dk=32, Dv=64,
                fmap={1: 64, 2: 72, 3: 80}, cache=160, bm=24, bn=40)

    ctx = _rope_backward_case(device, "tiny", softcap=SOFTCAP, seed=90,
                              **tiny)
    _rope_backward_case(device, "tiny/nocap", softcap=None, seed=92, **tiny)
    _rope_backward_case(device, "eviction", softcap=SOFTCAP, seed=94, **evic)

    # Negative control: the position_mode="none" backward on the same rope
    # forward outputs must NOT match -- proves the comparison has power.
    (q_leaf, packed_detached, out, lse, dout, dq_ref, fmap, cache,
     bm, bn, softcap) = ctx
    dq_bad, _, _, _, _ = multilevel_attention_backward(
        q_leaf.detach(), packed_detached, out.detach(), lse.detach(), dout,
        fmap, cache, block_m=bm, block_n=bn, softcap=softcap,
        position_mode="none")
    err = (dq_bad - dq_ref).abs().max().item()
    if err <= GRAD_ATOL:
        raise AssertionError(
            f"negative control: unrotated backward still matched rope "
            f"autograd (max err {err:.3e})")
    print("  rope-explicit-backward  tiny/nocap/eviction == autograd "
          "(+neg control) PASS")


def _relative_backward_case(device, name, B, N, Hq, Hkv, Dk, Dv, fmap, cache,
                            bm, bn, d_rel, max_bins, softcap, seed):
    """
    Packed-boundary comparison for the relative mode: q, packed K/V,
    relative_states and relative_proj are all leaves, so one autograd call
    yields every reference; the explicit backward's drel_logits seed is
    composed to dstates/dproj via the closed-form VJPs of
    rel_logits = states @ proj and must match.
    """
    torch.manual_seed(seed)
    q = torch.randn(B, N, Hq, Dk, device=device)
    k = torch.randn(B, N, Hkv, Dk, device=device)
    v = torch.randn(B, N, Hkv, Dv, device=device)
    states = torch.randn(B, N, Hq, d_rel, device=device)
    proj = torch.randn(d_rel, max_bins, device=device) / d_rel ** 0.5
    spec = RangeSpec.from_fmap(fmap, cache, N)
    with torch.no_grad():
        _, _, packed = _build_and_pack(q, k, v, spec)

    q_leaf = q.detach().clone().requires_grad_(True)
    k_leaf = packed.k.detach().clone().requires_grad_(True)
    v_leaf = packed.v.detach().clone().requires_grad_(True)
    states_leaf = states.detach().clone().requires_grad_(True)
    proj_leaf = proj.detach().clone().requires_grad_(True)
    packed_leaf = PackedKV(k_leaf, v_leaf, packed.level_offsets)

    out, lse = multilevel_attention_forward(
        q_leaf, packed_leaf, fmap, cache, block_m=bm, block_n=bn,
        softcap=softcap, position_mode="relative",
        relative_states=states_leaf, relative_proj=proj_leaf)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    dout = torch.randn(out.shape, generator=g).to(device)
    dq_ref, dk_ref, dv_ref, ds_ref, dp_ref = torch.autograd.grad(
        out, (q_leaf, k_leaf, v_leaf, states_leaf, proj_leaf), dout)

    packed_detached = PackedKV(
        k_leaf.detach(), v_leaf.detach(), packed.level_offsets)
    dq, dk_p, dv_p, drel, _ = multilevel_attention_backward(
        q_leaf.detach(), packed_detached, out.detach(), lse.detach(), dout,
        fmap, cache, block_m=bm, block_n=bn, softcap=softcap,
        position_mode="relative",
        relative_states=states_leaf.detach(),
        relative_proj=proj_leaf.detach())

    if (drel is None or tuple(drel.shape) != (B, N, Hq, max_bins)
            or drel.dtype != torch.float32):
        raise AssertionError(
            f"{name}: dposition must be float32 [B, N, Hq, max_bins], got "
            f"{None if drel is None else (tuple(drel.shape), drel.dtype)}")
    # Closed-form composition of the drel_logits seed through
    # rel_logits = states @ proj.
    dstates = torch.einsum("bnhr,dr->bnhd", drel, proj.float())
    dproj = torch.einsum("bnhd,bnhr->dr", states.float(), drel)

    for nm, got, ref in [(f"dq/{name}", dq, dq_ref),
                         (f"dk-packed/{name}", dk_p, dk_ref),
                         (f"dv-packed/{name}", dv_p, dv_ref),
                         (f"dstates/{name}", dstates, ds_ref),
                         (f"dproj/{name}", dproj, dp_ref)]:
        if got.shape != ref.shape or got.dtype != ref.dtype:
            raise AssertionError(
                f"{nm}: shape/dtype {tuple(got.shape)}/{got.dtype} != "
                f"{tuple(ref.shape)}/{ref.dtype}")
        if not bool(torch.isfinite(got).all()):
            raise AssertionError(f"{nm}: non-finite entries")
        torch.testing.assert_close(got, ref, rtol=GRAD_RTOL, atol=GRAD_ATOL)

    return (q_leaf, packed_detached, out, lse, dout, dq_ref, fmap, cache,
            bm, bn, softcap)


def test_relative_explicit_backward(device):
    # Same geometry rationale as the rope test: tiny = misaligned blocks;
    # eviction = full cache, coarsest-level eviction, partial tiles, GQA,
    # Dv != Dk -- with max_bins = cache_size (always sufficient).
    tiny = dict(B=2, N=8, Hq=4, Hkv=2, Dk=8, Dv=16,
                fmap={1: 2, 2: 3}, cache=6, bm=3, bn=2,
                d_rel=6, max_bins=8)
    evic = dict(B=1, N=1024, Hq=4, Hkv=2, Dk=32, Dv=64,
                fmap={1: 64, 2: 72, 3: 80}, cache=160, bm=24, bn=40,
                d_rel=8, max_bins=160)

    ctx = _relative_backward_case(device, "tiny", softcap=SOFTCAP, seed=100,
                                  **tiny)
    _relative_backward_case(device, "tiny/nocap", softcap=None, seed=102,
                            **tiny)
    _relative_backward_case(device, "eviction", softcap=SOFTCAP, seed=104,
                            **evic)

    # Negative control: the position_mode="none" backward on the same
    # relative forward outputs must NOT match.
    (q_leaf, packed_detached, out, lse, dout, dq_ref, fmap, cache,
     bm, bn, softcap) = ctx
    dq_bad, _, _, dposition, _ = multilevel_attention_backward(
        q_leaf.detach(), packed_detached, out.detach(), lse.detach(), dout,
        fmap, cache, block_m=bm, block_n=bn, softcap=softcap,
        position_mode="none")
    if dposition is not None:
        raise AssertionError("dposition must be None outside relative mode")
    err = (dq_bad - dq_ref).abs().max().item()
    if err <= GRAD_ATOL:
        raise AssertionError(
            f"negative control: bias-free backward still matched relative "
            f"autograd (max err {err:.3e})")
    print("  relative-explicit-bwd   tiny/nocap/eviction == autograd "
          "(+neg control) PASS")


def _positional_sink_case(device, mode, softcap, seed, neg_control):
    """
    Sink-dlse composition under one positional mode, at the packed
    boundary:

        dout_pre, dlse, dsinks = attention_sink_backward(out, lse, sinks, g)
        multilevel_attention_backward(..., dout_pre, dlse=dlse,
                                      position_mode=mode, ...)

    must reproduce autograd through [forward -> apply_attention_sink],
    including the lse -> logits path (and, for relative, its flow into the
    drel_logits bins via db = dX). The dlse fold itself is per-row and
    happens before any tile loop, so the tiny geometry fully exercises
    what is new here: its interaction with the rotation/bias paths.
    """
    B, N, Hq, Hkv, Dk, Dv = 2, 8, 4, 2, 8, 16
    fmap, cache = {1: 2, 2: 3}, 6
    bm, bn = 3, 2
    d_rel, max_bins = 6, 8
    torch.manual_seed(seed)
    q = torch.randn(B, N, Hq, Dk, device=device)
    k = torch.randn(B, N, Hkv, Dk, device=device)
    v = torch.randn(B, N, Hkv, Dv, device=device)
    spec = RangeSpec.from_fmap(fmap, cache, N)
    with torch.no_grad():
        _, _, packed = _build_and_pack(q, k, v, spec)

    q_leaf = q.detach().clone().requires_grad_(True)
    k_leaf = packed.k.detach().clone().requires_grad_(True)
    v_leaf = packed.v.detach().clone().requires_grad_(True)
    packed_leaf = PackedKV(k_leaf, v_leaf, packed.level_offsets)

    if mode == "rope":
        cos, sin = rope_tables(N, Dk, device=device)
        fw_kwargs = dict(rope_cos=cos, rope_sin=sin)
        bw_kwargs = dict(rope_cos=cos, rope_sin=sin)
        extra_leaves = ()
    else:
        states = torch.randn(B, N, Hq, d_rel, device=device)
        proj = torch.randn(d_rel, max_bins, device=device) / d_rel ** 0.5
        states_leaf = states.detach().clone().requires_grad_(True)
        proj_leaf = proj.detach().clone().requires_grad_(True)
        fw_kwargs = dict(relative_states=states_leaf,
                         relative_proj=proj_leaf)
        bw_kwargs = dict(relative_states=states_leaf.detach(),
                         relative_proj=proj_leaf.detach())
        extra_leaves = (states_leaf, proj_leaf)

    out, lse = multilevel_attention_forward(
        q_leaf, packed_leaf, fmap, cache, block_m=bm, block_n=bn,
        softcap=softcap, position_mode=mode, **fw_kwargs)

    # Sinks centered on THIS mode's lse: r ~ 0.5, r(1-r) ~ 0.25, so the
    # dlse branch is maximally load-bearing (test_sink_gate's rationale;
    # the real forward already ran, no probe needed).
    torch.manual_seed(seed + 1)
    sinks0 = (lse.detach().float().mean(dim=(0, 1))
              + 0.1 * torch.randn(Hq, device=device))
    sinks_leaf = sinks0.detach().clone().requires_grad_(True)

    out_post = apply_attention_sink(out, lse, sinks_leaf)
    g = torch.Generator(device="cpu").manual_seed(seed + 2)
    dout_post = torch.randn(out_post.shape, generator=g).to(device)
    grads = torch.autograd.grad(
        out_post, (q_leaf, k_leaf, v_leaf) + extra_leaves + (sinks_leaf,),
        dout_post)
    dq_ref, dk_ref, dv_ref = grads[:3]
    dsinks_ref = grads[-1]

    dout_pre, dlse, dsinks = attention_sink_backward(
        out.detach(), lse.detach(), sinks0.detach(), dout_post)
    packed_detached = PackedKV(
        k_leaf.detach(), v_leaf.detach(), packed.level_offsets)
    dq, dk_p, dv_p, dposition, _ = multilevel_attention_backward(
        q_leaf.detach(), packed_detached, out.detach(), lse.detach(),
        dout_pre, fmap, cache, block_m=bm, block_n=bn, softcap=softcap,
        dlse=dlse, position_mode=mode, **bw_kwargs)

    name = f"{mode}/cap{softcap:g}"
    checks = [(f"dq/{name}", dq, dq_ref),
              (f"dk-packed/{name}", dk_p, dk_ref),
              (f"dv-packed/{name}", dv_p, dv_ref),
              (f"dsinks/{name}", dsinks, dsinks_ref)]
    if mode == "rope":
        if dposition is not None:
            raise AssertionError(
                f"{name}: dposition must be None outside relative mode")
    else:
        ds_ref, dp_ref = grads[3:5]
        dstates = torch.einsum("bnhr,dr->bnhd", dposition, proj.float())
        dproj = torch.einsum("bnhd,bnhr->dr", states.float(), dposition)
        checks += [(f"dstates/{name}", dstates, ds_ref),
                   (f"dproj/{name}", dproj, dp_ref)]
    for nm, got, ref in checks:
        if not bool(torch.isfinite(got).all()):
            raise AssertionError(f"{nm}: non-finite entries")
        torch.testing.assert_close(got, ref, rtol=GRAD_RTOL, atol=GRAD_ATOL)

    if not neg_control:
        return
    # Negative control: same positional mode, dlse dropped -- the ONLY
    # difference. The missing seed must visibly perturb Q or K, and in
    # relative mode the positional path itself.
    dq_bad, dk_bad, _, dpos_bad, _ = multilevel_attention_backward(
        q_leaf.detach(), packed_detached, out.detach(), lse.detach(),
        dout_pre, fmap, cache, block_m=bm, block_n=bn, softcap=softcap,
        position_mode=mode, **bw_kwargs)
    err_q = (dq_bad - dq_ref).abs().max().item()
    err_k = (dk_bad - dk_ref).abs().max().item()
    if max(err_q, err_k) <= GRAD_ATOL:
        raise AssertionError(
            f"{name} negative control: dropping dlse still matched "
            f"dq/dk (max err {max(err_q, err_k):.3e})")
    if mode == "relative":
        dstates_bad = torch.einsum(
            "bnhr,dr->bnhd", dpos_bad, proj.float())
        err_pos = (dstates_bad - ds_ref).abs().max().item()
        if err_pos <= GRAD_ATOL:
            raise AssertionError(
                f"{name} negative control: dropping dlse still matched "
                f"the relative-position gradient path (max err "
                f"{err_pos:.3e})")


def test_positional_backward_with_sink(device):
    _positional_sink_case(device, "rope", 20.0, seed=110, neg_control=True)
    _positional_sink_case(device, "relative", 20.0, seed=114,
                          neg_control=True)
    # Stronger softcap: 1 - tanh^2(X/c) visibly below 1, so the
    # dLSE -> dS -> dX -> db chain is diagnostic of db = dX (not dS).
    _positional_sink_case(device, "relative", 2.0, seed=118,
                          neg_control=False)
    print("  positional-bwd-sink     rope/relative dlse composition == "
          "autograd (+neg controls) PASS")


def _e2e_case(device, mode, name, B, N, Hq, Hkv, Dk, Dv, fmap, cache,
              bm, bn, d_rel, max_bins, softcap, seed):
    """
    Phase-2 + Phase-1 composition (test_backward.part3_composition, for a
    positional mode): tree graph kept alive, forward under no_grad,
    explicit dk_p/dv_p fed back through (packed.k, packed.v) as VJP seeds,

        dq_total = dq_attn + dq_tree    (dq_tree = the q -> w merge path)

    must equal end-to-end autograd through [tree -> forward]. Relative
    states/proj feed the forward directly (never the tree), so their
    gradients come purely from the dposition seed's closed forms.
    """
    torch.manual_seed(seed)
    q0 = torch.randn(B, N, Hq, Dk, device=device)
    k0 = torch.randn(B, N, Hkv, Dk, device=device)
    v0 = torch.randn(B, N, Hkv, Dv, device=device)
    if mode == "relative":
        states0 = torch.randn(B, N, Hq, d_rel, device=device)
        proj0 = torch.randn(d_rel, max_bins, device=device) / d_rel ** 0.5
    else:
        cos, sin = rope_tables(N, Dk, device=device)
    spec = RangeSpec.from_fmap(fmap, cache, N)

    # Base tensors drawn ONCE; both paths take literal clones so draw-order
    # changes can never silently diverge them.
    def leaves():
        ts = [t.detach().clone().requires_grad_(True) for t in (q0, k0, v0)]
        if mode == "relative":
            ts += [states0.detach().clone().requires_grad_(True),
                   proj0.detach().clone().requires_grad_(True)]
        return ts

    def mode_kwargs(s, p):
        if mode == "rope":
            return dict(rope_cos=cos, rope_sin=sin)
        return dict(relative_states=s, relative_proj=p)

    # --- end-to-end autograd oracle (tree + forward, one graph) ----------
    oracle_leaves = leaves()
    q_a, k_a, v_a = oracle_leaves[:3]
    s_a, p_a = (oracle_leaves[3:5] if mode == "relative" else (None, None))
    _, _, packed_a = _build_and_pack(q_a, k_a, v_a, spec)
    out_a, _ = multilevel_attention_forward(
        q_a, packed_a, fmap, cache, block_m=bm, block_n=bn, softcap=softcap,
        position_mode=mode, **mode_kwargs(s_a, p_a))
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    dout = torch.randn(out_a.shape, generator=g).to(device)
    oracle = torch.autograd.grad(out_a, tuple(oracle_leaves), dout)
    dq_ref, dk_ref, dv_ref = oracle[:3]

    # --- explicit path: graph-intact tree, no_grad forward + Phase 2 -----
    expl_leaves = leaves()
    q_l, k_l, v_l = expl_leaves[:3]
    s_l, p_l = (expl_leaves[3:5] if mode == "relative" else (None, None))
    _, _, packed_g = _build_and_pack(q_l, k_l, v_l, spec)
    with torch.no_grad():
        out, lse = multilevel_attention_forward(
            q_l, packed_g, fmap, cache, block_m=bm, block_n=bn,
            softcap=softcap, position_mode=mode, **mode_kwargs(s_l, p_l))
    dq_attn, dk_p, dv_p, dpos, _ = multilevel_attention_backward(
        q_l, packed_g, out, lse, dout, fmap, cache,
        block_m=bm, block_n=bn, softcap=softcap, position_mode=mode,
        **mode_kwargs(None if s_l is None else s_l.detach(),
                      None if p_l is None else p_l.detach()))
    dq_tree, dk, dv = torch.autograd.grad(
        (packed_g.k, packed_g.v), (q_l, k_l, v_l), (dk_p, dv_p))
    dq_total = dq_attn + dq_tree

    # q -> w path must exist AND be load-bearing: dq_attn alone (tree
    # contribution dropped) must visibly fail the end-to-end comparison.
    if dq_tree.norm().item() <= GRAD_MIN_NORM:
        raise AssertionError(f"{name}: q->w tree path carries no gradient")
    err_no_tree = (dq_attn - dq_ref).abs().max().item()
    if err_no_tree <= GRAD_ATOL:
        raise AssertionError(
            f"{name}: dropping the q->w tree contribution still matched "
            f"end-to-end dq (max err {err_no_tree:.3e})")

    checks = [(f"dq-e2e/{name}", dq_total, dq_ref),
              (f"dk-e2e/{name}", dk, dk_ref),
              (f"dv-e2e/{name}", dv, dv_ref)]
    if mode == "rope":
        if dpos is not None:
            raise AssertionError(
                f"{name}: dposition must be None outside relative mode")
    else:
        ds_ref, dp_ref = oracle[3:5]
        dstates = torch.einsum("bnhr,dr->bnhd", dpos, proj0.float())
        dproj = torch.einsum("bnhd,bnhr->dr", states0.float(), dpos)
        checks += [(f"dstates-e2e/{name}", dstates, ds_ref),
                   (f"dproj-e2e/{name}", dproj, dp_ref)]
    for nm, got, ref in checks:
        if not bool(torch.isfinite(got).all()):
            raise AssertionError(f"{nm}: non-finite entries")
        torch.testing.assert_close(got, ref, rtol=GRAD_RTOL, atol=GRAD_ATOL)


def test_positional_e2e_composition(device):
    tiny = dict(B=2, N=8, Hq=4, Hkv=2, Dk=8, Dv=16,
                fmap={1: 2, 2: 3}, cache=6, bm=3, bn=2,
                d_rel=6, max_bins=8)
    evic = dict(B=1, N=1024, Hq=4, Hkv=2, Dk=32, Dv=64,
                fmap={1: 64, 2: 72, 3: 80}, cache=160, bm=24, bn=40,
                d_rel=8, max_bins=160)
    _e2e_case(device, "rope", "rope/tiny", softcap=SOFTCAP, seed=120, **tiny)
    _e2e_case(device, "relative", "relative/tiny", softcap=SOFTCAP,
              seed=124, **tiny)
    _e2e_case(device, "rope", "rope/eviction", softcap=SOFTCAP,
              seed=128, **evic)
    _e2e_case(device, "relative", "relative/eviction", softcap=SOFTCAP,
              seed=132, **evic)
    print("  positional-e2e          explicit + tree VJP == autograd "
          "(4 cases, +q->w controls) PASS")


def _full_block_case(device, mode, weight_mode, seed):
    """
    Full attention-block composition, fms/train.py's canonical op order
    with reference-semantics positional encoding:

        x -> Wq/Wk/Wv -> K/V short conv (flat channels, inside
        build_dyadic_summaries) -> summary weights + tree -> positional
        attention -> sink -> SiLU gate (from the ORIGINAL x) -> o_proj

    Explicit composition = the two explicit backwards reference.py
    provides (Phase-2 attention, attention sink) + autograd VJPs at the
    documented seams (Phase 1 upstream; gate + o_proj downstream), checked
    against pure autograd for ALL trainable parameters plus x. dx is a
    genuine multi-path sum: projections (+ w_proj / rel_w per run)
    upstream, gate downstream.
    """
    B, N, emb, Hq, Hkv, Dk, Dv = 2, 8, 32, 4, 2, 8, 16
    fmap, cache = {1: 2, 2: 3}, 6
    bm, bn = 3, 2
    Kc = 4
    d_rel, max_bins = 6, 8
    spec = RangeSpec.from_fmap(fmap, cache, N)
    L = spec.num_levels - 1
    name = f"{mode}/{weight_mode}"

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
        base["rel_w"] = (torch.randn(Hq * d_rel, emb, device=device)
                         / emb ** 0.5)
        base["rel_proj"] = (torch.randn(d_rel, max_bins, device=device)
                            / d_rel ** 0.5)
    if mode == "rope":
        cos, sin = rope_tables(N, Dk, device=device)

    def chain(t, attn_no_grad=False):
        x = t["x"]
        q4 = x.matmul(t["Wq"].t()).view(B, N, Hq, Dk)
        k4 = x.matmul(t["Wk"].t()).view(B, N, Hkv, Dk)
        v4 = x.matmul(t["Wv"].t()).view(B, N, Hkv, Dv)
        bd_kwargs = dict(k_conv_weight=t["kcw"], v_conv_weight=t["vcw"])
        if weight_mode == "linear":
            bd_kwargs.update(x=x, w_proj=t["w_proj"])
        kl, vl, _ = build_dyadic_summaries(q4, k4, v4, L, **bd_kwargs)
        packed = pack_levels(kl, vl)
        states = None
        fw_kwargs = {}
        if mode == "rope":
            fw_kwargs = dict(rope_cos=cos, rope_sin=sin)
        elif mode == "relative":
            states = compute_relative_states(x, t["rel_w"], Hq)
            fw_kwargs = dict(relative_states=states,
                             relative_proj=t["rel_proj"])
        if attn_no_grad:
            with torch.no_grad():
                out, lse = multilevel_attention_forward(
                    q4, packed, fmap, cache, block_m=bm, block_n=bn,
                    softcap=SOFTCAP, position_mode=mode, **fw_kwargs)
        else:
            out, lse = multilevel_attention_forward(
                q4, packed, fmap, cache, block_m=bm, block_n=bn,
                softcap=SOFTCAP, position_mode=mode, **fw_kwargs)
        return dict(q4=q4, packed=packed, states=states, out=out, lse=lse)

    # Sinks centered on this configuration's lse (probe on base tensors,
    # values only): r ~ 0.5 keeps the dlse branch load-bearing.
    with torch.no_grad():
        probe = chain(base, attn_no_grad=True)
    torch.manual_seed(seed + 1)
    base["sinks"] = (probe["lse"].float().mean(dim=(0, 1))
                     + 0.1 * torch.randn(Hq, device=device))

    # Base tensors drawn once; both paths take literal clones.
    def leaves():
        return {k: v.detach().clone().requires_grad_(True)
                for k, v in base.items()}

    keys = list(base.keys())

    # --- pure-autograd oracle, one graph over the whole block ------------
    a = leaves()
    ca = chain(a)
    out_post_a = apply_attention_sink(ca["out"], ca["lse"], a["sinks"])
    gated_a = apply_output_gate(out_post_a, a["x"], a["gate_w"])
    final_a = gated_a.reshape(B, N, Hq * Dv).matmul(a["o_w"].t())
    g = torch.Generator(device="cpu").manual_seed(seed + 2)
    dfinal = torch.randn(final_a.shape, generator=g).to(device)
    oracle = dict(zip(keys, torch.autograd.grad(
        final_a, tuple(a[k] for k in keys), dfinal)))

    # --- explicit composition -------------------------------------------
    e = leaves()
    ce = chain(e, attn_no_grad=True)   # Phase-1 graph alive, attention not
    out, lse = ce["out"], ce["lse"]

    # Downstream VJP (gate + o_proj) via autograd on a small subgraph; the
    # shared x leaf yields only its gate contribution here.
    out_sink_leaf = apply_attention_sink(
        out, lse, e["sinks"].detach()).detach().requires_grad_(True)
    gated_e = apply_output_gate(out_sink_leaf, e["x"], e["gate_w"])
    final_e = gated_e.reshape(B, N, Hq * Dv).matmul(e["o_w"].t())
    if not torch.equal(final_e, final_a):
        raise AssertionError(f"{name}: forward paths diverged (bitwise)")
    dout_sink, dx_gate, dgate_w, do_w = torch.autograd.grad(
        final_e, (out_sink_leaf, e["x"], e["gate_w"], e["o_w"]), dfinal)

    # Explicit sink + explicit Phase-2 attention backward.
    dout_pre, dlse, dsinks = attention_sink_backward(
        out, lse, e["sinks"].detach(), dout_sink)
    bw_kwargs = {}
    if mode == "rope":
        bw_kwargs = dict(rope_cos=cos, rope_sin=sin)
    elif mode == "relative":
        bw_kwargs = dict(relative_states=ce["states"].detach(),
                         relative_proj=e["rel_proj"].detach())
    dq_attn, dk_p, dv_p, dpos, _ = multilevel_attention_backward(
        ce["q4"], ce["packed"], out, lse, dout_pre, fmap, cache,
        block_m=bm, block_n=bn, softcap=SOFTCAP, dlse=dlse,
        position_mode=mode, **bw_kwargs)

    # Upstream VJP in one call: seeds at the documented boundaries.
    up_outputs = [ce["packed"].k, ce["packed"].v, ce["q4"]]
    up_seeds = [dk_p, dv_p, dq_attn]
    up_keys = ["x", "Wq", "Wk", "Wv", "kcw", "vcw"]
    if weight_mode == "linear":
        up_keys.append("w_proj")
    if mode == "relative":
        if dpos is None:
            raise AssertionError(f"{name}: missing dposition")
        dstates = torch.einsum(
            "bnhr,dr->bnhd", dpos, e["rel_proj"].detach().float())
        up_outputs.append(ce["states"])
        up_seeds.append(dstates)
        up_keys.append("rel_w")
    else:
        if dpos is not None:
            raise AssertionError(
                f"{name}: dposition must be None outside relative mode")
    up = dict(zip(up_keys, torch.autograd.grad(
        tuple(up_outputs), tuple(e[k] for k in up_keys), tuple(up_seeds),
        retain_graph=(mode == "rope"))))

    got = dict(up)
    got["x"] = up["x"] + dx_gate      # multi-path sum: upstream + gate
    got["gate_w"] = dgate_w
    got["o_w"] = do_w
    got["sinks"] = dsinks
    if mode == "relative":
        got["rel_proj"] = torch.einsum(
            "bnhd,bnhr->dr", ce["states"].detach().float(), dpos)

    for kname in keys:
        ref, gg = oracle[kname], got[kname]
        if not bool(torch.isfinite(gg).all()):
            raise AssertionError(f"d{kname}/{name}: non-finite entries")
        if ref.norm().item() <= GRAD_MIN_NORM:
            raise AssertionError(
                f"d{kname}/{name}: oracle gradient is dead "
                f"({ref.norm().item():.3e}) -- test has no power here")
        torch.testing.assert_close(gg, ref, rtol=GRAD_RTOL, atol=GRAD_ATOL)

    # Negative controls (rope run): each drops exactly one composition
    # term and must visibly fail.
    if mode == "rope":
        err_gate = (up["x"] - oracle["x"]).abs().max().item()
        if err_gate <= GRAD_ATOL:
            raise AssertionError(
                f"{name} negative control: dropping the gate path still "
                f"matched dx (max err {err_gate:.3e})")
        dq_b, dk_b, dv_b, _, _ = multilevel_attention_backward(
            ce["q4"], ce["packed"], out, lse, dout_pre, fmap, cache,
            block_m=bm, block_n=bn, softcap=SOFTCAP,
            position_mode=mode, **bw_kwargs)   # dlse dropped
        dx_up_bad, = torch.autograd.grad(
            (ce["packed"].k, ce["packed"].v, ce["q4"]), (e["x"],),
            (dk_b, dv_b, dq_b))
        err_dlse = (dx_up_bad + dx_gate - oracle["x"]).abs().max().item()
        if err_dlse <= GRAD_ATOL:
            raise AssertionError(
                f"{name} negative control: dropping dlse still matched "
                f"dx through the full block (max err {err_dlse:.3e})")


def test_full_block_composition(device):
    _full_block_case(device, "none", "linear", seed=140)
    _full_block_case(device, "rope", "qk", seed=144)
    _full_block_case(device, "relative", "qk", seed=148)
    print("  full-block              proj/conv/tree/position/sink/gate/"
          "o_proj explicit == autograd (3 modes, +controls) PASS")


def test_error_paths(device):
    q, k, v, spec, fmap, cache = _tiny_case(device, seed=80)
    N, Hq, Dk = q.shape[1], q.shape[2], q.shape[-1]
    _, _, packed = _build_and_pack(q, k, v, spec)
    cos, sin = rope_tables(N, Dk, device=device)
    states = torch.randn(*q.shape[:2], Hq, 6, device=device)
    proj = torch.randn(6, 8, device=device)

    def fwd(**kw):
        return multilevel_attention_forward(
            q, packed, fmap, cache, block_m=3, block_n=2, **kw)

    def expect(err, label, fn):
        try:
            fn()
        except err:
            return
        raise AssertionError(f"{label}: no {err.__name__} raised")

    expect(ValueError, "unknown mode", lambda: fwd(position_mode="alibi"))
    expect(ValueError, "rope args with none",
           lambda: fwd(position_mode="none", rope_cos=cos, rope_sin=sin))
    expect(ValueError, "relative args with rope",
           lambda: fwd(position_mode="rope", rope_cos=cos, rope_sin=sin,
                       relative_states=states, relative_proj=proj))
    expect(ValueError, "rope args with relative",
           lambda: fwd(position_mode="relative", relative_states=states,
                       relative_proj=proj, rope_cos=cos, rope_sin=sin))
    expect(ValueError, "missing sin",
           lambda: fwd(position_mode="rope", rope_cos=cos))
    expect(ValueError, "short rope table",
           lambda: fwd(position_mode="rope",
                       rope_cos=cos[:N - 1], rope_sin=sin[:N - 1]))
    expect(ValueError, "mismatched cos/sin",
           lambda: fwd(position_mode="rope",
                       rope_cos=cos, rope_sin=sin[:N - 1]))
    expect(ValueError, "wrong proj shape",
           lambda: fwd(position_mode="relative", relative_states=states,
                       relative_proj=torch.randn(5, 8, device=device)))
    expect(ValueError, "1-D proj",
           lambda: fwd(position_mode="relative", relative_states=states,
                       relative_proj=torch.randn(8, device=device)))
    # Valid distance >= max_relative_bins: q=7 has max distance 3.
    expect(ValueError, "table too small",
           lambda: fwd(position_mode="relative", relative_states=states,
                       relative_proj=torch.randn(6, 2, device=device)))
    # Odd Dk for rope.
    q_odd = torch.randn(2, N, Hq, 7, device=device)
    expect(ValueError, "odd Dk", lambda: multilevel_attention_forward(
        q_odd, packed, fmap, cache, block_m=3, block_n=2,
        position_mode="rope", rope_cos=cos, rope_sin=sin))
    # relative_bin_distance on an invisible entry.
    expect(ValueError, "invisible entry",
           lambda: relative_bin_distance(spec, 0, 0, 5))
    # Explicit backward: unknown modes are a ValueError; the positional
    # arg-validation matrix mirrors the forward's for all three modes.
    out, lse = fwd(position_mode="none")

    def bwd(**kw):
        return multilevel_attention_backward(
            q, packed, out, lse, torch.randn_like(out), fmap, cache,
            block_m=3, block_n=2, **kw)

    expect(ValueError, "backward unknown mode",
           lambda: bwd(position_mode="alibi"))
    expect(ValueError, "backward rope args with none",
           lambda: bwd(position_mode="none", rope_cos=cos, rope_sin=sin))
    expect(ValueError, "backward missing sin",
           lambda: bwd(position_mode="rope", rope_cos=cos))
    expect(ValueError, "backward short rope table",
           lambda: bwd(position_mode="rope",
                       rope_cos=cos[:N - 1], rope_sin=sin[:N - 1]))
    expect(ValueError, "backward mismatched cos/sin",
           lambda: bwd(position_mode="rope",
                       rope_cos=cos, rope_sin=sin[:N - 1]))
    expect(ValueError, "backward missing proj",
           lambda: bwd(position_mode="relative", relative_states=states))
    expect(ValueError, "backward relative args with none",
           lambda: bwd(position_mode="none", relative_states=states,
                       relative_proj=proj))
    expect(ValueError, "backward rope args with relative",
           lambda: bwd(position_mode="relative", relative_states=states,
                       relative_proj=proj, rope_cos=cos, rope_sin=sin))
    expect(ValueError, "backward relative args with rope",
           lambda: bwd(position_mode="rope", rope_cos=cos, rope_sin=sin,
                       relative_states=states, relative_proj=proj))
    expect(ValueError, "backward wrong proj shape",
           lambda: bwd(position_mode="relative", relative_states=states,
                       relative_proj=torch.randn(5, 8, device=device)))
    # Valid distance >= max_relative_bins: q=7 has max distance 3.
    expect(ValueError, "backward table too small",
           lambda: bwd(position_mode="relative", relative_states=states,
                       relative_proj=torch.randn(6, 2, device=device)))
    print("  error-paths             ValueError on misuse PASS")


TESTS = [
    test_none_bitwise_regression,
    test_rope_l0_dense_equivalence,
    test_summary_token_position_units,
    test_rope_helpers_units,
    test_relative_bins_vs_oracle,
    test_relative_span_independence,
    test_relative_dense_oracle,
    test_relative_gradient_flow,
    test_tree_mode_independence,
    test_rope_explicit_backward,
    test_relative_explicit_backward,
    test_positional_backward_with_sink,
    test_positional_e2e_composition,
    test_full_block_composition,
    test_error_paths,
]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Positional-encoding contract for reference.py "
                    "(position_mode none/rope/relative)."
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
        raise SystemExit(f"Unknown test(s) {unknown}; available: {sorted(names)}")
    selected = [t for t in TESTS if not args.tests or t.__name__ in args.tests]

    print(f"=== positional-encoding contract: device={device} ===")
    for t in selected:
        t(device)
    print(f"\nALL POSITION TESTS PASSED ({len(selected)} tests)")


if __name__ == "__main__":
    main()
