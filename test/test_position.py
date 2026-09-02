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
dense RoPE and dense relative-bias oracles, gradient flow, and the
position-independence of the summary tree. Training-only; the explicit
backward raises NotImplementedError for non-none modes.
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
    apply_rope,
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
    # Explicit backward: executable limitation.
    out, lse = fwd(position_mode="none")
    expect(NotImplementedError, "backward rope mode",
           lambda: multilevel_attention_backward(
               q, packed, out, lse, torch.randn_like(out), fmap, cache,
               block_m=3, block_n=2, position_mode="rope"))
    print("  error-paths             ValueError/NotImplementedError PASS")


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
