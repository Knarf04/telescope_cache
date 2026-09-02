"""
Unit contract for the learned attention sink and the SiLU output gate:

    telescope_cache.reference.apply_attention_sink   out * sigmoid(lse - s_h)
    telescope_cache.reference.apply_output_gate      SiLU(x @ Wg^T) * out

Both are post-ops over the frozen (out, lse) contract of
multilevel_attention_forward; the kernel/tree suites are untouched by these
features. The sink is checked against a literal augmented-softmax oracle
(one extra logit per query head, zero value) in forward AND gradients --
the gradient comparison is what catches a dropped lse gradient path.
Training-only; kernel fusion and decode tests are TODOs on the functions.
"""

import argparse
import os
import sys

import torch

# Make `telescope_cache` importable (namespace package).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from telescope_cache.range_spec import RangeSpec  # noqa: E402
from telescope_cache.reference import (  # noqa: E402
    apply_attention_sink,
    apply_output_gate,
    attention_sink_backward,
    build_dyadic_summaries,
    multilevel_attention_backward,
    multilevel_attention_forward,
    pack_levels,
)

GRAD_MIN_NORM = 1e-6


def dense_softmax_attention(z, v):
    """z: [B, H, N, M] logits, v: [B, H, M, Dv] -> (out [B,H,N,Dv], lse)."""
    lse = torch.logsumexp(z.float(), dim=-1)
    out = torch.softmax(z.float(), dim=-1).matmul(v.float()).to(v.dtype)
    return out, lse


def augmented_sink_attention(z, v, sinks):
    """The doc's oracle: one extra softmax entry per query head with logit
    sinks[h] and an identically-zero value row."""
    B, H, N, M = z.shape
    sink_col = sinks.float()[None, :, None, None].expand(B, H, N, 1)
    z_aug = torch.cat([z.float(), sink_col], dim=-1)
    v_aug = torch.cat(
        [v.float(), torch.zeros(B, H, 1, v.shape[-1], device=v.device)],
        dim=-2,
    )
    return torch.softmax(z_aug, dim=-1).matmul(v_aug).to(v.dtype)


def sink_via_lse(z, v, sinks):
    """LSE-rescale path in the [B,H,N,*] layout used by the oracles."""
    out, lse = dense_softmax_attention(z, v)
    # apply_attention_sink expects [B, N, Hq, Dv] / [B, N, Hq].
    out_s = apply_attention_sink(
        out.permute(0, 2, 1, 3), lse.permute(0, 2, 1), sinks
    )
    return out_s.permute(0, 2, 1, 3)


def test_sink_augmented_softmax(device):
    torch.manual_seed(0)
    for H in (2, 4):
        for N, M in ((1, 1), (5, 7), (8, 8), (16, 33)):
            for dtype in (torch.float32, torch.bfloat16):
                for sink_kind in ("zero", "random"):
                    z = torch.randn(2, H, N, M, device=device)
                    v = torch.randn(2, H, M, 6, device=device).to(dtype)
                    sinks = (torch.zeros(H, device=device)
                             if sink_kind == "zero"
                             else torch.randn(H, device=device))
                    if N == M:  # causal variant (diag keeps rows nonempty)
                        z = z.masked_fill(
                            torch.ones(N, M, device=device,
                                       dtype=torch.bool).triu(1),
                            -float("inf"),
                        )
                    ref = augmented_sink_attention(z, v, sinks)
                    got = sink_via_lse(z, v, sinks)
                    torch.testing.assert_close(got, ref)
    print("  sink-augmented-softmax  H x (N,M) x dtype x {0,rand} PASS")


def test_sink_gradients(device):
    # Same math from q/k/v leaves; grads of the LSE-rescale path must match
    # the augmented-softmax path for (dq, dk, dv, dsinks). This catches the
    # failure mode where forward is right but the lse gradient path (the
    # sink's route into the logits) is dropped.
    torch.manual_seed(1)
    B, H, N, M, D, Dv = 2, 3, 6, 9, 8, 5
    q0 = torch.randn(B, H, N, D, device=device)
    k0 = torch.randn(B, H, M, D, device=device)
    v0 = torch.randn(B, H, M, Dv, device=device)
    s0 = torch.randn(H, device=device)
    dout = torch.randn(B, H, N, Dv, device=device)
    scale = 1.0 / D ** 0.5

    def run(path):
        q, k, v, s = (t.detach().clone().requires_grad_(True)
                      for t in (q0, k0, v0, s0))
        z = q.matmul(k.transpose(-1, -2)) * scale
        out = (sink_via_lse(z, v, s) if path == "lse"
               else augmented_sink_attention(z, v, s))
        return torch.autograd.grad(out, (q, k, v, s), dout)

    g_lse, g_aug = run("lse"), run("aug")
    for nm, a, b in zip(("dq", "dk", "dv", "dsinks"), g_lse, g_aug):
        if not bool(torch.isfinite(a).all()):
            raise AssertionError(f"{nm}: non-finite")
        torch.testing.assert_close(a, b)
    if g_lse[3].norm().item() <= GRAD_MIN_NORM:
        raise AssertionError("dsinks carries no gradient")
    print("  sink-gradients          lse-rescale == augmented oracle PASS")


def test_sink_zero_is_not_identity(device):
    torch.manual_seed(2)
    B, N, Hq, Dv = 2, 7, 4, 6
    out = torch.randn(B, N, Hq, Dv, device=device)
    lse = torch.randn(B, N, Hq, device=device)
    got = apply_attention_sink(out, lse, torch.zeros(Hq, device=device))
    if torch.equal(got, out):
        raise AssertionError("zero sink behaved as identity")
    expected = out * torch.sigmoid(lse.float()).unsqueeze(-1).to(out.dtype)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    print("  sink-zero-not-identity  sinks=0 -> out * sigmoid(lse) PASS")


def test_gate_reference(device):
    torch.manual_seed(3)
    # (Hq, Hkv) covers MHA and GQA shapes; (Dk, Dv) covers Dk==Dv and not.
    for Hq, Dk, Dv in ((4, 8, 8), (4, 8, 16), (2, 16, 8)):
        for dtype in (torch.float32, torch.bfloat16):
            B, N, emb_dim = 2, 6, 24
            out = torch.randn(B, N, Hq, Dv, device=device).to(dtype)
            x = torch.randn(B, N, emb_dim, device=device).to(dtype)
            wg = (torch.randn(Hq * Dv, emb_dim, device=device)
                  / emb_dim ** 0.5).to(dtype)
            expected = (
                torch.nn.functional.silu(x.matmul(wg.t()))
                .reshape(B, N, Hq, Dv) * out
            )
            got = apply_output_gate(out, x, wg)
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
            # SiLU-vs-sigmoid guard: the two gates must not coincide.
            sig = torch.sigmoid(x.matmul(wg.t())).reshape(B, N, Hq, Dv) * out
            if torch.equal(got, sig):
                raise AssertionError("gate matches sigmoid, not SiLU")
    # Gradients reach the gate weight and x.
    out = torch.randn(2, 6, 4, 8, device=device)
    x = torch.randn(2, 6, 24, device=device, requires_grad=True)
    wg = (torch.randn(32, 24, device=device) / 24 ** 0.5).requires_grad_(True)
    dx, dwg = torch.autograd.grad(
        apply_output_gate(out, x, wg).sum(), (x, wg))
    for nm, g in (("dx", dx), ("dgate_w", dwg)):
        if not bool(torch.isfinite(g).all()) or g.norm() <= GRAD_MIN_NORM:
            raise AssertionError(f"{nm}: missing/non-finite gradient")
    print("  gate-reference          SiLU(x@Wg^T)*out exact + grads PASS")


def _tiny_case(device, seed):
    torch.manual_seed(seed)
    B, N, Hq, Hkv, Dk, Dv, emb_dim = 2, 8, 4, 2, 8, 16, 32
    fmap, cache_size = {1: 2, 2: 3}, 6
    q = torch.randn(B, N, Hq, Dk, device=device)
    k = torch.randn(B, N, Hkv, Dk, device=device)
    v = torch.randn(B, N, Hkv, Dv, device=device)
    x = torch.randn(B, N, emb_dim, device=device)
    L = RangeSpec.from_fmap(fmap, cache_size, N).num_levels - 1
    return q, k, v, x, L, fmap, cache_size


def test_e2e_multilevel_composition(device):
    q0, k0, v0, x0, L, fmap, cache_size = _tiny_case(device, seed=4)
    B, N, Hq, Dv = q0.shape[0], q0.shape[1], q0.shape[2], v0.shape[-1]
    emb_dim = x0.shape[-1]

    for use_sinks in (False, True):
        for use_gate in (False, True):
            q, k, v, x = (t.detach().clone().requires_grad_(True)
                          for t in (q0, k0, v0, x0))
            leaves = [q, k, v]
            names = ["dq", "dk", "dv"]
            torch.manual_seed(5)
            sinks = torch.randn(Hq, device=device, requires_grad=True)
            gate_w = (torch.randn(Hq * Dv, emb_dim, device=device)
                      / emb_dim ** 0.5).requires_grad_(True)
            # o_proj as a plain leaf matmul (doc SS7 combined check).
            o_w = (torch.randn(emb_dim, Hq * Dv, device=device)
                   / (Hq * Dv) ** 0.5).requires_grad_(True)

            kl, vl, _ = build_dyadic_summaries(q, k, v, L)
            packed = pack_levels(kl, vl)
            out, lse = multilevel_attention_forward(
                q, packed, fmap, cache_size, block_m=3, block_n=2)
            if use_sinks:
                out = apply_attention_sink(out, lse, sinks)
                leaves.append(sinks)
                names.append("dsinks")
            if use_gate:
                out = apply_output_gate(out, x, gate_w)
                leaves += [x, gate_w]
                names += ["dx", "dgate_w"]
            final = out.reshape(B, N, Hq * Dv).matmul(o_w.t())
            leaves.append(o_w)
            names.append("do_w")
            assert tuple(final.shape) == (B, N, emb_dim)
            assert bool(torch.isfinite(final).all())

            g = torch.Generator(device="cpu").manual_seed(6)
            dout = torch.randn(final.shape, generator=g).to(device)
            grads = torch.autograd.grad(final, tuple(leaves), dout)
            for nm, grad in zip(names, grads):
                if not bool(torch.isfinite(grad).all()):
                    raise AssertionError(
                        f"{nm} non-finite (sinks={use_sinks}, "
                        f"gate={use_gate})")
                if grad.norm().item() <= GRAD_MIN_NORM:
                    raise AssertionError(
                        f"{nm} norm {grad.norm().item():.3e} <= "
                        f"{GRAD_MIN_NORM} (sinks={use_sinks}, "
                        f"gate={use_gate})")
            if not use_sinks and not use_gate:
                # Both-off composition is exactly the plain forward.
                torch.testing.assert_close(
                    out, multilevel_attention_forward(
                        q, packed, fmap, cache_size, block_m=3, block_n=2
                    )[0], rtol=0, atol=0)
    print("  e2e-composition         4 flag combos, grads to all leaves PASS")


def test_composition_order(device):
    # Combined path == sink first, then gate (then o_proj outside):
    # SiLU(x@Wg^T) * (out * sigmoid(lse - s)), bitwise.
    q, k, v, x, L, fmap, cache_size = _tiny_case(device, seed=7)
    Hq, Dv, emb_dim = q.shape[2], v.shape[-1], x.shape[-1]
    torch.manual_seed(8)
    sinks = torch.randn(Hq, device=device)
    gate_w = torch.randn(Hq * Dv, emb_dim, device=device) / emb_dim ** 0.5

    kl, vl, _ = build_dyadic_summaries(q, k, v, L)
    packed = pack_levels(kl, vl)
    out, lse = multilevel_attention_forward(
        q, packed, fmap, cache_size, block_m=3, block_n=2)

    combined = apply_output_gate(
        apply_attention_sink(out, lse, sinks), x, gate_w)
    B, N = out.shape[0], out.shape[1]
    inlined = (
        torch.nn.functional.silu(x.matmul(gate_w.t())).reshape(B, N, Hq, Dv)
        * (out * torch.sigmoid(
            lse.float() - sinks.float()[None, None, :]
        ).unsqueeze(-1).to(out.dtype))
    )
    torch.testing.assert_close(combined, inlined, rtol=0, atol=0)
    # Independence: each feature alone equals the combined path with the
    # other feature skipped (they only compose, never interact).
    torch.testing.assert_close(
        apply_output_gate(out, x, gate_w),
        torch.nn.functional.silu(x.matmul(gate_w.t()))
        .reshape(B, N, Hq, Dv) * out,
        rtol=0, atol=0)
    print("  composition-order       sink -> gate -> o_proj, bitwise PASS")


def test_explicit_backward_with_sink(device):
    # The explicit Phase-2 backward composed with the sink VJP must
    # reproduce autograd through [forward -> apply_attention_sink]:
    #   dout_pre, dlse, dsinks = attention_sink_backward(...)
    #   multilevel_attention_backward(..., dout_pre, dlse=dlse) + tree VJP.
    # GRAD tolerances match test_backward.py's.
    GRAD_RTOL = GRAD_ATOL = 1e-4
    q0, k0, v0, _, L, fmap, cache_size = _tiny_case(device, seed=10)
    Hq = q0.shape[2]

    # Throwaway forward to center the sinks on each head's typical lse so
    # r ~ 0.5 and r(1-r) ~ 0.25: the dlse branch is maximally load-bearing
    # (zero-centered sinks with lse >> 0 give r ~ 1 and a vanishing term).
    with torch.no_grad():
        kl0, vl0, _ = build_dyadic_summaries(q0, k0, v0, L)
        _, lse_probe = multilevel_attention_forward(
            q0, pack_levels(kl0, vl0), fmap, cache_size, block_m=3, block_n=2)
    torch.manual_seed(11)
    sinks0 = (lse_probe.float().mean(dim=(0, 1))
              + 0.1 * torch.randn(Hq, device=device))

    # --- autograd oracle -------------------------------------------------
    q_a, k_a, v_a = (t.detach().clone().requires_grad_(True)
                     for t in (q0, k0, v0))
    s_a = sinks0.detach().clone().requires_grad_(True)
    kl, vl, _ = build_dyadic_summaries(q_a, k_a, v_a, L)
    out_a, lse_a = multilevel_attention_forward(
        q_a, pack_levels(kl, vl), fmap, cache_size, block_m=3, block_n=2)
    out_post = apply_attention_sink(out_a, lse_a, s_a)
    g = torch.Generator(device="cpu").manual_seed(12)
    dout_post = torch.randn(out_post.shape, generator=g).to(device)
    dq_o, dk_o, dv_o, ds_o = torch.autograd.grad(
        out_post, (q_a, k_a, v_a, s_a), dout_post)

    # --- explicit composition (graph-intact tree, no_grad attention) -----
    q_l, k_l, v_l = (t.detach().clone().requires_grad_(True)
                     for t in (q0, k0, v0))
    kl_g, vl_g, _ = build_dyadic_summaries(q_l, k_l, v_l, L)
    packed_g = pack_levels(kl_g, vl_g)
    with torch.no_grad():
        out, lse = multilevel_attention_forward(
            q_l, packed_g, fmap, cache_size, block_m=3, block_n=2)
    dout_pre, dlse, dsinks = attention_sink_backward(
        out, lse, sinks0, dout_post)
    dq_attn, dk_p, dv_p, _ = multilevel_attention_backward(
        q_l, packed_g, out, lse, dout_pre, fmap, cache_size,
        block_m=3, block_n=2, dlse=dlse)
    dq_tree, dk_tree, dv_tree = torch.autograd.grad(
        (packed_g.k, packed_g.v), (q_l, k_l, v_l), (dk_p, dv_p),
        retain_graph=True)  # the negative control reuses this tree graph

    torch.testing.assert_close(dq_attn + dq_tree, dq_o,
                               rtol=GRAD_RTOL, atol=GRAD_ATOL)
    torch.testing.assert_close(dk_tree, dk_o, rtol=GRAD_RTOL, atol=GRAD_ATOL)
    torch.testing.assert_close(dv_tree, dv_o, rtol=GRAD_RTOL, atol=GRAD_ATOL)
    torch.testing.assert_close(dsinks, ds_o, rtol=GRAD_RTOL, atol=GRAD_ATOL)

    # Negative control: dropping the dlse seed (LSE path) must NOT match --
    # proves the dlse term is load-bearing, not noise.
    dq_bad, dk_pb, dv_pb, _ = multilevel_attention_backward(
        q_l, packed_g, out, lse, dout_pre, fmap, cache_size,
        block_m=3, block_n=2)
    dq_tb, _, _ = torch.autograd.grad(
        (packed_g.k, packed_g.v), (q_l, k_l, v_l), (dk_pb, dv_pb))
    err = (dq_bad + dq_tb - dq_o).abs().max().item()
    if err <= GRAD_ATOL:
        raise AssertionError(
            f"negative control: dropping dlse still matched (max err "
            f"{err:.3e}); the dlse path is not being exercised")
    print("  explicit-bwd-sink       dlse composition == autograd PASS")


def test_error_paths(device):
    torch.manual_seed(9)
    B, N, Hq, Dv, emb_dim = 2, 4, 4, 8, 16
    out = torch.randn(B, N, Hq, Dv, device=device)
    lse = torch.randn(B, N, Hq, device=device)
    x = torch.randn(B, N, emb_dim, device=device)

    def expect_value_error(label, fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"{label}: no ValueError raised")

    expect_value_error(
        "wrong sinks length",
        lambda: apply_attention_sink(out, lse,
                                     torch.zeros(Hq + 1, device=device)))
    expect_value_error(
        "sinks per KV head",
        lambda: apply_attention_sink(out, lse,
                                     torch.zeros(2, device=device)))
    expect_value_error(
        "lse/out mismatch",
        lambda: apply_attention_sink(out, lse[:, :-1],
                                     torch.zeros(Hq, device=device)))
    expect_value_error(
        "wrong gate_weight shape",
        lambda: apply_output_gate(
            out, x, torch.randn(Hq * Dv + 1, emb_dim, device=device)))
    expect_value_error(
        "gate x mismatch",
        lambda: apply_output_gate(
            out, x[:, :-1], torch.randn(Hq * Dv, emb_dim, device=device)))
    expect_value_error(
        "sink-backward wrong sinks",
        lambda: attention_sink_backward(
            out, lse, torch.zeros(Hq + 1, device=device), out))
    expect_value_error(
        "sink-backward dout mismatch",
        lambda: attention_sink_backward(
            out, lse, torch.zeros(Hq, device=device), out[:, :-1]))
    expect_value_error(
        "sink-backward dout dtype",
        lambda: attention_sink_backward(
            out, lse, torch.zeros(Hq, device=device),
            out.to(torch.bfloat16)))
    # dlse validation on the explicit backward, via a real tiny case (the
    # check runs before any tile work).
    q, k, v, _, L, fmap, cache_size = _tiny_case(device, seed=13)
    with torch.no_grad():
        kl, vl, _ = build_dyadic_summaries(q, k, v, L)
        packed = pack_levels(kl, vl)
        o, l = multilevel_attention_forward(
            q, packed, fmap, cache_size, block_m=3, block_n=2)
    expect_value_error(
        "dlse wrong shape",
        lambda: multilevel_attention_backward(
            q, packed, o, l, torch.randn_like(o), fmap, cache_size,
            block_m=3, block_n=2, dlse=l[:, :-1]))
    expect_value_error(
        "dlse non-floating",
        lambda: multilevel_attention_backward(
            q, packed, o, l, torch.randn_like(o), fmap, cache_size,
            block_m=3, block_n=2,
            dlse=torch.zeros_like(l, dtype=torch.long)))
    print("  error-paths             ValueError on misuse PASS")


TESTS = [
    test_sink_augmented_softmax,
    test_sink_gradients,
    test_sink_zero_is_not_identity,
    test_gate_reference,
    test_e2e_multilevel_composition,
    test_composition_order,
    test_explicit_backward_with_sink,
    test_error_paths,
]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Learned attention sink + SiLU output gate contract "
                    "(reference.apply_attention_sink / apply_output_gate)."
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

    print(f"=== sink + gate contract: device={device} ===")
    for t in selected:
        t(device)
    print(f"\nALL SINK/GATE TESTS PASSED ({len(selected)} tests)")


if __name__ == "__main__":
    main()
