"""
Unit contract for the optional short causal depthwise K/V convolution.

Two deliberately independent implementations are under test:

    telescope_cache.reference.short_conv          functional oracle-side op
    telescope_cache.fms.fms_template.ShortConv1d  production training module

Both are checked against a hand-written double-loop oracle (mathematical
independence) and against each other bitwise (drift guard). Properties:
causality, explicit-reference equivalence across N/K/C/dtype, FP32
residual-add ordering, kernel_size=1, channel (feature) independence,
zero-init identity, K/V parameter independence, error paths, and end-to-end
gradients through the full telescoping pipeline.

Training/full-sequence only; decode and packing TODOs live on the
implementations themselves.
"""

import argparse
import os
import sys

import torch

# Make `telescope_cache` importable (namespace package).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from telescope_cache.fms.fms_template import ShortConv1d  # noqa: E402
from telescope_cache.range_spec import RangeSpec  # noqa: E402
from telescope_cache.reference import (  # noqa: E402
    build_dyadic_summaries,
    multilevel_attention_forward,
    pack_levels,
    short_conv,
)

GRAD_MIN_NORM = 1e-6


def hand_oracle(x, w):
    """
    Independent double-loop realization of the contract, with the SAME FP32
    ordering: convolve in fp32, add the residual in fp32, cast the sum back.
    """
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


def rand_weight(C, K, device, dtype=torch.float32, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(C, K, generator=g, dtype=torch.float32)
            / K ** 0.5).to(device=device, dtype=dtype)


def module_with_weight(w):
    m = ShortConv1d(w.shape[0], w.shape[1])
    with torch.no_grad():
        m.weight.copy_(w)
    return m


def test_reference_equivalence(device):
    for dtype in (torch.float32, torch.bfloat16):
        for N in (1, 2, 5, 8, 33):
            for K in (1, 2, 4, 16):
                for C in (1, 3, 8):
                    torch.manual_seed(1000 + N * 100 + K * 10 + C)
                    x = torch.randn(2, N, C, device=device).to(dtype)
                    w = rand_weight(C, K, device, seed=N * 100 + K)
                    ref = hand_oracle(x, w)
                    for name, out in [
                        ("short_conv", short_conv(x, w)),
                        ("ShortConv1d", module_with_weight(w).to(device)(x)),
                    ]:
                        assert out.dtype == dtype, (name, out.dtype)
                        # Only fp32 summation order can differ (<= K terms).
                        torch.testing.assert_close(out, ref)
    print("  reference-equivalence   N x K x C x {fp32,bf16} sweep PASS")


def test_causality(device):
    torch.manual_seed(0)
    B, N, C, K = 2, 16, 6, 4
    x = torch.randn(B, N, C, device=device)
    w = rand_weight(C, K, device, seed=1)
    y = short_conv(x, w)
    for t in (0, 5, 14):
        x2 = x.clone()
        x2[:, t + 1] += 100.0
        y2 = short_conv(x2, w)
        if not torch.equal(y2[:, : t + 1], y[:, : t + 1]):
            raise AssertionError(
                f"future token {t + 1} leaked into positions <= {t}"
            )
        if torch.equal(y2[:, t + 1], y[:, t + 1]):
            raise AssertionError(f"perturbation at {t + 1} had no effect")
    print("  causality               future tokens never leak PASS")


def test_kernel_size_one(device):
    torch.manual_seed(2)
    B, N, C = 2, 7, 5
    x = torch.randn(B, N, C, device=device)
    w = rand_weight(C, 1, device, seed=2)
    expected = (x.float() + w.float()[:, 0] * x.float()).to(x.dtype)
    for name, out in [
        ("short_conv", short_conv(x, w)),
        ("ShortConv1d", module_with_weight(w).to(device)(x)),
    ]:
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    print("  kernel-size-1           y = x + w_c * x exact PASS")


def test_feature_independence(device):
    torch.manual_seed(3)
    B, N, C, K = 2, 12, 8, 4
    x = torch.randn(B, N, C, device=device)
    w = rand_weight(C, K, device, seed=3)
    y = short_conv(x, w)
    c1 = 2
    x2 = x.clone()
    x2[:, :, c1] += 100.0
    y2 = short_conv(x2, w)
    others = [c for c in range(C) if c != c1]
    if not torch.equal(y2[:, :, others], y[:, :, others]):
        raise AssertionError("depthwise conv mixed across channels")
    print("  feature-independence    depthwise channels isolated PASS")


def test_zero_weight_identity(device):
    torch.manual_seed(4)
    B, N, C, K = 2, 9, 6, 4
    x = torch.randn(B, N, C, device=device)
    m = ShortConv1d(C, K).to(device)  # fresh module: zero init
    if not torch.equal(m(x), x):
        raise AssertionError("zero-init ShortConv1d is not an identity")
    if not torch.equal(short_conv(x, torch.zeros(C, K, device=device)), x):
        raise AssertionError("zero-weight short_conv is not an identity")
    print("  zero-weight-identity    enabled-at-init == disabled PASS")


def test_fp32_add_ordering(device):
    # Pin the contract: residual added in FP32, THEN cast -- not
    # x + conv(x).to(dtype). With bf16 inputs the two orderings must
    # actually differ somewhere (otherwise this test would be vacuous),
    # and both implementations must match the FP32 ordering bitwise.
    torch.manual_seed(5)
    B, N, C, K = 4, 64, 32, 4
    x = (torch.randn(B, N, C, device=device) * 100.0).to(torch.bfloat16)
    w = rand_weight(C, K, device, seed=5)

    xf = x.float()
    y = torch.nn.functional.conv1d(
        xf.transpose(1, 2), w.float().unsqueeze(1), padding=K - 1, groups=C
    )[:, :, :N].transpose(1, 2)
    fp32_order = (xf + y).to(torch.bfloat16)
    cast_order = x + y.to(torch.bfloat16)

    if torch.equal(fp32_order, cast_order):
        raise AssertionError(
            "orderings agree everywhere; test inputs are vacuous"
        )
    for name, out in [
        ("short_conv", short_conv(x, w)),
        ("ShortConv1d", module_with_weight(w).to(device)(x)),
    ]:
        if not torch.equal(out, fp32_order):
            raise AssertionError(f"{name} does not add the residual in FP32")
        if torch.equal(out, cast_order):
            raise AssertionError(f"{name} matches the cast-then-add ordering")
    print("  fp32-add-ordering       residual added in FP32 PASS")


def test_module_matches_functional(device):
    torch.manual_seed(6)
    B, N = 2, 11
    # train.py shapes: K conv Hkv*Dk channels, V conv Hkv*Dv channels.
    Hkv, Dk, Dv, K = 2, 8, 16, 4
    for C in (Hkv * Dk, Hkv * Dv):
        x = torch.randn(B, N, C, device=device)
        w = rand_weight(C, K, device, seed=C)
        m = module_with_weight(w).to(device)
        if not torch.equal(m(x), short_conv(x, w)):
            raise AssertionError(
                f"ShortConv1d != short_conv for C={C} (implementations "
                f"drifted)"
            )
    # K/V parameter independence: distinct storage, writing one leaves the
    # other untouched.
    k_sconv = ShortConv1d(Hkv * Dk, K)
    v_sconv = ShortConv1d(Hkv * Dv, K)
    assert k_sconv.weight.data_ptr() != v_sconv.weight.data_ptr()
    with torch.no_grad():
        k_sconv.weight.fill_(1.0)
    if not torch.equal(v_sconv.weight, torch.zeros(Hkv * Dv, K)):
        raise AssertionError("K/V conv modules share parameters")
    try:
        import telescope_cache.fms.train  # noqa: F401
    except Exception as e:  # torch < 2.5 lacks flex_attention
        print(
            "  module-vs-functional    bitwise + K/V independence PASS "
            f"(train.py import SKIP: {type(e).__name__})"
        )
        return
    print("  module-vs-functional    bitwise + K/V independence PASS")


def test_error_paths(device):
    torch.manual_seed(7)
    B, N, Hq, Hkv, Dk, Dv, L = 1, 8, 4, 2, 8, 16, 2
    q = torch.randn(B, N, Hq, Dk, device=device)
    k = torch.randn(B, N, Hkv, Dk, device=device)
    v = torch.randn(B, N, Hkv, Dv, device=device)
    kw = rand_weight(Hkv * Dk, 4, device, seed=7)
    vw = rand_weight(Hkv * Dv, 4, device, seed=8)

    def expect_value_error(label, fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"{label}: no ValueError raised")

    expect_value_error(
        "only k_conv_weight",
        lambda: build_dyadic_summaries(q, k, v, L, k_conv_weight=kw),
    )
    expect_value_error(
        "only v_conv_weight",
        lambda: build_dyadic_summaries(q, k, v, L, v_conv_weight=vw),
    )
    expect_value_error(
        "wrong channel count",
        lambda: build_dyadic_summaries(
            q, k, v, L,
            k_conv_weight=rand_weight(3, 4, device), v_conv_weight=vw,
        ),
    )
    expect_value_error(
        "1-D weight",
        lambda: short_conv(torch.randn(1, 4, 4, device=device),
                           torch.randn(4, device=device)),
    )
    expect_value_error(
        "zero-width weight",
        lambda: short_conv(torch.randn(1, 4, 4, device=device),
                           torch.empty(4, 0, device=device)),
    )
    expect_value_error("ShortConv1d(C, 0)", lambda: ShortConv1d(4, 0))
    print("  error-paths             ValueError on misuse PASS")


def _tiny_case(device, seed):
    torch.manual_seed(seed)
    B, N, Hq, Hkv, Dk, Dv = 2, 8, 4, 2, 8, 16
    fmap, cache_size = {1: 2, 2: 3}, 6
    q = torch.randn(B, N, Hq, Dk, device=device)
    k = torch.randn(B, N, Hkv, Dk, device=device)
    v = torch.randn(B, N, Hkv, Dv, device=device)
    L = RangeSpec.from_fmap(fmap, cache_size, N).num_levels - 1
    return q, k, v, L, fmap, cache_size


def test_disabled_and_zero_equivalence(device):
    q, k, v, L, _, _ = _tiny_case(device, seed=8)
    Hkv, Dk, Dv = k.shape[2], k.shape[-1], v.shape[-1]
    base = build_dyadic_summaries(q, k, v, L)
    zeroed = build_dyadic_summaries(
        q, k, v, L,
        k_conv_weight=torch.zeros(Hkv * Dk, 4, device=device),
        v_conv_weight=torch.zeros(Hkv * Dv, 4, device=device),
    )
    for name, a_levels, b_levels in [
        ("k", base[0], zeroed[0]),
        ("v", base[1], zeroed[1]),
        ("w", base[2], zeroed[2]),
    ]:
        for lvl, (a, b) in enumerate(zip(a_levels, b_levels)):
            if not torch.equal(a, b):
                raise AssertionError(
                    f"{name}_levels[{lvl}]: zero-weight conv != disabled"
                )
    print("  disabled-equivalence    zero weights == None path PASS")


def test_e2e_pipeline(device):
    q, k, v, L, fmap, cache_size = _tiny_case(device, seed=9)
    Hkv, Dk, Dv = k.shape[2], k.shape[-1], v.shape[-1]
    B, N, Hq = q.shape[0], q.shape[1], q.shape[2]

    for detach in (False, True):
        q_l = q.detach().clone().requires_grad_(True)
        k_l = k.detach().clone().requires_grad_(True)
        v_l = v.detach().clone().requires_grad_(True)
        kcw = rand_weight(Hkv * Dk, 4, device, seed=10).requires_grad_(True)
        vcw = rand_weight(Hkv * Dv, 4, device, seed=11).requires_grad_(True)

        kl, vl, _ = build_dyadic_summaries(
            q_l, k_l, v_l, L, detach_weights=detach,
            k_conv_weight=kcw, v_conv_weight=vcw,
        )
        packed = pack_levels(kl, vl)
        out, lse = multilevel_attention_forward(
            q_l, packed, fmap, cache_size, block_m=3, block_n=2
        )
        assert tuple(out.shape) == (B, N, Hq, Dv)
        assert tuple(lse.shape) == (B, N, Hq) and lse.dtype == torch.float32
        assert bool(torch.isfinite(out).all()) and bool(
            torch.isfinite(lse).all()
        )

        g = torch.Generator(device="cpu").manual_seed(12)
        dout = torch.randn(out.shape, generator=g).to(device)
        grads = torch.autograd.grad(out, (q_l, k_l, v_l, kcw, vcw), dout)
        for name, grad in zip(("dq", "dk", "dv", "dk_conv_w", "dv_conv_w"),
                              grads):
            if not bool(torch.isfinite(grad).all()):
                raise AssertionError(f"{name} non-finite (detach={detach})")
            if grad.norm().item() <= GRAD_MIN_NORM:
                raise AssertionError(
                    f"{name} norm {grad.norm().item():.3e} <= "
                    f"{GRAD_MIN_NORM} (detach={detach})"
                )
    print("  e2e-pipeline            grads reach conv weights PASS")


TESTS = [
    test_reference_equivalence,
    test_causality,
    test_kernel_size_one,
    test_feature_independence,
    test_zero_weight_identity,
    test_fp32_add_ordering,
    test_module_matches_functional,
    test_error_paths,
    test_disabled_and_zero_equivalence,
    test_e2e_pipeline,
]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Short causal depthwise K/V conv contract "
                    "(reference.short_conv / fms_template.ShortConv1d)."
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

    print(f"=== short-conv contract: device={device} ===")
    for t in selected:
        t(device)
    print(f"\nALL SHORT-CONV TESTS PASSED ({len(selected)} tests)")


if __name__ == "__main__":
    main()
