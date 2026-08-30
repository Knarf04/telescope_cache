"""
Step 5: end-to-end autograd gradient oracle for telescoping attention.

Runs the ORIGINAL reference chain and the NEW two-phase packed/tiled POC on
identical q, k, v (requires_grad=True), applies the same random upstream
gradient dout to `out`, and compares dq, dk, dv via plain PyTorch autograd.

    reference:  q,k,v -> build_plan_based_summaries -> dense_reference_attention
    new path:   q,k,v -> build_dyadic_summaries -> pack_levels
                      -> multilevel_attention_tiled_poc

This validates the ENTIRE gradient path, including the dyadic summary tree
and the summary weights w = LSE_h(q_h.k / sqrt(d)), which give q and k a
second gradient route besides the attention scores:

    dq = dq_attn + dq_w      dk = dk_attn/tree + dk_w      dv = dv_attn/tree

This is the ground truth for the explicit tiled Phase-2 backward (next
step). It deliberately does NOT exercise bwd_bounds, P reconstruction from
the saved lse, or explicit dQ/dK/dV formulas. `lse` receives no gradient:
it is saved state for the explicit backward, not part of the modeled loss.

Runtime: backward through the tiled POC builds a graph of ~1e4-1e5 small
ops at N=1024; fine on CPU (about a minute per large case), sync-heavy on
CUDA. Not representative of the intended kernel.

    python test_backward.py [--device cpu|cuda] [case ...]
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from test_tiled import (  # noqa: E402
    CASES,
    build_dyadic_summaries,
    build_plan_based_summaries,
    dense_reference_attention,
    get_structured_plan,
    multilevel_attention_tiled_poc,
    pack_levels,
)

FWD_RTOL = FWD_ATOL = 1e-5
# Uniform gradient tolerance for all cases: two materially different reduction
# orders (dense joint softmax vs. level/tile streaming online softmax). This is
# the correctness headroom for the eventual kernel; do not tune per case.
GRAD_RTOL = GRAD_ATOL = 1e-4
# Absolute noise-level threshold proving the w path carries gradient.
W_PATH_MIN_NORM = 1e-6


def rel_err(a, b, eps=1e-12):
    """Global norm ratio ||a - b|| / max(||b||, eps)."""
    return ((a - b).norm() / max(b.norm().item(), eps)).item()


def report(name, a, b, rtol, atol):
    err = (a - b).abs().max().item()
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    print(f"  {name:<12} max={err:.2e} rel={rel_err(a, b):.2e} PASS")


def check_grad_contract(name, g, ref_input):
    if tuple(g.shape) != tuple(ref_input.shape):
        raise AssertionError(f"{name}: shape {tuple(g.shape)} != input")
    if g.dtype != torch.float32:
        raise AssertionError(f"{name}: dtype {g.dtype} != float32")
    if not bool(torch.isfinite(g).all().item()):
        raise AssertionError(f"{name}: non-finite gradient")


def leaves(q, k, v):
    return (
        q.detach().clone().requires_grad_(True),
        k.detach().clone().requires_grad_(True),
        v.detach().clone().requires_grad_(True),
    )


def reference_path(q, k, v, merge_plan, select_level, select_index,
                   detach_weights):
    old_k, old_v, _, old_valid = build_plan_based_summaries(
        q, k, v, merge_plan, detach_weights=detach_weights
    )
    out, _ = dense_reference_attention(
        q, old_k, old_v, old_valid, select_level, select_index
    )
    return out


def new_path(q, k, v, num_summary_levels, fmap, cache_size, block_m, block_n,
             detach_weights):
    dk_levels, dv_levels, _ = build_dyadic_summaries(
        q, k, v, num_summary_levels, detach_weights=detach_weights
    )
    packed = pack_levels(dk_levels, dv_levels)
    out, _lse = multilevel_attention_tiled_poc(
        q, packed, fmap, cache_size, block_m=block_m, block_n=block_n
    )
    return out


def grads(out, q, k, v, dout):
    # autograd.grad: no .grad accumulation; a missing dependency raises
    # (allow_unused=False is the default).
    return torch.autograd.grad(out, (q, k, v), dout)


def run_case(name, *, B, N, Hq, Hkv, Dk, Dv, cache_size, fmap, block_m,
             block_n, device, seed=0, dtype=torch.float32, **_ignored):
    print(
        f"\n=== {name}: B={B} N={N} Hq={Hq} Hkv={Hkv} Dk={Dk} Dv={Dv} "
        f"cache={cache_size} fmap={fmap} BLOCK_M={block_m} "
        f"BLOCK_N={block_n} device={device} ==="
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    q = torch.randn(B, N, Hq, Dk, device=device, dtype=dtype)
    k = torch.randn(B, N, Hkv, Dk, device=device, dtype=dtype)
    v = torch.randn(B, N, Hkv, Dv, device=device, dtype=dtype)

    # Index-only plan (no_grad); shared by both reference evaluations.
    merge_plan, select_level, select_index = get_structured_plan(
        N, fmap, cache_size, device
    )
    num_summary_levels = len(merge_plan) - 2

    # ---- forward + backward, full gradient path ----
    q_ref, k_ref, v_ref = leaves(q, k, v)
    q_new, k_new, v_new = leaves(q, k, v)
    out_ref = reference_path(q_ref, k_ref, v_ref, merge_plan,
                             select_level, select_index, False)
    out_new = new_path(q_new, k_new, v_new, num_summary_levels, fmap,
                       cache_size, block_m, block_n, False)
    report("out", out_new, out_ref, FWD_RTOL, FWD_ATOL)

    dout = torch.randn_like(out_ref)
    dq_ref, dk_ref, dv_ref = grads(out_ref, q_ref, k_ref, v_ref, dout)
    dq_new, dk_new, dv_new = grads(out_new, q_new, k_new, v_new, dout)

    for nm, g, x in [("dq_ref", dq_ref, q), ("dk_ref", dk_ref, k),
                     ("dv_ref", dv_ref, v), ("dq_new", dq_new, q),
                     ("dk_new", dk_new, k), ("dv_new", dv_new, v)]:
        check_grad_contract(nm, g, x)

    report("dq", dq_new, dq_ref, GRAD_RTOL, GRAD_ATOL)
    report("dk", dk_new, dk_ref, GRAD_RTOL, GRAD_ATOL)
    report("dv", dv_new, dv_ref, GRAD_RTOL, GRAD_ATOL)

    # ---- summary-weight path experiment: detach w (gradient-only) ----
    q_rd, k_rd, v_rd = leaves(q, k, v)
    q_nd, k_nd, v_nd = leaves(q, k, v)
    out_ref_det = reference_path(q_rd, k_rd, v_rd, merge_plan,
                                 select_level, select_index, True)
    out_new_det = new_path(q_nd, k_nd, v_nd, num_summary_levels, fmap,
                           cache_size, block_m, block_n, True)

    # 1. detach is strictly gradient-only: forward values unchanged.
    torch.testing.assert_close(out_ref_det, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(out_new_det, out_new, rtol=0, atol=0)

    dq_rd, dk_rd, dv_rd = grads(out_ref_det, q_rd, k_rd, v_rd, dout)
    dq_nd, dk_nd, dv_nd = grads(out_new_det, q_nd, k_nd, v_nd, dout)
    for nm, g, x in [("dq_ref_det", dq_rd, q), ("dk_ref_det", dk_rd, k),
                     ("dv_ref_det", dv_rd, v), ("dq_new_det", dq_nd, q),
                     ("dk_new_det", dk_nd, k), ("dv_new_det", dv_nd, v)]:
        check_grad_contract(nm, g, x)

    # 2. detached paths still agree with each other.
    torch.testing.assert_close(dq_nd, dq_rd, rtol=GRAD_RTOL, atol=GRAD_ATOL)
    torch.testing.assert_close(dk_nd, dk_rd, rtol=GRAD_RTOL, atol=GRAD_ATOL)
    torch.testing.assert_close(dv_nd, dv_rd, rtol=GRAD_RTOL, atol=GRAD_ATOL)

    # 3. the w path carries gradient (correctness, not magnitude).
    dq_w = dq_ref - dq_rd
    dk_w = dk_ref - dk_rd
    for nm, g in [("dq_w", dq_w), ("dk_w", dk_w)]:
        if not bool(torch.isfinite(g).all().item()):
            raise AssertionError(f"{nm}: non-finite")
        if g.norm().item() <= W_PATH_MIN_NORM:
            raise AssertionError(
                f"{nm}: norm {g.norm().item():.3e} <= {W_PATH_MIN_NORM}; "
                f"the summary-weight path carries no gradient"
            )

    # 3b. Diagnostic: the ISOLATED w contributions agree between the two
    # implementations. Implied by the comparisons above, but a regression
    # that breaks only the weight path shows up here by name.
    report("dq_w", dq_new - dq_nd, dq_w, GRAD_RTOL, GRAD_ATOL)
    report("dk_w", dk_new - dk_nd, dk_w, GRAD_RTOL, GRAD_ATOL)

    # 4. dv is exactly unaffected by w (w does not depend on v and enters
    #    only through the K/V mixing weights). atol guards CUDA reduction
    #    nondeterminism only; on CPU this is bitwise.
    torch.testing.assert_close(dv_rd, dv_ref, rtol=0, atol=1e-7)
    torch.testing.assert_close(dv_nd, dv_new, rtol=0, atol=1e-7)

    print(
        f"  {'w-path':<12} det-fwd-equal ✓ det-agree ✓ dv-invariant ✓ "
        f"|dq_w|={dq_w.norm().item():.2e} "
        f"(rel {dq_w.norm().item() / max(dq_ref.norm().item(), 1e-12):.2e}) "
        f"|dk_w|={dk_w.norm().item():.2e} "
        f"(rel {dk_w.norm().item() / max(dk_ref.norm().item(), 1e-12):.2e})"
    )
    print(f"\n[{name}] PASS")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Telescoping-cache end-to-end autograd gradient oracle."
    )
    parser.add_argument("--device", default=None,
                        help="cpu or cuda (default: cuda if available)")
    parser.add_argument("cases", nargs="*",
                        help=f"case names (default: all). "
                             f"Available: {[c['name'] for c in CASES]}")
    args = parser.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    names = {c["name"] for c in CASES}
    unknown = [c for c in args.cases if c not in names]
    if unknown:
        raise SystemExit(f"Unknown case(s) {unknown}; available: {sorted(names)}")
    selected = [c for c in CASES if not args.cases or c["name"] in args.cases]

    for case in selected:
        run_case(device=device, **case)

    print(f"\nALL BACKWARD TESTS PASSED ({len(selected)} cases)")


if __name__ == "__main__":
    main()
