"""
Backward correctness suite for telescope_cache/reference.py.

Part 1 -- end-to-end autograd gradient oracle
    Runs the ORIGINAL reference chain and the NEW two-phase packed path on
    identical q, k, v (requires_grad=True), applies the same random upstream
    gradient dout to `out`, and compares dq, dk, dv via plain autograd.

        reference:  q,k,v -> build_plan_based_summaries -> dense_reference_attention
        new path:   q,k,v -> build_dyadic_summaries -> pack_levels
                          -> multilevel_attention_forward

    This validates the ENTIRE gradient path, including the dyadic summary
    tree and the summary weights w = LSE_h(q_h.k / sqrt(d)), which give q
    and k a second gradient route besides the attention scores:

        dq = dq_attn + dq_w      dk = dk_attn/tree + dk_w      dv = dv_attn/tree

    The `detach_weights` experiment proves that route carries gradient and
    that dv is invariant to it.

Part 2 -- explicit Phase-2 backward at the packed boundary
    Q, K_packed, V_packed as independent leaves: autograd gives dQ_attn,
    dK_packed, dV_packed with no summary-tree path. The explicit two-pass
    backward (reference.multilevel_attention_backward: Pass A Q-owned dQ via
    fwd_bounds, Pass B KV-owned dK/dV via bwd_bounds, P rebuilt from the
    SAVED forward lse) must match. Also: never-visible packed nodes get
    exactly zero gradient; both softcap branches; a targeted conservative-
    hull tile (readme_tiny with BLOCK_N=3).

Part 3 -- composition
    explicit Phase-2 backward + autograd Phase-1 VJP through the dyadic tree
    must reproduce the Part-1 oracle:

        dq = dQ_attn + dq_tree      dk = dk_tree      dv = dv_tree

`lse` receives no gradient anywhere: it is saved state for the explicit
backward, not part of the modeled loss.

Runtime: autograd through the reference forward builds ~1e4-1e5 small ops
at N=1024; fine on CPU (a couple of minutes per large case), sync-heavy on
CUDA. Not representative of the intended kernel.

    python test_backward.py [--device cpu|cuda] [case ...]
"""

import argparse
import copy
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

from telescope_cache.range_spec import (  # noqa: E402
    EMPTY,
    RangeSpec,
    node_query_bounds,
)
from telescope_cache.reference import (  # noqa: E402
    PackedKV,
    build_dyadic_summaries,
    multilevel_attention_backward,
    multilevel_attention_forward,
    pack_levels,
)
from test_forward import (  # noqa: E402  (oracles only)
    CASES,
    build_plan_based_summaries,
    dense_reference_attention,
    get_structured_plan,
)

FWD_RTOL = FWD_ATOL = 1e-5
# Uniform gradient tolerance for all cases: two materially different reduction
# orders (dense joint softmax vs. level/tile streaming online softmax). This is
# the correctness headroom for the eventual kernel; do not tune per case.
GRAD_RTOL = GRAD_ATOL = 1e-4
# Absolute noise-level threshold proving the w path carries gradient.
W_PATH_MIN_NORM = 1e-6


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def rel_err(a, b, eps=1e-12):
    """Global norm ratio ||a - b|| / max(||b||, eps)."""
    return ((a - b).norm() / max(b.norm().item(), eps)).item()


def report(name, a, b, rtol=GRAD_RTOL, atol=GRAD_ATOL):
    err = (a - b).abs().max().item()
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    print(f"  {name:<16} max={err:.2e} rel={rel_err(a, b):.2e} PASS")


def check_tensor(name, t, shape, dtype=torch.float32):
    if tuple(t.shape) != tuple(shape):
        raise AssertionError(f"{name}: shape {tuple(t.shape)} != {tuple(shape)}")
    if t.dtype != dtype:
        raise AssertionError(f"{name}: dtype {t.dtype} != {dtype}")
    if not bool(torch.isfinite(t).all().item()):
        raise AssertionError(f"{name}: non-finite")


def leaves(q, k, v):
    return (
        q.detach().clone().requires_grad_(True),
        k.detach().clone().requires_grad_(True),
        v.detach().clone().requires_grad_(True),
    )


def seeded_randn_like(t, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(t.shape, generator=g, dtype=t.dtype).to(t.device)


def reference_path(q, k, v, merge_plan, select_level, select_index,
                   detach_weights=False):
    old_k, old_v, _, old_valid = build_plan_based_summaries(
        q, k, v, merge_plan, detach_weights=detach_weights
    )
    out, _ = dense_reference_attention(
        q, old_k, old_v, old_valid, select_level, select_index
    )
    return out


def new_path(q, k, v, num_summary_levels, fmap, cache_size, block_m, block_n,
             detach_weights=False):
    dk_levels, dv_levels, _ = build_dyadic_summaries(
        q, k, v, num_summary_levels, detach_weights=detach_weights
    )
    packed = pack_levels(dk_levels, dv_levels)
    out, _lse = multilevel_attention_forward(
        q, packed, fmap, cache_size, block_m=block_m, block_n=block_n
    )
    return out


def grads(out, q, k, v, dout):
    # autograd.grad: no .grad accumulation; a missing dependency raises
    # (allow_unused=False is the default).
    return torch.autograd.grad(out, (q, k, v), dout)


def never_visible_mask(spec, device):
    """Boolean [sumN] mask of packed nodes that are never attended."""
    off = spec.level_offsets()
    mask = torch.zeros(spec.total_len, dtype=torch.bool, device=device)
    for level in range(spec.num_levels):
        for kk in range(spec.level_len(level)):
            if node_query_bounds(spec, kk, level) == EMPTY:
                mask[off[level] + kk] = True
    return mask


# ----------------------------------------------------------------------------
# Part 1: end-to-end autograd oracle (+ summary-weight path experiment)
# ----------------------------------------------------------------------------

def part1_end_to_end(q, k, v, merge_plan, sel_level, sel_index, L, fmap,
                     cache_size, block_m, block_n, dout_seed):
    q_ref, k_ref, v_ref = leaves(q, k, v)
    q_new, k_new, v_new = leaves(q, k, v)
    out_ref = reference_path(q_ref, k_ref, v_ref, merge_plan, sel_level, sel_index)
    out_new = new_path(q_new, k_new, v_new, L, fmap, cache_size, block_m, block_n)
    report("out", out_new, out_ref, FWD_RTOL, FWD_ATOL)

    dout = seeded_randn_like(out_ref, dout_seed)
    dq_ref, dk_ref, dv_ref = grads(out_ref, q_ref, k_ref, v_ref, dout)
    dq_new, dk_new, dv_new = grads(out_new, q_new, k_new, v_new, dout)
    for nm, g, x in [("dq_ref", dq_ref, q), ("dk_ref", dk_ref, k),
                     ("dv_ref", dv_ref, v), ("dq_new", dq_new, q),
                     ("dk_new", dk_new, k), ("dv_new", dv_new, v)]:
        check_tensor(nm, g, x.shape)
    report("dq", dq_new, dq_ref)
    report("dk", dk_new, dk_ref)
    report("dv", dv_new, dv_ref)

    # ---- summary-weight path experiment: detach w (gradient-only) ----
    q_rd, k_rd, v_rd = leaves(q, k, v)
    q_nd, k_nd, v_nd = leaves(q, k, v)
    out_ref_det = reference_path(q_rd, k_rd, v_rd, merge_plan, sel_level,
                                 sel_index, detach_weights=True)
    out_new_det = new_path(q_nd, k_nd, v_nd, L, fmap, cache_size, block_m,
                           block_n, detach_weights=True)
    # 1. detach is strictly gradient-only: forward values unchanged.
    torch.testing.assert_close(out_ref_det, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(out_new_det, out_new, rtol=0, atol=0)

    dq_rd, dk_rd, dv_rd = grads(out_ref_det, q_rd, k_rd, v_rd, dout)
    dq_nd, dk_nd, dv_nd = grads(out_new_det, q_nd, k_nd, v_nd, dout)
    for nm, g, x in [("dq_ref_det", dq_rd, q), ("dk_ref_det", dk_rd, k),
                     ("dv_ref_det", dv_rd, v), ("dq_new_det", dq_nd, q),
                     ("dk_new_det", dk_nd, k), ("dv_new_det", dv_nd, v)]:
        check_tensor(nm, g, x.shape)
    # 2. detached paths still agree with each other.
    torch.testing.assert_close(dq_nd, dq_rd, rtol=GRAD_RTOL, atol=GRAD_ATOL)
    torch.testing.assert_close(dk_nd, dk_rd, rtol=GRAD_RTOL, atol=GRAD_ATOL)
    torch.testing.assert_close(dv_nd, dv_rd, rtol=GRAD_RTOL, atol=GRAD_ATOL)
    # 3. the w path carries gradient (correctness, not magnitude), and the
    #    isolated contributions agree between the two implementations.
    dq_w, dk_w = dq_ref - dq_rd, dk_ref - dk_rd
    for nm, g in [("dq_w", dq_w), ("dk_w", dk_w)]:
        if not bool(torch.isfinite(g).all().item()):
            raise AssertionError(f"{nm}: non-finite")
        if g.norm().item() <= W_PATH_MIN_NORM:
            raise AssertionError(
                f"{nm}: norm {g.norm().item():.3e} <= {W_PATH_MIN_NORM}; "
                f"the summary-weight path carries no gradient")
    report("dq_w", dq_new - dq_nd, dq_w)
    report("dk_w", dk_new - dk_nd, dk_w)
    # 4. dv is exactly unaffected by w (w does not depend on v and enters
    #    only through the K/V mixing weights). atol guards CUDA reduction
    #    nondeterminism only; on CPU this is bitwise.
    torch.testing.assert_close(dv_rd, dv_ref, rtol=0, atol=1e-7)
    torch.testing.assert_close(dv_nd, dv_new, rtol=0, atol=1e-7)
    print(
        f"  {'w-path':<16} det-fwd-equal ✓ det-agree ✓ dv-invariant ✓ "
        f"|dq_w|={dq_w.norm().item():.2e} "
        f"(rel {dq_w.norm().item() / max(dq_ref.norm().item(), 1e-12):.2e}) "
        f"|dk_w|={dk_w.norm().item():.2e} "
        f"(rel {dk_w.norm().item() / max(dk_ref.norm().item(), 1e-12):.2e})"
    )
    return dout, (dq_ref, dk_ref, dv_ref)


# ----------------------------------------------------------------------------
# Part 2: explicit Phase-2 backward at the packed boundary
# ----------------------------------------------------------------------------

def part2_phase2(tag, q, packed, fmap, cache_size, block_m, block_n, softcap,
                 spec, dout_seed):
    q_leaf = q.detach().clone().requires_grad_(True)
    k_leaf = packed.k.detach().clone().requires_grad_(True)
    v_leaf = packed.v.detach().clone().requires_grad_(True)
    packed_leaf = PackedKV(k_leaf, v_leaf, packed.level_offsets)

    out, lse = multilevel_attention_forward(
        q_leaf, packed_leaf, fmap, cache_size,
        block_m=block_m, block_n=block_n, softcap=softcap)
    dout = seeded_randn_like(out, dout_seed)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out, (q_leaf, k_leaf, v_leaf), dout)

    dq, dk_p, dv_p, stats = multilevel_attention_backward(
        q_leaf.detach(),
        PackedKV(k_leaf.detach(), v_leaf.detach(), packed.level_offsets),
        out.detach(), lse.detach(), dout, fmap, cache_size,
        block_m=block_m, block_n=block_n, softcap=softcap)

    check_tensor("dq", dq, q.shape, q.dtype)
    check_tensor("dk_packed", dk_p, packed.k.shape, packed.k.dtype)
    check_tensor("dv_packed", dv_p, packed.v.shape, packed.v.dtype)
    report(f"dq-phase2{tag}", dq, dq_ref)
    report(f"dk-packed{tag}", dk_p, dk_ref)
    report(f"dv-packed{tag}", dv_p, dv_ref)

    nv = never_visible_mask(spec, q.device)
    if nv.any():
        for nm, t in [("dk_p", dk_p), ("dv_p", dv_p),
                      ("dk_ref", dk_ref), ("dv_ref", dv_ref)]:
            if not bool((t[:, nv] == 0).all().item()):
                raise AssertionError(f"{nm}: nonzero gradient on never-visible nodes")
    print(
        f"  {'bwd-enum' + tag:<16} kv_tiles={stats['kv_tiles']} "
        f"q_tiles={stats['q_tiles_enumerated']} "
        f"all_false={stats['all_false_qk_tiles']} "
        f"empty_hulls={stats['empty_bwd_hulls']} "
        f"softcap {'✓' if stats['softcap_used'] else '(none)'} "
        f"never_visible={int(nv.sum())} zero ✓"
    )


# ----------------------------------------------------------------------------
# Part 3: explicit Phase 2 + autograd Phase-1 VJP == Part-1 oracle
# ----------------------------------------------------------------------------

def part3_composition(q, k, v, L, fmap, cache_size, block_m, block_n, dout,
                      oracle_grads):
    q_l, k_l, v_l = leaves(q, k, v)
    kl_g, vl_g, _ = build_dyadic_summaries(q_l, k_l, v_l, L)
    packed_g = pack_levels(kl_g, vl_g)  # graph intact
    with torch.no_grad():
        out, lse = multilevel_attention_forward(
            q_l, packed_g, fmap, cache_size, block_m=block_m, block_n=block_n)
    dq_attn, dk_p, dv_p, _ = multilevel_attention_backward(
        q_l, packed_g, out, lse, dout, fmap, cache_size,
        block_m=block_m, block_n=block_n)
    dq_tree, dk_tree, dv_tree = torch.autograd.grad(
        (packed_g.k, packed_g.v), (q_l, k_l, v_l), (dk_p, dv_p))
    dq_ref, dk_ref, dv_ref = oracle_grads
    report("dq-e2e", dq_attn + dq_tree, dq_ref)
    report("dk-e2e", dk_tree, dk_ref)
    report("dv-e2e", dv_tree, dv_ref)


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def run_case(name, *, B, N, Hq, Hkv, Dk, Dv, cache_size, fmap, block_m,
             block_n, device, seed=0, dtype=torch.float32,
             softcap_none_too=False, **_ignored):
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

    spec = RangeSpec.from_fmap(fmap, cache_size, N)
    L = spec.num_levels - 1
    merge_plan, sel_level, sel_index = get_structured_plan(N, fmap, cache_size, device)
    assert len(merge_plan) - 2 == L

    print("-- part 1: end-to-end autograd oracle")
    dout, oracle_grads = part1_end_to_end(
        q, k, v, merge_plan, sel_level, sel_index, L, fmap, cache_size,
        block_m, block_n, dout_seed=seed + 1)

    print("-- part 2: explicit Phase-2 backward at the packed boundary")
    with torch.no_grad():
        kl, vl, _ = build_dyadic_summaries(q, k, v, L)
        packed = pack_levels(kl, vl)
    part2_phase2("", q, packed, fmap, cache_size, block_m, block_n, 20.0,
                 spec, dout_seed=seed + 2)
    if softcap_none_too:
        part2_phase2("/nocap", q, packed, fmap, cache_size, block_m, block_n,
                     None, spec, dout_seed=seed + 2)

    print("-- part 3: explicit Phase 2 + autograd Phase-1 VJP vs oracle")
    part3_composition(q, k, v, L, fmap, cache_size, block_m, block_n, dout,
                      oracle_grads)
    print(f"\n[{name}] PASS")


def build_cases():
    cases = [copy.deepcopy(c) for c in CASES]
    for c in cases:
        if c["name"] in ("readme_tiny", "baseline"):
            c["softcap_none_too"] = True
    # Targeted conservative-hull invocation: on readme_tiny the L1 window
    # [0,3) is the one test_range reports as conservative (its hull [3,8)
    # contains q=6 where L1 is empty). BLOCK_N=3 makes it a real tile.
    tiny = copy.deepcopy(next(c for c in CASES if c["name"] == "readme_tiny"))
    tiny["name"], tiny["block_n"] = "readme_tiny_bn3", 3
    cases.append(tiny)
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Telescoping-cache backward suite (reference.py vs oracles).")
    parser.add_argument("--device", default=None,
                        help="cpu or cuda (default: cuda if available)")
    parser.add_argument("cases", nargs="*", help="case names (default: all)")
    args = parser.parse_args(argv)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cases = build_cases()
    names = {c["name"] for c in cases}
    unknown = [c for c in args.cases if c not in names]
    if unknown:
        raise SystemExit(f"Unknown case(s) {unknown}; available: {sorted(names)}")
    selected = [c for c in cases if not args.cases or c["name"] in args.cases]
    for case in selected:
        run_case(device=device, **case)
    print(f"\nALL BACKWARD TESTS PASSED ({len(selected)} cases)")


if __name__ == "__main__":
    main()
