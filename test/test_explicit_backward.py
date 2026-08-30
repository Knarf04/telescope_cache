"""
Step 6: explicit FlashAttention-style Phase-2 backward at the packed boundary.

Primary oracle (Phase 2 only): Q, K_packed, V_packed are independent leaves,

    (Q, K_packed, V_packed) -> O,

so autograd gives dQ_attn, dK_packed, dV_packed with no summary-tree path.
The explicit two-pass backward (multilevel_attention_backward_poc: Pass A
Q-owned dQ via fwd_bounds, Pass B KV-owned dK/dV via bwd_bounds, P rebuilt
from the SAVED forward lse) must match it.

Composition check: explicit Phase-2 backward + autograd Phase-1 VJP through
the dyadic tree must reproduce the Step-5 end-to-end oracle:

    dq = dQ_attn + dq_tree      dk = dk_tree      dv = dv_tree

    python test_explicit_backward.py [--device cpu|cuda] [case ...]
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
from test_tiled import (  # noqa: E402
    CASES,
    PackedKV,
    build_dyadic_summaries,
    get_structured_plan,
    multilevel_attention_backward_poc,
    multilevel_attention_tiled_poc,
    pack_levels,
)
from test_backward import grads, leaves, reference_path  # noqa: E402

RTOL = ATOL = 1e-4


def rel_err(a, b, eps=1e-12):
    return ((a - b).norm() / max(b.norm().item(), eps)).item()


def report(name, a, b):
    err = (a - b).abs().max().item()
    torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL)
    print(f"  {name:<12} max={err:.2e} rel={rel_err(a, b):.2e} PASS")


def check_tensor(name, t, shape, dtype):
    if tuple(t.shape) != tuple(shape):
        raise AssertionError(f"{name}: shape {tuple(t.shape)} != {tuple(shape)}")
    if t.dtype != dtype:
        raise AssertionError(f"{name}: dtype {t.dtype} != {dtype}")
    if not bool(torch.isfinite(t).all().item()):
        raise AssertionError(f"{name}: non-finite")


def never_visible_mask(spec, device):
    """Boolean [sumN] mask of packed nodes that are never attended."""
    off = spec.level_offsets()
    mask = torch.zeros(spec.total_len, dtype=torch.bool, device=device)
    for level in range(spec.num_levels):
        for k in range(spec.level_len(level)):
            if node_query_bounds(spec, k, level) == EMPTY:
                mask[off[level] + k] = True
    return mask


def phase2_check(tag, q, packed, fmap, cache_size, block_m, block_n, softcap,
                 spec, dout_seed):
    """Packed-boundary oracle vs explicit backward. Returns nothing; asserts."""
    B, N, Hq, Dk = q.shape
    q_leaf = q.detach().clone().requires_grad_(True)
    k_leaf = packed.k.detach().clone().requires_grad_(True)
    v_leaf = packed.v.detach().clone().requires_grad_(True)
    packed_leaf = PackedKV(k_leaf, v_leaf, packed.level_offsets)

    out, lse = multilevel_attention_tiled_poc(
        q_leaf, packed_leaf, fmap, cache_size,
        block_m=block_m, block_n=block_n, softcap=softcap,
    )
    g = torch.Generator(device="cpu").manual_seed(dout_seed)
    dout = torch.randn(out.shape, generator=g, dtype=out.dtype).to(out.device)

    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        out, (q_leaf, k_leaf, v_leaf), dout
    )

    dq, dk_p, dv_p, stats = multilevel_attention_backward_poc(
        q_leaf.detach(), PackedKV(k_leaf.detach(), v_leaf.detach(),
                                  packed.level_offsets),
        out.detach(), lse.detach(), dout, fmap, cache_size,
        block_m=block_m, block_n=block_n, softcap=softcap,
    )

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
        f"  {'bwd-enum' + tag:<12} kv_tiles={stats['kv_tiles']} "
        f"q_tiles={stats['q_tiles_enumerated']} "
        f"all_false={stats['all_false_qk_tiles']} "
        f"empty_hulls={stats['empty_bwd_hulls']} "
        f"softcap {'✓' if stats['softcap_used'] else '(none)'} "
        f"never_visible={int(nv.sum())} zero ✓"
    )


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

    # ---- 1-3: packed-boundary oracle (values only from Phase 1) ----
    with torch.no_grad():
        kl, vl, _ = build_dyadic_summaries(q, k, v, L)
        packed = pack_levels(kl, vl)
    phase2_check("", q, packed, fmap, cache_size, block_m, block_n, 20.0,
                 spec, dout_seed=seed + 1)

    # ---- 4: softcap=None branch ----
    if softcap_none_too:
        phase2_check("/nocap", q, packed, fmap, cache_size, block_m, block_n,
                     None, spec, dout_seed=seed + 1)

    # ---- 5: composition = explicit Phase 2 + autograd Phase-1 VJP ----
    q_l, k_l, v_l = leaves(q, k, v)
    kl_g, vl_g, _ = build_dyadic_summaries(q_l, k_l, v_l, L)
    packed_g = pack_levels(kl_g, vl_g)  # graph intact
    with torch.no_grad():
        out, lse = multilevel_attention_tiled_poc(
            q_l, packed_g, fmap, cache_size, block_m=block_m, block_n=block_n
        )
    g = torch.Generator(device="cpu").manual_seed(seed + 2)
    dout = torch.randn(out.shape, generator=g, dtype=out.dtype).to(device)

    dq_attn, dk_p, dv_p, _ = multilevel_attention_backward_poc(
        q_l, packed_g, out, lse, dout, fmap, cache_size,
        block_m=block_m, block_n=block_n,
    )
    dq_tree, dk_tree, dv_tree = torch.autograd.grad(
        (packed_g.k, packed_g.v), (q_l, k_l, v_l), (dk_p, dv_p)
    )
    dq_e2e = dq_attn + dq_tree
    dk_e2e, dv_e2e = dk_tree, dv_tree

    # Step-5 oracle: original reference chain on fresh leaves, same dout.
    merge_plan, sel_level, sel_index = get_structured_plan(N, fmap, cache_size, device)
    q_r, k_r, v_r = leaves(q, k, v)
    out_ref = reference_path(q_r, k_r, v_r, merge_plan, sel_level, sel_index, False)
    dq_ref, dk_ref, dv_ref = grads(out_ref, q_r, k_r, v_r, dout)

    report("dq-e2e", dq_e2e, dq_ref)
    report("dk-e2e", dk_e2e, dk_ref)
    report("dv-e2e", dv_e2e, dv_ref)
    print(f"\n[{name}] PASS")


def build_cases():
    cases = [copy.deepcopy(c) for c in CASES]
    for c in cases:
        if c["name"] in ("readme_tiny", "baseline"):
            c["softcap_none_too"] = True
    # Targeted conservative-hull invocation: on readme_tiny the L1 window
    # [0,3) is the one test_range_spec reports as conservative (its hull
    # [3,8) contains q=6 where L1 is empty). BLOCK_N=3 makes it a real tile.
    tiny = copy.deepcopy(next(c for c in CASES if c["name"] == "readme_tiny"))
    tiny["name"] = "readme_tiny_bn3"
    tiny["block_n"] = 3
    cases.append(tiny)
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Explicit Phase-2 backward vs autograd at the packed boundary."
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("cases", nargs="*")
    args = parser.parse_args(argv)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cases = build_cases()
    names = {c["name"] for c in cases}
    unknown = [c for c in args.cases if c not in names]
    if unknown:
        raise SystemExit(f"Unknown case(s) {unknown}; available: {sorted(names)}")
    selected = [c for c in cases if not args.cases or c["name"] in args.cases]
    for case in selected:
        run_case(device=device, **case)
    print(f"\nALL EXPLICIT-BACKWARD TESTS PASSED ({len(selected)} cases)")


if __name__ == "__main__":
    main()
