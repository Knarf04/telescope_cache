"""
Extraction check: telescope_cache/reference.py (the clean implementation)
must be exactly what the frozen test files define.

    clean Phase 1 + packing        == test_tiled.py            (bitwise)
    clean forward (out, lse)       == test_tiled.py forward    (bitwise)
    clean Phase-2 backward         == test_tiled.py explicit backward (bitwise)
    clean Phase 2 + autograd Phase-1 VJP  ~= test_backward.py end-to-end oracle (1e-4)

Bitwise equality is expected because the code was extracted verbatim; any
drift between reference.py and test_tiled.py shows up here by name.

    python test_reference.py [--device cpu|cuda] [case ...]
"""

import argparse
import copy
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

from telescope_cache import reference as ref  # noqa: E402
import test_tiled as frozen  # noqa: E402
from test_backward import grads, leaves, reference_path  # noqa: E402

GRAD_RTOL = GRAD_ATOL = 1e-4


def rel_err(a, b, eps=1e-12):
    return ((a - b).norm() / max(b.norm().item(), eps)).item()


def bitwise(name, a, b):
    if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
        raise AssertionError(f"{name}: clean != frozen (max abs "
                             f"{(a.float() - b.float()).abs().max().item():.3e})")
    print(f"  {name:<18} bitwise ✓")


def close(name, a, b):
    err = (a - b).abs().max().item()
    torch.testing.assert_close(a, b, rtol=GRAD_RTOL, atol=GRAD_ATOL)
    print(f"  {name:<18} max={err:.2e} rel={rel_err(a, b):.2e} PASS")


def run_case(name, *, B, N, Hq, Hkv, Dk, Dv, cache_size, fmap, block_m,
             block_n, device, seed=0, dtype=torch.float32, softcap=20.0,
             **_ignored):
    print(
        f"\n=== {name}: B={B} N={N} Hq={Hq} Hkv={Hkv} Dk={Dk} Dv={Dv} "
        f"cache={cache_size} fmap={fmap} BLOCK_M={block_m} BLOCK_N={block_n} "
        f"softcap={softcap} device={device} ==="
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    q = torch.randn(B, N, Hq, Dk, device=device, dtype=dtype)
    k = torch.randn(B, N, Hkv, Dk, device=device, dtype=dtype)
    v = torch.randn(B, N, Hkv, Dv, device=device, dtype=dtype)
    L = len(fmap)

    # ---- Phase 1 + packing: bitwise ----
    with torch.no_grad():
        ck, cv, cw = ref.build_dyadic_summaries(q, k, v, L)
        fk, fv, fw = frozen.build_dyadic_summaries(q, k, v, L)
        for l in range(L + 1):
            bitwise(f"L{l} k/v/w", torch.cat([ck[l].flatten(), cv[l].flatten(), cw[l].flatten()]),
                    torch.cat([fk[l].flatten(), fv[l].flatten(), fw[l].flatten()]))
        cpacked = ref.pack_levels(ck, cv)
        fpacked = frozen.pack_levels(fk, fv)
        assert cpacked.level_offsets == fpacked.level_offsets
        bitwise("packed k", cpacked.k, fpacked.k)
        bitwise("packed v", cpacked.v, fpacked.v)

        # ---- forward: bitwise ----
        c_out, c_lse = ref.multilevel_attention_forward(
            q, cpacked, fmap, cache_size, block_m=block_m, block_n=block_n,
            softcap=softcap)
        f_out, f_lse = frozen.multilevel_attention_tiled_poc(
            q, fpacked, fmap, cache_size, block_m=block_m, block_n=block_n,
            softcap=softcap)
        bitwise("forward out", c_out, f_out)
        bitwise("forward lse", c_lse, f_lse)

        # ---- explicit Phase-2 backward: bitwise ----
        g = torch.Generator(device="cpu").manual_seed(seed + 1)
        dout = torch.randn(c_out.shape, generator=g, dtype=c_out.dtype).to(device)
        c_dq, c_dk, c_dv, c_stats = ref.multilevel_attention_backward(
            q, cpacked, c_out, c_lse, dout, fmap, cache_size,
            block_m=block_m, block_n=block_n, softcap=softcap)
        f_dq, f_dk, f_dv, f_stats = frozen.multilevel_attention_backward_poc(
            q, fpacked, f_out, f_lse, dout, fmap, cache_size,
            block_m=block_m, block_n=block_n, softcap=softcap)
        bitwise("backward dq", c_dq, f_dq)
        bitwise("backward dk_p", c_dk, f_dk)
        bitwise("backward dv_p", c_dv, f_dv)
        assert c_stats == f_stats, (c_stats, f_stats)
        print(f"  {'backward stats':<18} equal ✓ {c_stats}")

    # ---- clean Phase 2 + autograd Phase-1 VJP vs Step-5 oracle ----
    if softcap != 20.0:
        print(f"\n[{name}] PASS")
        return  # the end-to-end oracle is defined at softcap=20 only
    q_l, k_l, v_l = leaves(q, k, v)
    gk, gv, _ = ref.build_dyadic_summaries(q_l, k_l, v_l, L)
    gpacked = ref.pack_levels(gk, gv)
    with torch.no_grad():
        out, lse = ref.multilevel_attention_forward(
            q_l, gpacked, fmap, cache_size, block_m=block_m, block_n=block_n)
    g = torch.Generator(device="cpu").manual_seed(seed + 2)
    dout = torch.randn(out.shape, generator=g, dtype=out.dtype).to(device)
    dq_attn, dk_p, dv_p, _ = ref.multilevel_attention_backward(
        q_l, gpacked, out, lse, dout, fmap, cache_size,
        block_m=block_m, block_n=block_n)
    dq_tree, dk_tree, dv_tree = torch.autograd.grad(
        (gpacked.k, gpacked.v), (q_l, k_l, v_l), (dk_p, dv_p))

    merge_plan, sel_level, sel_index = frozen.get_structured_plan(
        N, fmap, cache_size, device)
    q_r, k_r, v_r = leaves(q, k, v)
    out_ref = reference_path(q_r, k_r, v_r, merge_plan, sel_level, sel_index, False)
    dq_ref, dk_ref, dv_ref = grads(out_ref, q_r, k_r, v_r, dout)
    close("dq-e2e", dq_attn + dq_tree, dq_ref)
    close("dk-e2e", dk_tree, dk_ref)
    close("dv-e2e", dv_tree, dv_ref)
    print(f"\n[{name}] PASS")


def build_cases():
    cases = [copy.deepcopy(c) for c in frozen.CASES]
    tiny = copy.deepcopy(next(c for c in frozen.CASES if c["name"] == "readme_tiny"))
    tiny["name"], tiny["block_n"] = "readme_tiny_bn3", 3
    cases.append(tiny)
    for base in ("readme_tiny", "baseline"):
        c = copy.deepcopy(next(c for c in frozen.CASES if c["name"] == base))
        c["name"], c["softcap"] = f"{base}_nocap", None
        cases.append(c)
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="reference.py extraction check against the frozen tests.")
    parser.add_argument("--device", default=None)
    parser.add_argument("cases", nargs="*")
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
    print(f"\nALL REFERENCE-EXTRACTION TESTS PASSED ({len(selected)} cases)")


if __name__ == "__main__":
    main()
