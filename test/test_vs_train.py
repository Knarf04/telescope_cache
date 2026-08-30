"""
Closes the last gap in the equivalence chain:

    train.py (get_scan_plan + scan + flex_attention)
        == test_tiled.dense_reference_attention      (re-implementation)
        == ... == multilevel_attention_tiled_poc

test_tiled.py only compares against a *re-implementation* of train.py's
cache path. This file runs train.py's actual functions and its compiled
flex_attention on the same q/k/v and compares to the tiled POC.

Requires a GPU (flex_attention is torch.compile'd; run on the remote box).
Run from telescope_cache/test:

    python test_vs_train.py
"""

import functools
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# Make `telescope_cache` importable (train.py does
# `from telescope_cache.fms.fms_template import *`).
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

from telescope_cache.fms.train import (  # noqa: E402
    MultiHeadAttention,
    create_block_mask,
    flex_attention,
    get_scan_plan,
)

from test_tiled import (  # noqa: E402
    build_dyadic_summaries,
    build_plan_based_summaries,
    dense_reference_attention,
    get_structured_plan,
    multilevel_attention_tiled_poc,
    pack_levels,
    validate_dyadic_fmap,
)


@torch.no_grad()
def train_py_reference(
    queries: torch.Tensor,   # [B, N, Hq, Dk]  (post-RoPE in train.py)
    keys: torch.Tensor,      # [B, N, Hkv, Dk]
    values: torch.Tensor,    # [B, N, Hkv, Dv]
    fmap,
    cache_size: int,
):
    """
    Verbatim replay of MultiHeadAttention.forward() from
    "# Build telescoping cache" through flex_attention, using train.py's
    own get_scan_plan / scan / flex_attention objects.
    """
    B, N, Hq, Dk = queries.shape
    Hkv = keys.shape[2]
    expansion = Hq // Hkv
    device = queries.device

    # get_scan_plan only uses x for (b, n, device); train passes k as [b n d].
    plan = get_scan_plan(keys.flatten(2), fmap, cache_size)

    # scan() never touches self, so call it unbound.
    scan = MultiHeadAttention.scan

    q_g = queries.unflatten(2, (Hkv, expansion))  # b l h e d
    w = (
        q_g.div(Dk**0.5)
        .matmul(keys.unsqueeze(-1))
        .squeeze(-1)
        .logsumexp(-1, True)
    )  # b l h 1

    k_cache = scan(None, keys, plan, w)    # b n' h d
    v_cache = scan(None, values, plan, w)  # b n' h d
    cache_len = k_cache.size(1)

    mask = torch.zeros(N, cache_len, device=device, dtype=torch.bool)
    mask.scatter_(1, plan[-1], True)
    flags = torch.ones(1, N, device=device)
    flags = scan(None, flags, plan, flags)
    flags = flags[0].bool().logical_not()
    mask[:, flags] = False

    if expansion != 1:
        keys_e = (
            k_cache.transpose(1, 2).unsqueeze(2)
            .expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        )
        values_e = (
            v_cache.transpose(1, 2).unsqueeze(2)
            .expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        )
    else:
        keys_e = k_cache.transpose(1, 2)
        values_e = v_cache.transpose(1, 2)

    Q = queries.transpose(1, 2)  # b h n d

    def mask_index(mask, b, h, q_i, k_i):
        return mask[
            q_i.clamp(min=0, max=mask.size(0) - 1),
            k_i.clamp(min=0, max=mask.size(1) - 1),
        ]

    block_mask = create_block_mask(
        functools.partial(mask_index, mask), 1, 1, N, cache_len,
        device=str(device),
    )

    def soft_cap(score, b, h, q_i, kv_i):
        return 20 * score.div(20).tanh()

    attn = flex_attention(
        Q, keys_e, values_e, block_mask=block_mask, score_mod=soft_cap
    )
    return attn.transpose(1, 2), plan, k_cache, v_cache, mask


def report(name, a, b, rtol, atol):
    err = (a - b).abs()
    print(f"\n{name}:")
    print("  max abs error :", err.max().item())
    print("  mean abs error:", err.mean().item())
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    print("  PASS")


def run(B, N, Hq, Hkv, Dk, Dv, cache_size, fmap, dtype, block_m, block_n):
    device = torch.device("cuda")
    print(
        f"\n=== B={B} N={N} Hq={Hq} Hkv={Hkv} Dk={Dk} Dv={Dv} "
        f"cache={cache_size} fmap={fmap} dtype={dtype} ==="
    )
    torch.manual_seed(0)
    q = torch.randn(B, N, Hq, Dk, device=device, dtype=dtype)
    k = torch.randn(B, N, Hkv, Dk, device=device, dtype=dtype)
    v = torch.randn(B, N, Hkv, Dv, device=device, dtype=dtype)

    # ---- train.py path (ground truth) ----
    out_train, plan, k_cache, v_cache, mask = train_py_reference(
        q, k, v, fmap, cache_size
    )

    # ---- test_tiled's re-implementation of the same thing ----
    merge_plan, sel_level, sel_index = get_structured_plan(
        N, fmap, cache_size, device
    )
    old_k, old_v, _, old_valid = build_plan_based_summaries(
        q, k, v, merge_plan
    )

    # (a) plan bookkeeping is bit-identical
    for l in range(1, len(merge_plan)):
        assert torch.equal(plan[l].long(), merge_plan[l].long()), \
            f"merge plan differs at level {l}"
    # (b) flattened cache is identical to train.py's scan() output
    k_flat = torch.cat(old_k[1:], dim=1)
    v_flat = torch.cat(old_v[1:], dim=1)
    report("scan() K cache vs re-implementation", k_cache, k_flat, 1e-5, 1e-5)
    report("scan() V cache vs re-implementation", v_cache, v_flat, 1e-5, 1e-5)
    # (c) dense mask is identical
    offsets = torch.zeros(len(old_k), dtype=torch.long, device=device)
    c = 0
    for l in range(1, len(old_k)):
        offsets[l] = c
        c += old_k[l].shape[1]
    flat_idx = offsets[sel_level] + sel_index
    mask_re = torch.zeros(N, c, dtype=torch.bool, device=device)
    mask_re.scatter_(1, flat_idx, True)
    mask_re[:, ~torch.cat(old_valid[1:])] = False
    assert torch.equal(mask, mask_re), "attention mask differs from train.py"
    print("\nmerge plan / attention mask: bit-identical to train.py  PASS")

    # flex_attention is compiled (Triton); allow looser tolerance vs exact
    # fp32 math, and bf16-appropriate tolerance in bf16.
    if dtype == torch.float32:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 2e-2, 2e-2

    out_dense, _ = dense_reference_attention(
        q, old_k, old_v, old_valid, sel_level, sel_index
    )
    report("train.py flex_attention vs test_tiled dense reference",
           out_train.float(), out_dense.float(), rtol, atol)

    # ---- the new path: dyadic tree + tiled attention ----
    num_summary_levels = len(fmap)
    dk_levels, dv_levels, _ = build_dyadic_summaries(
        q, k, v, num_summary_levels
    )
    validate_dyadic_fmap(fmap)
    out_tiled, _ = multilevel_attention_tiled_poc(
        q, pack_levels(dk_levels, dv_levels), fmap, cache_size,
        block_m=block_m, block_n=block_n,
    )
    report("train.py flex_attention vs tiled dyadic POC",
           out_train.float(), out_tiled.float(), rtol, atol)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("flex_attention path in train.py needs CUDA; "
                         "run this on the remote GPU box.")

    fmap = {1: 64, 2: 72, 3: 80}   # exactly train.py's config
    cache_size = 512               # exactly train.py's config

    # 1. train.py's exact config, fp32.
    run(B=2, N=128, Hq=8, Hkv=2, Dk=32, Dv=32,
        cache_size=cache_size, fmap=fmap, dtype=torch.float32,
        block_m=16, block_n=32)

    # 2. Long enough to fill the cache and exercise coarsest-level eviction;
    #    asymmetric Dk/Dv and partial tiles.
    run(B=1, N=1024, Hq=4, Hkv=2, Dk=32, Dv=64,
        cache_size=160, fmap=fmap, dtype=torch.float32,
        block_m=24, block_n=40)

    # 3. Training-like dtype.
    run(B=2, N=256, Hq=8, Hkv=2, Dk=64, Dv=64,
        cache_size=cache_size, fmap=fmap, dtype=torch.bfloat16,
        block_m=16, block_n=32)

    print("\nALL train.py EQUIVALENCE TESTS PASSED")


if __name__ == "__main__":
    main()
