"""
Clean PyTorch reference implementation of telescoping (multiresolution)
attention: the algorithm to be implemented in CuTeDSL.

This module contains ONLY the implementation; every historical oracle (scan
plan, dense flattened-mask reference, loop POCs, analytic range oracles)
lives in the test files. Behavioral contracts are enforced by:

    test/test_forward.py    forward equivalence chain, (out, lse) contract
    test/test_backward.py   end-to-end gradient oracle, Phase-2 backward at
                            the packed boundary, and their composition
    test/test_range.py      range_spec properties

Pipeline
--------
    q, k, v
      -> compute_summary_weights          w = LSE_h(q_h.k / sqrt(Dk))   [Phase 1]
         (or compute_linear_weights       w = x.w_proj^T, trained mode)
      -> short_conv (optional)            causal depthwise conv + residual on K/V
      -> build_dyadic_summaries           canonical dyadic K/V tree
      -> pack_levels                      one level-major buffer per tensor
      -> multilevel_attention_forward     (out, lse)                    [Phase 2]
      -> multilevel_attention_backward    (dq, dk_packed, dv_packed)    [Phase 2]

Phase-1 backward (gradients of packed K/V back to raw q, k, v, including the
q/k -> w -> merge-weight path) is left to autograd; see test_backward.py.

Coordinates
-----------
Range arithmetic (telescope_cache.range_spec) is level-local. Storage
addressing offset[level] + local happens only in _packed_slice. Physical
layout of the packed buffers is BSHD, [B, sumN, H, D]; a kernel may choose
[B, H, sumN, D] (contiguous per-head streams) -- that decision touches
pack_levels, _packed_slice/_kv_tile and the packed-geometry test only.
"""

import math
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

from telescope_cache.range_spec import (
    RangeSpec,
    bwd_bounds,
    fwd_bounds,
    range_bounds,
)

__all__ = [
    "PackedKV",
    "compute_summary_weights",
    "compute_linear_weights",
    "short_conv",
    "build_dyadic_summaries",
    "pack_levels",
    "multilevel_attention_forward",
    "multilevel_attention_backward",
]


# ============================================================
# Phase 1: summary weights and the canonical dyadic K/V tree.
# ============================================================

def compute_summary_weights(
    q: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    q: [B, N, Hq, Dk]   k: [B, N, Hkv, Dk]   ->   w: [B, N, Hkv, 1]

    For each KV head, groups its associated query heads and computes

        w = logsumexp_h(q_h^T k / sqrt(Dk)).

    This is the Phase-1 merge weight; it is unrelated to the Phase-2
    attention LSE returned by multilevel_attention_forward.
    """
    B, N, Hq, Dk = q.shape
    _, Nk, Hkv, Dkk = k.shape

    assert N == Nk
    assert Dk == Dkk
    assert Hq % Hkv == 0

    expansion = Hq // Hkv
    q_grouped = q.reshape(B, N, Hkv, expansion, Dk)

    scores = (
        q_grouped * k.unsqueeze(3)
    ).sum(dim=-1) / math.sqrt(Dk)  # [B, N, Hkv, expansion]

    return torch.logsumexp(scores, dim=3, keepdim=True)


def compute_linear_weights(
    x: torch.Tensor,
    w_proj: torch.Tensor,
) -> torch.Tensor:
    """
    x: [B, N, emb_dim]   w_proj: [Hkv, emb_dim]   ->   w: [B, N, Hkv, 1]

    Trained-linear alternative to compute_summary_weights. Matches
    nn.Linear(emb_dim, Hkv, bias=False) applied to hidden states:
    w = x @ w_proj^T. Unlike the QK mode, w does not depend on q or k.
    """
    B, N, E = x.shape
    Hkv, Ew = w_proj.shape

    assert E == Ew

    return x.matmul(w_proj.t()).unsqueeze(-1)


def short_conv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Causal depthwise short convolution with a mandatory residual.

    x: [B, N, C]   weight: [C, K]   ->   [B, N, C]

        y[b, t, c] = x[b, t, c] + sum_j weight[c, j] * x[b, t-K+1+j, c]

    Only current and past tokens contribute (left zero-padding); depthwise
    (groups = C, no mixing across channels), no bias, no activation. The
    input is cast to FP32, convolved, the residual is ADDED IN FP32, and
    only the final sum is cast back to x.dtype -- this ordering is part of
    the contract (see test_shortconv.test_fp32_add_ordering), not
    x + conv(x).to(dtype).

    TODO(decode): incremental decoding requires a rolling state of
    pre-convolution projected K/V values. Current implementation supports
    full-sequence training/prefill only.

    TODO(padding/packing): current implementation assumes one continuous
    causal sequence per batch row. It does not reset convolution state
    across padding or packed-example boundaries (would need seq_idx/segment
    boundaries).
    """
    B, N, C = x.shape
    if weight.dim() != 2 or weight.shape[0] != C:
        raise ValueError(
            f"conv weight shape {tuple(weight.shape)} incompatible with "
            f"C={C}; expected [C, K]"
        )
    K = weight.shape[1]
    if K <= 0:
        raise ValueError(f"conv kernel size must be positive, got {K}")

    input_dtype = x.dtype
    x_fp32 = x.float()
    # conv1d is cross-correlation; padding=K-1 plus the [:N] crop yields
    # exactly the causal sum above (no kernel flip).
    y = torch.nn.functional.conv1d(
        x_fp32.transpose(1, 2),
        weight.float().unsqueeze(1),
        padding=K - 1,
        groups=C,
    )[:, :, :N].transpose(1, 2)
    return (x_fp32 + y).to(input_dtype)


def build_dyadic_summaries(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_summary_levels: int,
    detach_weights: bool = False,
    x: Optional[torch.Tensor] = None,
    w_proj: Optional[torch.Tensor] = None,
    k_conv_weight: Optional[torch.Tensor] = None,
    v_conv_weight: Optional[torch.Tensor] = None,
):
    """
    Build a canonical dyadic K/V summary tree directly.

    Canonical level numbering:
        L0: raw tokens, span 1
        L1: 2-token summaries
        L2: 4-token summaries
        ...

    Returns (k_levels, v_levels, w_levels), lists of length
    num_summary_levels + 1 with k_levels[l]: [B, N >> l, Hkv, Dk].

    For odd level lengths, only complete adjacent pairs are merged. This is
    exactly the set of complete aligned dyadic intervals available at the
    next level.

    Weight mode: with x and w_proj both None (default), merge weights come
    from compute_summary_weights(q, k) (QK mode). With both given, they come
    from compute_linear_weights(x, w_proj) (trained-linear mode) and do not
    depend on q or k. Providing exactly one raises.

    detach_weights: gradient-only switch (forward values unchanged) that cuts
    the w -> merge-weight gradient branch (q/k in QK mode, x/w_proj in linear
    mode); used by test_backward.py.

    Short-conv mode: with k_conv_weight [Hkv*Dk, K] and v_conv_weight
    [Hkv*Dv, K] both given, short_conv is applied to k and v (flattened over
    heads, channel c = h*D + d, matching fms/train.py's flat post-projection
    layout) BEFORE the merge weights are computed and before the tree is
    built. Both-or-neither; providing exactly one raises. With both None this
    function is bitwise-identical to the pre-conv behavior. See the
    TODO(decode)/TODO(padding/packing) notes on short_conv.
    """
    if (k_conv_weight is None) != (v_conv_weight is None):
        raise ValueError(
            "k_conv_weight and v_conv_weight must both be provided or both "
            "be None"
        )
    if k_conv_weight is not None:
        B, N, Hkv, Dk = k.shape
        Dv = v.shape[-1]
        k = short_conv(
            k.reshape(B, N, Hkv * Dk), k_conv_weight
        ).reshape(B, N, Hkv, Dk)
        v = short_conv(
            v.reshape(B, N, Hkv * Dv), v_conv_weight
        ).reshape(B, N, Hkv, Dv)

    if (x is None) != (w_proj is None):
        raise ValueError(
            "x and w_proj must both be provided (linear weight mode) or "
            "both be None (QK weight mode)"
        )
    if x is not None:
        w = compute_linear_weights(x, w_proj)
        expected = (k.shape[0], k.shape[1], k.shape[2], 1)
        if tuple(w.shape) != expected:
            raise ValueError(
                f"linear weights shape {tuple(w.shape)} != {expected}"
            )
    else:
        w = compute_summary_weights(q, k)
    if detach_weights:
        w = w.detach()

    k_levels = [k]
    v_levels = [v]
    w_levels = [w]

    for _level in range(1, num_summary_levels + 1):
        k_prev = k_levels[-1]
        v_prev = v_levels[-1]
        w_prev = w_levels[-1]

        usable = (k_prev.shape[1] // 2) * 2

        if usable == 0:
            B, _, Hkv, Dk = k_prev.shape
            Dv = v_prev.shape[-1]
            k_levels.append(k_prev.new_empty(B, 0, Hkv, Dk))
            v_levels.append(v_prev.new_empty(B, 0, Hkv, Dv))
            w_levels.append(w_prev.new_empty(B, 0, Hkv, 1))
            continue

        # [B, N_parent, 2, Hkv, D]
        k_children = torch.stack(
            [k_prev[:, 0:usable:2], k_prev[:, 1:usable:2]],
            dim=2,
        )
        v_children = torch.stack(
            [v_prev[:, 0:usable:2], v_prev[:, 1:usable:2]],
            dim=2,
        )
        w_children = torch.stack(
            [w_prev[:, 0:usable:2], w_prev[:, 1:usable:2]],
            dim=2,
        )

        alpha = torch.softmax(w_children, dim=2)

        k_parent = (k_children * alpha).sum(dim=2)
        v_parent = (v_children * alpha).sum(dim=2)
        w_parent = torch.logsumexp(w_children, dim=2)

        k_levels.append(k_parent)
        v_levels.append(v_parent)
        w_levels.append(w_parent)

    return k_levels, v_levels, w_levels


# ============================================================
# Packed level-major K/V representation.
#
#     [ L0 | L1 | ... | LL ]     level_offsets[l] = sum_{j<l} N_j
#
# torch.cat keeps the autograd graph intact (Phase-1 VJP via autograd); a
# Phase-1 kernel should instead allocate the packed buffer once and write
# each level into its interval. w_levels are deliberately not packed:
# Phase-2 attention needs only K and V.
# ============================================================

class PackedKV(NamedTuple):
    k: torch.Tensor                 # [B, sumN, Hkv, Dk]
    v: torch.Tensor                 # [B, sumN, Hkv, Dv]
    level_offsets: Tuple[int, ...]  # len num_levels + 1, derived from tensors


def pack_levels(
    k_levels: List[torch.Tensor],
    v_levels: List[torch.Tensor],
) -> PackedKV:
    if len(k_levels) != len(v_levels) or not k_levels:
        raise ValueError("k_levels and v_levels must be nonempty, same length")
    B, _, Hkv, Dk = k_levels[0].shape
    Dv = v_levels[0].shape[-1]
    device = k_levels[0].device
    k_dtype, v_dtype = k_levels[0].dtype, v_levels[0].dtype

    # Offsets come from the actual tensors, never from RangeSpec, so that the
    # test's comparison against spec.level_offsets() is a real check.
    offsets = [0]
    for level, (k_l, v_l) in enumerate(zip(k_levels, v_levels)):
        if k_l.shape[1] != v_l.shape[1]:
            raise ValueError(
                f"level {level}: K length {k_l.shape[1]} != V length "
                f"{v_l.shape[1]}"
            )
        if k_l.shape[0] != B or v_l.shape[0] != B:
            raise ValueError(f"level {level}: batch mismatch")
        if k_l.shape[2] != Hkv or v_l.shape[2] != Hkv:
            raise ValueError(f"level {level}: Hkv mismatch")
        if k_l.shape[-1] != Dk or v_l.shape[-1] != Dv:
            raise ValueError(f"level {level}: head-dim mismatch")
        if k_l.device != device or v_l.device != device:
            raise ValueError(f"level {level}: device mismatch")
        if k_l.dtype != k_dtype or v_l.dtype != v_dtype:
            raise ValueError(f"level {level}: dtype mismatch")
        offsets.append(offsets[-1] + k_l.shape[1])

    return PackedKV(
        k=torch.cat(k_levels, dim=1),
        v=torch.cat(v_levels, dim=1),
        level_offsets=tuple(offsets),
    )


def _packed_slice(
    packed: PackedKV, level: int, k_start: int, k_end: int
) -> Tuple[int, int]:
    """
    The boundary k_local (range spec) -> k_physical (packed storage):
    level-local [k_start, k_end) -> physical [p_start, p_end). Shared by the
    forward reads (_kv_tile) and the backward writes so the boundary checks
    live in one place.
    """
    lo = packed.level_offsets[level]
    hi = packed.level_offsets[level + 1]
    level_len = hi - lo
    # Local bounds: catches a packed index passed where a local one belongs.
    assert 0 <= k_start <= k_end <= level_len, (level, k_start, k_end, level_len)
    p_start, p_end = lo + k_start, lo + k_end
    # Physical bounds: a tile never crosses into the next level.
    assert lo <= p_start <= p_end <= hi, (level, p_start, p_end, lo, hi)
    return p_start, p_end


def _kv_tile(
    packed: PackedKV, b: int, hkv: int, level: int, k_start: int, k_end: int
):
    """Read one level-local KV tile from the packed buffers."""
    p_start, p_end = _packed_slice(packed, level, k_start, k_end)
    return packed.k[b, p_start:p_end, hkv], packed.v[b, p_start:p_end, hkv]


# ============================================================
# Phase 2 forward: query-block / KV-tile FlashAttention shape.
# ============================================================

def multilevel_attention_forward(
    q: torch.Tensor,
    packed: PackedKV,
    fmap: Dict[int, int],
    cache_size: int,
    block_m: int = 16,
    block_n: int = 32,
    softcap: float = 20.0,
):
    """
    FlashAttention-shaped multiresolution attention over the packed
    level-major K/V buffers.

      for batch
        for KV head                     # exploit GQA sharing
          for BLOCK_M query positions
            keep Q and online-softmax state live
            for hierarchy level
              [tile_k_lo, tile_k_hi) = fwd_bounds(q_start, q_end, level)
              per-row [row_k_lo, row_k_hi) = range_bounds(q, level)
              for BLOCK_N KV tile in [tile_k_lo, tile_k_hi):     (level-local)
                K, V = packed[offset[level] + tile]              (storage)
                QK matmul, softcap
                valid = k >= row_k_lo AND k < row_k_hi           (elem_mask)
                online-softmax update

    Only the semantics are frozen: a kernel computes the hull in O(1) from
    the first/last nonempty rows (monotone bounds, see range_spec) and must
    predicate padded lanes of a BLOCK_N tile before the semantic mask.

    Forward contract
    ----------------
        out: [B, N, Hq, Dv], dtype = V.dtype
        lse: [B, N, Hq],     dtype = torch.float32, finite everywhere.
             Natural-log log-sum-exp of the final attention scores S over
             exactly the multiresolution KV set A(q) of each query, where

                 S[q,k] = c * tanh( (q.k / sqrt(Dk)) / c )   if softcap=c
                        = q.k / sqrt(Dk)                     if softcap=None

                 LSE[b,q,h] = log sum_{k in A(q)} exp( S[q,k] ).

    Natural log is the API; a kernel that uses exp2 internally must convert.
    It is exactly the state the backward needs to recompute P = exp(S - LSE).
    """
    B, N, Hq, Dk = q.shape
    Hkv = packed.k.shape[2]
    Dv = packed.v.shape[-1]

    if Hq % Hkv != 0:
        raise ValueError(
            f"Hq={Hq} must be divisible by Hkv={Hkv}."
        )

    expansion = Hq // Hkv
    spec = RangeSpec.from_fmap(fmap, cache_size, N)

    if tuple(packed.level_offsets) != spec.level_offsets():
        raise ValueError(
            f"packed level_offsets {tuple(packed.level_offsets)} != "
            f"spec {spec.level_offsets()}"
        )

    out = torch.empty(
        B, N, Hq, Dv, device=q.device, dtype=packed.v.dtype
    )
    # NaN-initialized so an isfinite() check doubles as an "every row was
    # written" check (reference only; a kernel need not do this).
    attn_lse = torch.full(
        (B, N, Hq), float("nan"), device=q.device, dtype=torch.float32
    )
    scale = 1.0 / math.sqrt(Dk)

    for b in range(B):
        for hkv in range(Hkv):
            hq_start = hkv * expansion
            hq_end = (hkv + 1) * expansion
            hq_slice = slice(hq_start, hq_end)

            for q_start in range(0, N, block_m):
                q_end = min(q_start + block_m, N)
                M = q_end - q_start

                Q = q[b, q_start:q_end, hq_slice]  # [M, E, Dk]

                m = torch.full(
                    (M, expansion),
                    -float("inf"),
                    device=q.device,
                    dtype=torch.float32,
                )
                ell = torch.zeros(
                    M, expansion, device=q.device, dtype=torch.float32
                )
                acc = torch.zeros(
                    M, expansion, Dv,
                    device=q.device,
                    dtype=torch.float32,
                )

                for level in range(spec.num_levels):
                    tile_k_lo, tile_k_hi = fwd_bounds(
                        spec, q_start, q_end, level
                    )
                    if tile_k_lo == tile_k_hi:
                        continue

                    rows = [
                        range_bounds(spec, qi, level)
                        for qi in range(q_start, q_end)
                    ]
                    level_start = torch.tensor(
                        [lo for lo, _ in rows],
                        device=q.device, dtype=torch.long,
                    )
                    level_end = torch.tensor(
                        [hi for _, hi in rows],
                        device=q.device, dtype=torch.long,
                    )

                    for k_start in range(
                        tile_k_lo, tile_k_hi, block_n
                    ):
                        k_end = min(k_start + block_n, tile_k_hi)

                        K, V = _kv_tile(
                            packed, b, hkv, level, k_start, k_end
                        )  # [Ktile, Dk], [Ktile, Dv]

                        scores = torch.einsum(
                            "med,kd->mek", Q, K
                        ) * scale

                        if softcap is not None:
                            scores = softcap * torch.tanh(
                                scores / softcap
                            )

                        kv_indices = torch.arange(
                            k_start,
                            k_end,
                            device=q.device,
                            dtype=torch.long,
                        )

                        # Vectorized realization of range_spec.elem_mask.
                        valid = (
                            (kv_indices[None, :] >= level_start[:, None])
                            & (kv_indices[None, :] < level_end[:, None])
                        )  # [M, Ktile]

                        scores = scores.masked_fill(
                            ~valid[:, None, :], -float("inf")
                        )

                        # FlashAttention online-softmax update.
                        active = valid.any(dim=1)[:, None].expand(
                            -1, expansion
                        )
                        block_max = scores.float().max(dim=-1).values
                        m_candidate = torch.maximum(m, block_max)
                        m_new = torch.where(active, m_candidate, m)

                        # where() INSIDE exp(): inactive rows have
                        # m = m_new = -inf and exp must never consume
                        # (-inf) - (-inf) = nan (its autograd backward would
                        # poison dq/dk/dv with 0 * nan). Values identical.
                        old_scale = torch.exp(
                            torch.where(
                                active,
                                m - m_new,
                                torch.zeros_like(m),
                            )
                        )

                        shifted = torch.where(
                            valid[:, None, :],
                            scores.float() - m_new[:, :, None],
                            torch.full_like(
                                scores.float(), -float("inf")
                            ),
                        )
                        p = torch.exp(shifted)

                        ell = ell * old_scale + p.sum(dim=-1)
                        acc = (
                            acc * old_scale[:, :, None]
                            + torch.einsum(
                                "mek,kd->med", p, V.float()
                            )
                        )
                        m = m_new

                if (ell == 0).any():
                    raise RuntimeError(
                        f"Found query rows with no attended KV entries in "
                        f"block [{q_start},{q_end})."
                    )

                out[b, q_start:q_end, hq_slice] = (
                    acc / ell[:, :, None]
                ).to(out.dtype)
                attn_lse[b, q_start:q_end, hq_slice] = m + torch.log(ell)

    return out, attn_lse


# ============================================================
# Phase 2 backward: explicit two-pass FlashAttention-style.
#
#   Pass A (Q-owned dQ):    Q-block --fwd_bounds--> K-tiles
#   Pass B (KV-owned dK/dV): K-tile --bwd_bounds--> Q-tiles
#
# Per Q x K tile (all FP32):
#   X  = Q K^T * scale                     scale = 1/sqrt(Dk)
#   S  = c*tanh(X/c)  (softcap=c) | X      softcap_grad = 1 - tanh(X/c)^2 | 1
#   P  = exp(S - LSE_q) on valid (q,k), else 0     <- SAVED forward lse
#   Delta_q = sum_d dO_qd O_qd                     <- computed once
#   dV += P^T dO      dP = dO V^T      dS = P * (dP - Delta)
#   dX  = dS * softcap_grad
#   dQ += scale * dX K                 dK += scale * dX^T Q
#
# The hulls from fwd_bounds / bwd_bounds (and BLOCK_M alignment of the
# latter) are conservative; the tile-local element mask is exact, so
# invalid entries have P = dS = dX = 0 and over-enumeration changes only
# work, never semantics. `lse` must be the forward's saved value; it is
# never recomputed.
# ============================================================

def _tile_grads(
    Q, K, V, dO, lse_t, delta_t, valid, scale, softcap, *, need_dq, need_dkv
):
    """
    Shared per-tile backward math for both passes.

    Q [M,E,Dk]  K [Kt,Dk]  V [Kt,Dv]  dO [M,E,Dv]  lse_t/delta_t [M,E]
    valid [M,Kt] (tile-local elem_mask, vectorized)
    Returns (dQ [M,E,Dk] | None, dK [Kt,Dk] | None, dV [Kt,Dv] | None), FP32.
    """
    Qf, Kf, Vf, dOf = Q.float(), K.float(), V.float(), dO.float()

    x = torch.einsum("med,kd->mek", Qf, Kf) * scale
    if softcap is not None:
        t = torch.tanh(x / softcap)
        s = softcap * t
        softcap_grad = 1.0 - t * t
    else:
        s = x
        softcap_grad = None

    valid3 = valid[:, None, :]
    zeros = torch.zeros_like(s)
    # P reconstructed from the SAVED forward LSE. where() before exp() so
    # exp never sees an invalid score (unconstrained by the row's LSE):
    #   P = exp(S - LSE_q) on valid (q,k), else exp(-inf) = 0.
    shifted = torch.where(
        valid3, s - lse_t[:, :, None], torch.full_like(s, -float("inf"))
    )
    p = torch.exp(shifted)

    dP = torch.einsum("med,kd->mek", dOf, Vf)
    dS = p * (dP - delta_t[:, :, None])
    dX = dS * softcap_grad if softcap_grad is not None else dS
    dX = torch.where(valid3, dX, zeros)  # already 0 via p; keep explicit

    dQ = scale * torch.einsum("mek,kd->med", dX, Kf) if need_dq else None
    dK = scale * torch.einsum("mek,med->kd", dX, Qf) if need_dkv else None
    dV = torch.einsum("mek,med->kd", p, dOf) if need_dkv else None
    return dQ, dK, dV


@torch.no_grad()
def multilevel_attention_backward(
    q: torch.Tensor,
    packed: PackedKV,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    fmap: Dict[int, int],
    cache_size: int,
    block_m: int = 16,
    block_n: int = 32,
    softcap: float = 20.0,
):
    """
    Explicit Phase-2 backward at the packed boundary (see block comment).

    Contracts
    ---------
        q        [B, N, Hq, Dk]      out   [B, N, Hq, Dv]
        packed.k [B, sumN, Hkv, Dk]  lse   [B, N, Hq] float32 (saved forward)
        packed.v [B, sumN, Hkv, Dv]  dout  [B, N, Hq, Dv]

        dq        [B, N, Hq, Dk]     (q.dtype)
        dk_packed [B, sumN, Hkv, Dk] (packed.k.dtype)
        dv_packed [B, sumN, Hkv, Dv] (packed.v.dtype)
        stats     diagnostic dict (never asserted on)

    Non-differentiable by construction (@torch.no_grad): it reads graph-
    attached packed.k/v VALUES without building a higher-order graph, so a
    caller may still use the returned dk/dv_packed as VJP seeds into the
    Phase-1 graph that produced packed.k/v:

        dq_tree, dk, dv = torch.autograd.grad(
            (packed.k, packed.v), (q, k, v), (dk_packed, dv_packed))
        dq_total = dq + dq_tree
    """
    B, N, Hq, Dk = q.shape
    Hkv = packed.k.shape[2]
    Dv = packed.v.shape[-1]
    if Hq % Hkv != 0:
        raise ValueError(f"Hq={Hq} must be divisible by Hkv={Hkv}.")
    E = Hq // Hkv
    if tuple(out.shape) != (B, N, Hq, Dv):
        raise ValueError(f"out shape {tuple(out.shape)} != {(B, N, Hq, Dv)}")
    if tuple(dout.shape) != (B, N, Hq, Dv):
        raise ValueError(f"dout shape {tuple(dout.shape)} != {(B, N, Hq, Dv)}")
    if tuple(lse.shape) != (B, N, Hq) or lse.dtype != torch.float32:
        raise ValueError("lse must be float32 [B, N, Hq]")

    spec = RangeSpec.from_fmap(fmap, cache_size, N)
    if tuple(packed.level_offsets) != spec.level_offsets():
        raise ValueError("packed level_offsets do not match the schedule")

    scale = 1.0 / math.sqrt(Dk)
    device = q.device

    # Delta_q = sum_d dO_qd O_qd = sum_k P_qk dP_qk: no pre-pass over KV.
    delta = (dout.float() * out.float()).sum(dim=-1)  # [B, N, Hq]

    dq_acc = torch.zeros(B, N, Hq, Dk, device=device, dtype=torch.float32)
    dk_acc = torch.zeros_like(packed.k, dtype=torch.float32)
    dv_acc = torch.zeros_like(packed.v, dtype=torch.float32)

    stats = {
        "kv_tiles": 0,
        "q_tiles_enumerated": 0,
        "all_false_qk_tiles": 0,
        "empty_bwd_hulls": 0,
        "softcap_used": softcap is not None,
    }

    def row_bounds(q0, q1, level):
        rows = [range_bounds(spec, qi, level) for qi in range(q0, q1)]
        lo = torch.tensor([r[0] for r in rows], device=device, dtype=torch.long)
        hi = torch.tensor([r[1] for r in rows], device=device, dtype=torch.long)
        return lo, hi

    # ------------------------------------------------------------------
    # Pass A: Q-owned dQ, same traversal as the forward.
    # ------------------------------------------------------------------
    for b in range(B):
        for hkv in range(Hkv):
            hq_slice = slice(hkv * E, (hkv + 1) * E)
            for q_start in range(0, N, block_m):
                q_end = min(q_start + block_m, N)
                M = q_end - q_start
                Q = q[b, q_start:q_end, hq_slice]
                dO = dout[b, q_start:q_end, hq_slice]
                lse_t = lse[b, q_start:q_end, hq_slice]
                delta_t = delta[b, q_start:q_end, hq_slice]
                dQ_tile = torch.zeros(M, E, Dk, device=device, dtype=torch.float32)

                for level in range(spec.num_levels):
                    tile_k_lo, tile_k_hi = fwd_bounds(spec, q_start, q_end, level)
                    if tile_k_lo == tile_k_hi:
                        continue
                    row_lo, row_hi = row_bounds(q_start, q_end, level)
                    for k_start in range(tile_k_lo, tile_k_hi, block_n):
                        k_end = min(k_start + block_n, tile_k_hi)
                        K, V = _kv_tile(packed, b, hkv, level, k_start, k_end)
                        kv_idx = torch.arange(k_start, k_end, device=device)
                        valid = (kv_idx[None, :] >= row_lo[:, None]) & (
                            kv_idx[None, :] < row_hi[:, None]
                        )
                        dQ_t, _, _ = _tile_grads(
                            Q, K, V, dO, lse_t, delta_t, valid, scale, softcap,
                            need_dq=True, need_dkv=False,
                        )
                        dQ_tile += dQ_t

                dq_acc[b, q_start:q_end, hq_slice] = dQ_tile

    # ------------------------------------------------------------------
    # Pass B: KV-owned dK/dV, K-tile -> candidate Q-tiles via bwd_bounds.
    # ------------------------------------------------------------------
    for b in range(B):
        for hkv in range(Hkv):
            hq_slice = slice(hkv * E, (hkv + 1) * E)
            for level in range(spec.num_levels):
                level_len = spec.level_len(level)
                for k_start in range(0, level_len, block_n):
                    k_end = min(k_start + block_n, level_len)
                    stats["kv_tiles"] += 1

                    # Level-local coordinates only; hull may be conservative.
                    q_lo, q_hi = bwd_bounds(spec, k_start, k_end, level)
                    if q_lo == q_hi:
                        stats["empty_bwd_hulls"] += 1
                        continue

                    K, V = _kv_tile(packed, b, hkv, level, k_start, k_end)
                    kv_idx = torch.arange(k_start, k_end, device=device)
                    Kt = k_end - k_start
                    dK_tile = torch.zeros(Kt, Dk, device=device, dtype=torch.float32)
                    dV_tile = torch.zeros(Kt, Dv, device=device, dtype=torch.float32)

                    # Aligned BLOCK_M Q tiles intersecting the hull.
                    for q0 in range((q_lo // block_m) * block_m, q_hi, block_m):
                        q1 = min(q0 + block_m, N)
                        stats["q_tiles_enumerated"] += 1
                        row_lo, row_hi = row_bounds(q0, q1, level)
                        valid = (kv_idx[None, :] >= row_lo[:, None]) & (
                            kv_idx[None, :] < row_hi[:, None]
                        )
                        if not bool(valid.any().item()):
                            stats["all_false_qk_tiles"] += 1
                            continue
                        Q = q[b, q0:q1, hq_slice]
                        dO = dout[b, q0:q1, hq_slice]
                        lse_t = lse[b, q0:q1, hq_slice]
                        delta_t = delta[b, q0:q1, hq_slice]
                        _, dK_t, dV_t = _tile_grads(
                            Q, K, V, dO, lse_t, delta_t, valid, scale, softcap,
                            need_dq=False, need_dkv=True,
                        )
                        dK_tile += dK_t
                        dV_tile += dV_t

                    # Storage addressing: offset[level] + local, checked once.
                    p_start, p_end = _packed_slice(packed, level, k_start, k_end)
                    dk_acc[b, p_start:p_end, hkv] = dK_tile
                    dv_acc[b, p_start:p_end, hkv] = dV_tile

    return (
        dq_acc.to(q.dtype),
        dk_acc.to(packed.k.dtype),
        dv_acc.to(packed.v.dtype),
        stats,
    )
