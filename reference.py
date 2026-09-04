"""
Clean PyTorch reference implementation of telescoping (multiresolution)
attention: the algorithm to be implemented in CuTeDSL.

This module contains ONLY the implementation; the analytic range oracles
live in the consolidated test suite. Behavioral contracts are enforced by
test/test_reference.py: the (out, lse) forward contract and gradients
against telescope_cache.minimal_reference (the semantic oracle), the
explicit Phase-2 backward at the packed boundary, and the range_spec
properties.

Pipeline
--------
    q, k, v
      -> short_conv (optional)            causal depthwise conv + residual on K/V
      -> compute_summary_weights          w = LSE_h(q_h.k / sqrt(Dk))   [Phase 1]
         (or compute_linear_weights       w = x.w_proj^T, trained mode)
      -> build_dyadic_summaries           canonical dyadic K/V tree
      -> pack_levels                      one level-major buffer per tensor
      -> multilevel_attention_forward     (out, lse)                    [Phase 2]
         (position_mode: none | rope | relative -- Phase-2 only; the
          summary tree is position-independent in every mode)
      -> apply_attention_sink (optional)  out * sigmoid(lse - sinks)
      -> apply_output_gate (optional)     SiLU(x.gate_w^T) * out
      -> multilevel_attention_backward    (dq, dk_packed, dv_packed,
                                           dposition, stats)            [Phase 2]

Phase-1 backward (gradients of packed K/V back to raw q, k, v, including the
q/k -> w -> merge-weight path) is left to autograd; see
test/test_reference.py.

Incremental decode
------------------
    init_decode_state          prefill hand-off: the live nodes of the dyadic
                               tree at query N-1 -> per-level modular rings
                               (+ the pre-conv rolling state)
    decode_step, per token t:
      short_conv_step          rolling K-1 pre-conv rows       (item: conv)
      -> merge weight w_t      qk logsumexp | linear
      -> advance_decode_state  the ONE level whose node activates at t is
                               merged from its two live children, THEN
                               token t is written at level 0
      -> multilevel_attention_decode
                               query t over range_bounds(spec_{t+1}, t, l),
                               level by level, same online softmax and the
                               same positional rules as the forward
                               (post-summary RoPE at right endpoints,
                               summary-bin relative bias)
      -> apply_attention_sink / apply_output_gate       (unchanged post-ops)

The decode cache is addressed by the SAME level-local coordinates as the
packed prefill buffer: node j of level l lives at physical slot
ring_offsets[l] + j % caps[l] (ring_slot), and range_bounds alone decides
what is live -- no dummies, no plan replay, no eviction bookkeeping (an
evicted node simply leaves the range; its slot is reused caps[l] nodes
later). Feeding tokens one at a time reproduces the full-sequence rows
(test/test_reference.py, Section 5).

Coordinates
-----------
Range arithmetic (telescope_cache.range_spec) is level-local. Storage
addressing offset[level] + local happens only in _packed_slice. Physical
layout of the packed buffers is BSHD, [B, sumN, H, D]; a kernel may choose
[B, H, sumN, D] (contiguous per-head streams) -- that decision touches
pack_levels, _packed_slice/_kv_tile and the packed-geometry test only.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

from telescope_cache.range_spec import (
    RangeSpec,
    activation_times_from_fmap,
    bwd_bounds,
    elem_mask,
    fwd_bounds,
    range_bounds,
)

__all__ = [
    "PackedKV",
    "compute_summary_weights",
    "compute_linear_weights",
    "short_conv",
    "summary_token_position",
    "rope_tables",
    "apply_rope",
    "relative_bin_distance",
    "validate_summary_causality",
    "compute_relative_states",
    "build_dyadic_summaries",
    "pack_levels",
    "multilevel_attention_forward",
    "apply_attention_sink",
    "attention_sink_backward",
    "apply_output_gate",
    "multilevel_attention_backward",
    # Incremental decode.
    "DecodeState",
    "decode_capacities",
    "ring_slot",
    "short_conv_step",
    "init_decode_state",
    "advance_decode_state",
    "multilevel_attention_decode",
    "decode_step",
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
    the contract (see test_reference.test_mixed_precision), not
    x + conv(x).to(dtype).

    Decode: the one-token counterpart is short_conv_step, driven by the
    rolling state of the last K-1 PRE-conv rows kept in DecodeState
    (init_decode_state seeds it from the prefill rows).

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


# ============================================================
# Positional-encoding helpers (Phase-2 only; the summary tree is
# position-independent in every mode).
# ============================================================

def summary_token_position(level: int, local_index: int) -> int:
    """
    Original-token anchor of a summary for post-summary RoPE: the RIGHT
    ENDPOINT of the represented dyadic interval [j*2^l, (j+1)*2^l):

        position = (local_index + 1) * (1 << level) - 1

    Level 0 degenerates to the token index itself (standard RoPE). The
    summary is treated as a virtual token located at the newest token it
    contains ("the state as of its newest token").

    TODO(ablation): midpoint summary position; content-weighted position.
    """
    if level < 0 or local_index < 0:
        raise ValueError(
            f"level and local_index must be nonnegative, got "
            f"({level}, {local_index})"
        )
    return (local_index + 1) * (1 << level) - 1


def rope_tables(
    n_pos: int,
    dim: int,
    base: float = 10000.0,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotary tables (cos, sin), each [n_pos, dim // 2], float32.

    Convention (GPT-NeoX rotate-half, the convention for this repo):
        x1, x2 = x[..., :dim/2], x[..., dim/2:]
        rope(x) = cat(x1*cos - x2*sin, x2*cos + x1*sin)
    with theta_i = base ** (-2i / dim) for i in 0..dim/2-1.
    """
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"dim must be positive and even, got {dim}")
    if n_pos <= 0:
        raise ValueError(f"n_pos must be positive, got {n_pos}")
    inv_freq = base ** (
        -torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
    )
    angles = torch.arange(
        n_pos, device=device, dtype=torch.float32
    )[:, None] * inv_freq[None, :]
    return torch.cos(angles), torch.sin(angles)


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """
    x: [..., D]   positions: integer tensor broadcastable to x.shape[:-1]
    cos/sin: [n_pos, D/2] (see rope_tables) -> rotated x, x.dtype.

    Orthogonal per-position rotation (rotate-half), computed in FP32 and
    cast back. Norm-preserving.

    inverse=True rotates by -theta, i.e. applies R(p)^T = R(p)^{-1} = R(-p):
    the mathematical VJP of the forward rotation. The explicit Phase-2
    backward uses it to rotate accumulated dQ/dK back out of rotated space
    (multilevel_attention_backward, position_mode="rope").
    """
    if cos.shape != sin.shape:
        raise ValueError(
            f"cos shape {tuple(cos.shape)} != sin shape {tuple(sin.shape)}"
        )
    D = x.shape[-1]
    if 2 * cos.shape[-1] != D:
        raise ValueError(
            f"rope tables cover dim {2 * cos.shape[-1]}, x has dim {D}"
        )
    c = cos[positions]  # [..., D/2]
    s = sin[positions]
    if inverse:
        s = -s
    x_fp32 = x.float()
    x1, x2 = x_fp32[..., : D // 2], x_fp32[..., D // 2:]
    rotated = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    return rotated.to(x.dtype)


def validate_summary_causality(spec: RangeSpec) -> None:
    """
    Positional semantics require every visible summary to be entirely
    past-or-current: node (l, j) first becomes visible at
    q = a[l] + 2^l * j and its right endpoint is 2^l * j + 2^l - 1, so the
    schedule must satisfy a[l] >= 2^l - 1 for every level. This holds for
    all standard telescoping schedules (a parent forms only after both
    children complete) but is not enforced by fmap alignment alone.

    Independent of the positional scheme; called by
    multilevel_attention_forward for position_mode "rope"/"relative" only.
    """
    for level in range(1, spec.num_levels):
        if spec.activation_times[level] < (1 << level) - 1:
            raise ValueError(
                f"schedule makes a level-{level} summary visible before its "
                f"interval completes (a[{level}]="
                f"{spec.activation_times[level]} < {(1 << level) - 1}); "
                f"positional semantics undefined"
            )


def relative_bin_distance(
    spec: RangeSpec, q: int, level: int, k_local: int
) -> int:
    """
    0-based chronological bin distance from query q to visible entry
    (level, k_local), in SUMMARY-BIN space.

    The query's visible entries partition a contiguous span of the past
    disjointly across levels (coarse -> fine in time). Ordered oldest ->
    newest they occupy virtual bins with distances M-1 .. 0 from the
    query, where M is the number of visible entries:

        distance 0 = the query's own L0 token (current bin, Inkling's
                     diagonal-at-0 convention: one step = one memory bin)
        distance 1 = the previous memory bin
        ...

    Every visible entry counts as exactly ONE bin regardless of its
    span/level. Derivation: with per-level visible ranges [lo_l', hi_l')
    and t = k_local * 2^level, the entries older than (level, k_local) are
    those whose interval start is < t; per level that count is
    clamp(ceil(t / 2^l'), lo, hi) - lo (starts are unique by disjointness,
    the entry itself excluded by the strict inequality), and

        rank = sum_l' counts,   distance = (M - 1) - rank.

    Distances lie in [0, M-1] and M <= cache_size, so a relative table
    with max_relative_bins >= cache_size always suffices.

    Raises ValueError if the entry is not visible to q.
    """
    if not elem_mask(spec, q, k_local, level):
        raise ValueError(
            f"entry (level={level}, k_local={k_local}) is not visible to "
            f"query {q}"
        )
    t = k_local << level
    total = 0
    rank = 0
    for lp in range(spec.num_levels):
        lo, hi = range_bounds(spec, q, lp)
        total += hi - lo
        ceil_t = (t + (1 << lp) - 1) >> lp
        rank += min(max(ceil_t, lo), hi) - lo
    return (total - 1) - rank


def compute_relative_states(
    x: torch.Tensor,
    relative_weight: torch.Tensor,
    Hq: int,
) -> torch.Tensor:
    """
    x: [B, N, emb_dim]   relative_weight: [Hq * d_rel, emb_dim]
    ->  relative_states: [B, N, Hq, d_rel]

    Query-conditioned relative states from the ORIGINAL hidden states
    (matches nn.Linear(emb_dim, Hq * d_rel, bias=False)); consumed by
    multilevel_attention_forward(position_mode="relative") together with a
    learned relative_proj [d_rel, max_relative_bins].
    """
    B, N, E = x.shape
    rows, Ew = relative_weight.shape
    if Ew != E or rows % Hq != 0:
        raise ValueError(
            f"relative_weight shape {tuple(relative_weight.shape)} "
            f"incompatible with emb_dim={E}, Hq={Hq}"
        )
    return x.matmul(relative_weight.t()).reshape(B, N, Hq, rows // Hq)


def _tile_bin_distances(
    spec: RangeSpec,
    row_lo: torch.Tensor,
    row_hi: torch.Tensor,
    kv_indices: torch.Tensor,
    level: int,
    max_relative_bins: int,
    valid: torch.Tensor,
    q_start: int,
    q_end: int,
) -> torch.Tensor:
    """
    Vectorized relative_bin_distance for one query-block x KV tile.

    row_lo/row_hi: [M, num_levels] per-row visible bounds for ALL levels
    kv_indices:    [Kt] level-local entry indices at `level`
    valid:         [M, Kt] tile-local elem_mask
    ->             dist [M, Kt] long, the 0-based chronological bin distance.

    Valid lanes must satisfy 0 <= dist < max_relative_bins; either violation
    raises (no silent clamping -- a table with >= cache_size bins always
    suffices, and a negative distance on a valid lane means broken range
    invariants; negative indices could silently alias the tensor end).
    Hull-only (invalid) lanes may carry negative distances -- callers clamp
    their gather index to 0 and the element mask keeps them gradient-dead.
    Shared by the forward and the explicit backward so both reconstruct
    identical biases.
    """
    M = row_lo.shape[0]
    t = kv_indices << level  # [Kt]
    rank = torch.zeros(
        M, t.shape[0], device=kv_indices.device, dtype=torch.long
    )
    for lp in range(spec.num_levels):
        ceil_t = (
            (t + (1 << lp) - 1) >> lp
        )[None, :].expand(M, -1)
        rank = rank + torch.maximum(
            torch.minimum(ceil_t, row_hi[:, lp:lp + 1]),
            row_lo[:, lp:lp + 1],
        ) - row_lo[:, lp:lp + 1]
    m_total = (row_hi - row_lo).sum(dim=1)  # [M]
    dist = (m_total[:, None] - 1) - rank
    bad = valid & (dist >= max_relative_bins)
    if bool(bad.any()):
        raise ValueError(
            f"relative bin distance "
            f"{int(dist[valid].max())} in query "
            f"block [{q_start},{q_end}) at level "
            f"{level} exceeds max_relative_bins="
            f"{max_relative_bins}; no silent "
            f"clamping -- a table with >= "
            f"cache_size={spec.cache_size} bins "
            f"always suffices"
        )
    bad_low = valid & (dist < 0)
    if bool(bad_low.any()):
        raise ValueError(
            f"negative relative bin distance "
            f"{int(dist[valid].min())} on a valid lane in query block "
            f"[{q_start},{q_end}) at level {level}; range invariants "
            f"guarantee 0 <= dist for visible entries, so this input is "
            f"semantically broken"
        )
    return dist


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
    mode); used by test/test_reference.py.

    Short-conv mode: with k_conv_weight [Hkv*Dk, K] and v_conv_weight
    [Hkv*Dv, K] both given, short_conv is applied to k and v (flattened over
    heads, channel c = h*D + d, matching fms/train.py's flat post-projection
    layout) BEFORE the merge weights are computed and before the tree is
    built. Both-or-neither; providing exactly one raises. With both None this
    function is bitwise-identical to the pre-conv behavior. See the
    Decode / TODO(padding/packing) notes on short_conv.
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

def _positional_setup(
    position_mode: str,
    rope_cos: Optional[torch.Tensor],
    rope_sin: Optional[torch.Tensor],
    relative_states: Optional[torch.Tensor],
    relative_proj: Optional[torch.Tensor],
    *,
    B: int,
    N: int,
    Hq: int,
    Dk: int,
    spec: RangeSpec,
    rope_positions: int,
) -> Tuple[Optional[torch.Tensor], int]:
    """
    Positional-argument validation shared by the forward, the backward and
    the decode path -- one error matrix, one place. relative_states must be
    [B, N, Hq, d_rel] (N = 1 at decode); rope tables must cover positions
    [0, rope_positions) (N for full sequences, t + 1 at decode).
    Returns (rel_logits [B, N, Hq, max_relative_bins] float32 | None,
    max_relative_bins).
    """
    if position_mode not in ("none", "rope", "relative"):
        raise ValueError(
            f"position_mode must be 'none', 'rope' or 'relative', got "
            f"{position_mode!r}"
        )
    rel_logits = None
    max_relative_bins = 0
    if position_mode == "none":
        if any(t is not None for t in (rope_cos, rope_sin,
                                       relative_states, relative_proj)):
            raise ValueError(
                "positional arguments provided with position_mode='none'"
            )
    elif position_mode == "rope":
        if rope_cos is None or rope_sin is None:
            raise ValueError(
                "position_mode='rope' requires rope_cos and rope_sin"
            )
        if relative_states is not None or relative_proj is not None:
            raise ValueError(
                "relative_* arguments provided with position_mode='rope'"
            )
        if Dk % 2 != 0:
            raise ValueError(f"RoPE requires an even Dk, got {Dk}")
        if rope_cos.shape != rope_sin.shape:
            raise ValueError(
                f"rope_cos shape {tuple(rope_cos.shape)} != rope_sin "
                f"shape {tuple(rope_sin.shape)}"
            )
        if rope_cos.dim() != 2 or rope_cos.shape[-1] * 2 != Dk:
            raise ValueError(
                f"rope tables must be [n_pos, {Dk // 2}], got "
                f"{tuple(rope_cos.shape)}"
            )
        if rope_cos.shape[0] < rope_positions:
            raise ValueError(
                f"rope tables cover {rope_cos.shape[0]} positions; "
                f"summary positions reach N-1={rope_positions - 1}"
            )
        validate_summary_causality(spec)
    else:  # "relative"
        if relative_states is None or relative_proj is None:
            raise ValueError(
                "position_mode='relative' requires relative_states and "
                "relative_proj"
            )
        if rope_cos is not None or rope_sin is not None:
            raise ValueError(
                "rope_* arguments provided with position_mode='relative'"
            )
        if (relative_states.dim() != 4
                or tuple(relative_states.shape[:3]) != (B, N, Hq)):
            raise ValueError(
                f"relative_states shape {tuple(relative_states.shape)} != "
                f"[{B}, {N}, {Hq}, d_rel]"
            )
        if (relative_proj.dim() != 2
                or relative_proj.shape[0] != relative_states.shape[-1]):
            raise ValueError(
                f"relative_proj shape {tuple(relative_proj.shape)} "
                f"incompatible with d_rel={relative_states.shape[-1]}"
            )
        max_relative_bins = relative_proj.shape[1]
        if max_relative_bins < 1:
            raise ValueError("relative_proj must have >= 1 bins")
        validate_summary_causality(spec)
        # Query-conditioned logits over bin distances, computed once.
        rel_logits = torch.einsum(
            "bnhd,dr->bnhr",
            relative_states.float(), relative_proj.float(),
        )  # [B, N, Hq, max_relative_bins]
    return rel_logits, max_relative_bins


def multilevel_attention_forward(
    q: torch.Tensor,
    packed: PackedKV,
    fmap: Dict[int, int],
    cache_size: int,
    block_m: int = 16,
    block_n: int = 32,
    softcap: float = 20.0,
    *,
    position_mode: str = "none",
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    relative_states: Optional[torch.Tensor] = None,
    relative_proj: Optional[torch.Tensor] = None,
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

    Positional encoding (keyword-only; the summary tree is position-
    independent in EVERY mode -- no transform happens before
    build_dyadic_summaries)
    -----------------------------------------------------------------
    position_mode="none" (default): exactly the contract above, bitwise.

    position_mode="rope": post-summary RoPE. Q rows are rotated at their
        token positions; each already-summarized K entry is rotated at the
        RIGHT ENDPOINT of its dyadic interval,
        summary_token_position(l, j) = (j+1)*2^l - 1 -- the summary is a
        virtual token located at the newest token it contains. Level 0
        reduces to standard causal RoPE. V is untouched. This is
        deliberately R(p_s)(sum_i a_i K_i), NOT sum_i a_i R(p_i) K_i:
        rotation does not commute with the softmax-weighted merge.
        Requires rope_cos/rope_sin from rope_tables covering positions
        0..N-1 (rotate-half convention).

    position_mode="relative": learned query-conditioned additive bias in
        SUMMARY-BIN space, added to the raw logit BEFORE softcap:
        X = QK/sqrt(Dk) + b, S = c*tanh(X/c). Every visible entry is one
        bin regardless of span; distance is 0-based chronological rank
        (0 = the query's own L0 token / current bin, 1 = previous memory
        bin, ...; see relative_bin_distance). No rel_extent cutoff: every
        attended entry gets a bias; a valid distance >= relative_proj's
        bin count raises (max_relative_bins >= cache_size always
        suffices). Takes relative_states [B, N, Hq, d_rel] (from
        compute_relative_states over the ORIGINAL hidden states) and
        relative_proj [d_rel, max_relative_bins];
        rel_logits = relative_states @ relative_proj, gathered per pair.
        The bias add promotes scores to FP32 -- the kernel port's
        precision point.

    Decode: multilevel_attention_decode applies the identical rules to the
    ring cache -- RoPE positions from (level, j) as here, bin distances from
    the same _tile_bin_distances at q = t -- so no positional state is ever
    cached; only unrotated K is.
    TODO(kernel): CuTeDSL RoPE forward/backward; learned-relative score
    bias + parameter gradients in the tile loop.
    TODO(ablation): midpoint / content-weighted summary positions;
    level/span embedding in relative mode; original-token-distance
    relative bias.
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

    rel_logits, max_relative_bins = _positional_setup(
        position_mode, rope_cos, rope_sin, relative_states, relative_proj,
        B=B, N=N, Hq=Hq, Dk=Dk, spec=spec, rope_positions=N,
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

                # Per-row visible bounds for ALL levels, [M, num_levels]
                # (integer-identical to the previous per-level lists; the
                # relative mode needs the cross-level view for bin ranks).
                bounds = [
                    [range_bounds(spec, qi, lv)
                     for lv in range(spec.num_levels)]
                    for qi in range(q_start, q_end)
                ]
                row_lo = torch.tensor(
                    [[lo for lo, _ in row] for row in bounds],
                    device=q.device, dtype=torch.long,
                )
                row_hi = torch.tensor(
                    [[hi for _, hi in row] for row in bounds],
                    device=q.device, dtype=torch.long,
                )

                if position_mode == "rope":
                    q_pos = torch.arange(
                        q_start, q_end, device=q.device
                    )[:, None]  # broadcast over E
                    Q = apply_rope(Q, q_pos, rope_cos, rope_sin)
                elif position_mode == "relative":
                    rel_t = rel_logits[b, q_start:q_end, hq_slice]

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

                    level_start = row_lo[:, level]
                    level_end = row_hi[:, level]

                    for k_start in range(
                        tile_k_lo, tile_k_hi, block_n
                    ):
                        k_end = min(k_start + block_n, tile_k_hi)

                        K, V = _kv_tile(
                            packed, b, hkv, level, k_start, k_end
                        )  # [Ktile, Dk], [Ktile, Dv]

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

                        if position_mode == "rope":
                            # summary_token_position, vectorized: the
                            # summary is a virtual token at its interval's
                            # right endpoint. V is untouched.
                            k_pos = (kv_indices + 1) * (1 << level) - 1
                            K = apply_rope(K, k_pos, rope_cos, rope_sin)

                        scores = torch.einsum(
                            "med,kd->mek", Q, K
                        ) * scale

                        if position_mode == "relative":
                            # 0-based chronological bin distance; shared
                            # with the explicit backward.
                            dist = _tile_bin_distances(
                                spec, row_lo, row_hi, kv_indices, level,
                                max_relative_bins, valid, q_start, q_end,
                            )
                            # Hull-only (invalid) lanes may have dist < 0:
                            # clamp their gather index to 0; the -inf mask
                            # below kills any gradient into those bins.
                            idx = torch.where(
                                valid, dist, torch.zeros_like(dist)
                            )
                            bias = torch.gather(
                                rel_t, 2,
                                idx[:, None, :].expand(M, expansion, -1),
                            )
                            # X = QK/sqrt(Dk) + b, BEFORE softcap; the
                            # fp32 bias promotes scores to fp32.
                            scores = scores + bias

                        if softcap is not None:
                            scores = softcap * torch.tanh(
                                scores / softcap
                            )

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
# Optional post-attention composition: learned sink, output gate.
#
# Both are pure composable post-ops over the frozen (out, lse) contract;
# the attention kernel and the tree are untouched, and autograd provides
# their backward.
# ============================================================

def apply_attention_sink(
    out: torch.Tensor,
    lse: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    """
    out: [B, N, Hq, Dv]   lse: [B, N, Hq] (fp32, natural log)   sinks: [Hq]

        out_sink = out * sigmoid(lse - sinks[h])

    Equivalent to one extra softmax entry per query head with logit
    sinks[h] and an identically-zero value: the denominator gains
    exp(sinks[h]), the numerator is unchanged. No sink K/V token exists and
    nothing enters the KV cache. Sinks index QUERY heads (GQA: [Hq], not
    [Hkv]). The scale is computed in FP32 and cast to out.dtype.

    A zero sink is NOT an identity (it still adds e^0 = 1 to the
    denominator); only skipping this call reproduces plain attention.

    NOTE: gradient flows into lse (and through it into the attention
    logits). Autograd handles this; for the explicit path, use
    attention_sink_backward to get (dout_pre, dlse, dsinks) and pass dlse
    into multilevel_attention_backward(..., dlse=dlse).

    TODO(kernel): fuse into the online softmax (the sink joins the m/l
    running statistics as one extra logit, contributing zero to acc); stop
    exposing lse once fused -- the dlse contract in
    multilevel_attention_backward already specifies the extra backward term.
    """
    B, N, Hq, Dv = out.shape
    if tuple(lse.shape) != (B, N, Hq):
        raise ValueError(
            f"lse shape {tuple(lse.shape)} != {(B, N, Hq)}"
        )
    if tuple(sinks.shape) != (Hq,):
        raise ValueError(
            f"sinks shape {tuple(sinks.shape)} != {(Hq,)} (one logit per "
            f"query head)"
        )
    scale = torch.sigmoid(lse.float() - sinks.float()[None, None, :])
    return out * scale.unsqueeze(-1).to(out.dtype)


def attention_sink_backward(
    out: torch.Tensor,
    lse: torch.Tensor,
    sinks: torch.Tensor,
    dout: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VJP of apply_attention_sink at the (out, lse) boundary.

    out/lse/sinks: as in apply_attention_sink.  dout: [B, N, Hq, Dv], the
    gradient wrt the POST-sink output.  Returns

        dout_pre [B, N, Hq, Dv] (dout.dtype)  gradient wrt the pre-sink out
        dlse     [B, N, Hq]     (float32)     gradient wrt the forward lse
        dsinks   [Hq]           (sinks.dtype)

    With r = sigmoid(lse - sinks):  dout_pre = r * dout,
    dlse = <dout, out> * r * (1 - r),  dsinks = -sum_{b,n} dlse.

    This mirrors apply_attention_sink's mixed-precision graph, including
    the forward cast of r to out.dtype. The VJP matches torch.autograd
    within floating-point precision (dout_pre is typically bitwise; dlse
    and dsinks may differ by fp32 rounding since autograd's sigmoid
    backward need not use the same operation ordering as r * (1 - r)).

    Requires dout.dtype == out.dtype (what autograd produces); a manually
    supplied fp32 dout for a bf16 out would not represent the VJP of the
    actual mixed-precision forward.

    Composition with the explicit Phase-2 backward:

        dout_pre, dlse, dsinks = attention_sink_backward(out, lse, sinks, g)
        dq, dk_p, dv_p, _, _ = multilevel_attention_backward(
            q, packed, out, lse, dout_pre, fmap, cache_size, dlse=dlse)
    """
    B, N, Hq, Dv = out.shape
    if tuple(lse.shape) != (B, N, Hq):
        raise ValueError(f"lse shape {tuple(lse.shape)} != {(B, N, Hq)}")
    if tuple(sinks.shape) != (Hq,):
        raise ValueError(
            f"sinks shape {tuple(sinks.shape)} != {(Hq,)} (one logit per "
            f"query head)"
        )
    if tuple(dout.shape) != (B, N, Hq, Dv):
        raise ValueError(
            f"dout shape {tuple(dout.shape)} != {(B, N, Hq, Dv)}"
        )
    if dout.dtype != out.dtype:
        raise ValueError(
            f"dout dtype {dout.dtype} must match out dtype {out.dtype}"
        )
    r = torch.sigmoid(lse.float() - sinks.float()[None, None, :])  # fp32
    r_out = r.to(out.dtype)                     # the forward's exact cast
    dout_pre = dout * r_out.unsqueeze(-1)       # same-dtype math as forward
    # VJP through (out * r_out): grad wrt r_out is <dout, out> summed over
    # Dv in the output dtype; the cast-backward then promotes to fp32.
    g_dot = (dout * out).sum(dim=-1).float()
    dlse = g_dot * r * (1.0 - r)                # sigmoid backward, fp32
    dsinks = (-dlse).sum(dim=(0, 1)).to(sinks.dtype)
    return dout_pre, dlse, dsinks


def apply_output_gate(
    out: torch.Tensor,
    x: torch.Tensor,
    gate_weight: torch.Tensor,
) -> torch.Tensor:
    """
    out: [B, N, Hq, Dv]   x: [B, N, emb_dim]   gate_weight: [Hq*Dv, emb_dim]

        gated = SiLU(x @ gate_weight^T).reshape(B, N, Hq, Dv) * out

    x is the ORIGINAL attention-input hidden state (pre-projection
    query-side, no RoPE/positional transforms). SiLU, not sigmoid -- the
    gate may suppress, amplify, or go slightly negative. Applied after
    attention (and after any sink rescale), before the output projection.
    Ordinary mixed precision; no FP32 requirement. No cache/decode state:
    at decode the gate is recomputed from the current token's hidden state.

    TODO(perf): optionally fuse SiLU(gate) * out into the output-projection
    input epilogue if profiling shows it matters.
    """
    B, N, Hq, Dv = out.shape
    if x.dim() != 3 or x.shape[0] != B or x.shape[1] != N:
        raise ValueError(
            f"x shape {tuple(x.shape)} incompatible with out "
            f"{tuple(out.shape)}; expected [B, N, emb_dim]"
        )
    if tuple(gate_weight.shape) != (Hq * Dv, x.shape[-1]):
        raise ValueError(
            f"gate_weight shape {tuple(gate_weight.shape)} != "
            f"{(Hq * Dv, x.shape[-1])}"
        )
    gate = torch.nn.functional.silu(x.matmul(gate_weight.t()))
    return gate.reshape(B, N, Hq, Dv) * out


# ============================================================
# Phase 2 backward: explicit two-pass FlashAttention-style.
#
#   Pass A (Q-owned dQ):    Q-block --fwd_bounds--> K-tiles
#   Pass B (KV-owned dK/dV): K-tile --bwd_bounds--> Q-tiles
#
# Per Q x K tile (all FP32):
#   X  = Q K^T * scale [+ b]               scale = 1/sqrt(Dk); b only in
#                                          relative mode (pre-softcap bias)
#   S  = c*tanh(X/c)  (softcap=c) | X      softcap_grad = 1 - tanh(X/c)^2 | 1
#   P  = exp(S - LSE_q) on valid (q,k), else 0     <- SAVED forward lse
#   Delta_q     = sum_d dO_qd O_qd                 <- computed once
#   Delta_eff_q = Delta_q - dLSE_q                 <- only if a dlse seed is
#             given (dLSE/dS_i = P_i, so dS = P(dP - Delta + dlse)
#             = P(dP - Delta_eff)); the definition of Delta is unchanged
#   dV += P^T dO      dP = dO V^T      dS = P * (dP - Delta_eff)
#   dX  = dS * softcap_grad
#   dQ += scale * dX K                 dK += scale * dX^T Q
#   db  = dX (X affine in b): Pass A scatter-adds dX into each row's bins
#
# The hulls from fwd_bounds / bwd_bounds (and BLOCK_M alignment of the
# latter) are conservative; the tile-local element mask is exact, so
# invalid entries have P = dS = dX = 0 and over-enumeration changes only
# work, never semantics. `lse` must be the forward's saved value; it is
# never recomputed.
# ============================================================

def _tile_grads(
    Q, K, V, dO, lse_t, delta_t, valid, scale, softcap, *, need_dq, need_dkv,
    bias=None, need_dbias=False,
):
    """
    Shared per-tile backward math for both passes.

    Q [M,E,Dk]  K [Kt,Dk]  V [Kt,Dv]  dO [M,E,Dv]  lse_t/delta_t [M,E]
    valid [M,Kt] (tile-local elem_mask, vectorized)
    bias [M,E,Kt] fp32 | None: additive pre-softcap score bias (the
    forward's relative mode: X = QK*scale + bias). Since X is affine in
    the bias, its gradient is exactly dX -- returned as dBias when
    need_dbias (already zeroed on invalid lanes). Invalid-lane bias values
    are irrelevant: the P reconstruction overrides them with -inf.
    Returns (dQ [M,E,Dk] | None, dK [Kt,Dk] | None, dV [Kt,Dv] | None,
    dBias [M,E,Kt] | None), FP32.
    """
    if need_dbias and bias is None:
        raise ValueError("need_dbias=True requires bias")
    Qf, Kf, Vf, dOf = Q.float(), K.float(), V.float(), dO.float()

    x = torch.einsum("med,kd->mek", Qf, Kf) * scale
    if bias is not None:
        if bias.shape != x.shape:
            raise ValueError(
                f"bias shape {tuple(bias.shape)} != scores shape "
                f"{tuple(x.shape)}"
            )
        x = x + bias
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
    dBias = dX if need_dbias else None
    return dQ, dK, dV, dBias


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
    *,
    dlse: Optional[torch.Tensor] = None,
    position_mode: str = "none",
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    relative_states: Optional[torch.Tensor] = None,
    relative_proj: Optional[torch.Tensor] = None,
):
    """
    Explicit Phase-2 backward at the packed boundary (see block comment).

    position_mode="none" (default): recomputes the scores from Q/K with no
    rotation or positional bias; rope_cos/rope_sin must be None.

    position_mode="rope": recomputes the scores in ROTATED space, exactly
    as the forward -- Q rows rotated at their token positions, K entries
    rotated at their summary positions (j+1)*2^l - 1, V untouched -- then
    applies the inverse (orthogonal) rotation to the accumulated dQ/dK
    before writing them out: dX = R(p)^T dX_rot. Deferring the inverse to
    the accumulated tile is exact because every contribution to a row
    (resp. a K entry) shares the same rotation, so R^T distributes over
    the sum. Requires rope_cos/rope_sin (same tables as the forward). The
    saved `lse` is reused unchanged: it is the LSE of the rotated scores,
    which is what the rotated-space P reconstruction needs. Inverse RoPE
    is the mathematical VJP of the rotation; this backward accumulates in
    FP32 and casts once at its output boundary, so for non-FP32 inputs
    tiny rounding differences vs. autograd (which casts per apply_rope
    call) are possible.

    position_mode="relative": recomputes the biased scores exactly as the
    forward -- X = QK*scale + b before softcap, b gathered per (row, bin
    distance) from rel_logits = relative_states.float() @
    relative_proj.float() -- in BOTH passes (P depends on the bias), and
    accumulates the positional gradient in Pass A only (the Q-owned
    traversal visits every valid (q, k) pair exactly once, and rel_logits
    is Q-indexed). Since X is affine in b, db = dX (post-softcap-chain,
    NOT dS): each tile's dX is scatter-added into its rows' bins,
    producing the returned dposition = drel_logits. Requires
    relative_states [B, N, Hq, d_rel] and relative_proj
    [d_rel, max_relative_bins], the forward's exact arguments.

    Contracts
    ---------
        q        [B, N, Hq, Dk]      out   [B, N, Hq, Dv]
        packed.k [B, sumN, Hkv, Dk]  lse   [B, N, Hq] float32 (saved forward)
        packed.v [B, sumN, Hkv, Dv]  dout  [B, N, Hq, Dv]
        dlse     [B, N, Hq] floating, optional (keyword-only): gradient seed
                 wrt the RETURNED lse, e.g. from a downstream attention sink
                 (see attention_sink_backward). Since dLSE/dS_i = P_i it
                 folds into Delta once: dS = P(dP - (Delta - dlse)). None
                 reproduces the pre-sink contract bitwise. Orthogonal to
                 the positional modes.

        dq        [B, N, Hq, Dk]     (q.dtype)
        dk_packed [B, sumN, Hkv, Dk] (packed.k.dtype)
        dv_packed [B, sumN, Hkv, Dv] (packed.v.dtype)
        dposition None                            ("none"/"rope")
                  drel_logits [B, N, Hq, max_relative_bins] float32
                                                  ("relative")
        stats     diagnostic dict (never asserted on)

    The return arity is FIXED at five fields for every mode; dposition is
    simply None when the mode has no positional parameters.

    Non-differentiable by construction (@torch.no_grad): it reads graph-
    attached packed.k/v VALUES without building a higher-order graph, so a
    caller may still use the returned dk/dv_packed as VJP seeds into the
    Phase-1 graph that produced packed.k/v:

        dq_tree, dk, dv = torch.autograd.grad(
            (packed.k, packed.v), (q, k, v), (dk_packed, dv_packed))
        dq_total = dq + dq_tree

    Likewise drel_logits is the VJP seed at the rel_logits boundary;
    either compose in closed form,

        drel_states = einsum("bnhr,dr->bnhd", drel_logits, proj.float())
        drel_proj   = einsum("bnhd,bnhr->dr", states.float(), drel_logits)

    or recompute a graph-attached rel_logits and
    torch.autograd.grad(rel_logits, (states, proj), drel_logits).

    TODO(kernel): the reference materializes drel_logits to make the
    contract verifiable; [B, N, Hq, max_relative_bins] gets large at long
    context, so the kernel may instead fold tile-local dX directly into
    drelative_states and a reduced drelative_proj.
    """
    if position_mode not in ("none", "rope", "relative"):
        raise ValueError(
            f"position_mode must be 'none', 'rope' or 'relative', got "
            f"{position_mode!r}"
        )
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

    # Positional-argument validation mirrors multilevel_attention_forward's
    # (shared helper); rel_logits is the forward's exact value, recomputed.
    rel_logits, max_relative_bins = _positional_setup(
        position_mode, rope_cos, rope_sin, relative_states, relative_proj,
        B=B, N=N, Hq=Hq, Dk=Dk, spec=spec, rope_positions=N,
    )

    scale = 1.0 / math.sqrt(Dk)
    device = q.device

    # Delta_q = sum_d dO_qd O_qd = sum_k P_qk dP_qk: no pre-pass over KV.
    delta = (dout.float() * out.float()).sum(dim=-1)  # [B, N, Hq]
    if dlse is not None:
        if tuple(dlse.shape) != (B, N, Hq):
            raise ValueError(
                f"dlse shape {tuple(dlse.shape)} != {(B, N, Hq)}"
            )
        if not dlse.dtype.is_floating_point:
            raise ValueError("dlse must have floating dtype")
        # dS = P (dP - Delta + dlse) since dLSE/dS_i = P_i: fold once here.
        delta = delta - dlse.float()

    dq_acc = torch.zeros(B, N, Hq, Dk, device=device, dtype=torch.float32)
    dk_acc = torch.zeros_like(packed.k, dtype=torch.float32)
    dv_acc = torch.zeros_like(packed.v, dtype=torch.float32)
    drel_acc = None
    if position_mode == "relative":
        drel_acc = torch.zeros(
            B, N, Hq, max_relative_bins, device=device, dtype=torch.float32
        )

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

    def row_bounds_all(q0, q1):
        # All-levels [Mq, num_levels] view; the relative mode needs the
        # cross-level matrices for bin ranks (same values as the forward's).
        rows = [
            [range_bounds(spec, qi, lv) for lv in range(spec.num_levels)]
            for qi in range(q0, q1)
        ]
        lo = torch.tensor(
            [[a for a, _ in row] for row in rows],
            device=device, dtype=torch.long,
        )
        hi = torch.tensor(
            [[c for _, c in row] for row in rows],
            device=device, dtype=torch.long,
        )
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

                if position_mode == "rope":
                    q_pos = torch.arange(
                        q_start, q_end, device=device
                    )[:, None]
                    Q = apply_rope(Q, q_pos, rope_cos, rope_sin)
                elif position_mode == "relative":
                    row_lo_all, row_hi_all = row_bounds_all(q_start, q_end)
                    rel_t = rel_logits[b, q_start:q_end, hq_slice]
                    drel_t = torch.zeros(
                        M, E, max_relative_bins,
                        device=device, dtype=torch.float32,
                    )

                for level in range(spec.num_levels):
                    tile_k_lo, tile_k_hi = fwd_bounds(spec, q_start, q_end, level)
                    if tile_k_lo == tile_k_hi:
                        continue
                    if position_mode == "relative":
                        row_lo = row_lo_all[:, level]
                        row_hi = row_hi_all[:, level]
                    else:
                        row_lo, row_hi = row_bounds(q_start, q_end, level)
                    for k_start in range(tile_k_lo, tile_k_hi, block_n):
                        k_end = min(k_start + block_n, tile_k_hi)
                        K, V = _kv_tile(packed, b, hkv, level, k_start, k_end)
                        kv_idx = torch.arange(k_start, k_end, device=device)
                        valid = (kv_idx[None, :] >= row_lo[:, None]) & (
                            kv_idx[None, :] < row_hi[:, None]
                        )
                        bias = None
                        idx3 = None
                        if position_mode == "rope":
                            k_pos = (kv_idx + 1) * (1 << level) - 1
                            K = apply_rope(K, k_pos, rope_cos, rope_sin)
                        elif position_mode == "relative":
                            # The forward's exact bias reconstruction.
                            dist = _tile_bin_distances(
                                spec, row_lo_all, row_hi_all, kv_idx, level,
                                max_relative_bins, valid, q_start, q_end,
                            )
                            idx = torch.where(
                                valid, dist, torch.zeros_like(dist)
                            )
                            idx3 = idx[:, None, :].expand(M, E, -1)
                            bias = torch.gather(rel_t, 2, idx3)
                        dQ_t, _, _, dB = _tile_grads(
                            Q, K, V, dO, lse_t, delta_t, valid, scale, softcap,
                            need_dq=True, need_dkv=False,
                            bias=bias,
                            need_dbias=position_mode == "relative",
                        )
                        dQ_tile += dQ_t
                        if position_mode == "relative":
                            # db = dX (bias enters X before softcap); dX is
                            # zero off-mask, so clamped invalid lanes add
                            # exact zeros to bin 0.
                            drel_t.scatter_add_(2, idx3, dB)

                if position_mode == "rope":
                    # dQ accumulated in rotated space; one inverse rotation
                    # per row (all contributions share R(p_q)).
                    dQ_tile = apply_rope(
                        dQ_tile, q_pos, rope_cos, rope_sin, inverse=True
                    )
                elif position_mode == "relative":
                    # Pass A is the sole owner of the positional gradient:
                    # the Q-owned traversal visits every valid (q, k) pair
                    # exactly once (Pass B would double-count).
                    drel_acc[b, q_start:q_end, hq_slice] = drel_t
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

                    if position_mode == "rope":
                        k_pos = (kv_idx + 1) * (1 << level) - 1
                        K = apply_rope(K, k_pos, rope_cos, rope_sin)

                    # Aligned BLOCK_M Q tiles intersecting the hull.
                    for q0 in range((q_lo // block_m) * block_m, q_hi, block_m):
                        q1 = min(q0 + block_m, N)
                        stats["q_tiles_enumerated"] += 1
                        if position_mode == "relative":
                            row_lo_all, row_hi_all = row_bounds_all(q0, q1)
                            row_lo = row_lo_all[:, level]
                            row_hi = row_hi_all[:, level]
                        else:
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
                        bias = None
                        if position_mode == "rope":
                            q_pos = torch.arange(
                                q0, q1, device=device
                            )[:, None]
                            Q = apply_rope(Q, q_pos, rope_cos, rope_sin)
                        elif position_mode == "relative":
                            # P depends on the bias, so Pass B must
                            # reconstruct it too; it accumulates no
                            # positional gradient (Pass A owns that).
                            dist = _tile_bin_distances(
                                spec, row_lo_all, row_hi_all, kv_idx, level,
                                max_relative_bins, valid, q0, q1,
                            )
                            idx = torch.where(
                                valid, dist, torch.zeros_like(dist)
                            )
                            bias = torch.gather(
                                rel_logits[b, q0:q1, hq_slice], 2,
                                idx[:, None, :].expand(q1 - q0, E, -1),
                            )
                        _, dK_t, dV_t, _ = _tile_grads(
                            Q, K, V, dO, lse_t, delta_t, valid, scale, softcap,
                            need_dq=False, need_dkv=True, bias=bias,
                        )
                        dK_tile += dK_t
                        dV_tile += dV_t

                    if position_mode == "rope":
                        # dK accumulated in rotated space; one inverse
                        # rotation per entry (all contributions share its
                        # summary-position rotation). dV: V never rotated.
                        dK_tile = apply_rope(
                            dK_tile, k_pos, rope_cos, rope_sin, inverse=True
                        )

                    # Storage addressing: offset[level] + local, checked once.
                    p_start, p_end = _packed_slice(packed, level, k_start, k_end)
                    dk_acc[b, p_start:p_end, hkv] = dK_tile
                    dv_acc[b, p_start:p_end, hkv] = dV_tile

    return (
        dq_acc.to(q.dtype),
        dk_acc.to(packed.k.dtype),
        dv_acc.to(packed.v.dtype),
        drel_acc,  # None unless position_mode == "relative"
        stats,
    )


# ============================================================
# Incremental decode: per-level modular rings driven by range_spec.
#
# State after consuming tokens 0..t-1: for every level l the nodes in
# range_bounds(spec_t, t-1, l), spec_t = RangeSpec(a, cache_size, t), at
# physical slots ring_offsets[l] + j % caps[l]. Step t:
#
#   1. the one level l >= 1 with t >= a[l] and (t - a[l]) % 2^l == 0 (the
#      dyadic alignment a[l] = 2^(l-1) mod 2^l makes it unique -- the ruler
#      tick) creates node j = (t - a[l]) >> l from children (l-1, 2j) and
#      (l-1, 2j+1), which leave level l-1's range exactly at t and are
#      therefore still live; the merge is build_dyadic_summaries' pair
#      merge, softmax(w)-weighted with logsumexp(w) as the parent weight;
#   2. token t is written at level 0 AFTER the merge (its slot can coincide
#      with child 2j's when t - 2j == caps[0]);
#   3. query t attends range_bounds(spec_{t+1}, t, l) per level.
#
# caps[l] bounds the live count of level l over the whole schedule (fine
# counts are periodic once q >= a[l+1]; the coarsest count is bounded by
# the slots the fine levels leave over), so a contiguous live interval never
# collides with itself modulo caps[l]. Eviction at the coarsest level is
# implicit: an evicted node leaves the range and its slot is reused caps[L]
# nodes later. Every gathered lane is valid by construction (the per-row
# ranges are exact), so the decode attention needs no element mask; it
# keeps the forward's level/tile loop only so a decode kernel inherits the
# same skeleton.
# ============================================================

@dataclass
class DecodeState:
    """
    One layer's decode cache. Tensors are level-major rings:

        k [B, sum(caps), Hkv, Dk]   UNROTATED, post-conv keys
        v [B, sum(caps), Hkv, Dv]
        w [B, sum(caps), Hkv, 1]    merge log-weights (future pair merges)
        conv_k / conv_v [B, K-1, Hkv*D] | None
                                    last K-1 PRE-conv flat rows, newest last
        t                           tokens consumed so far == next query index

    Node j of level l lives at ring_slot(state, l, j); range_bounds alone
    says which nodes are live. Mutated in place by advance_decode_state.
    """

    k: torch.Tensor
    v: torch.Tensor
    w: torch.Tensor
    conv_k: Optional[torch.Tensor]
    conv_v: Optional[torch.Tensor]
    t: int
    activation_times: Tuple[int, ...]
    cache_size: int
    caps: Tuple[int, ...]
    ring_offsets: Tuple[int, ...]

    @property
    def num_levels(self) -> int:
        return len(self.activation_times)

    def spec(self, seq_len: int) -> RangeSpec:
        """The prefill schedule truncated to `seq_len` tokens."""
        return RangeSpec(self.activation_times, self.cache_size, seq_len)


def decode_capacities(
    activation_times: Tuple[int, ...], cache_size: int
) -> Tuple[int, ...]:
    """
    Per-level ring capacities: caps[l] = max_q |R_l(q)| for fine levels and
    max_q (cache_size - fine usage at q) for the coarsest (its live count is
    min(remaining, newest + 1) <= remaining). Scanned over two schedule
    periods past the last activation: every fine count is periodic with
    period 2^(l+1) once q >= a[l+1], so the joint pattern repeats with
    period 2^L after a[L]. sum(caps) may exceed cache_size (each ring holds
    its own level's peak, not the joint peak).
    """
    a = tuple(activation_times)
    if not a or a[0] != 0:
        raise ValueError("activation_times must start with a[0] = 0")
    if cache_size <= 0:
        raise ValueError("cache_size must be positive")
    L = len(a) - 1
    horizon = 2 * (a[L] + (1 << (L + 1))) + 1
    spec = RangeSpec(a, cache_size, horizon)
    caps = [0] * (L + 1)
    for q in range(horizon):
        used = 0
        for level in range(L):
            lo, hi = range_bounds(spec, q, level)
            caps[level] = max(caps[level], hi - lo)
            used += hi - lo
        if used > cache_size:
            raise ValueError(
                f"fine levels need {used} slots > cache_size={cache_size} "
                f"at q={q}"
            )
        caps[L] = max(caps[L], cache_size - used)
    return tuple(caps)


def ring_slot(state: DecodeState, level: int, j: int) -> int:
    """Physical slot of node j at `level`: ring_offsets[level] + j % caps[level].
    The ONLY place that maps a level-local node index to storage."""
    if not 0 <= level < state.num_levels:
        raise ValueError(
            f"level {level} out of range [0, {state.num_levels - 1}]"
        )
    if j < 0:
        raise ValueError(f"node index must be nonnegative, got {j}")
    return state.ring_offsets[level] + j % state.caps[level]


def _ring_slots(
    state: DecodeState, level: int, indices: torch.Tensor
) -> torch.Tensor:
    """Vectorized ring_slot for a level-local index tensor (may wrap)."""
    return state.ring_offsets[level] + torch.remainder(
        indices, state.caps[level]
    )


def _conv_history(x_pre: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    The last K-1 rows of x_pre [B, N, C], left zero-padded when N < K-1
    (newest row last) -- the rolling state short_conv_step consumes.
    K == 1 yields an empty [B, 0, C] (Python's x[:, -0:] would be the whole
    sequence, hence the explicit branch).
    """
    history = kernel_size - 1
    if history == 0:
        return x_pre[:, :0]
    pad_rows = max(history - x_pre.shape[1], 0)
    return torch.nn.functional.pad(x_pre, (0, 0, pad_rows, 0))[:, -history:]


def short_conv_step(
    x: torch.Tensor, state: torch.Tensor, weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One-token counterpart of short_conv.

    x: [B, 1, C] current PRE-conv row   state: [B, K-1, C] previous pre-conv
    rows, newest last   weight: [C, K]   ->  (y [B, 1, C], new_state [B, K-1, C])

        full = cat(state, x)                          [B, K, C]
        y[b, 0, c] = x[b, 0, c] + sum_j weight[c, j] * full[b, j, c]

    Tap K-1 is the current token, i.e. short_conv's sum at t = the newest
    row, with the same contract: FP32 taps, residual ADDED IN FP32, one cast
    back. new_state drops the oldest row. Note the explicit [B, C] + [B, C]
    residual add (x[:, 0] + y), not [B, 1, C] + [B, C].
    """
    if x.dim() != 3 or x.shape[1] != 1:
        raise ValueError(f"x must be [B, 1, C], got {tuple(x.shape)}")
    B, _, C = x.shape
    if weight.dim() != 2 or weight.shape[0] != C:
        raise ValueError(
            f"conv weight shape {tuple(weight.shape)} incompatible with "
            f"C={C}; expected [C, K]"
        )
    K = weight.shape[1]
    if K <= 0:
        raise ValueError(f"conv kernel size must be positive, got {K}")
    if tuple(state.shape) != (B, K - 1, C):
        raise ValueError(
            f"conv state shape {tuple(state.shape)} != {(B, K - 1, C)}"
        )
    full = torch.cat([state, x], dim=1)  # [B, K, C]
    y = (full.float() * weight.float().t().unsqueeze(0)).sum(dim=1)  # [B, C]
    out = (x[:, 0].float() + y).to(x.dtype).unsqueeze(1)  # [B, 1, C]
    return out, full[:, 1:]


def init_decode_state(
    k_levels: List[torch.Tensor],
    v_levels: List[torch.Tensor],
    w_levels: List[torch.Tensor],
    fmap: Dict[int, int],
    cache_size: int,
    *,
    k_pre_conv: Optional[torch.Tensor] = None,
    v_pre_conv: Optional[torch.Tensor] = None,
    conv_kernel_size: Optional[int] = None,
) -> DecodeState:
    """
    Prefill -> decode hand-off. k_levels/v_levels/w_levels are
    build_dyadic_summaries' outputs for the N prefill tokens (post-conv,
    unrotated); the state receives, for every level, the nodes live at the
    last prefill query N-1 (range_bounds(spec_N, N-1, l)), and t = N.

    Short-conv mode: pass the RAW pre-conv k/v that went into
    build_dyadic_summaries ([B, N, Hkv, D] or flat [B, N, Hkv*D]) together
    with conv_kernel_size; the last K-1 flat rows seed conv_k/conv_v. All
    three or none.
    """
    if (k_pre_conv is None) != (v_pre_conv is None):
        raise ValueError(
            "k_pre_conv and v_pre_conv must both be provided or both be None"
        )
    if (k_pre_conv is None) != (conv_kernel_size is None):
        raise ValueError(
            "conv_kernel_size must accompany the pre-conv rows (and vice versa)"
        )
    a = activation_times_from_fmap(fmap)
    L = len(a) - 1
    if min(len(k_levels), len(v_levels), len(w_levels)) < L + 1:
        raise ValueError(
            f"need {L + 1} levels, got {len(k_levels)}/{len(v_levels)}/"
            f"{len(w_levels)}"
        )
    B, N, Hkv, Dk = k_levels[0].shape
    Dv = v_levels[0].shape[-1]
    if N < 1:
        raise ValueError("prefill must contain at least one token")
    if tuple(w_levels[0].shape) != (B, N, Hkv, 1):
        raise ValueError(
            f"w_levels[0] shape {tuple(w_levels[0].shape)} != "
            f"{(B, N, Hkv, 1)}"
        )
    caps = decode_capacities(a, cache_size)
    offsets = [0]
    for c in caps:
        offsets.append(offsets[-1] + c)
    total = offsets[-1]
    state = DecodeState(
        k=k_levels[0].new_zeros(B, total, Hkv, Dk),
        v=v_levels[0].new_zeros(B, total, Hkv, Dv),
        w=w_levels[0].new_zeros(B, total, Hkv, 1),
        conv_k=None,
        conv_v=None,
        t=N,
        activation_times=a,
        cache_size=cache_size,
        caps=caps,
        ring_offsets=tuple(offsets),
    )
    spec = RangeSpec(a, cache_size, N)
    for level in range(L + 1):
        lo, hi = range_bounds(spec, N - 1, level)
        if lo == hi:
            continue
        if hi > k_levels[level].shape[1] or hi > v_levels[level].shape[1] \
                or hi > w_levels[level].shape[1]:
            raise ValueError(
                f"level {level}: live range [{lo},{hi}) exceeds the given "
                f"level tensors"
            )
        slots = _ring_slots(
            state, level,
            torch.arange(lo, hi, device=state.k.device, dtype=torch.long),
        )
        state.k[:, slots] = k_levels[level][:, lo:hi]
        state.v[:, slots] = v_levels[level][:, lo:hi]
        state.w[:, slots] = w_levels[level][:, lo:hi]
    if k_pre_conv is not None:
        if conv_kernel_size <= 0:
            raise ValueError(
                f"conv kernel size must be positive, got {conv_kernel_size}"
            )
        if k_pre_conv.shape[:2] != (B, N) or v_pre_conv.shape[:2] != (B, N):
            raise ValueError(
                f"pre-conv rows must be [B={B}, N={N}, ...], got "
                f"{tuple(k_pre_conv.shape)} / {tuple(v_pre_conv.shape)}"
            )
        state.conv_k = _conv_history(
            k_pre_conv.reshape(B, N, -1), conv_kernel_size
        )
        state.conv_v = _conv_history(
            v_pre_conv.reshape(B, N, -1), conv_kernel_size
        )
    return state


def advance_decode_state(
    state: DecodeState,
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    *,
    x_t: Optional[torch.Tensor] = None,
    w_proj: Optional[torch.Tensor] = None,
    k_conv_weight: Optional[torch.Tensor] = None,
    v_conv_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Consume token t = state.t (in place; state.t becomes t + 1).

    q_t [B, 1, Hq, Dk], k_t [B, 1, Hkv, Dk], v_t [B, 1, Hkv, Dv] are the RAW
    projections (pre-conv, unrotated). Steps, in this order:

      1. short conv (both weights or neither; the state must have been
         initialised with pre-conv rows iff weights are given): k/v are
         convolved on the flat channel layout c = h*D + d via
         short_conv_step, advancing conv_k/conv_v;
      2. merge weight w_t: compute_summary_weights(q_t, k_t) on the
         POST-conv key (QK mode) or compute_linear_weights(x_t, w_proj)
         (both x_t and w_proj, or neither);
      3. the one level whose node activates at t is merged from its two
         live children (build_dyadic_summaries' pair merge);
      4. token t is written at level 0 -- after the merge.

    Returns (k_t, v_t) as stored (post-conv), for inspection.
    """
    t = state.t
    B, _, Hkv, Dk = state.k.shape
    Dv = state.v.shape[-1]
    if q_t.dim() != 4 or q_t.shape[0] != B or q_t.shape[1] != 1 \
            or q_t.shape[-1] != Dk or q_t.shape[2] % Hkv != 0:
        raise ValueError(
            f"q_t shape {tuple(q_t.shape)} incompatible with [B={B}, 1, "
            f"Hq (multiple of {Hkv}), Dk={Dk}]"
        )
    if tuple(k_t.shape) != (B, 1, Hkv, Dk):
        raise ValueError(
            f"k_t shape {tuple(k_t.shape)} != {(B, 1, Hkv, Dk)}"
        )
    if tuple(v_t.shape) != (B, 1, Hkv, Dv):
        raise ValueError(
            f"v_t shape {tuple(v_t.shape)} != {(B, 1, Hkv, Dv)}"
        )

    # 1. short conv with its rolling pre-conv state.
    if (k_conv_weight is None) != (v_conv_weight is None):
        raise ValueError(
            "k_conv_weight and v_conv_weight must both be provided or both "
            "be None"
        )
    if k_conv_weight is not None:
        if state.conv_k is None or state.conv_v is None:
            raise ValueError(
                "conv weights given but the state was initialised without "
                "pre-conv rows (init_decode_state k_pre_conv/v_pre_conv)"
            )
        k_flat, state.conv_k = short_conv_step(
            k_t.reshape(B, 1, Hkv * Dk), state.conv_k, k_conv_weight
        )
        v_flat, state.conv_v = short_conv_step(
            v_t.reshape(B, 1, Hkv * Dv), state.conv_v, v_conv_weight
        )
        k_t = k_flat.reshape(B, 1, Hkv, Dk)
        v_t = v_flat.reshape(B, 1, Hkv, Dv)
    elif state.conv_k is not None:
        raise ValueError(
            "the state carries a conv state but no conv weights were given"
        )

    # 2. merge weight (post-conv key; unpositioned q/k).
    if (x_t is None) != (w_proj is None):
        raise ValueError(
            "x_t and w_proj must both be provided (linear weight mode) or "
            "both be None (QK weight mode)"
        )
    if x_t is not None:
        w_t = compute_linear_weights(x_t, w_proj)
        if tuple(w_t.shape) != (B, 1, Hkv, 1):
            raise ValueError(
                f"linear weights shape {tuple(w_t.shape)} != {(B, 1, Hkv, 1)}"
            )
    else:
        w_t = compute_summary_weights(q_t, k_t)

    # 3. the ruler tick: at most one level >= 1 activates a node at t.
    a = state.activation_times
    L = len(a) - 1
    fired = None
    for level in range(1, L + 1):
        span = 1 << level
        if t < a[level] or (t - a[level]) % span != 0:
            continue
        if fired is not None:
            raise AssertionError(
                f"levels {fired} and {level} both activate at t={t}; the "
                f"schedule is not dyadically aligned"
            )
        fired = level
        j = (t - a[level]) >> level
        # Children (level-1, 2j) and (level-1, 2j+1) leave level-1's range
        # exactly at t, so they are live at t-1 (and t >= 1 here).
        lo_c, hi_c = range_bounds(state.spec(t), t - 1, level - 1)
        if not (lo_c <= 2 * j and 2 * j + 1 < hi_c):
            raise AssertionError(
                f"children of node ({level}, {j}) not live at t-1={t - 1}: "
                f"level-{level - 1} range [{lo_c},{hi_c})"
            )
        c0 = ring_slot(state, level - 1, 2 * j)
        c1 = ring_slot(state, level - 1, 2 * j + 1)
        parent = ring_slot(state, level, j)
        w_children = torch.stack(
            [state.w[:, c0], state.w[:, c1]], dim=1
        )  # [B, 2, Hkv, 1] -- the child axis is dim 1
        alpha = torch.softmax(w_children, dim=1)
        state.k[:, parent] = (
            torch.stack([state.k[:, c0], state.k[:, c1]], dim=1) * alpha
        ).sum(dim=1)
        state.v[:, parent] = (
            torch.stack([state.v[:, c0], state.v[:, c1]], dim=1) * alpha
        ).sum(dim=1)
        state.w[:, parent] = torch.logsumexp(w_children, dim=1)

    # Capacity invariant at the new query (also validates the schedule).
    spec_next = state.spec(t + 1)
    for level in range(L + 1):
        lo, hi = range_bounds(spec_next, t, level)
        if hi - lo > state.caps[level]:
            raise AssertionError(
                f"level {level} needs {hi - lo} live slots > cap "
                f"{state.caps[level]} at t={t}"
            )

    # 4. token t at level 0, after the merge.
    slot0 = ring_slot(state, 0, t)
    state.k[:, slot0] = k_t[:, 0]
    state.v[:, slot0] = v_t[:, 0]
    state.w[:, slot0] = w_t[:, 0]
    state.t = t + 1
    return k_t, v_t


def multilevel_attention_decode(
    q_t: torch.Tensor,
    state: DecodeState,
    *,
    block_n: int = 32,
    softcap: float = 20.0,
    position_mode: str = "none",
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    relative_states: Optional[torch.Tensor] = None,
    relative_proj: Optional[torch.Tensor] = None,
):
    """
    Attention of the most recently consumed token (query index t = state.t
    - 1) over its multiresolution KV set, read from the ring cache:

      for level
        [k_lo, k_hi) = range_bounds(spec_{t+1}, t, level)        (exact)
        for BLOCK_N tile in [k_lo, k_hi):                        (level-local)
          K, V = rings[ring_offsets[level] + tile % caps[level]]  (storage)
          QK matmul [+ relative bias], softcap, online-softmax update

    Same contract as multilevel_attention_forward for one row: out
    [B, 1, Hq, Dv] in V's dtype, lse [B, 1, Hq] float32 natural-log LSE of
    the softcapped scores over exactly A(t). Positional modes are identical
    to the forward's: "rope" rotates Q at t and each K entry at
    summary_token_position(level, j) = (j+1)*2^level - 1 (rotate-half,
    tables must cover [0, t]); "relative" adds the learned summary-bin bias
    before softcap with distances from the shared _tile_bin_distances
    (relative_states is [B, 1, Hq, d_rel] from compute_relative_states over
    the token's ORIGINAL hidden state).

    Also returns `stats`, a diagnostic dict (never asserted on in
    production): entries [(level, j)] in gather order plus the aligned
    `positions` (rope) / `distances` (relative) lists.
    """
    if q_t.dim() != 4 or q_t.shape[1] != 1:
        raise ValueError(f"q_t must be [B, 1, Hq, Dk], got {tuple(q_t.shape)}")
    if state.t < 1:
        raise ValueError("no token has been consumed yet")
    B, _, Hq, Dk = q_t.shape
    Bs, _, Hkv, Dks = state.k.shape
    Dv = state.v.shape[-1]
    if Bs != B or Dks != Dk:
        raise ValueError(
            f"q_t {tuple(q_t.shape)} incompatible with the state's keys "
            f"{tuple(state.k.shape)}"
        )
    if Hq % Hkv != 0:
        raise ValueError(f"Hq={Hq} must be divisible by Hkv={Hkv}.")
    expansion = Hq // Hkv
    t = state.t - 1
    device = q_t.device
    spec = state.spec(t + 1)

    rel_logits, max_relative_bins = _positional_setup(
        position_mode, rope_cos, rope_sin, relative_states, relative_proj,
        B=B, N=1, Hq=Hq, Dk=Dk, spec=spec, rope_positions=t + 1,
    )

    # Per-level visible bounds of the single row, [1, num_levels].
    bounds = [range_bounds(spec, t, lv) for lv in range(spec.num_levels)]
    row_lo = torch.tensor([[lo for lo, _ in bounds]], device=device,
                          dtype=torch.long)
    row_hi = torch.tensor([[hi for _, hi in bounds]], device=device,
                          dtype=torch.long)

    Q = q_t[:, 0]  # [B, Hq, Dk]
    if position_mode == "rope":
        Q = apply_rope(
            Q, torch.tensor(t, device=device, dtype=torch.long),
            rope_cos, rope_sin,
        )
    Q = Q.view(B, Hkv, expansion, Dk)  # GQA grouping: hq = hkv*E + e
    scale = 1.0 / math.sqrt(Dk)

    m = torch.full((B, Hkv, expansion), -float("inf"), device=device,
                   dtype=torch.float32)
    ell = torch.zeros(B, Hkv, expansion, device=device, dtype=torch.float32)
    acc = torch.zeros(B, Hkv, expansion, Dv, device=device,
                      dtype=torch.float32)
    stats = {"entries": [], "positions": [], "distances": []}

    for level in range(spec.num_levels):
        k_lo, k_hi = bounds[level]
        if k_lo == k_hi:
            continue
        for k_start in range(k_lo, k_hi, block_n):
            k_end = min(k_start + block_n, k_hi)
            kv_indices = torch.arange(
                k_start, k_end, device=device, dtype=torch.long
            )
            slots = _ring_slots(state, level, kv_indices)
            K = state.k[:, slots]  # [B, Kt, Hkv, Dk]
            V = state.v[:, slots]  # [B, Kt, Hkv, Dv]
            stats["entries"].extend((level, j) for j in range(k_start, k_end))

            if position_mode == "rope":
                # summary_token_position, vectorized; V untouched.
                k_pos = (kv_indices + 1) * (1 << level) - 1
                K = apply_rope(K, k_pos.view(1, -1, 1), rope_cos, rope_sin)
                stats["positions"].extend(k_pos.tolist())

            scores = torch.einsum("bhed,bkhd->bhek", Q, K) * scale

            if position_mode == "relative":
                # Every lane is valid: the row's range is exact.
                valid = torch.ones(
                    1, k_end - k_start, device=device, dtype=torch.bool
                )
                dist = _tile_bin_distances(
                    spec, row_lo, row_hi, kv_indices, level,
                    max_relative_bins, valid, t, t + 1,
                )  # [1, Kt]
                stats["distances"].extend(dist[0].tolist())
                rel_t = rel_logits[:, 0].view(
                    B, Hkv, expansion, max_relative_bins
                )
                bias = torch.gather(
                    rel_t, 3,
                    dist.view(1, 1, 1, -1).expand(B, Hkv, expansion, -1),
                )
                # X = QK/sqrt(Dk) + b, BEFORE softcap; fp32 promotion.
                scores = scores + bias

            if softcap is not None:
                scores = softcap * torch.tanh(scores / softcap)

            # FlashAttention online-softmax update; no masking needed.
            block_max = scores.float().max(dim=-1).values
            m_new = torch.maximum(m, block_max)
            old_scale = torch.exp(m - m_new)  # exp(-inf) = 0 on the first tile
            p = torch.exp(scores.float() - m_new[..., None])
            ell = ell * old_scale + p.sum(dim=-1)
            acc = (
                acc * old_scale[..., None]
                + torch.einsum("bhek,bkhd->bhed", p, V.float())
            )
            m = m_new

    if (ell == 0).any():
        raise RuntimeError(
            f"decode query t={t} has no attended KV entries."
        )
    out = (acc / ell[..., None]).reshape(B, 1, Hq, Dv).to(state.v.dtype)
    lse = (m + torch.log(ell)).reshape(B, 1, Hq)
    return out, lse, stats


def decode_step(
    x_t: torch.Tensor,
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    state: DecodeState,
    *,
    weight_mode: str = "qk",
    w_proj: Optional[torch.Tensor] = None,
    k_conv_weight: Optional[torch.Tensor] = None,
    v_conv_weight: Optional[torch.Tensor] = None,
    block_n: int = 32,
    softcap: float = 20.0,
    position_mode: str = "none",
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    relative_weight: Optional[torch.Tensor] = None,
    relative_proj: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    gate_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The complete telescope block for ONE token, after the Q/K/V projections
    (the one-token analogue of minimal_attention_block, minus the output
    projection):

        (x_t, q_t, k_t, v_t)  [B, 1, ...]
          -> advance_decode_state    (short conv + merge weight + tree update)
          -> multilevel_attention_decode   (positional mode)  -> (out, lse)
          -> apply_attention_sink(out, lse, sinks)             (optional)
          -> apply_output_gate(out, x_t, gate_weight)          (optional)

    x_t is the token's ORIGINAL hidden state [B, 1, emb] (linear weights,
    relative states, gate). Returns (out [B, 1, Hq, Dv] post-sink/gate,
    lse [B, 1, Hq] pre-sink float32) -- the frozen attention-boundary
    terminology. The state is advanced in place.
    """
    if weight_mode == "qk":
        if w_proj is not None:
            raise ValueError("w_proj provided with weight_mode='qk'")
        weight_kwargs = {}
    elif weight_mode == "linear":
        if w_proj is None:
            raise ValueError("weight_mode='linear' requires w_proj")
        weight_kwargs = dict(x_t=x_t, w_proj=w_proj)
    else:
        raise ValueError(f"unknown weight_mode {weight_mode!r}")
    advance_decode_state(
        state, q_t, k_t, v_t,
        k_conv_weight=k_conv_weight, v_conv_weight=v_conv_weight,
        **weight_kwargs,
    )
    relative_states = None
    if position_mode == "relative":
        if relative_weight is None or relative_proj is None:
            raise ValueError(
                "position_mode='relative' requires relative_weight and "
                "relative_proj"
            )
        relative_states = compute_relative_states(
            x_t, relative_weight, q_t.shape[2]
        )
    elif relative_weight is not None:
        raise ValueError(
            f"relative_weight provided with position_mode={position_mode!r}"
        )
    out, lse, _ = multilevel_attention_decode(
        q_t, state, block_n=block_n, softcap=softcap,
        position_mode=position_mode, rope_cos=rope_cos, rope_sin=rope_sin,
        relative_states=relative_states, relative_proj=relative_proj,
    )
    if sinks is not None:
        out = apply_attention_sink(out, lse, sinks)
    if gate_weight is not None:
        out = apply_output_gate(out, x_t, gate_weight)
    return out, lse
