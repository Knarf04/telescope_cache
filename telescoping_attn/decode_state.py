"""Incremental decode state: per-level rings, capacities, tree update."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from telescope_cache.telescoping_attn.range_spec import (
    RangeSpec,
    activation_times_from_fmap,
    range_bounds,
)
from telescope_cache.telescoping_attn.summaries import (
    compute_linear_weights,
    compute_summary_weights,
)

@dataclass
class DecodeState:
    """
    One layer's decode cache, as level-major rings:

        k [B, sum(caps), Hkv, Dk]   UNROTATED, post-conv keys
        v [B, sum(caps), Hkv, Dv]
        w [B, sum(caps), Hkv, 1]    merge log-weights
        conv_k / conv_v [B, K-1, Hkv*D] | None   last K-1 PRE-conv flat rows
        t                           tokens consumed == next query index

    Node j of level l lives at ring_slot(state, l, j). Keys are unrotated
    because the merge re-reads them and RoPE rotates after aggregation.
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
    -> caps[l], the ring size per level. sum(caps) may exceed cache_size: each
    ring holds its own level's peak, not the joint peak.
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
    """
    Physical slot of node j at `level`: ring_offsets[level] + j % caps[level].
    """
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
    """
    Vectorized ring_slot for a level-local index tensor.
    """
    return state.ring_offsets[level] + torch.remainder(
        indices, state.caps[level]
    )

def _conv_history(x_pre: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    -> the last K-1 rows of x_pre [B, N, C], newest last, left zero-padded if
    N < K-1. K == 1 needs the explicit branch: x[:, -0:] is the whole tensor.
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
    x: [B, 1, C] current PRE-conv row  state: [B, K-1, C] previous rows, newest
    last  weight: [C, K]  -> (y [B, 1, C], new_state [B, K-1, C])
    Same FP32 residual contract as short_conv.
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
    Prefill -> decode hand-off: k/v/w_levels are build_dyadic_summaries' outputs
    for N tokens, and the state receives the nodes live at query N-1, t = N.
    Short-conv mode also takes the RAW pre-conv k/v and conv_kernel_size.
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
    Consume token t = state.t in place; state.t becomes t + 1.
    q_t/k_t/v_t: [B, 1, H*, D*] RAW projections. -> (k_t, v_t) as stored.

    Order matters: short conv -> merge weight -> the one level activating at t
    merges its two live children -> token t written at level 0, AFTER the merge.
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

    # 1. short conv, rolling pre-conv state
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

    # 2. merge weight from the post-conv key, unpositioned q/k
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

    # 3. ruler tick: at most one level >= 1 activates a node at t
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
        # Both children leave level-1's range exactly at t, so they are live
        # at t-1 (and t >= 1 here).
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

    # capacity invariant at the new query; also validates the schedule
    spec_next = state.spec(t + 1)
    for level in range(L + 1):
        lo, hi = range_bounds(spec_next, t, level)
        if hi - lo > state.caps[level]:
            raise AssertionError(
                f"level {level} needs {hi - lo} live slots > cap "
                f"{state.caps[level]} at t={t}"
            )

    # 4. token t at level 0, after the merge
    slot0 = ring_slot(state, 0, t)
    state.k[:, slot0] = k_t[:, 0]
    state.v[:, slot0] = v_t[:, 0]
    state.w[:, slot0] = w_t[:, 0]
    state.t = t + 1
    return k_t, v_t
