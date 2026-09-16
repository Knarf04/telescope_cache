"""Positional encoding: post-summary RoPE and the learned summary-bin bias.

minimal_reference.py deliberately does NOT import this -- its naive
rope/bin-distance are an independent oracle, and the mini<->ref comparison
only has force while the two share nothing but range_spec.
"""

import math
from typing import Optional, Tuple

import torch

from telescope_cache.telescoping_attn.range_spec import RangeSpec, elem_mask, range_bounds

def summary_token_position(level: int, local_index: int) -> int:
    """
    RoPE anchor of a summary: (local_index + 1) * 2**level - 1, the RIGHT
    ENDPOINT of its dyadic interval.

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
    -> (cos, sin), each [n_pos, dim // 2] fp32. GPT-NeoX rotate-half,
    theta_i = base ** (-2i / dim).
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
    x: [..., D]  positions: ints broadcastable to x.shape[:-1]
    cos/sin: [n_pos, D/2]  -> rotated x in x.dtype, computed fp32.
    inverse=True applies R(-p), the VJP of the forward rotation.
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
    Raises unless a[l] >= 2^l - 1 at every level, i.e. every visible summary is
    entirely past-or-current. Not implied by fmap alignment.
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
    THE DEFINITION: 0-based chronological rank of a visible entry among the M
    visible to q, 0 = newest. -> int in [0, M-1]; M <= cache_size. Raises if the
    entry is not visible.

    Written for an arbitrary token position, hence the clamp/ceil sum; for
    visible entries flex_tree.bin_distances uses the cheaper base[l] - (j - lo_l)
    form, checked against this.
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
    x: [B, N, emb_dim]  relative_weight: [Hq * d_rel, emb_dim]
    -> [B, N, Hq, d_rel], QUERY-conditioned, from the ORIGINAL hidden states.
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
    Tile-vectorized relative_bin_distance.
    row_lo/row_hi: [M, num_levels]  kv_indices: [Kt]  valid: [M, Kt]
    -> dist [M, Kt] long.

    Valid lanes must satisfy 0 <= dist < max_relative_bins or this raises: a
    negative index would alias the tensor end. Invalid lanes may go negative.
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
