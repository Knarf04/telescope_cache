import argparse
import math
import os
import sys
from typing import Dict, List, NamedTuple, Tuple

import torch

# The tiled POC consumes the pure-integer range specification. Make
# `telescope_cache` importable (namespace package, as in test_vs_train.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from telescope_cache.range_spec import (  # noqa: E402
    RangeSpec,
    bwd_bounds,
    fwd_bounds,
    range_bounds,
)


# ============================================================
# Original structured plan, kept only as a correctness oracle
# for which semantic cache entries each query should see.
# ============================================================

@torch.no_grad()
def get_structured_plan(
    n: int,
    fmap: Dict[int, int],
    cache_size: int,
    device: torch.device,
):
    """
    Same cache-selection/merge-plan logic as the original get_scan_plan(),
    except that the final indices remain as (level, local_index) pairs instead
    of being flattened into one global cache index.

    Old level numbering:
        level 1: padded raw tokens (index 0 is dummy)
        level 2: 2-token summaries
        level 3: 4-token summaries
        ...

    Returns
    -------
    merge_plan:
        merge_plan[l]: [num_nodes_l, 2]. Each row gives the two child indices
        in old level l-1 used to construct one node in old level l.

    select_level:
        [N, C], old level selected by each logical cache slot.

    select_index:
        [N, C], local index within select_level.
    """
    levels = sum(
        [
            torch.arange(n, device=device)
            .remainder(2**i)
            .sub(2**i - 1)
            .sign()
            .add(1)
            for i in range(n.bit_length())
        ]
    ).roll(1, 0)

    merge_plan = [
        torch.zeros(0, 2, device=device, dtype=torch.long)
        for _ in range(len(fmap) + 2)
    ]

    # Padded raw level: 0=dummy, 1=token0, 2=token1, ...
    merge_plan[1] = (
        torch.arange(n + 1, device=device, dtype=torch.long)
        .unsqueeze(1)
        .expand(-1, 2)
    )

    # [query_position, cache_slot, (old_level, old_local_index)]
    inds = torch.zeros(n, cache_size, 2, device=device, dtype=torch.long)
    inds[:, 0, 1] = torch.arange(n, device=device, dtype=torch.long) + 1
    inds[:, :, 0] = 1

    for i in range(1, n):
        level = int(levels[i].item())
        m = fmap.get(level, cache_size)

        inds[i, 1:m] = inds[i - 1, : m - 1]

        if m < cache_size:
            inds[i, m + 1 :] = inds[i - 1, m + 1 :]

            # Same order as the original code.
            children = inds[i - 1, m - 1 : m + 1].flip(0)
            parent_level = level + 1
            parent_index = merge_plan[parent_level].shape[0]

            inds[i, m, 0] = parent_level
            inds[i, m, 1] = parent_index

            merge_plan[parent_level] = torch.cat(
                [merge_plan[parent_level], children[:, 1][None]],
                dim=0,
            )

    return merge_plan, inds[..., 0], inds[..., 1]


# ============================================================
# Shared leaf-weight computation: identical semantics to the
# original q-k score used to weight the hierarchy.
# ============================================================

def compute_summary_weights(
    q: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    Parameters
    ----------
    q: [B, N, Hq, Dk]
    k: [B, N, Hkv, Dk]

    Returns
    -------
    w: [B, N, Hkv, 1]

    For each KV head, groups its associated query heads and computes

        w = logsumexp_h(q_h^T k / sqrt(Dk)).
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


# ============================================================
# Reference Phase 1: original plan-based recursive scan, but
# kept level-major instead of flattening the hierarchy.
# ============================================================

def build_plan_based_summaries(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    merge_plan: List[torch.Tensor],
    detach_weights: bool = False,
):
    """
    Exact level-major analogue of the original weighted scan().

    This function exists only as a correctness reference for the new dyadic
    Phase 1. It intentionally retains the original dummy/warm-up nodes.

    detach_weights: gradient-only switch used by test_backward.py. Detaching
    the leaf weights w cuts the entire merge-weight gradient branch at every
    level while leaving all forward values unchanged.
    """
    B, N, _, _ = q.shape
    Hkv = k.shape[2]

    w = compute_summary_weights(q, k)
    if detach_weights:
        w = w.detach()

    num_old_levels = len(merge_plan)
    k_levels = [None] * num_old_levels
    v_levels = [None] * num_old_levels
    w_levels = [None] * num_old_levels
    valid_levels = [None] * num_old_levels

    # Old level 1: padded raw K/V.
    k_levels[1] = torch.cat([torch.zeros_like(k[:, :1]), k], dim=1)
    v_levels[1] = torch.cat([torch.zeros_like(v[:, :1]), v], dim=1)
    w_levels[1] = torch.cat(
        [torch.full_like(w[:, :1], -1000.0), w],
        dim=1,
    )
    valid_levels[1] = torch.cat(
        [
            torch.zeros(1, dtype=torch.bool, device=q.device),
            torch.ones(N, dtype=torch.bool, device=q.device),
        ]
    )

    for old_level in range(2, num_old_levels):
        pairs = merge_plan[old_level].long()

        if pairs.shape[0] == 0:
            k_levels[old_level] = k.new_empty(B, 0, Hkv, k.shape[-1])
            v_levels[old_level] = v.new_empty(B, 0, Hkv, v.shape[-1])
            w_levels[old_level] = w.new_empty(B, 0, Hkv, 1)
            valid_levels[old_level] = torch.empty(
                0, dtype=torch.bool, device=q.device
            )
            continue

        child_w = w_levels[old_level - 1][:, pairs]
        alpha = torch.softmax(child_w, dim=2)

        child_k = k_levels[old_level - 1][:, pairs]
        child_v = v_levels[old_level - 1][:, pairs]

        k_levels[old_level] = (child_k * alpha).sum(dim=2)
        v_levels[old_level] = (child_v * alpha).sum(dim=2)
        w_levels[old_level] = torch.logsumexp(child_w, dim=2)

        valid_levels[old_level] = (
            valid_levels[old_level - 1][pairs].any(dim=1)
        )

    return k_levels, v_levels, w_levels, valid_levels


# ============================================================
# New Phase 1: canonical dyadic hierarchy.
# ============================================================

def build_dyadic_summaries(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_summary_levels: int,
    detach_weights: bool = False,
):
    """
    Build a canonical dyadic K/V summary tree directly.

    detach_weights: gradient-only switch (see build_plan_based_summaries).

    Canonical level numbering:
        L0: raw tokens, span 1
        L1: 2-token summaries
        L2: 4-token summaries
        L3: 8-token summaries
        ...

    No merge_plan, dummy nodes, index_select, or flattened cache is needed.

    For odd level lengths, only complete adjacent pairs are merged. This is
    exactly the set of complete aligned dyadic intervals available at the
    next level.
    """
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
# Level-local range [start, end) from range_spec addresses physical entries
# offset[level] + [start, end). POC physical layout is BSHD, [B, sumN, H, D]
# (kernel layout undecided; [B, H, sumN, D] gives contiguous per-head
# streams). torch.cat keeps the autograd graph intact, which the upcoming
# backward oracle relies on; the Phase-1 kernel should instead allocate the
# packed buffer once and write each level into its interval. w_levels are
# deliberately not packed: Phase-2 attention needs only K and V.
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
    """
    Read one level-local KV tile from the packed buffers. The only place the
    forward POC slices the packed buffer; switching physical layout changes
    this function (and pack_levels / the pack oracle / the backward writes).
    """
    p_start, p_end = _packed_slice(packed, level, k_start, k_end)
    return packed.k[b, p_start:p_end, hkv], packed.v[b, p_start:p_end, hkv]


# ============================================================
# Remap the original semantic selection plan into canonical
# dyadic (level, index) coordinates.
# ============================================================

@torch.no_grad()
def remap_plan_to_dyadic(
    merge_plan: List[torch.Tensor],
    select_level: torch.Tensor,
    select_index: torch.Tensor,
    n: int,
):
    """
    Translate nodes from the original scan numbering into the canonical
    dyadic tree.

    Original numbering:
        old level 1, index 0: dummy
        old level 1, index i+1: raw token i
        old level l: recursively created nodes, including warm-up dummies

    Canonical numbering:
        level 0, index i: raw token i
        level 1, index i: tokens [2i, 2i+1]
        level 2, index i: tokens [4i, ..., 4i+3]
        ...

    Dummy/warm-up nodes map to index -1.
    """
    device = select_index.device
    old_to_dyadic = [None] * len(merge_plan)

    # Old padded raw level -> canonical raw level.
    raw_map = torch.full(
        (n + 1,),
        -1,
        dtype=torch.long,
        device=device,
    )
    raw_map[1:] = torch.arange(n, device=device)
    old_to_dyadic[1] = raw_map

    for old_level in range(2, len(merge_plan)):
        pairs = merge_plan[old_level].long()
        child_idx = old_to_dyadic[old_level - 1][pairs]

        mapped = torch.full(
            (pairs.shape[0],),
            -1,
            dtype=torch.long,
            device=device,
        )

        if pairs.shape[0] == 0:
            old_to_dyadic[old_level] = mapped
            continue

        child_valid = child_idx >= 0

        # The original construction should produce either two dummy children
        # or two real children for a meaningful parent.
        if not torch.equal(child_valid[:, 0], child_valid[:, 1]):
            bad = torch.nonzero(
                child_valid[:, 0] != child_valid[:, 1], as_tuple=False
            ).flatten()
            raise RuntimeError(
                f"Found mixed real/dummy children at old level {old_level}; "
                f"first bad parent indices: {bad[:8].tolist()}"
            )

        valid = child_valid[:, 0]

        if valid.any():
            children = child_idx[valid]

            if not torch.all(children[:, 1] == children[:, 0] + 1):
                raise RuntimeError(
                    f"Non-adjacent canonical children at old level {old_level}."
                )

            if not torch.all(children[:, 0] % 2 == 0):
                raise RuntimeError(
                    f"Non-dyadically-aligned children at old level {old_level}."
                )

            mapped[valid] = children[:, 0] // 2

        old_to_dyadic[old_level] = mapped

    dyadic_level = select_level - 1
    dyadic_index = torch.full_like(select_index, -1)

    for old_level in range(1, len(merge_plan)):
        mask = select_level == old_level
        dyadic_index[mask] = old_to_dyadic[old_level][select_index[mask]]

    return dyadic_level, dyadic_index, old_to_dyadic


# ============================================================
# Phase 2, version A: original level-major hierarchy.
# Used only as a reference.
# ============================================================

def multilevel_attention_plan_poc(
    q: torch.Tensor,
    k_levels: List[torch.Tensor],
    v_levels: List[torch.Tensor],
    valid_levels: List[torch.Tensor],
    select_level: torch.Tensor,
    select_index: torch.Tensor,
    softcap: float = 20.0,
):
    """
    Slow loop-based correctness implementation using original level numbering.

    It directly reads selected (level, index) entries and performs one joint
    softmax over all selected levels.
    """
    B, N, Hq, Dk = q.shape
    Hkv = k_levels[1].shape[2]
    expansion = Hq // Hkv

    rows = []

    for qi in range(N):
        selected = []
        seen = set()

        for slot in range(select_level.shape[1]):
            level = int(select_level[qi, slot].item())
            index = int(select_index[qi, slot].item())

            if not bool(valid_levels[level][index].item()):
                continue

            entry = (level, index)
            if entry in seen:
                continue

            seen.add(entry)
            selected.append(entry)

        if not selected:
            raise RuntimeError(f"Query {qi} has no valid cache entries.")

        head_outputs = []

        for hq in range(Hq):
            hkv = hq // expansion

            K = torch.stack(
                [k_levels[level][:, index, hkv] for level, index in selected],
                dim=1,
            )
            V = torch.stack(
                [v_levels[level][:, index, hkv] for level, index in selected],
                dim=1,
            )

            query = q[:, qi, hq]
            scores = torch.einsum("bd,bsd->bs", query, K) / math.sqrt(Dk)

            if softcap is not None:
                scores = softcap * torch.tanh(scores / softcap)

            probs = torch.softmax(scores, dim=-1)
            output = torch.einsum("bs,bsd->bd", probs, V)
            head_outputs.append(output)

        rows.append(torch.stack(head_outputs, dim=1))

    return torch.stack(rows, dim=1)


# ============================================================
# Phase 2, version B: canonical dyadic hierarchy.
# This is the POC path we ultimately care about.
# ============================================================

def multilevel_attention_dyadic_poc(
    q: torch.Tensor,
    k_levels: List[torch.Tensor],
    v_levels: List[torch.Tensor],
    select_level: torch.Tensor,
    select_index: torch.Tensor,
    softcap: float = 20.0,
):
    """
    Slow loop-based multilevel attention over canonical dyadic levels.

    select_level:
        0 = raw
        1 = 2-token summary
        2 = 4-token summary
        ...

    select_index == -1 means the original plan selected a dummy/warm-up node.
    """
    B, N, Hq, Dk = q.shape
    Hkv = k_levels[0].shape[2]
    expansion = Hq // Hkv

    rows = []

    for qi in range(N):
        selected = []
        seen = set()

        for slot in range(select_level.shape[1]):
            level = int(select_level[qi, slot].item())
            index = int(select_index[qi, slot].item())

            if index < 0:
                continue

            if level < 0 or level >= len(k_levels):
                raise RuntimeError(
                    f"Query {qi}: invalid dyadic level {level}."
                )
            if index >= k_levels[level].shape[1]:
                raise RuntimeError(
                    f"Query {qi}: index {index} out of range for L{level} "
                    f"with size {k_levels[level].shape[1]}."
                )

            entry = (level, index)
            if entry in seen:
                continue

            seen.add(entry)
            selected.append(entry)

        if not selected:
            raise RuntimeError(f"Query {qi} has no valid cache entries.")

        head_outputs = []

        for hq in range(Hq):
            hkv = hq // expansion

            K = torch.stack(
                [k_levels[level][:, index, hkv] for level, index in selected],
                dim=1,
            )
            V = torch.stack(
                [v_levels[level][:, index, hkv] for level, index in selected],
                dim=1,
            )

            query = q[:, qi, hq]
            scores = torch.einsum("bd,bsd->bs", query, K) / math.sqrt(Dk)

            if softcap is not None:
                scores = softcap * torch.tanh(scores / softcap)

            probs = torch.softmax(scores, dim=-1)
            output = torch.einsum("bs,bsd->bd", probs, V)
            head_outputs.append(output)

        rows.append(torch.stack(head_outputs, dim=1))

    return torch.stack(rows, dim=1)


# ============================================================
# New Phase 2 metadata: analytic per-level contiguous ranges.
# No scan/select plan is needed on the actual computation path.
# ============================================================


def compute_dyadic_activation_times(fmap: Dict[int, int]):
    """
    For a dyadically aligned fmap, return a[l], the first query position at
    which canonical dyadic node 0 at level l becomes visible.

    Canonical levels:
        L0 span 1
        L1 span 2
        L2 span 4
        ...

    For the current fmap {1:64, 2:72, 3:80}, this gives
        a = [0, 65, 82, 116].

    These constants are induced by the original ruler/slot-update policy.
    """
    if not fmap:
        return [0]

    L = max(fmap)
    expected = list(range(1, L + 1))
    if sorted(fmap) != expected:
        raise ValueError(
            f"fmap keys must be consecutive 1..L; got {sorted(fmap)}"
        )

    a = [0] * (L + 1)
    a[0] = 0
    a[1] = fmap[1] + 1

    for level in range(2, L + 1):
        a[level] = (
            a[level - 1]
            + (2 ** (level - 1)) * (fmap[level] - fmap[level - 1])
            + 2 ** (level - 2)
        )

    return a


def validate_dyadic_fmap(fmap: Dict[int, int]):
    """
    Sanity-check that the original slot schedule is compatible with the
    *canonical aligned* dyadic tree used by build_dyadic_summaries().

    For canonical level l, node 0 must enter on the correct ruler phase:

        a[l] mod 2^l == 2^(l-1),  l >= 1.

    The current fmap {1:64,2:72,3:80} satisfies this condition.

    This check is useful if you later add more levels: not every arbitrary
    sequence of fmap boundaries preserves canonical dyadic alignment.
    """
    a = compute_dyadic_activation_times(fmap)

    for level in range(1, len(a)):
        expected_phase = 2 ** (level - 1)
        actual_phase = a[level] % (2**level)
        if actual_phase != expected_phase:
            raise ValueError(
                f"fmap is not compatible with the canonical dyadic tree at "
                f"level {level}: activation a[{level}]={a[level]} has phase "
                f"{actual_phase} mod {2**level}, expected {expected_phase}."
            )

    return a


def dyadic_ranges_for_query(
    q_index: int,
    fmap: Dict[int, int],
    cache_size: int,
    activation_times=None,
):
    """
    Analytically compute the contiguous cache range used from every dyadic
    level for one query.

    Returns
    -------
    ranges: List[(start, end)]
        Half-open local-index ranges [start, end) for L0, L1, ..., LL.

    There is no select_level/select_index tensor and no dense mask.

    Intuition
    ---------
    A canonical node j at level l first becomes visible at

        q = a[l] + 2^l * j.

    For every non-coarsest level, it remains present until its pair is merged
    into the next level. Therefore the live nodes at each level form one
    contiguous index interval.

    The coarsest level uses all remaining logical cache slots, so once the
    cache is full its lower bound simply slides forward.
    """
    if activation_times is None:
        activation_times = validate_dyadic_fmap(fmap)

    L = len(activation_times) - 1
    ranges = []
    used_slots = 0

    # Fine/intermediate levels. Their lower bound is determined by how many
    # nodes have already been promoted into the next level.
    for level in range(L):
        span = 2**level
        a = activation_times[level]

        if q_index < a:
            start = end = 0
        else:
            newest = (q_index - a) // span

            next_a = activation_times[level + 1]
            if q_index < next_a:
                oldest = 0
            else:
                parent_span = 2 * span
                newest_parent = (q_index - next_a) // parent_span

                # Every completed parent removes its two children from this
                # finer level.
                oldest = 2 * (newest_parent + 1)

            if oldest > newest:
                start = end = 0
            else:
                start = oldest
                end = newest + 1

        ranges.append((start, end))
        used_slots += end - start

    # Coarsest level. It never promotes further in this model; instead it
    # occupies whatever cache slots remain and evicts the oldest summaries.
    level = L
    span = 2**level
    a = activation_times[level]
    remaining = cache_size - used_slots

    if remaining < 0:
        raise RuntimeError(
            f"Fine levels already require {used_slots} slots, larger than "
            f"cache_size={cache_size}."
        )

    if q_index < a or remaining == 0:
        start = end = 0
    else:
        newest = (q_index - a) // span
        count = min(remaining, newest + 1)
        start = newest - count + 1
        end = newest + 1

    ranges.append((start, end))

    return ranges


@torch.no_grad()
def validate_analytic_ranges_against_plan(
    dyadic_select_level: torch.Tensor,
    dyadic_select_index: torch.Tensor,
    fmap: Dict[int, int],
    cache_size: int,
):
    """
    Correctness-only oracle check.

    Compare the new analytic ranges against the old plan after remapping that
    plan into canonical dyadic coordinates. The *new path itself* does not
    use the plan.
    """
    N = dyadic_select_level.shape[0]
    activation_times = validate_dyadic_fmap(fmap)
    num_levels = len(activation_times)

    for qi in range(N):
        ranges = dyadic_ranges_for_query(
            qi,
            fmap,
            cache_size,
            activation_times,
        )

        for level in range(num_levels):
            start, end = ranges[level]
            analytic = set(range(start, end))

            mask = (
                (dyadic_select_level[qi] == level)
                & (dyadic_select_index[qi] >= 0)
            )
            oracle = set(dyadic_select_index[qi][mask].tolist())

            if analytic != oracle:
                raise AssertionError(
                    f"Range mismatch at query={qi}, level={level}:\n"
                    f"  analytic=[{start}, {end})\n"
                    f"  oracle={sorted(oracle)}"
                )


def multilevel_attention_ranges_poc(
    q: torch.Tensor,
    k_levels: List[torch.Tensor],
    v_levels: List[torch.Tensor],
    fmap: Dict[int, int],
    cache_size: int,
    softcap: float = 20.0,
):
    """
    Simple correctness implementation of Phase 2 using only analytic
    contiguous ranges.

    It concatenates the selected slices from all levels and performs one
    ordinary softmax. This has the intended semantics but is not intended to
    be efficient.

    Importantly, this path uses:
        - no get_structured_plan()
        - no select_level/select_index
        - no flattened hierarchical cache
        - no dense Boolean attention mask
    """
    B, N, Hq, Dk = q.shape
    Hkv = k_levels[0].shape[2]
    expansion = Hq // Hkv

    activation_times = validate_dyadic_fmap(fmap)
    if len(k_levels) != len(activation_times):
        raise ValueError(
            f"Have {len(k_levels)} hierarchy levels but fmap implies "
            f"{len(activation_times)}."
        )

    rows = []

    for qi in range(N):
        ranges = dyadic_ranges_for_query(
            qi,
            fmap,
            cache_size,
            activation_times,
        )

        head_outputs = []

        for hq in range(Hq):
            hkv = hq // expansion

            K_parts = []
            V_parts = []

            for level, (start, end) in enumerate(ranges):
                if start == end:
                    continue

                # Each level contributes one contiguous slice.
                K_parts.append(k_levels[level][:, start:end, hkv])
                V_parts.append(v_levels[level][:, start:end, hkv])

            if not K_parts:
                raise RuntimeError(f"Query {qi} has no cache entries.")

            K = torch.cat(K_parts, dim=1)
            V = torch.cat(V_parts, dim=1)

            query = q[:, qi, hq]
            scores = torch.einsum("bd,bsd->bs", query, K) / math.sqrt(Dk)

            if softcap is not None:
                scores = softcap * torch.tanh(scores / softcap)

            probs = torch.softmax(scores, dim=-1)
            output = torch.einsum("bs,bsd->bd", probs, V)
            head_outputs.append(output)

        rows.append(torch.stack(head_outputs, dim=1))

    return torch.stack(rows, dim=1)


def multilevel_attention_ranges_online_poc(
    q: torch.Tensor,
    k_levels: List[torch.Tensor],
    v_levels: List[torch.Tensor],
    fmap: Dict[int, int],
    cache_size: int,
    softcap: float = 20.0,
):
    """
    Same analytic ranges as multilevel_attention_ranges_poc(), but processes
    each hierarchy level one at a time with a single FlashAttention-style
    online-softmax state (m, l, acc).

    This is closer to the eventual custom multiresolution attention kernel:
    stream one contiguous range from L0, then L1, ..., while preserving one
    global softmax across all levels.
    """
    B, N, Hq, Dk = q.shape
    Hkv = k_levels[0].shape[2]
    Dv = v_levels[0].shape[-1]
    expansion = Hq // Hkv

    activation_times = validate_dyadic_fmap(fmap)
    rows = []
    # NaN-initialized so the test's isfinite() check doubles as an
    # "every row was written" check (POC only; a kernel need not do this).
    attn_lse = torch.full(
        (B, N, Hq), float("nan"), device=q.device, dtype=torch.float32
    )

    for qi in range(N):
        ranges = dyadic_ranges_for_query(
            qi,
            fmap,
            cache_size,
            activation_times,
        )

        head_outputs = []

        for hq in range(Hq):
            hkv = hq // expansion
            query = q[:, qi, hq]

            # FP32 is natural for the running softmax statistics. The POC
            # inputs are FP32 already, but these explicit dtypes make the
            # intended kernel structure clear.
            m = torch.full(
                (B,),
                -float("inf"),
                device=q.device,
                dtype=torch.float32,
            )
            ell = torch.zeros(B, device=q.device, dtype=torch.float32)
            acc = torch.zeros(
                B, Dv, device=q.device, dtype=torch.float32
            )

            for level, (start, end) in enumerate(ranges):
                if start == end:
                    continue

                K = k_levels[level][:, start:end, hkv]
                V = v_levels[level][:, start:end, hkv]

                scores = torch.einsum(
                    "bd,bsd->bs",
                    query,
                    K,
                ) / math.sqrt(Dk)

                if softcap is not None:
                    scores = softcap * torch.tanh(scores / softcap)

                # One online-softmax update for this level's contiguous slice.
                block_max = scores.max(dim=1).values.float()
                m_new = torch.maximum(m, block_max)

                old_scale = torch.exp(m - m_new)
                p = torch.exp(scores.float() - m_new[:, None])

                ell = ell * old_scale + p.sum(dim=1)
                acc = (
                    acc * old_scale[:, None]
                    + torch.einsum("bs,bsd->bd", p, V.float())
                )
                m = m_new

            output = (acc / ell[:, None]).to(v_levels[0].dtype)
            head_outputs.append(output)
            attn_lse[:, qi, hq] = m + torch.log(ell)

        rows.append(torch.stack(head_outputs, dim=1))

    return torch.stack(rows, dim=1), attn_lse


# ============================================================
# Phase 2, version E: query-block / KV-tile FlashAttention POC.
# This is the last Python bridge before a custom GPU kernel.
# ============================================================

def dyadic_ranges_for_query_block(
    q_indices: torch.Tensor,
    fmap: Dict[int, int],
    cache_size: int,
    activation_times=None,
):
    """
    Vectorized version of dyadic_ranges_for_query() for one query block.

    Parameters
    ----------
    q_indices:
        [M] integer query positions.

    Returns
    -------
    starts, ends:
        [M, num_levels] tensors. For query row i and hierarchy level l,
        the valid KV interval is

            [starts[i,l], ends[i,l]).

    This is the metadata the eventual attention CTA/program needs. No dense
    mask or per-query list of KV indices is materialized.
    """
    if activation_times is None:
        activation_times = validate_dyadic_fmap(fmap)

    q_indices = q_indices.long()
    device = q_indices.device
    L = len(activation_times) - 1
    M = q_indices.numel()

    starts = torch.zeros(
        M, L + 1, dtype=torch.long, device=device
    )
    ends = torch.zeros_like(starts)
    used_slots = torch.zeros(M, dtype=torch.long, device=device)

    # Fine/intermediate levels.
    for level in range(L):
        span = 2**level
        a = activation_times[level]
        next_a = activation_times[level + 1]

        visible = q_indices >= a
        newest = torch.div(
            q_indices - a, span, rounding_mode="floor"
        )

        parent_visible = q_indices >= next_a
        newest_parent = torch.div(
            q_indices - next_a, 2 * span, rounding_mode="floor"
        )

        oldest = torch.where(
            parent_visible,
            2 * (newest_parent + 1),
            torch.zeros_like(q_indices),
        )

        nonempty = visible & (oldest <= newest)
        start = torch.where(
            nonempty, oldest, torch.zeros_like(oldest)
        )
        end = torch.where(
            nonempty, newest + 1, torch.zeros_like(newest)
        )

        starts[:, level] = start
        ends[:, level] = end
        used_slots += end - start

    # Coarsest level uses whatever logical cache slots remain.
    level = L
    span = 2**level
    a = activation_times[level]
    remaining = cache_size - used_slots

    if (remaining < 0).any():
        raise RuntimeError(
            "Fine hierarchy levels exceed cache_size for at least one "
            "query in the block."
        )

    visible = (q_indices >= a) & (remaining > 0)
    newest = torch.div(
        q_indices - a, span, rounding_mode="floor"
    )
    count = torch.minimum(remaining, newest + 1)
    start = newest - count + 1
    end = newest + 1

    starts[:, level] = torch.where(
        visible, start, torch.zeros_like(start)
    )
    ends[:, level] = torch.where(
        visible, end, torch.zeros_like(end)
    )

    return starts, ends


def multilevel_attention_tiled_poc(
    q: torch.Tensor,
    packed: PackedKV,
    fmap: Dict[int, int],
    cache_size: int,
    block_m: int = 16,
    block_n: int = 32,
    softcap: float = 20.0,
):
    """
    FlashAttention-shaped PyTorch POC for multiresolution attention over the
    packed level-major K/V buffers (see pack_levels).

    The computational organization is intentionally close to the eventual
    custom kernel:

      for batch
        for KV head                     # exploit GQA sharing
          for BLOCK_M query positions
            keep Q and online-softmax state live
            for hierarchy level
              find union KV interval for the Q block   (level-local)
              for BLOCK_N contiguous KV tile in that interval
                load K/V once from offset[level] + tile
                QK matmul
                mask each query to its own [start,end) range
                online-softmax update

    Important properties
    --------------------
    * No old scan plan.
    * No flattened *selection* cache; K/V live in one level-major buffer.
    * No N x cache_len Boolean mask.
    * KV accesses within each level are contiguous in the packed index.
    * All GQA query heads belonging to one KV head reuse the same K/V tile.
    * One global (m, ell, acc) state is maintained across every level/tile.

    This implementation still uses Python loops and PyTorch operations for
    correctness. It is not meant to benchmark performance.

    Forward contract
    ----------------
        out, lse = multilevel_attention_tiled_poc(...)

        out: [B, N, Hq, Dv], dtype = V.dtype
        lse: [B, N, Hq],     dtype = torch.float32, finite everywhere.
             Natural-log log-sum-exp of the final attention scores S over
             exactly the multiresolution KV set A(q) of each query, where

                 S[q,k] = c * tanh( (q.k / sqrt(Dk)) / c )   if softcap=c
                        = q.k / sqrt(Dk)                     if softcap=None

                 LSE[b,q,h] = log sum_{k in A(q)} exp( S[q,k] ).

    Natural log is the API; a kernel that uses exp2 internally must convert.
    This is the Phase-2 *attention* LSE and is unrelated to the Phase-1
    summary weight w = logsumexp_h(q_h.k / sqrt(Dk)) from
    compute_summary_weights(). It is exactly the state the backward needs to
    recompute P = exp(S - LSE) without storing P.
    """
    B, N, Hq, Dk = q.shape
    Hkv = packed.k.shape[2]
    Dv = packed.v.shape[-1]

    if Hq % Hkv != 0:
        raise ValueError(
            f"Hq={Hq} must be divisible by Hkv={Hkv}."
        )

    expansion = Hq // Hkv

    # All range arithmetic comes from the pure-integer spec (range_spec.py),
    # in level-local coordinates:
    #   fwd_bounds   -> the KV tile range a query block streams per level
    #   range_bounds -> each row's own [k_lo, k_hi)
    #   elem_mask    -> realized vectorized below as (k >= k_lo) & (k < k_hi)
    # Storage addressing (offset[level] + local) happens only in _kv_tile.
    spec = RangeSpec.from_fmap(fmap, cache_size, N)

    # The packed buffer's geometry (derived from the tensors) must match the
    # geometry the schedule predicts.
    if tuple(packed.level_offsets) != spec.level_offsets():
        raise ValueError(
            f"packed level_offsets {tuple(packed.level_offsets)} != "
            f"spec {spec.level_offsets()}"
        )

    out = torch.empty(
        B, N, Hq, Dv, device=q.device, dtype=packed.v.dtype
    )
    # NaN-initialized so the test's isfinite() check doubles as an
    # "every row was written" check (POC only; a kernel need not do this).
    attn_lse = torch.full(
        (B, N, Hq), float("nan"), device=q.device, dtype=torch.float32
    )
    scale = 1.0 / math.sqrt(Dk)

    # POC loops over batch explicitly. A GPU kernel would normally make batch
    # part of the program/grid index.
    for b in range(B):
        # One logical program family per KV head. All associated query heads
        # share the K/V tiles loaded for this KV head.
        for hkv in range(Hkv):
            hq_start = hkv * expansion
            hq_end = (hkv + 1) * expansion
            hq_slice = slice(hq_start, hq_end)

            for q_start in range(0, N, block_m):
                q_end = min(q_start + block_m, N)
                M = q_end - q_start

                # [M, E, Dk], where E=GQA expansion.
                # The custom kernel would keep this Q tile in registers/shared
                # memory while streaming all hierarchy levels.
                Q = q[b, q_start:q_end, hq_slice]

                # FlashAttention online-softmax state for every
                # (query position, query head) row in this program.
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

                # Stream hierarchy levels. This loop mirrors the logical
                # kernel spec:
                #
                #   for level:
                #     [tile_k_lo, tile_k_hi) = fwd_bounds(q_start, q_end, level)
                #     per-row [row_k_lo, row_k_hi) = range_bounds(q, level)
                #     for KV tile in [tile_k_lo, tile_k_hi):        (level-local)
                #       K, V = packed[offset[level] + tile]        (storage)
                #       valid = physical_bounds AND k >= row_k_lo AND k < row_k_hi
                #
                # Only the semantics are frozen: a real kernel computes the
                # hull in O(1) from the first/last nonempty rows (monotone
                # bounds, see range_spec) rather than scanning BLOCK_M rows,
                # and derives row bounds in registers rather than from lists.
                # Physical bounds are implicit here (slices never exceed the
                # level length); a real kernel must predicate padded lanes of
                # a BLOCK_N tile before applying this semantic mask.
                for level in range(spec.num_levels):
                    # Hull of all rows' ranges: the tile range to stream,
                    # exactly as SWA kernels load tiles intersecting a query
                    # block's band and mask per element.
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

                    # Stream contiguous KV tiles from this hierarchy level.
                    for k_start in range(
                        tile_k_lo, tile_k_hi, block_n
                    ):
                        k_end = min(k_start + block_n, tile_k_hi)

                        # Contiguous reads (in packed index) from the
                        # level-major K/V buffers at offset[level] + tile,
                        # shared across BLOCK_M queries and all E GQA heads.
                        K, V = _kv_tile(
                            packed, b, hkv, level, k_start, k_end
                        )  # [Ktile, Dk], [Ktile, Dv]

                        # Tensor-Core-shaped operation conceptually:
                        #   [M*E, Dk] @ [Dk, Ktile]
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

                        # Per-query range mask INSIDE the tile: the vectorized
                        # realization of range_spec.elem_mask. This is only
                        # [M, Ktile], never [N, cache_len]. Query heads sharing
                        # one query position use the same mask.
                        valid = (
                            (kv_indices[None, :] >= level_start[:, None])
                            & (kv_indices[None, :] < level_end[:, None])
                        )  # [M, Ktile]

                        scores = scores.masked_fill(
                            ~valid[:, None, :], -float("inf")
                        )

                        # --------------------------------------------------
                        # FlashAttention online-softmax update.
                        # --------------------------------------------------
                        active = valid.any(dim=1)[:, None].expand(
                            -1, expansion
                        )
                        block_max = scores.float().max(dim=-1).values
                        m_candidate = torch.maximum(m, block_max)
                        m_new = torch.where(active, m_candidate, m)

                        # For an active row whose previous m=-inf, this is
                        # exp(-inf - finite)=0, which is exactly what we want.
                        # Inactive rows use scale exp(0)=1 and leave state
                        # unchanged. The where() sits INSIDE exp() on purpose:
                        # inactive rows have m = m_new = -inf, and exp() must
                        # never consume (-inf) - (-inf) = nan, because exp's
                        # autograd backward (grad * exp(x)) would turn the
                        # zero gradient of the unselected branch into 0 * nan
                        # and poison dq/dk/dv. Forward values are identical.
                        old_scale = torch.exp(
                            torch.where(
                                active,
                                m - m_new,
                                torch.zeros_like(m),
                            )
                        )

                        # Avoid -inf - -inf for masked/inactive positions.
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
                # Online-softmax state -> per-row LSE (natural log, FP32).
                attn_lse[b, q_start:q_end, hq_slice] = m + torch.log(ell)

    return out, attn_lse


# ============================================================
# Explicit FlashAttention-style Phase-2 backward POC over the packed
# K/V buffers. Two passes, no atomics, no materialized P:
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
# work, never semantics. `lse` must be the forward's saved value (this is
# where the Step-2 (out, lse) contract is consumed); it is never recomputed.
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
    # P reconstructed from the SAVED forward LSE. The where() sits before
    # exp() so exp never sees an invalid score (which is unconstrained by
    # the row's LSE and could overflow to a useless inf temporary):
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
def multilevel_attention_backward_poc(
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
    Phase-1 graph that produced packed.k/v.
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


# ============================================================
# Dense flattened-mask reference corresponding to the original
# implementation. Deliberately expensive; use only at small N.
# ============================================================

def dense_reference_attention(
    q: torch.Tensor,
    k_levels: List[torch.Tensor],
    v_levels: List[torch.Tensor],
    valid_levels: List[torch.Tensor],
    select_level: torch.Tensor,
    select_index: torch.Tensor,
    softcap: float = 20.0,
):
    """
    Reconstruct the original behavior:

        level-major hierarchy
          -> flatten all old levels
          -> dense Boolean [N, cache_len] mask
          -> masked attention

    This is only a correctness oracle.
    """
    B, N, Hq, Dk = q.shape
    Hkv = k_levels[1].shape[2]
    expansion = Hq // Hkv

    K_flat = torch.cat(k_levels[1:], dim=1)
    V_flat = torch.cat(v_levels[1:], dim=1)
    valid_flat = torch.cat(valid_levels[1:])

    offsets = torch.zeros(
        len(k_levels), dtype=torch.long, device=q.device
    )
    cursor = 0
    for old_level in range(1, len(k_levels)):
        offsets[old_level] = cursor
        cursor += k_levels[old_level].shape[1]

    flat_indices = offsets[select_level] + select_index

    mask = torch.zeros(
        N,
        cursor,
        dtype=torch.bool,
        device=q.device,
    )
    mask.scatter_(1, flat_indices, True)
    mask[:, ~valid_flat] = False

    K = K_flat.transpose(1, 2).repeat_interleave(expansion, dim=1)
    V = V_flat.transpose(1, 2).repeat_interleave(expansion, dim=1)
    Q = q.transpose(1, 2)

    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / math.sqrt(Dk)

    if softcap is not None:
        scores = softcap * torch.tanh(scores / softcap)

    scores = scores.masked_fill(~mask[None, None], float("-inf"))

    # Direct LSE(S) over the masked, softcapped scores: the reference for the
    # forward contract's `lse`. Computed before softmax so it is visibly the
    # direct quantity, not reconstructed from probabilities.
    attn_lse = torch.logsumexp(scores.float(), dim=-1).transpose(1, 2)

    probs = torch.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bhkd->bhqd", probs, V)

    return output.transpose(1, 2), attn_lse


# ============================================================
# Extra direct check: every real node in the old hierarchy must
# equal its corresponding canonical dyadic node.
# ============================================================

def check_summary_equivalence(
    old_k_levels,
    old_v_levels,
    old_valid_levels,
    dyadic_k_levels,
    dyadic_v_levels,
    old_to_dyadic,
    rtol=1e-5,
    atol=1e-5,
):
    for old_level in range(1, len(old_k_levels)):
        dyadic_level = old_level - 1
        mapping = old_to_dyadic[old_level]
        real = mapping >= 0

        # Also require the old reference's own validity bookkeeping to agree.
        assert torch.equal(real, old_valid_levels[old_level])

        if not real.any():
            continue

        mapped_idx = mapping[real]

        old_k = old_k_levels[old_level][:, real]
        old_v = old_v_levels[old_level][:, real]
        new_k = dyadic_k_levels[dyadic_level][:, mapped_idx]
        new_v = dyadic_v_levels[dyadic_level][:, mapped_idx]

        torch.testing.assert_close(old_k, new_k, rtol=rtol, atol=atol)
        torch.testing.assert_close(old_v, new_v, rtol=rtol, atol=atol)


# ============================================================
# Test harness: multi-configuration equivalence suite.
#
# Every case runs the full chain
#
#     plan POC == dense reference == dyadic POC
#              == analytic-range POC == online POC == tiled POC (packed K/V)
#
# plus the analytic-ranges-vs-remapped-plan oracle, the range_spec and
# packed-geometry oracles, and asserts a set of
# coverage flags so a config edit cannot silently stop exercising a regime
# (warm-up, eviction, odd hierarchy lengths, partial tiles, Dv != Dk, GQA).
#
# Runtime note: the loop-based POCs are intentionally Python-heavy and may be
# slow -- particularly on CUDA, because multilevel_attention_plan_poc /
# _dyadic_poc do ~N*cache_size scalar .item() calls per case (about 164k at
# N=1024, cache_size=160), and every CUDA .item() is a host-device
# synchronization. The tiled POC intentionally performs its range arithmetic
# as Python integer calls into range_spec and constructs small metadata
# tensors per query block and level. Runtime is not representative of the
# intended kernel. Use filtered cases and/or --device cpu during development;
# the full suite is a correctness regression test.
#
#     python test_tiled.py [--device cpu|cuda] [case ...]
# ============================================================

RTOL = 1e-5
ATOL = 1e-5
VERBOSE = False


def report(name, a, b, rtol=RTOL, atol=ATOL):
    err = (a - b).abs()
    if VERBOSE:
        print(f"\n{name}:")
        print("  max abs error :", err.max().item())
        print("  mean abs error:", err.mean().item())
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    if VERBOSE:
        print("  PASS")
    else:
        print(f"  {name:<24} max={err.max().item():.2e} "
              f"mean={err.mean().item():.2e} PASS")


def check_lse_contract(name, lse, B, N, Hq):
    """Forward-contract invariants for a returned attention LSE."""
    if tuple(lse.shape) != (B, N, Hq):
        raise AssertionError(
            f"{name}: lse shape {tuple(lse.shape)} != {(B, N, Hq)}"
        )
    if lse.dtype != torch.float32:
        raise AssertionError(f"{name}: lse dtype {lse.dtype} != float32")
    if not bool(torch.isfinite(lse).all().item()):
        raise AssertionError(f"{name}: lse contains non-finite values")


def check_block_ranges_match_scalar(
    N: int,
    fmap: Dict[int, int],
    cache_size: int,
    device: torch.device,
):
    """
    The tiled POC is the only consumer of the vectorized range computation.
    Check it row-by-row against the scalar version for every query.
    Returns (starts, ends) as [N, num_levels] long tensors.
    """
    activation_times = validate_dyadic_fmap(fmap)
    starts, ends = dyadic_ranges_for_query_block(
        torch.arange(N, device=device), fmap, cache_size, activation_times
    )
    starts_l = starts.tolist()
    ends_l = ends.tolist()
    for qi in range(N):
        scalar = dyadic_ranges_for_query(
            qi, fmap, cache_size, activation_times
        )
        block = list(zip(starts_l[qi], ends_l[qi]))
        if scalar != block:
            raise AssertionError(
                f"Block/scalar range mismatch at query={qi}:\n"
                f"  scalar={scalar}\n  block ={block}"
            )
    return starts, ends


def fmt_ranges(ranges):
    return ", ".join(
        f"L{level}=[{start},{end})"
        for level, (start, end) in enumerate(ranges)
        if start != end
    )


# Hand-derived from the README replay table for N=8, cache_size=6,
# fmap={1:2,2:3} (an oracle independent of the original implementation):
#
#   q=0: [t0]                    q=4: [t4,t3,t2,P1,Q0]
#   q=1: [t1,t0,P0(dummy)]       q=5: [t5,t4,P2,P1,Q0]
#   q=2: [t2,t1,t0,Q0(dummy)]    q=6: [t6,t5,t4,Q1,Q0]
#   q=3: [t3,t2,P1,Q0]           q=7: [t7,t6,P3,Q1,Q0,D]
#
# P1=merge(t0,t1)=L1[0], P2=L1[1], P3=merge(t4,t5)=L1[2],
# Q1=merge(P1,P2)=tokens 0..3=L2[0].
README_TINY_GOLDEN = {
    0: [(0, 1), (0, 0), (0, 0)],
    1: [(0, 2), (0, 0), (0, 0)],
    2: [(0, 3), (0, 0), (0, 0)],
    3: [(2, 4), (0, 1), (0, 0)],
    4: [(2, 5), (0, 1), (0, 0)],
    5: [(4, 6), (0, 2), (0, 0)],
    6: [(4, 7), (0, 0), (0, 1)],
    7: [(6, 8), (2, 3), (0, 1)],
}


def check_coverage(
    name,
    *,
    N, Hq, Hkv, Dk, Dv, cache_size, fmap, block_m, block_n,
    merge_plan, dyadic_select_index, dyadic_k_levels, starts, ends,
    expect,
):
    """
    Compute what the instantiated config actually exercises, compare to
    `expect`, and print a summary. Runs before any attention POC.
    """
    activation_times = validate_dyadic_fmap(fmap)
    level_lens = [k.shape[1] for k in dyadic_k_levels]

    totals = (ends - starts).sum(dim=1)
    full_q = torch.nonzero(totals == cache_size).flatten()
    evict_q = torch.nonzero(starts[:, -1] > 0).flatten()

    actual = {
        "activation_times": activation_times,
        "num_summary_levels": len(merge_plan) - 2,
        "warm_up": bool((dyadic_select_index < 0).any().item()),
        "full_cache": full_q.numel() > 0,
        "eviction": evict_q.numel() > 0,
        "odd_levels": [l for l, n in enumerate(level_lens) if n % 2 == 1],
        "partial_m": N % block_m != 0,
        "partial_n": any(n % block_n != 0 for n in level_lens),
        "dv_ne_dk": Dv != Dk,
        "expansion": Hq // Hkv,
    }
    assert len(dyadic_k_levels) == len(activation_times)

    first_full = int(full_q[0].item()) if actual["full_cache"] else None
    first_evict = int(evict_q[0].item()) if actual["eviction"] else None

    def tick(flag, q=None):
        if not flag:
            return "-"
        return "✓" if q is None else f"✓(q={q})"

    print(f"activation: {activation_times}  "
          f"levels={actual['num_summary_levels']}")
    print("hierarchy: " + " / ".join(str(n) for n in level_lens)
          + f"  packed={sum(level_lens)}")
    print(
        f"coverage: warm_up {tick(actual['warm_up'])} "
        f"full_cache {tick(actual['full_cache'], first_full)} "
        f"eviction {tick(actual['eviction'], first_evict)} "
        f"odd_levels={actual['odd_levels']}"
    )
    print(
        f"          partial_M {tick(actual['partial_m'])} "
        f"partial_N {tick(actual['partial_n'])} "
        f"Dv!=Dk {tick(actual['dv_ne_dk'])} "
        f"GQA={actual['expansion']}"
    )

    unknown = set(expect) - set(actual)
    if unknown:
        raise KeyError(f"[{name}] unknown expect keys: {sorted(unknown)}")
    mismatches = {
        key: (expect[key], actual[key])
        for key in expect
        if expect[key] != actual[key]
    }
    if mismatches:
        raise AssertionError(
            f"[{name}] coverage mismatch (expected, actual): {mismatches}"
        )
    if actual["eviction"] and not actual["full_cache"]:
        raise AssertionError(
            f"[{name}] eviction without a full cache is impossible."
        )

    return activation_times, first_evict


def run_case(
    name,
    *,
    B, N, Hq, Hkv, Dk, Dv, cache_size, fmap, block_m, block_n,
    expect, device, golden_ranges=None, seed=0,
    dtype=torch.float32,
):
    print(
        f"\n=== {name}: B={B} N={N} Hq={Hq} Hkv={Hkv} Dk={Dk} Dv={Dv} "
        f"cache={cache_size} fmap={fmap} BLOCK_M={block_m} "
        f"BLOCK_N={block_n} device={device} ==="
    )

    # Per-case seed: a filtered run draws exactly the same tensors as the
    # full run, so a filtered failure is reproducible.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    q = torch.randn(B, N, Hq, Dk, device=device, dtype=dtype)
    k = torch.randn(B, N, Hkv, Dk, device=device, dtype=dtype)
    v = torch.randn(B, N, Hkv, Dv, device=device, dtype=dtype)

    # --------------------------------------------------------
    # Original selection/merge plan and level-major reference summaries.
    # --------------------------------------------------------
    merge_plan, select_level, select_index = get_structured_plan(
        N, fmap, cache_size, device
    )
    old_k_levels, old_v_levels, _, old_valid_levels = (
        build_plan_based_summaries(q, k, v, merge_plan)
    )

    # --------------------------------------------------------
    # Canonical dyadic summaries + remapped plan (cheap; needed for
    # the coverage check below).
    # --------------------------------------------------------
    num_summary_levels = len(merge_plan) - 2
    dyadic_k_levels, dyadic_v_levels, _ = build_dyadic_summaries(
        q, k, v, num_summary_levels
    )
    dyadic_select_level, dyadic_select_index, old_to_dyadic = (
        remap_plan_to_dyadic(merge_plan, select_level, select_index, N)
    )
    packed = pack_levels(dyadic_k_levels, dyadic_v_levels)

    # --------------------------------------------------------
    # Analytic ranges: vectorized vs scalar, then coverage flags.
    # --------------------------------------------------------
    starts, ends = check_block_ranges_match_scalar(
        N, fmap, cache_size, device
    )
    activation_times, first_evict = check_coverage(
        name,
        N=N, Hq=Hq, Hkv=Hkv, Dk=Dk, Dv=Dv, cache_size=cache_size,
        fmap=fmap, block_m=block_m, block_n=block_n,
        merge_plan=merge_plan, dyadic_select_index=dyadic_select_index,
        dyadic_k_levels=dyadic_k_levels, starts=starts, ends=ends,
        expect=expect,
    )

    if VERBOSE:
        print("\nOriginal hierarchy sizes (includes dummy/warm-up nodes):")
        for old_level in range(1, len(old_k_levels)):
            print(f"  old L{old_level - 1}: "
                  f"{old_k_levels[old_level].shape[1]} entries")

        L = len(activation_times) - 1
        print("\nExample analytic ranges [start, end):")
        sample = [0, activation_times[1] - 1, activation_times[1],
                  activation_times[L], first_evict, N - 1]
        seen = set()
        for qi in sample:
            if qi is None or qi < 0 or qi >= N or qi in seen:
                continue
            seen.add(qi)
            ranges = dyadic_ranges_for_query(
                qi, fmap, cache_size, activation_times
            )
            total = sum(end - start for start, end in ranges)
            print(f"  q={qi:>4}: {fmt_ranges(ranges)}  (total={total})")

    # --------------------------------------------------------
    # Structural oracles: analytic ranges vs remapped original plan
    # (the key one), summaries, and optional hand-derived golden ranges.
    # --------------------------------------------------------
    oracles = []

    validate_analytic_ranges_against_plan(
        dyadic_select_level, dyadic_select_index, fmap, cache_size
    )
    oracles.append("ranges-plan ✓")
    oracles.append("block-scalar ✓")  # check_block_ranges_match_scalar above

    # The pure-integer spec the tiled POC consumes must agree with the
    # block oracle for every query and level (ties range_spec.py to this
    # suite; test_range_spec.py holds the exhaustive property tests).
    spec = RangeSpec.from_fmap(fmap, cache_size, N)
    starts_l, ends_l = starts.tolist(), ends.tolist()
    for qi in range(N):
        for level in range(spec.num_levels):
            got = range_bounds(spec, qi, level)
            if got != (starts_l[qi][level], ends_l[qi][level]):
                raise AssertionError(
                    f"[{name}] range_spec mismatch at q={qi}, L{level}: "
                    f"spec={got} oracle="
                    f"{(starts_l[qi][level], ends_l[qi][level])}"
                )
    oracles.append("spec ✓")

    # Packed geometry (derived from the tensors) vs the geometry RangeSpec
    # predicts, and each packed interval equals its level tensor.
    if tuple(packed.level_offsets) != spec.level_offsets():
        raise AssertionError(
            f"[{name}] packed offsets {packed.level_offsets} != "
            f"spec {spec.level_offsets()}"
        )
    if packed.k.shape[1] != spec.total_len or packed.v.shape[1] != spec.total_len:
        raise AssertionError(f"[{name}] packed length != spec.total_len")
    off = packed.level_offsets
    for level in range(spec.num_levels):
        if not torch.equal(packed.k[:, off[level]:off[level + 1]],
                           dyadic_k_levels[level]):
            raise AssertionError(f"[{name}] packed K level {level} mismatch")
        if not torch.equal(packed.v[:, off[level]:off[level + 1]],
                           dyadic_v_levels[level]):
            raise AssertionError(f"[{name}] packed V level {level} mismatch")
    oracles.append("pack ✓")

    check_summary_equivalence(
        old_k_levels, old_v_levels, old_valid_levels,
        dyadic_k_levels, dyadic_v_levels, old_to_dyadic,
        rtol=RTOL, atol=ATOL,
    )
    oracles.append("summaries ✓")

    if golden_ranges is not None:
        assert len(golden_ranges) == N
        for qi in range(N):
            got = dyadic_ranges_for_query(
                qi, fmap, cache_size, activation_times
            )
            if got != golden_ranges[qi]:
                raise AssertionError(
                    f"[{name}] golden range mismatch at q={qi}: "
                    f"expected {golden_ranges[qi]}, got {got}"
                )
        oracles.append("golden ✓")

    print("oracles: " + " ".join(oracles))

    # --------------------------------------------------------
    # Attention equivalence chain for `out`.
    # --------------------------------------------------------
    print()
    out_plan = multilevel_attention_plan_poc(
        q, old_k_levels, old_v_levels, old_valid_levels,
        select_level, select_index,
    )
    out_dense, lse_dense = dense_reference_attention(
        q, old_k_levels, old_v_levels, old_valid_levels,
        select_level, select_index,
    )
    report("out: plan vs dense", out_plan, out_dense)

    out_dyadic = multilevel_attention_dyadic_poc(
        q, dyadic_k_levels, dyadic_v_levels,
        dyadic_select_level, dyadic_select_index,
    )
    report("out: dyadic vs plan", out_dyadic, out_plan)
    report("out: dyadic vs dense", out_dyadic, out_dense)

    out_ranges = multilevel_attention_ranges_poc(
        q, dyadic_k_levels, dyadic_v_levels, fmap, cache_size
    )
    report("out: ranges vs dyadic", out_ranges, out_dyadic)

    out_online, lse_online = multilevel_attention_ranges_online_poc(
        q, dyadic_k_levels, dyadic_v_levels, fmap, cache_size
    )
    report("out: online vs ranges", out_online, out_ranges)
    report("out: online vs dense", out_online, out_dense)

    out_tiled, lse_tiled = multilevel_attention_tiled_poc(
        q, packed, fmap, cache_size,
        block_m=block_m, block_n=block_n,
    )
    report("out: tiled vs online", out_tiled, out_online)
    report("out: tiled vs dense", out_tiled, out_dense)

    # --------------------------------------------------------
    # Forward contract: lse = m + log(ell), FP32 [B, N, Hq].
    # Triangular check: dense direct logsumexp is the reference; the
    # online-per-level and tiled paths accumulate in different orders.
    # --------------------------------------------------------
    print()
    check_lse_contract("dense", lse_dense, B, N, Hq)
    check_lse_contract("online", lse_online, B, N, Hq)
    check_lse_contract("tiled", lse_tiled, B, N, Hq)
    report("lse: online vs dense", lse_online, lse_dense)
    report("lse: tiled vs dense", lse_tiled, lse_dense)
    report("lse: tiled vs online", lse_tiled, lse_online)

    print(f"\n[{name}] PASS")


FMAP_DEFAULT = {1: 64, 2: 72, 3: 80}

CASES = [
    # Current single clean config, kept as a regression: warm-up only.
    dict(
        name="baseline",
        B=2, N=128, Hq=8, Hkv=2, Dk=32, Dv=32,
        cache_size=512, fmap=FMAP_DEFAULT, block_m=16, block_n=32,
        expect=dict(
            activation_times=[0, 65, 82, 116], num_summary_levels=3,
            warm_up=True, full_cache=False, eviction=False,
            # L3 has 16 entries < BLOCK_N=32, so its single tile is partial.
            odd_levels=[], partial_m=False, partial_n=True,
            dv_ne_dk=False, expansion=4,
        ),
    ),
    # Steady-state coarsest-level eviction, partial Q/K tiles, Dv != Dk.
    # Same config as test_vs_train.py case 2.
    dict(
        name="eviction",
        B=1, N=1024, Hq=4, Hkv=2, Dk=32, Dv=64,
        cache_size=160, fmap=FMAP_DEFAULT, block_m=24, block_n=40,
        expect=dict(
            activation_times=[0, 65, 82, 116], num_summary_levels=3,
            warm_up=True, full_cache=True, eviction=True,
            odd_levels=[], partial_m=True, partial_n=True,
            dv_ne_dk=True, expansion=2,
        ),
    ),
    # Every hierarchy length odd: 1023 -> 511 -> 255 -> 127.
    dict(
        name="odd_n",
        B=1, N=1023, Hq=6, Hkv=2, Dk=32, Dv=64,
        cache_size=160, fmap=FMAP_DEFAULT, block_m=24, block_n=40,
        expect=dict(
            activation_times=[0, 65, 82, 116], num_summary_levels=3,
            warm_up=True, full_cache=True, eviction=True,
            odd_levels=[0, 1, 2, 3], partial_m=True, partial_n=True,
            dv_ne_dk=True, expansion=3,
        ),
    ),
    # Alternate dyadically aligned schedule with different activation times.
    dict(
        name="alt_fmap",
        B=1, N=1024, Hq=8, Hkv=4, Dk=32, Dv=48,
        cache_size=160, fmap={1: 64, 2: 68, 3: 80}, block_m=16, block_n=48,
        expect=dict(
            activation_times=[0, 65, 74, 124], num_summary_levels=3,
            warm_up=True, full_cache=True, eviction=True,
            odd_levels=[], partial_m=False, partial_n=True,
            dv_ne_dk=True, expansion=2,
        ),
    ),
    # Deeper tree (4 summary levels), odd N, eviction.
    dict(
        name="four_levels",
        B=1, N=1001, Hq=8, Hkv=4, Dk=32, Dv=32,
        cache_size=96, fmap={1: 32, 2: 40, 3: 48, 4: 56},
        block_m=24, block_n=40,
        expect=dict(
            activation_times=[0, 33, 50, 84, 152], num_summary_levels=4,
            warm_up=True, full_cache=True, eviction=True,
            odd_levels=[0, 3], partial_m=True, partial_n=True,
            dv_ne_dk=False, expansion=2,
        ),
    ),
    # The README worked example, with hand-derived golden ranges.
    dict(
        name="readme_tiny",
        B=2, N=8, Hq=4, Hkv=2, Dk=8, Dv=16,
        cache_size=6, fmap={1: 2, 2: 3}, block_m=3, block_n=2,
        expect=dict(
            activation_times=[0, 3, 6], num_summary_levels=2,
            warm_up=True, full_cache=False, eviction=False,
            odd_levels=[], partial_m=True, partial_n=False,
            dv_ne_dk=True, expansion=2,
        ),
        golden_ranges=README_TINY_GOLDEN,
    ),
]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Telescoping-cache forward POC equivalence suite."
    )
    parser.add_argument(
        "--device", default=None,
        help="cpu or cuda (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="print hierarchy sizes, example ranges and full error blocks",
    )
    parser.add_argument(
        "cases", nargs="*",
        help=f"case names to run (default: all). "
             f"Available: {[c['name'] for c in CASES]}",
    )
    args = parser.parse_args(argv)

    global VERBOSE
    VERBOSE = args.verbose

    if args.device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    names = {c["name"] for c in CASES}
    unknown = [c for c in args.cases if c not in names]
    if unknown:
        raise SystemExit(
            f"Unknown case(s) {unknown}; available: {sorted(names)}"
        )
    selected = [
        c for c in CASES if not args.cases or c["name"] in args.cases
    ]

    for case in selected:
        run_case(device=device, **case)

    print(f"\nALL TESTS PASSED ({len(selected)} cases)")


if __name__ == "__main__":
    main()
