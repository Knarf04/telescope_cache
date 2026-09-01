"""
Telescoping attention as a FlashAttention-4 `mask_mod` over the packed buffer.

This is the *baseline* GPU path, not the fast one. It exists for two reasons:

  1. It validates the whole formulation -- packed level-major K/V, closed-form
     row bounds, GQA, softcap -- against `reference.py` on real hardware
     before any custom kernel exists.
  2. It is the honest performance floor to beat. FA4 walks all
     ceil(sumN / tile_n) key blocks per query block and evaluates the mask
     elementwise; the telescope kernel's whole point is to walk only the
     O(cache_size) blocks the schedule actually selects.

The key simplification
---------------------
`range_spec` keeps semantics in level-local coordinates and adds
`level_offsets[l]` only when addressing storage. In packed coordinates a
query's attended set is

    A(q) = U_l  [ off[l] + lo_l(q),  off[l] + hi_l(q) )

and those L+1 intervals are pairwise DISJOINT, because the spec guarantees
hi_l(q) <= level_len(l) so interval l never leaves level l's slab. So the
mask needs no "which level does this column belong to" decode at all -- it is
just an OR of L+1 interval tests on the packed column index. Empty levels
carry themselves: `cute.ranges` returns lo >= hi for them, and
`off+lo <= kv < off+hi` is then false for every kv.
"""

import math
from typing import Dict, Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute
from flash_attn.cute import utils

from telescope_cache.cute.ranges import IntOps, level_row_bounds
from telescope_cache.range_spec import RangeSpec

__all__ = ["make_telescope_mask_mod", "telescope_attn_mask_mod"]


# cutlass.max / cutlass.min are the Int32-SSA counterparts of the builtins
# that `cute.ranges` uses on the host. They are the only thing that differs.
CUTE_OPS = IntOps(minimum=cutlass.min, maximum=cutlass.max)


def make_telescope_mask_mod(
    activation_times: Tuple[int, ...],
    cache_size: int,
    level_offsets: Tuple[int, ...],
):
    """
    Build the mask_mod closure for one schedule.

    Every schedule constant is captured by value, so the DSL sees compile-time
    integers and unrolls the level loop; only q_idx and kv_idx are dynamic.
    FA4 keys its compile cache on `utils.hash_callable(mask_mod)`, so distinct
    schedules correctly compile to distinct kernels.
    """
    a = tuple(activation_times)
    offsets = tuple(level_offsets)
    num_levels = len(a)
    assert len(offsets) == num_levels + 1, (offsets, num_levels)

    @cute.jit
    def telescope_mask_mod(
        batch_idx,
        head_idx,
        q_idx,
        kv_idx,
        seqlen_info,
        aux_tensors=None,
        aux_scalars=None,
    ):
        # FA4 hands mask_mod indices as cute.TensorSSA, not scalars, and
        # cutlass.min/max are scalar ops. On the SM8x/SM90 path the vector
        # length is 1 (mask.py's `apply_mask` scalar branch calls
        # scalar_to_ssa per element), so unwrap, work in Int32, rewrap.
        # SM100's vectorized mask_mod path would need a vector formulation;
        # the real kernel does not go through mask_mod at all.
        q = utils.ssa_to_scalar(q_idx)
        kv = utils.ssa_to_scalar(kv_idx)

        bounds = level_row_bounds(q, a, cache_size, ops=CUTE_OPS)
        keep = cutlass.Boolean(False)
        for level in cutlass.range_constexpr(num_levels):
            lo, hi = bounds[level]
            base = offsets[level]
            keep = keep | ((kv >= base + lo) & (kv < base + hi))
        return utils.scalar_to_ssa(keep, cutlass.Boolean)

    return telescope_mask_mod


def telescope_attn_mask_mod(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    fmap: Dict[int, int],
    cache_size: int,
    softcap: float = 20.0,
    softmax_scale: Optional[float] = None,
    return_lse: bool = True,
):
    """
    q         [B, N, Hq, Dk]
    k_packed  [B, sumN, Hkv, Dk]      (telescope_cache.reference.pack_levels)
    v_packed  [B, sumN, Hkv, Dv]

    Returns (out [B, N, Hq, Dv], lse [B, N, Hq] float32) matching the
    `multilevel_attention_forward` contract -- note FA4 hands back lse as
    [B, Hq, N], which this transposes so callers can compare directly.
    """
    from flash_attn.cute.interface import flash_attn_func

    B, N, Hq, Dk = q.shape
    spec = RangeSpec.from_fmap(fmap, cache_size, N)
    offsets = spec.level_offsets()
    if k_packed.shape[1] != offsets[-1]:
        raise ValueError(
            f"packed length {k_packed.shape[1]} != schedule total {offsets[-1]}"
        )

    mask_mod = make_telescope_mask_mod(spec.activation_times, cache_size, offsets)
    out, lse = flash_attn_func(
        q,
        k_packed,
        v_packed,
        softmax_scale=softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(Dk),
        # No causal/window: the schedule is entirely in mask_mod.
        softcap=softcap if softcap is not None else 0.0,
        mask_mod=mask_mod,
        return_lse=True,
    )
    lse = lse.transpose(1, 2).contiguous()  # [B, Hq, N] -> [B, N, Hq]
    return (out, lse) if return_lse else out
