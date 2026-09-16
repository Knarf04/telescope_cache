"""Level-major packing of the summary tree into one K/V buffer."""

from typing import List, NamedTuple, Tuple

import torch

from telescope_cache.telescoping_attn.range_spec import RangeSpec

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

    # Offsets come from the tensors, never from RangeSpec, so the test's
    # comparison against spec.level_offsets() is a real check.
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
    level-local [k_start, k_end) -> physical [p_start, p_end). The single place
    k_local -> k_physical is translated.
    """
    lo = packed.level_offsets[level]
    hi = packed.level_offsets[level + 1]
    level_len = hi - lo
    # catches a packed index passed where a local one belongs
    assert 0 <= k_start <= k_end <= level_len, (level, k_start, k_end, level_len)
    p_start, p_end = lo + k_start, lo + k_end
    # a tile never crosses into the next level
    assert lo <= p_start <= p_end <= hi, (level, p_start, p_end, lo, hi)
    return p_start, p_end

def _kv_tile(
    packed: PackedKV, b: int, hkv: int, level: int, k_start: int, k_end: int
):
    """Read one level-local KV tile from the packed buffers."""
    p_start, p_end = _packed_slice(packed, level, k_start, k_end)
    return packed.k[b, p_start:p_end, hkv], packed.v[b, p_start:p_end, hkv]
