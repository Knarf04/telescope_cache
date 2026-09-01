"""
Block-aligned level-major packing: the storage layout the kernel wants.

`reference.pack_levels` concatenates the levels tightly:

    [ L0 | L1 | ... | LL ]      level_offsets[l] = sum_{j<l} level_len(j)

which is right for the semantics but wrong for a tiled kernel: level l starts
at an arbitrary offset, so a level-local KV tile [k_lo, k_hi) lands at
`offsets[l] + k_lo`, which is not a multiple of tile_n. A kernel indexing
`gK[..., n_block]` cannot express that without an unaligned base pointer and
per-level re-predication of every copy.

Padding each level up to a multiple of tile_n fixes it outright:

    [ L0 | pad | L1 | pad | ... | LL | pad ]    pad_offsets[l] % tile_n == 0

Now every level starts on a block boundary, a level-local block index k_lo //
tile_n is just an offset from `pad_offsets[l] // tile_n`, and -- because the
buffer length is itself a whole number of blocks -- *no* K/V copy in the
kernel needs seqlen predication at all. The cost is at most tile_n - 1
entries per level: for the shapes here (tile_n = 64, 4 levels) under 256 rows
against a buffer of ~2N.

The padding must be ZERO, not uninitialized. Masked lanes get score -inf, so
P = exp(-inf) = 0 and the padded K never affects the result -- but the PV
matmul still computes 0 * V on those lanes, and 0 * NaN is NaN. Garbage
bfloat16 can be NaN. `torch.zeros` is the whole guard.

Correctness of the mask over padded entries is automatic: `range_spec`
guarantees hi_l(q) <= level_len(l), so a padded row (local index >=
level_len) is outside every query's range and is masked at every level.
"""

from typing import List, NamedTuple, Sequence, Tuple

import torch

from telescope_cache.reference import PackedKV

__all__ = ["AlignedPackedKV", "aligned_level_offsets", "pack_levels_aligned", "repack_aligned"]


class AlignedPackedKV(NamedTuple):
    k: torch.Tensor                       # [B, padded_total, Hkv, Dk]
    v: torch.Tensor                       # [B, padded_total, Hkv, Dv]
    level_offsets: Tuple[int, ...]        # tight offsets, for cross-checks
    pad_offsets: Tuple[int, ...]          # len L+2, every entry % tile_n == 0
    level_lens: Tuple[int, ...]           # real entries per level (unpadded)
    tile_n: int


def aligned_level_offsets(level_lens: Sequence[int], tile_n: int) -> Tuple[int, ...]:
    """Prefix sums of each level's length rounded up to a multiple of tile_n."""
    offsets = [0]
    for n in level_lens:
        blocks = -(-n // tile_n)  # ceil
        offsets.append(offsets[-1] + blocks * tile_n)
    return tuple(offsets)


def pack_levels_aligned(
    k_levels: List[torch.Tensor],
    v_levels: List[torch.Tensor],
    tile_n: int,
) -> AlignedPackedKV:
    """
    Same inputs as `reference.pack_levels`, block-aligned output.

    Levels are copied into a zeroed buffer rather than concatenated, so the
    padding is zero by construction and there is no `torch.cat` temporary.
    """
    if len(k_levels) != len(v_levels) or not k_levels:
        raise ValueError("k_levels and v_levels must be nonempty, same length")

    B, _, Hkv, Dk = k_levels[0].shape
    Dv = v_levels[0].shape[-1]
    lens = tuple(k_l.shape[1] for k_l in k_levels)
    for level, (k_l, v_l) in enumerate(zip(k_levels, v_levels)):
        if k_l.shape[1] != v_l.shape[1]:
            raise ValueError(f"level {level}: K/V length mismatch")

    pad_offsets = aligned_level_offsets(lens, tile_n)
    total = pad_offsets[-1]

    k = k_levels[0].new_zeros(B, total, Hkv, Dk)
    v = v_levels[0].new_zeros(B, total, Hkv, Dv)
    for level, (k_l, v_l) in enumerate(zip(k_levels, v_levels)):
        start = pad_offsets[level]
        k[:, start : start + lens[level]] = k_l
        v[:, start : start + lens[level]] = v_l

    tight = [0]
    for n in lens:
        tight.append(tight[-1] + n)
    return AlignedPackedKV(k, v, tuple(tight), pad_offsets, lens, tile_n)


def repack_aligned(packed: PackedKV, tile_n: int) -> AlignedPackedKV:
    """Re-layout an existing tight `PackedKV` (used by the tests, which build
    the tight packing to feed the reference and need the same values here)."""
    offsets = packed.level_offsets
    lens = tuple(offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1))
    k_levels = [packed.k[:, offsets[i] : offsets[i + 1]] for i in range(len(lens))]
    v_levels = [packed.v[:, offsets[i] : offsets[i + 1]] for i in range(len(lens))]
    return pack_levels_aligned(k_levels, v_levels, tile_n)
