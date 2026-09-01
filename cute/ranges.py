"""
Branch-free closed forms of `telescope_cache.range_spec` for the GPU kernel.

`range_spec` is the frozen contract, written for clarity: it branches, it
loops over a query block to build a hull, and at the coarsest level it binary
searches. None of that belongs in a kernel. This module re-derives the same
integers using ONLY

    +   -   *   >> (by a compile-time constant)   min   max

so a single source runs three ways:

    * host, on Python ints          -> checked against range_spec exhaustively
    * host, on torch tensors        -> vectorized oracle for the tests
    * device, on CuTeDSL Int32 SSA  -> inside the kernel, no divergence

Every helper takes the (min, max) pair as arguments (`ops`) because that is
the only place the three backends differ. There is no `if` on a data value
anywhere below, which is what makes the device path branch-free.

Empty ranges
------------
`range_spec` normalizes empty to (0, 0). This module does NOT: it returns a
pair with lo >= hi. That is deliberate -- the only consumer is the predicate
`lo <= k < hi`, which is already false for any such pair, so normalizing
would cost instructions to make a value nothing reads. `normalize_empty()`
exists solely so the tests can compare against range_spec's convention.

Derivations
-----------
For x >= 0 and span = 2^l,  (x >> l) + 1 == (x + span) >> l.  Both closed
forms below use that to fold the "+1" of a half-open upper bound into the
shift, which then also absorbs the `q < a[l]` guard: when q < a[l] the
shifted quantity is already 0.

  Fine level l < L, from range_spec._fine_bounds:
      newest = (q - a[l]) >> l                    for q >= a[l]
      oldest = 2 * ((q - a[l+1]) // 2^(l+1) + 1)  for q >= a[l+1], else 0
  becomes
      hi = max(q - a[l]   +   span, 0) >> l
      lo = max(q - a[l+1] + 2*span, 0) >> (l+1)  * 2
  The `max(..., 0)` is exactly the activation guard: q < a[l] => hi = 0, and
  q < a[l+1] => lo = 0, because the added span is strictly less than the
  deficit only once q has passed the activation time.

  Coarsest level L, from range_spec.range_bounds:
      remaining = cache_size - sum_{l<L} width_l
      count     = min(remaining, newest + 1)
      lo, hi    = newest - count + 1, newest + 1
  becomes
      hi = max(q - a[L] + span_L, 0) >> L        (== newest + 1)
      lo = max(hi - remaining, 0)
  which handles remaining <= 0 without a branch: lo = hi - remaining > hi,
  an empty pair, and range_spec's ValueError case (fine levels overrunning
  the cache) is a schedule bug the host validates once, not a kernel concern.
"""

from typing import Callable, NamedTuple, Sequence, Tuple

__all__ = [
    "IntOps",
    "PY_OPS",
    "level_row_bounds",
    "level_row_bounds_one",
    "block_hull",
    "normalize_empty",
    "level_lens",
    "level_offsets",
    "node_query_bounds",
    "tile_query_hull",
    "coarsest_lifetime_span",
]


class IntOps(NamedTuple):
    """The only backend-dependent operations. Everything else is +, -, *, >>."""

    minimum: Callable
    maximum: Callable


PY_OPS = IntOps(minimum=min, maximum=max)


def level_row_bounds(
    q,
    activation_times: Sequence[int],
    cache_size: int,
    ops: IntOps = PY_OPS,
) -> Tuple[Tuple, ...]:
    """
    All levels' level-local [lo, hi) for one query row, in one pass.

    q may be a Python int, a torch tensor, or a CuTeDSL Int32; the level
    count and activation times are compile-time constants, so the loop below
    is fully unrolled by the DSL.

    The coarsest level needs the total width of the fine levels, so
    computing every level together is strictly cheaper than L+1 independent
    calls -- and the kernel wants all of them anyway, once per query row per
    m-block, hoisted out of the n-block loop.
    """
    a = tuple(activation_times)
    L = len(a) - 1
    mx = ops.maximum

    bounds = []
    used = 0
    for level in range(L):
        span = 1 << level
        hi = mx(q - a[level] + span, 0) >> level
        lo = (mx(q - a[level + 1] + 2 * span, 0) >> (level + 1)) * 2
        bounds.append((lo, hi))
        # Empty pairs have lo > hi; max(.., 0) keeps them from crediting the
        # cache with negative occupancy.
        used = used + mx(hi - lo, 0)

    span_L = 1 << L
    hi_L = mx(q - a[L] + span_L, 0) >> L
    lo_L = mx(hi_L - (cache_size - used), 0)
    bounds.append((lo_L, hi_L))
    return tuple(bounds)


def level_row_bounds_one(
    q,
    level: int,
    activation_times: Sequence[int],
    cache_size: int,
    ops: IntOps = PY_OPS,
) -> Tuple:
    """Single level, for tests and for readability at call sites."""
    return level_row_bounds(q, activation_times, cache_size, ops)[level]


def block_hull(
    q_lo: int,
    q_hi: int,
    level: int,
    activation_times: Sequence[int],
    cache_size: int,
    ops: IntOps = PY_OPS,
):
    """
    O(1) replacement for `range_spec.fwd_bounds`, which scans the block.

    Both lo(q) and hi(q) are non-decreasing in q (range_spec asserts this;
    test_range checks it), so

        hull = [ lo(q_lo), hi(q_hi - 1) )

    contains R_l(q) for every q in the block: lo(q) >= lo(q_lo) and
    hi(q) <= hi(q_hi-1). It is a valid hull whether or not individual rows
    are empty, and it collapses to an empty pair exactly when every row in
    the block is empty -- if any row were nonempty we would have
    lo(q_lo) <= lo(q) < hi(q) <= hi(q_hi-1).

    Returns a possibly-empty (lo >= hi) pair; the caller skips the level when
    lo >= hi.
    """
    lo, _ = level_row_bounds_one(q_lo, level, activation_times, cache_size, ops)
    _, hi = level_row_bounds_one(q_hi - 1, level, activation_times, cache_size, ops)
    return lo, hi


def normalize_empty(bounds: Tuple[int, int]) -> Tuple[int, int]:
    """This module's empty convention (lo >= hi) -> range_spec's (0, 0)."""
    lo, hi = bounds
    return (lo, hi) if lo < hi else (0, 0)


def level_lens(seq_len: int, num_levels: int) -> Tuple[int, ...]:
    """Host-side geometry; mirrors RangeSpec.level_len."""
    return tuple(seq_len >> level for level in range(num_levels))


def level_offsets(seq_len: int, num_levels: int) -> Tuple[int, ...]:
    """Host-side geometry; mirrors RangeSpec.level_offsets (len = L + 2)."""
    offsets = [0]
    for n in level_lens(seq_len, num_levels):
        offsets.append(offsets[-1] + n)
    return tuple(offsets)


# ---------------------------------------------------------------------------
# Backward: KV node -> query interval.
#
# `range_spec.node_query_bounds` is closed form at the fine levels and a
# BINARY SEARCH at the coarsest level -- it looks for the first q at which
# eviction has pushed lo_L(q) past the node. Neither the search nor its O(log
# N) trip count belongs in a kernel.
#
# The fine levels transcribe directly. The coarsest level does not: inverting
#
#     lo_L(q) = max( hi_L(q) - (cache_size - sum_{l<L} width_l(q)), 0 )
#
# for the smallest q with lo_L(q) > k means inverting a function that is
# linear in q with slope 2^-L plus a bounded, schedule-periodic wobble from
# the fine levels' occupancy. There is no tidy closed form.
#
# But `bwd_bounds` is only required to CONTAIN Q_l(k) -- exactness is a
# performance property, and the element mask is what makes the result exact
# (range_spec says as much, and notes bwd_bounds is already conservative at
# fine levels for a different reason). So the coarsest level uses
#
#     q_hi(k) = min(q_lo(k) + coarsest_span, N)
#
# with `coarsest_span` the longest lifetime any coarsest node actually has,
# measured once on the host by `coarsest_lifetime_span()`. That is the
# tightest constant that is valid for every node, and it costs one add.
# ---------------------------------------------------------------------------


def node_query_bounds(
    k,
    level: int,
    activation_times: Sequence[int],
    coarsest_span: int,
    seq_len: int,
    ops: IntOps = PY_OPS,
):
    """
    Branch-free hull of Q_l(k) = { q : k in R_l(q) }, in the same
    lo >= hi means empty convention as `level_row_bounds`.

    Fine level l < L, straight from range_spec.node_query_bounds:
        created at  a[l] + 2^l k
        removed when its pair is promoted, at a[l+1] + 2^(l+1) (k // 2)
    Coarsest level L: created at a[L] + 2^L k, removed no later than
    `coarsest_span` queries afterwards.

    Both ends are clamped to [0, N): a node created at or after N is never
    attended, and q_lo >= N >= q_hi then reports empty on its own.
    """
    a = tuple(activation_times)
    L = len(a) - 1
    mn = ops.minimum

    q_lo = a[level] + (k << level)
    if level < L:
        q_hi = a[level + 1] + ((k >> 1) << (level + 1))
    else:
        q_hi = q_lo + coarsest_span
    return q_lo, mn(q_hi, seq_len)


def tile_query_hull(
    k_lo: int,
    k_hi: int,
    level: int,
    activation_times: Sequence[int],
    coarsest_span: int,
    seq_len: int,
    ops: IntOps = PY_OPS,
):
    """
    O(1) replacement for `range_spec.bwd_bounds`, which scans the tile.

    q_lo(k) and q_hi(k) are both non-decreasing in k, so the hull over a KV
    tile is [q_lo(k_lo), q_hi(k_hi - 1)) -- the mirror of `block_hull`.
    """
    lo, _ = node_query_bounds(
        k_lo, level, activation_times, coarsest_span, seq_len, ops
    )
    _, hi = node_query_bounds(
        k_hi - 1, level, activation_times, coarsest_span, seq_len, ops
    )
    return lo, hi


def coarsest_lifetime_span(spec) -> int:
    """
    Host-side: the longest lifetime of any coarsest-level node, i.e.

        max_k ( q_hi(k) - q_lo(k) )   over nodes that are ever attended

    using `range_spec.node_query_bounds` as the oracle. Used as the single
    constant the kernel adds to a node's creation time. Nodes clipped by the
    end of the sequence are skipped -- their lifetime is truncated by N, not
    by eviction, and including them would understate the span.
    """
    from telescope_cache.range_spec import node_query_bounds as spec_bounds

    L = spec.coarsest
    span = 0
    for k in range(spec.level_len(L)):
        q_lo, q_hi = spec_bounds(spec, k, L)
        if q_lo == q_hi or q_hi >= spec.seq_len:
            continue
        span = max(span, q_hi - q_lo)
    if span == 0:
        # Every coarsest node runs to the end of the sequence (short N, or a
        # cache large enough that nothing is ever evicted): N bounds it.
        span = spec.seq_len
    return span
