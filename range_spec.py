"""
Pure-integer range specification for telescoping (multiresolution) attention.

This module is the mathematical contract a custom kernel implements. It has no
torch dependency and no Python-object cleverness: every function maps small
integers to small integers with half-open intervals throughout.

Setting
-------
Tokens 0..N-1 are summarized into a canonical dyadic tree. Level l holds
level_len(l) = floor-halving of N nodes; node k at level l summarizes tokens
[k * 2^l, (k+1) * 2^l). A schedule (`fmap`) induces activation times a[l]:
canonical node j at level l first becomes attendable by query q = a[l] + 2^l j.

For each query q and level l the attended nodes form ONE half-open interval

    R_l(q) = [k_lo, k_hi)                       (range_bounds)

and the inverse set of a node

    Q_l(k) = { q : k in R_l(q) }                (node_query_bounds gives its hull)

is also one interval, because under the ruler/slot policy a node is only ever
removed (merged upward or evicted), never re-added.

Kernel-facing functions
-----------------------
    range_bounds(spec, q, level)          -> (k_lo, k_hi)   exact per-query
    elem_mask(spec, q, k, level)          -> bool            k in R_l(q)
    fwd_bounds(spec, q_lo, q_hi, level)   -> hull( U_{q in [q_lo,q_hi)} R_l(q) )
    node_query_bounds(spec, k, level)     -> hull( Q_l(k) )
    bwd_bounds(spec, k_lo, k_hi, level)   -> hull( U_{k in [k_lo,k_hi)} Q_l(k) )

`fwd_bounds` / `bwd_bounds` are HULLS, not necessarily unions. The kernel may
load a conservative rectangular tile range and must then apply `elem_mask`
(vectorized: `(k >= row_k_lo) & (k < row_k_hi)`) for exact semantics, exactly
as sliding-window kernels enumerate blocks coarsely and mask elements. The
required properties are containment only:

    R_l(q) subset-of fwd_bounds(q_lo, q_hi, l)   for every q in the block
    Q_l(k) subset-of bwd_bounds(k_lo, k_hi, l)   for every k in the tile

Exactness of the hulls is a performance property that the tests measure.
Measured result (test/test_reference.py): every single node's inverse set Q_l(k)
is contiguous, but the union over a KV window can have holes at fine levels,
because at a merge step a level momentarily holds no attendable node (e.g.
the README example, N=8, cache 6, fmap {1:2, 2:3}: L1 is [0,2) at q=5, empty
at q=6, [2,3) at q=7). So `bwd_bounds` is exact at the coarsest level and
conservative by a few queries at fine levels; the backward MUST keep the
element mask. `fwd_bounds` was exact on every tested window.

Packed level-major storage
--------------------------
Storage holds all levels in ONE buffer per tensor, level-major:

    [ L0 | L1 | ... | LL ]      level_offsets[l] = sum_{j<l} level_len(j)

so a level-local range [k_lo, k_hi) addresses physical entries
[offset[l] + k_lo, offset[l] + k_hi) in a single address space. The
coordinate rule is strict:

    semantic geometry  = level-local coordinates
                         (range_bounds, elem_mask, fwd_bounds, bwd_bounds)
    storage addressing = offset[level] + local, only when touching the buffer

Nothing in this module ever consumes a packed index; `packed_bounds` is a
convenience at that boundary. The physical ordering of the packed buffer
([B, sumN, H, D] vs [B, H, sumN, D]) is a storage decision outside this spec.

Conventions
-----------
* Empty interval is always (0, 0).
* Nodes that exist in the buffer (k < level_len) but are never attended
  because a[l] + 2^l k >= N have Q_l(k) = {} and node_query_bounds = (0, 0).
* All functions raise ValueError on out-of-domain input instead of clamping.
  A physical BLOCK_N tile may contain padded k >= level_len(level); the
  kernel predicates those lanes physically BEFORE applying the semantic
  `elem_mask`. Physical bounds safety and attention semantics stay separate.

Domain contracts
----------------
    range_bounds:       0 <= q < N;  0 <= level <= L
    elem_mask:          0 <= q < N;  0 <= level <= L;  0 <= k < level_len(level)
    fwd_bounds:         0 <= q_lo <= q_hi <= N;  q_lo == q_hi -> (0, 0)
    node_query_bounds:  0 <= k < level_len(level);  never visible -> (0, 0)
    bwd_bounds:         0 <= k_lo <= k_hi <= level_len(level)
                        k_lo == k_hi -> (0, 0);  no visible node -> (0, 0)
"""

from dataclasses import dataclass
from typing import Dict, Tuple

EMPTY: Tuple[int, int] = (0, 0)


def activation_times_from_fmap(fmap: Dict[int, int]) -> Tuple[int, ...]:
    """
    a[l] = first query at which canonical node 0 of level l becomes visible.

        a[0] = 0
        a[1] = fmap[1] + 1
        a[l] = a[l-1] + 2^(l-1) * (fmap[l] - fmap[l-1]) + 2^(l-2)   (l >= 2)

    Also enforces dyadic alignment, a[l] mod 2^l == 2^(l-1), without which the
    schedule is not representable by the canonical aligned tree.

    Deliberately duplicates test_reference.compute_dyadic_activation_times /
    validate_dyadic_fmap (the oracle) in pure ints so this module has no test
    dependency; test_reference.test_schedule_properties asserts the two agree
    on every configuration.
    """
    if not fmap:
        return (0,)
    L = max(fmap)
    if sorted(fmap) != list(range(1, L + 1)):
        raise ValueError(
            f"fmap keys must be consecutive 1..L; got {sorted(fmap)}"
        )
    a = [0] * (L + 1)
    a[1] = fmap[1] + 1
    for level in range(2, L + 1):
        a[level] = (
            a[level - 1]
            + (1 << (level - 1)) * (fmap[level] - fmap[level - 1])
            + (1 << (level - 2))
        )
    for level in range(1, L + 1):
        if a[level] % (1 << level) != (1 << (level - 1)):
            raise ValueError(
                f"fmap not dyadically aligned at level {level}: "
                f"a[{level}]={a[level]} has phase {a[level] % (1 << level)} "
                f"mod {1 << level}, expected {1 << (level - 1)}."
            )
    return tuple(a)


@dataclass(frozen=True)
class RangeSpec:
    """Schedule parameters: activation times a[0..L], cache size C, seq len N."""

    activation_times: Tuple[int, ...]
    cache_size: int
    seq_len: int

    @classmethod
    def from_fmap(
        cls, fmap: Dict[int, int], cache_size: int, seq_len: int
    ) -> "RangeSpec":
        return cls(activation_times_from_fmap(fmap), cache_size, seq_len)

    def __post_init__(self):
        if self.cache_size <= 0 or self.seq_len <= 0:
            raise ValueError("cache_size and seq_len must be positive")
        if len(self.activation_times) < 1 or self.activation_times[0] != 0:
            raise ValueError("activation_times must start with a[0] = 0")

    @property
    def num_levels(self) -> int:
        return len(self.activation_times)

    @property
    def coarsest(self) -> int:
        return len(self.activation_times) - 1

    def level_len(self, level: int) -> int:
        """Number of nodes stored at `level`: N, N//2, N//4, ... (floor)."""
        self._check_level(level)
        n = self.seq_len
        for _ in range(level):
            n //= 2
        return n

    def level_offsets(self) -> Tuple[int, ...]:
        """
        Prefix sums of level_len: offsets[l] is where level l starts in the
        packed level-major buffer; len == num_levels + 1; offsets[-1] ==
        total_len.
        """
        offsets = [0]
        for level in range(self.num_levels):
            offsets.append(offsets[-1] + self.level_len(level))
        return tuple(offsets)

    @property
    def total_len(self) -> int:
        """Total number of packed entries, sum_l level_len(l)."""
        return self.level_offsets()[-1]

    # -- domain checks -----------------------------------------------------

    def _check_level(self, level: int) -> None:
        if not 0 <= level <= self.coarsest:
            raise ValueError(
                f"level {level} out of range [0, {self.coarsest}]"
            )

    def _check_q(self, q: int) -> None:
        if not 0 <= q < self.seq_len:
            raise ValueError(f"q={q} out of range [0, {self.seq_len})")

    def _check_k(self, k: int, level: int) -> None:
        n = self.level_len(level)
        if not 0 <= k < n:
            raise ValueError(
                f"k={k} out of range [0, {n}) at level {level}"
            )


# ---------------------------------------------------------------------------
# Forward: query -> KV interval
# ---------------------------------------------------------------------------

def _fine_bounds(spec: RangeSpec, q: int, level: int) -> Tuple[int, int]:
    """R_l(q) for a non-coarsest level l < L. No domain checks (internal)."""
    a = spec.activation_times
    span = 1 << level
    if q < a[level]:
        return EMPTY
    newest = (q - a[level]) // span
    next_a = a[level + 1]
    if q < next_a:
        oldest = 0
    else:
        # Every completed parent removes its two children from this level.
        oldest = 2 * ((q - next_a) // (2 * span) + 1)
    if oldest > newest:
        return EMPTY
    return (oldest, newest + 1)


def range_bounds(spec: RangeSpec, q: int, level: int) -> Tuple[int, int]:
    """
    Exact per-query interval R_l(q) = [k_lo, k_hi) of attended nodes.

    Fine levels l < L: nodes from the oldest not-yet-promoted node up to the
    newest complete node. Coarsest level L: the newest nodes that fit in the
    cache slots left over by the fine levels (oldest are evicted).
    """
    spec._check_q(q)
    spec._check_level(level)
    L = spec.coarsest
    if level < L:
        lo, hi = _fine_bounds(spec, q, level)
    else:
        used = 0
        for fine in range(L):
            flo, fhi = _fine_bounds(spec, q, fine)
            used += fhi - flo
        remaining = spec.cache_size - used
        if remaining < 0:
            raise ValueError(
                f"fine levels need {used} slots > cache_size="
                f"{spec.cache_size} at q={q}"
            )
        a = spec.activation_times[L]
        span = 1 << L
        if q < a or remaining == 0:
            return EMPTY
        newest = (q - a) // span
        count = min(remaining, newest + 1)
        lo, hi = newest - count + 1, newest + 1
    # Spec invariant: attended nodes always physically exist.
    if hi > spec.level_len(level):
        raise AssertionError(
            f"range [{lo},{hi}) exceeds level_len={spec.level_len(level)} "
            f"at q={q}, level={level}"
        )
    return (lo, hi)


def packed_bounds(spec: RangeSpec, q: int, level: int) -> Tuple[int, int]:
    """
    range_bounds translated into packed level-major coordinates:

        [k_lo, k_hi)  ->  [offset[level] + k_lo, offset[level] + k_hi)

    (0, 0) stays (0, 0) for an empty range; a nonempty physical range never
    equals (0, 0) because it has positive length.

    This is the ONLY function here that speaks packed coordinates, and it is
    a convenience for storage addressing. range_bounds, elem_mask, fwd_bounds
    and bwd_bounds all remain level-local; a backward implementation keeps
    calling bwd_bounds with level-local K indices and adds the offset only
    when it touches the packed buffer.
    """
    lo, hi = range_bounds(spec, q, level)
    if lo == hi:
        return EMPTY
    base = spec.level_offsets()[level]
    return (base + lo, base + hi)


def elem_mask(spec: RangeSpec, q: int, k_local: int, level: int) -> bool:
    """k_local in R_l(q). The kernel realizes this vectorized per tile."""
    spec._check_k(k_local, level)
    lo, hi = range_bounds(spec, q, level)
    return lo <= k_local < hi


def fwd_bounds(
    spec: RangeSpec, q_lo: int, q_hi: int, level: int
) -> Tuple[int, int]:
    """
    Hull of U_{q in [q_lo, q_hi)} R_l(q): the KV tile range a query block
    streams at `level`. (0, 0) if every row is empty.

    Implemented by scanning the block's rows (<= BLOCK_M). Because k_lo(q) and
    k_hi(q) are non-decreasing in q (asserted by the tests), a kernel may
    compute the same hull in O(1) from the first and last nonempty rows.
    """
    if not 0 <= q_lo <= q_hi <= spec.seq_len:
        raise ValueError(
            f"[q_lo, q_hi)=[{q_lo},{q_hi}) not within [0, {spec.seq_len}]"
        )
    spec._check_level(level)
    lo, hi = None, None
    for q in range(q_lo, q_hi):
        rlo, rhi = range_bounds(spec, q, level)
        if rlo == rhi:
            continue
        lo = rlo if lo is None else min(lo, rlo)
        hi = rhi if hi is None else max(hi, rhi)
    if lo is None:
        return EMPTY
    return (lo, hi)


# ---------------------------------------------------------------------------
# Backward: KV node -> query interval
# ---------------------------------------------------------------------------

def node_query_bounds(spec: RangeSpec, k: int, level: int) -> Tuple[int, int]:
    """
    Hull of Q_l(k) = { q : k in R_l(q) } for node k at `level`; (0, 0) if the
    node is never attended (created at or after N).

    Fine level l < L (closed form): the node is created at a[l] + 2^l k and
    removed when its pair is promoted, i.e. at the first q with
    oldest(q) > k  <=>  (q - a[l+1]) // 2^(l+1) >= k // 2:

        [ a[l] + 2^l k,  a[l+1] + 2^(l+1) (k // 2) )  intersected with [0, N)

    Coarsest level L: created at a[L] + 2^L k, removed by eviction at the first
    q with k_lo(q) > k. k_lo(q) is non-decreasing (asserted by the tests), so
    the removal time is found by binary search over q. A closed form for the
    eviction time is a later optimization.
    """
    spec._check_k(k, level)
    a = spec.activation_times
    N = spec.seq_len
    span = 1 << level
    q_lo = a[level] + span * k
    if q_lo >= N:
        return EMPTY
    if level < spec.coarsest:
        q_hi = a[level + 1] + 2 * span * (k // 2)
        q_hi = min(q_hi, N)
        if q_hi <= q_lo:
            raise AssertionError(
                f"fine-level inverse empty for a created node: "
                f"k={k}, level={level}, [{q_lo},{q_hi})"
            )
        return (q_lo, q_hi)

    # Coarsest level. visible(q) is True at creation and stays True until the
    # node is evicted, after which it never returns (monotone k_lo).
    def visible(q: int) -> bool:
        lo, hi = range_bounds(spec, q, level)
        return lo <= k < hi

    if not visible(q_lo):
        # Cache has no coarsest-level capacity at creation time; the node is
        # evicted immediately and never attended.
        return EMPTY
    # Binary search the first q in (q_lo, N] with visible(q) False (N counts
    # as not visible). Invariant: visible(lo_q) True, visible(hi_q) False.
    lo_q, hi_q = q_lo, N
    while hi_q - lo_q > 1:
        mid = (lo_q + hi_q) // 2
        if visible(mid):
            lo_q = mid
        else:
            hi_q = mid
    return (q_lo, hi_q)


def bwd_bounds(
    spec: RangeSpec, k_lo: int, k_hi: int, level: int
) -> Tuple[int, int]:
    """
    Hull of U_{k in [k_lo, k_hi)} Q_l(k): the query range a KV tile must be
    scattered back to in the backward. (0, 0) if k_lo == k_hi or no node in
    the tile is ever attended.

    Implemented by scanning the tile's nodes (<= BLOCK_N) and taking the hull
    over the NONEMPTY node_query_bounds. When q_lo(k) and q_hi(k) are
    non-decreasing over visible nodes (asserted by the tests) a kernel may use
    the first and last visible nodes only.
    """
    n = spec.level_len(level)
    if not 0 <= k_lo <= k_hi <= n:
        raise ValueError(
            f"[k_lo, k_hi)=[{k_lo},{k_hi}) not within [0, {n}] at level {level}"
        )
    lo, hi = None, None
    for k in range(k_lo, k_hi):
        qlo, qhi = node_query_bounds(spec, k, level)
        if qlo == qhi:
            continue
        lo = qlo if lo is None else min(lo, qlo)
        hi = qhi if hi is None else max(hi, qhi)
    if lo is None:
        return EMPTY
    return (lo, hi)


__all__ = [
    "EMPTY",
    "RangeSpec",
    "activation_times_from_fmap",
    "range_bounds",
    "packed_bounds",
    "elem_mask",
    "fwd_bounds",
    "node_query_bounds",
    "bwd_bounds",
]
