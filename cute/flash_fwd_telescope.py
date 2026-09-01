"""
Telescoping multiresolution attention, SM80 CuTeDSL forward.

Derived from `flash_attn.cute.flash_fwd.FlashAttentionForwardSm80` (FA4).
Everything below the mainloop -- Q/K/V loads, the two GEMMs, the online
softmax, the epilogue -- is inherited unchanged. Only the *block schedule*
and the *mask* are new, which is the whole claim of the design: telescoping
attention is sliding-window attention run once per hierarchy level, sharing
one online-softmax accumulator.

Why sliding window is the right template
----------------------------------------
FA4's local path does exactly two things beyond dense attention:

    BlockInfo.get_n_block_min_max   -> which n_blocks a query block touches
    AttentionMask.apply_mask        -> per-row [col_limit_left, col_limit_right)

For telescoping attention the same two hooks carry the whole schedule:

    ranges.block_hull(q_lo, q_hi, l) -> the level's n_block range, O(1)
    ranges.level_row_bounds(q)[l]    -> the level's per-row [lo, hi)

The difference is that a query attends to L+1 windows, one per level, at
different offsets in the packed buffer -- not one.

The virtual block index
-----------------------
The inherited pipeline (`compute_one_n_block` + `load_K`/`load_V`) is built
around a *contiguous descending* block sequence: it prefetches `n_block - 1`
and `n_block - num_stages` and guards on `>= 0`. The telescope schedule is a
union of L+1 disjoint block ranges, so those neighbours are wrong at every
level boundary.

Rather than rewrite the pipeline, this kernel introduces a **virtual block
index** v in [0, n_virtual) that enumerates the selected blocks contiguously,
level by level, and translates v -> (packed block, level) at the three points
that touch memory or indices. The pipeline sees a dense descending range and
is reused verbatim; the translation is a handful of integer selects, done
twice per block rather than per element.

    level l contributes cnt[l] blocks starting at packed block pbase[l]
    cum[l]  = cnt[0] + ... + cnt[l-1]
    v       -> pbase[l] + (v - cum[l])   for the l with cum[l] <= v < cum[l+1]

The select chain runs low level to high and keeps the last match; because
cum is non-decreasing that is the correct level, with no branching.

This is also where the padded packing earns its keep
(`packing.pack_levels_aligned`): pbase[l] is an integer block index only
because level l starts at a multiple of tile_n.
"""

from functools import partial
from types import SimpleNamespace
from typing import Callable, Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from quack import layout_utils

from flash_attn.cute import utils
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.softmax import Softmax
from flash_attn.cute.utils import AuxData

from telescope_cache.cute.ranges import IntOps, level_row_bounds

__all__ = ["TelescopeAttentionForwardSm80", "CUTE_OPS"]

CUTE_OPS = IntOps(minimum=cutlass.min, maximum=cutlass.max)


class TelescopeAttentionForwardSm80(FlashAttentionForwardSm80):
    """
    FA4's SM80 forward with a telescoping block schedule.

    The schedule is baked in as compile-time constants: a kernel is
    specialized to one (activation_times, cache_size, pad_offsets) triple,
    which is what makes every level bound a shift-and-add rather than a
    table lookup. Callers cache instances per schedule.
    """

    def __init__(
        self,
        *args,
        activation_times: Tuple[int, ...],
        cache_size: int,
        pad_offsets: Tuple[int, ...],
        **kwargs,
    ):
        super().__init__(*args, is_causal=False, is_local=False, **kwargs)
        self.activation_times = tuple(activation_times)
        self.cache_size = int(cache_size)
        self.pad_offsets = tuple(pad_offsets)
        self.num_levels = len(self.activation_times)
        if len(self.pad_offsets) != self.num_levels + 1:
            raise ValueError(
                f"pad_offsets has {len(self.pad_offsets)} entries, expected "
                f"{self.num_levels + 1} for {self.num_levels} levels"
            )
        if any(off % self.tile_n for off in self.pad_offsets):
            raise ValueError(
                f"pad_offsets {self.pad_offsets} must all be multiples of "
                f"tile_n={self.tile_n}; use packing.pack_levels_aligned"
            )
        if self.pack_gqa:
            raise NotImplementedError(
                "pack_gqa changes the row -> q_idx map; not wired up yet"
            )

    # -- schedule -------------------------------------------------------

    @cute.jit
    def _virtual_block_map(self, m_block: Int32, seqlen_q: Int32):
        """
        Per m-block: how many blocks each level contributes and where they
        start, as the (cum, pbase, n_virtual) triple the translation needs.

        The hull is `ranges.block_hull` inlined: lo of the block's first row,
        hi of its last row. Both bounds are monotone in q, so this contains
        every row's range, and it collapses to lo >= hi exactly when every
        row of the block is empty at this level.

        The last row is clamped to seqlen_q - 1. An m-block may overhang the
        sequence, and `range_spec`'s guarantee hi_l(q) <= level_len(l) only
        holds for q < N -- without the clamp an overhanging block could
        derive a block range running past the level's slab.
        """
        q_lo = m_block * self.tile_m
        q_hi = cutlass.min(q_lo + self.tile_m - 1, seqlen_q - 1)

        lo_first = level_row_bounds(q_lo, self.activation_times, self.cache_size, CUTE_OPS)
        hi_last = level_row_bounds(q_hi, self.activation_times, self.cache_size, CUTE_OPS)

        cum, pbase = [], []
        total = Int32(0)
        for level in cutlass.range_constexpr(self.num_levels):
            lo = lo_first[level][0]
            hi = hi_last[level][1]
            first = lo // self.tile_n
            last = cute.ceil_div(hi, self.tile_n)
            # lo >= hi means the level is empty for the whole block. Guard
            # explicitly: lo == hi > 0 would otherwise yield one block whose
            # rows are all masked -- correct, but a wasted tile.
            count = (last - first) if lo < hi else Int32(0)
            cum.append(total)
            pbase.append(self.pad_offsets[level] // self.tile_n + first)
            total = total + count
        return tuple(cum), tuple(pbase), total

    @cute.jit
    def _translate(self, v: Int32, cum, pbase):
        """virtual block index -> (packed block index, level). Branch-free."""
        packed_block = pbase[0] + v
        level = Int32(0)
        for l in cutlass.range_constexpr(1, self.num_levels):
            take = v >= cum[l]
            packed_block = (pbase[l] + (v - cum[l])) if take else packed_block
            level = Int32(l) if take else level
        return packed_block, level

    # -- virtual-index K/V loads ---------------------------------------
    #
    # The inherited load_K/load_V take a real block index. These translate
    # first, so `compute_one_n_block` -- which prefetches `n_block - 1` and
    # `n_block - num_stages` -- keeps working on the virtual index unchanged.
    # They are methods rather than closures because the DSL will not let a
    # closure capture a partial across staged control flow.

    @cute.jit
    def load_K_virtual(
        self, gmem_tiled_copy, tKgK, tKsK, tKcK, t0KcK, tKpK, cum, pbase,
        block: Int32, smem_pipe_write: Int32, seqlen: Int32,
        need_predicates: cutlass.Constexpr,
    ):
        packed_block, _ = self._translate(block, cum, pbase)
        self.load_K(
            gmem_tiled_copy, tKgK, tKsK, tKcK, t0KcK, tKpK,
            packed_block, smem_pipe_write, seqlen, need_predicates,
        )

    @cute.jit
    def load_V_virtual(
        self, gmem_tiled_copy, tVgV, tVsV, tVcV, t0VcV, tVpV, cum, pbase,
        block: Int32, smem_pipe_write: Int32, seqlen: Int32,
        need_predicates: cutlass.Constexpr,
    ):
        packed_block, _ = self._translate(block, cum, pbase)
        self.load_V(
            gmem_tiled_copy, tVgV, tVsV, tVcV, t0VcV, tVpV,
            packed_block, smem_pipe_write, seqlen, need_predicates,
        )

    # -- mask -----------------------------------------------------------

    @cute.jit
    def apply_telescope_mask(
        self,
        acc_S: cute.Tensor,
        n_block: Int32,
        thr_mma: cute.TiledMma,
        m_block: Int32,
        seqlen_q: Int32,
        cum,
        pbase,
    ):
        """
        Per-row [lo, hi) masking, structurally identical to FA4's local mask.

        FA4's local branch computes one `col_limit_left`/`col_limit_right`
        per row from an affine function of row_idx. Here the pair comes from
        `level_row_bounds` at the block's level instead -- still one interval
        per row, so the inner column loop is the same shape and the same cost.

        Two things stay out of the column loop, which is the hot part:
          * the level select, done once per row (the level is dynamic, so all
            L+1 bounds are computed and selected between);
          * the conversion to thread-local column coordinates, so the loop
            compares against `t0ScS` indices that are compile-time constants
            -- the same trick FA4 uses.
        """
        packed_block, level = self._translate(n_block, cum, pbase)

        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS))
        t0ScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cS))

        # Packed column index of this thread's column 0 within the block.
        col_base = packed_block * self.tile_n + tScS_mn[0][1]

        for r in cutlass.range_constexpr(cute.size(tScS_mn.shape[0])):
            q_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
            # Rows past the end of the sequence are dropped by the epilogue,
            # but their bounds still index this buffer, so clamp them.
            q_eff = cutlass.min(q_idx, seqlen_q - 1)
            bounds = level_row_bounds(
                q_eff, self.activation_times, self.cache_size, CUTE_OPS
            )
            lo = self.pad_offsets[0] + bounds[0][0]
            hi = self.pad_offsets[0] + bounds[0][1]
            for l in cutlass.range_constexpr(1, self.num_levels):
                take = level == l
                lo = (self.pad_offsets[l] + bounds[l][0]) if take else lo
                hi = (self.pad_offsets[l] + bounds[l][1]) if take else hi
            limit_lo = lo - col_base
            limit_hi = hi - col_base
            for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                col = t0ScS_mn[0, c][1]
                if col < limit_lo or col >= limit_hi:
                    acc_S_mn[r, c] = -Float32.inf

    # -- kernel ---------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params,
        TileScheduler: cutlass.Constexpr[Callable],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
    ):
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, num_head, batch_size, _ = work_tile.tile_idx

        seqlen = SeqlenInfoQK.create(
            batch_idx=batch_size,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=None,
            mCuSeqlensK=None,
            mSeqUsedQ=None,
            mSeqUsedK=None,
        )

        # The telescope schedule replaces BlockInfo.get_n_block_min_max.
        cum, pbase, n_virtual = self._virtual_block_map(m_block, seqlen.seqlen_q)
        n_block = cutlass.max(n_virtual - 1, 0)

        # ---------------------------------------------------------------
        # Tiles. Identical to the base kernel except that mK/mV are the
        # block-aligned packed buffers, so gK/gV are indexed by *packed*
        # block and every block is fully in bounds.
        # ---------------------------------------------------------------
        blkQ_shape = (self.tile_m, self.tile_hdim)
        blkK_shape = (self.tile_n, self.tile_hdim)
        blkV_shape = (self.tile_n, self.tile_hdimv)
        num_head_kv = num_head // self.qhead_per_kvhead
        mQ_cur = mQ[None, None, num_head, batch_size]
        mK_cur = mK[None, None, num_head_kv, batch_size]
        mV_cur = mV[None, None, num_head_kv, batch_size]
        gQ = cute.local_tile(mQ_cur, blkQ_shape, (m_block, 0))
        gK = cute.local_tile(mK_cur, blkK_shape, (None, 0))
        gV = cute.local_tile(mV_cur, blkV_shape, (None, 0))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout)
        else:
            sV = cute.make_tensor(cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout)
        sVt = layout_utils.transpose_view(sV)

        gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
        gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
        tKsK, tKgK = gmem_thr_copy_K.partition_D(sK), gmem_thr_copy_K.partition_S(gK)
        tVsV, tVgV = gmem_thr_copy_V.partition_D(sV), gmem_thr_copy_V.partition_S(gV)

        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
        acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
        acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
        acc_O.fill(0.0)

        from cutlass.cute.nvgpu import warp

        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )
        smem_thr_copy_Q = utils.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_K = utils.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_V = utils.make_tiled_copy_B(smem_copy_atom_V, tiled_mma_pv).get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)

        cK = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
        tKcK = gmem_thr_copy_K.partition_S(cK)
        t0KcK = gmem_thr_copy_K.get_slice(0).partition_S(cK)
        if const_expr(self.tile_hdim == self.tile_hdimv):
            tVcV, t0VcV = tKcK, t0KcK
        else:
            cV = cute.make_identity_tensor((self.tile_n, self.tile_hdimv))
            tVcV = gmem_thr_copy_V.partition_S(cV)
            t0VcV = gmem_thr_copy_V.get_slice(0).partition_S(cV)
        tKpK = utils.predicate_k(tKcK, limit=mK.shape[1])
        tVpV = tKpK if const_expr(self.same_hdim_kv) else utils.predicate_k(tVcV, limit=mV.shape[1])

        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )
        softmax.reset()

        mma_params = SimpleNamespace(
            thr_mma_qk=thr_mma_qk, thr_mma_pv=thr_mma_pv,
            tSrQ=tSrQ, tSrK=tSrK, tOrVt=tOrVt, acc_O=acc_O,
        )
        smem_copy_params = SimpleNamespace(
            smem_thr_copy_Q=smem_thr_copy_Q, smem_thr_copy_K=smem_thr_copy_K,
            smem_thr_copy_V=smem_thr_copy_V, tSsQ=tSsQ, tSsK=tSsK, tOsVt=tOsVt,
        )

        # The one place the virtual index meets memory. `seqlen` is the
        # padded buffer length and every selected block lies wholly inside
        # it, so these loads never need n-predication.
        load_K = partial(
            self.load_K_virtual, gmem_tiled_copy_K, tKgK, tKsK, tKcK, t0KcK,
            tKpK, cum, pbase, seqlen=seqlen.seqlen_k,
        )
        load_V = partial(
            self.load_V_virtual, gmem_tiled_copy_V, tVgV, tVsV, tVcV, t0VcV,
            tVpV, cum, pbase, seqlen=seqlen.seqlen_k,
        )

        mask_fn = partial(
            self.apply_telescope_mask,
            thr_mma=thr_mma_qk,
            m_block=m_block,
            seqlen_q=seqlen.seqlen_q,
            cum=cum,
            pbase=pbase,
        )

        compute_one_n_block = partial(
            self.compute_one_n_block,
            mma_params=mma_params,
            smem_copy_params=smem_copy_params,
            softmax=softmax,
            load_K=load_K,
            load_V=load_V,
            score_mod=self.score_mod,
            batch_idx=batch_size,
            head_idx=num_head,
            m_block=m_block,
            aux_data=aux_data,
            fastdiv_mods=fastdiv_mods,
        )

        # ---------------------------------------------------------------
        # Prologue
        # ---------------------------------------------------------------
        gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
        self.load_Q(
            gmem_thr_copy_Q, gQ, sQ, m_block,
            seqlen=seqlen.seqlen_q, headdim=mQ.shape[1],
        )
        cute.arch.cp_async_commit_group()

        def preprocess_Q():
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 1)
            if const_expr(self.Q_in_regs):
                cute.arch.barrier()
                tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
                cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)

        if const_expr(self.Q_in_regs):
            load_K(n_block, smem_pipe_write=0, need_predicates=True)
            cute.arch.cp_async_commit_group()
            preprocess_Q()
            cute.arch.barrier()

        for stage in cutlass.range_constexpr(self.num_stages):
            if const_expr(not self.Q_in_regs or stage > 0):
                if stage == 0 or n_block - stage >= 0:
                    load_K(n_block - stage, smem_pipe_write=stage, need_predicates=stage == 0)
                cute.arch.cp_async_commit_group()
            if const_expr(stage < self.num_stages - 1):
                if stage == 0 or n_block - stage >= 0:
                    load_V(n_block - stage, smem_pipe_write=stage, need_predicates=stage == 0)
                cute.arch.cp_async_commit_group()
        if const_expr(not self.Q_in_regs):
            preprocess_Q()

        # ---------------------------------------------------------------
        # Mainloop: v = n_virtual - 1 down to 0.
        #
        # Every block carries the mask. FA4 splits its loop into masked and
        # unmasked regions because a causal/local run has a long unmasked
        # interior; a telescope level contributes one or two blocks, of which
        # both ends are partial, so there is no interior to split off. The
        # rows-vs-columns comparison is the cheap part here anyway.
        # ---------------------------------------------------------------
        smem_pipe_read = Int32(0)
        smem_pipe_write = Int32(self.num_stages - 1)
        compute_one_n_block(
            n_block, smem_pipe_read, smem_pipe_write,
            is_first_n_block=True, seqlen=seqlen, mask_fn=mask_fn,
        )
        smem_pipe_read = self.advance_pipeline(smem_pipe_read)
        smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        for n_tile in cutlass.range(n_block, unroll=1):
            compute_one_n_block(
                n_block - n_tile - 1, smem_pipe_read, smem_pipe_write,
                is_first_n_block=False, seqlen=seqlen, mask_fn=mask_fn,
            )
            smem_pipe_read = self.advance_pipeline(smem_pipe_read)
            smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        row_scale = softmax.finalize()
        softmax.rescale_O(acc_O, row_scale)

        sO = cute.make_tensor(sQ.iterator, sO_layout)
        self.epilogue(
            acc_O, softmax.row_sum, mO, mLSE, sO, seqlen,
            gmem_tiled_copy_O, None, tiled_mma_pv, tidx,
            m_block, num_head, batch_size,
        )
