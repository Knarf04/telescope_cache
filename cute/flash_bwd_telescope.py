"""
Telescoping multiresolution attention, SM80 CuTeDSL backward.

Derived from `flash_attn.cute.flash_bwd.FlashAttentionBackwardSm80` the same
way the forward is derived from `FlashAttentionForwardSm80`: the loads, the
four GEMMs, the dQ/dK/dV accumulation and the epilogue are inherited
verbatim; only the block schedule and the mask are replaced. The two changed
regions in `kernel` are marked `TELESCOPE CHANGE 1/2 of 2`.

Why this one has to be forked rather than called
------------------------------------------------
FA4's public backward refuses Ampere outright --
`interface._flash_attn_bwd` asserts the compute capability is in
{9, 10, 11, 12}, and past that assert the arch-8 path raises
`UnboundLocalError: dQ_single_wg`, a variable only ever assigned in the SM90
and SM120 branches. That host path has never been executed.

The KERNEL, though, is fine: driven directly with a hand-written launcher it
reproduces a float32 torch reference for dq/dk/dv on sm86 to bf16 precision
(see `telescope_cache.cute.interface.telescope_attn_bwd`, which is that
launcher). So this file forks a working kernel and supplies the host plumbing
FA4 leaves unfinished on this architecture.

Structure, against reference.multilevel_attention_backward
----------------------------------------------------------
The reference splits the backward into two explicit passes -- Pass A walks
Q-blocks to build dQ, Pass B walks KV-tiles to build dK/dV. This kernel is
the fused FlashAttention shape: ONE pass owned by the KV tile, which
accumulates dK/dV in registers and scatters dQ into an fp32 `dq_accum`
buffer. Same arithmetic, same masks, one traversal instead of two.

Because of that, only the Pass-B direction of the range spec is needed:

    reference / range_spec          this kernel
    ----------------------          -----------
    bwd_bounds(k_lo, k_hi, l)       ranges.tile_query_hull, O(1)
    node_query_bounds(k, l)         ranges.node_query_bounds, branch-free
    elem_mask                       apply_telescope_mask_bwd

and unlike the forward there is no virtual block index, because the query
range one KV tile scatters back to is a single contiguous interval -- the
inherited m-loop already walks an arbitrary ascending range.

Level decode
------------
The forward derives the level per query block; here it is derived from
`n_block` alone. `pad_offsets` are compile-time constants, so

    level = the last l with n_block >= pad_offsets[l] // n_block_size

is a chain of comparisons against literals, evaluated once per CTA rather
than per block. That is the whole reason the packing is block-aligned.
"""

from types import SimpleNamespace
from typing import Callable, Optional, Tuple
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp
from cutlass import Int32, Float32

from quack import layout_utils
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute import utils
from flash_attn.cute.flash_bwd import FlashAttentionBackwardSm80
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.utils import AuxData

from telescope_cache.cute.ranges import (
    IntOps,
    level_row_bounds,
    node_query_bounds,
)

__all__ = ["TelescopeAttentionBackwardSm80"]

CUTE_OPS = IntOps(minimum=cutlass.min, maximum=cutlass.max)


class TelescopeAttentionBackwardSm80(FlashAttentionBackwardSm80):
    """
    FA4's SM80 backward with a telescoping block schedule.

    Specialized at compile time to one schedule, exactly as the forward is:
    `activation_times`, `cache_size`, `pad_offsets`, `level_lens` and
    `coarsest_span` are all baked in, which is what makes the level decode a
    chain of literal comparisons and every range bound a shift and an add.
    """

    def __init__(
        self,
        *args,
        activation_times: Tuple[int, ...],
        cache_size: int,
        pad_offsets: Tuple[int, ...],
        level_lens: Tuple[int, ...],
        coarsest_span: int,
        **kwargs,
    ):
        super().__init__(*args, is_causal=False, is_local=False, **kwargs)
        self.activation_times = tuple(activation_times)
        self.cache_size = int(cache_size)
        self.pad_offsets = tuple(pad_offsets)
        self.level_lens = tuple(level_lens)
        self.coarsest_span = int(coarsest_span)
        self.num_levels = len(self.activation_times)
        if len(self.pad_offsets) != self.num_levels + 1:
            raise ValueError(
                f"pad_offsets has {len(self.pad_offsets)} entries, expected "
                f"{self.num_levels + 1}"
            )
        if len(self.level_lens) != self.num_levels:
            raise ValueError(
                f"level_lens has {len(self.level_lens)} entries, expected "
                f"{self.num_levels}"
            )
        if any(off % self.n_block_size for off in self.pad_offsets):
            raise ValueError(
                f"pad_offsets {self.pad_offsets} must all be multiples of "
                f"n_block_size={self.n_block_size}; the packing must be built "
                f"with tile_n = n_block_size"
            )
        if self.pack_gqa:
            raise NotImplementedError("pack_gqa is not wired up for the backward")

    # -- schedule -------------------------------------------------------

    @cute.jit
    def _decode_n_block(self, n_block: Int32, seqlen_q: Int32):
        """
        packed block index -> (level, level-local base, query hull).

        One constexpr-unrolled pass over the levels. `pad_offsets` is
        increasing, so taking the LAST level whose block base is <= n_block
        selects correctly with no branching. Every quantity a level
        contributes is computed for all levels and selected at the end;
        that costs a few tens of integer ops, once per CTA, against a loop
        that then runs over many query blocks.

        The query hull is `ranges.tile_query_hull` inlined: q_lo of the
        tile's first node, q_hi of its last. Both are non-decreasing in k,
        so it contains Q_l(k) for every node in the tile -- the element
        mask is what makes the result exact, exactly as `range_spec` says.
        """
        a = self.activation_times
        level = Int32(0)
        k_base = Int32(0)
        q_lo = Int32(0)
        q_hi = Int32(0)
        for l in cutlass.range_constexpr(self.num_levels):
            block_base = self.pad_offsets[l] // self.n_block_size
            take = n_block >= block_base
            kl = (n_block - block_base) * self.n_block_size
            # The tile is clipped to the level's real length: everything past
            # it is zero padding, which no query attends.
            kh = cutlass.min(kl + self.n_block_size, self.level_lens[l])
            ql, _ = node_query_bounds(
                kl, l, a, self.coarsest_span, seqlen_q, CUTE_OPS
            )
            _, qh = node_query_bounds(
                kh - 1, l, a, self.coarsest_span, seqlen_q, CUTE_OPS
            )
            level = Int32(l) if take else level
            k_base = kl if take else k_base
            q_lo = ql if take else q_lo
            # An all-padding tile has kh <= kl, so its hull is empty.
            q_hi = (qh if kh > kl else ql) if take else q_hi
        return level, k_base, q_lo, q_hi

    # -- mask -----------------------------------------------------------

    @cute.jit
    def apply_telescope_mask_bwd(
        self,
        acc_S: cute.Tensor,
        m_block: Int32,
        thr_mma: cute.TiledMma,
        level: Int32,
        k_base: Int32,
        seqlen_q: Int32,
    ):
        """
        Per-row [lo, hi) masking in LEVEL-LOCAL coordinates.

        The forward masks in packed coordinates because a query block spans
        several levels at once. Here the whole CTA sits inside one level, so
        the column index is `k_base + col` and the bounds come straight from
        `level_row_bounds` with no offset -- the storage/semantics split
        `range_spec` insists on falls out naturally.

        Rows past the end of the sequence are masked wholesale by setting an
        empty limit pair, which is what `mask_seqlen=True` did upstream.
        """
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS))
        t0ScS_mn = layout_utils.reshape_acc_to_mn(
            thr_mma.get_slice(0).partition_C(cS)
        )
        # Level-local index of this thread's column 0 within the tile.
        col_base = k_base + tScS_mn[0][1]

        for r in cutlass.range_constexpr(cute.size(tScS_mn.shape[0])):
            q_idx = tScS_mn[r, 0][0] + m_block * self.m_block_size
            row_valid = q_idx < seqlen_q
            q_eff = cutlass.min(q_idx, seqlen_q - 1)
            bounds = level_row_bounds(
                q_eff, self.activation_times, self.cache_size, CUTE_OPS
            )
            lo = bounds[0][0]
            hi = bounds[0][1]
            for l in cutlass.range_constexpr(1, self.num_levels):
                take = level == l
                lo = bounds[l][0] if take else lo
                hi = bounds[l][1] if take else hi
            limit_lo = (lo - col_base) if row_valid else Int32(self.n_block_size)
            limit_hi = (hi - col_base) if row_valid else Int32(0)
            for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                col = t0ScS_mn[0, c][1]
                if col < limit_lo or col >= limit_hi:
                    acc_S_mn[r, c] = -Float32.inf

    # -- kernel ---------------------------------------------------------
    #
    # Everything below is FlashAttentionBackwardSm80.kernel verbatim except
    # the two regions marked TELESCOPE CHANGE.

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sLSEMma_layout: cute.Layout,
        gmem_tiled_copy_QK: cute.TiledCopy,
        gmem_tiled_copy_VdO: cute.TiledCopy,
        gmem_tiled_copy_dK: cute.TiledCopy,
        gmem_tiled_copy_dV: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
        tiled_mma_sdp: cute.TiledMma,
        tiled_mma_dkv: cute.TiledMma,
        tiled_mma_dq: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        aux_data: AuxData = AuxData(),
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()

        n_block, head_idx, batch_idx, _ = work_tile.tile_idx

        if work_tile.is_valid_tile:
            seqlen = SeqlenInfoQK.create(
                batch_idx,
                mQ.shape[1],
                mK.shape[1],
                mCuSeqlensQ=mCuSeqlensQ,
                mCuSeqlensK=mCuSeqlensK,
                mSeqUsedQ=mSeqUsedQ,
                mSeqUsedK=mSeqUsedK,
                tile_m=self.m_block_size,
                tile_n=self.n_block_size,
            )

            # == TELESCOPE CHANGE 1 of 2: the block schedule ==============
            # Upstream asks BlockInfo which query blocks a KV block scatters
            # back to. Here n_block indexes the block-aligned packed buffer,
            # so its level is a comparison chain over compile-time constants,
            # and the query range is `ranges.tile_query_hull` -- a single
            # contiguous interval. Unlike the forward there is no need for a
            # virtual block index: the inherited m-loop already walks an
            # arbitrary ascending range.
            level, k_base, q_lo, q_hi = self._decode_n_block(
                n_block, seqlen.seqlen_q
            )
            m_block_min = q_lo // self.m_block_size
            m_block_max = cute.ceil_div(q_hi, self.m_block_size)
            # A KV block no query ever reads (padding, or a node created at
            # or after N) yields q_lo >= q_hi; clamp so the m-loop is empty
            # rather than running backwards.
            m_block_min = cutlass.min(m_block_min, m_block_max)

            # ///////////////////////////////////////////////////////////////////////////////
            # Get the appropriate tiles for this thread block.
            # ///////////////////////////////////////////////////////////////////////////////
            blkQ_shape = (self.m_block_size, self.head_dim_padded)
            blkK_shape = (self.n_block_size, self.head_dim_padded)
            blkV_shape = (self.n_block_size, self.head_dim_v_padded)
            blkdO_shape = (self.m_block_size, self.head_dim_v_padded)

            if cutlass.const_expr(not seqlen.has_cu_seqlens_q):
                mQ_cur = mQ[batch_idx, None, head_idx, None]
                mLSE_cur = mLSE[batch_idx, head_idx, None]
                mdO_cur = mdO[batch_idx, None, head_idx, None]
                mdPsum_cur = mdPsum[batch_idx, head_idx, None]
                mdQaccum_cur = mdQaccum[batch_idx, head_idx, None]
            else:
                padded_offset_q = seqlen.padded_offset_q
                mQ_cur = cute.domain_offset((seqlen.offset_q, 0), mQ[None, head_idx, None])
                mLSE_cur = cute.domain_offset((padded_offset_q,), mLSE[head_idx, None])
                mdO_cur = cute.domain_offset((seqlen.offset_q, 0), mdO[None, head_idx, None])
                mdPsum_cur = cute.domain_offset((padded_offset_q,), mdPsum[head_idx, None])
                mdQaccum_cur = cute.domain_offset((padded_offset_q * self.head_dim_padded,), mdQaccum[head_idx, None])
            head_idx_kv = head_idx // self.qhead_per_kvhead if cutlass.const_expr(not self.pack_gqa) else head_idx

            if cutlass.const_expr(not seqlen.has_cu_seqlens_k):
                mK_cur, mV_cur = [t[batch_idx, None, head_idx_kv, None] for t in (mK, mV)]
            else:
                mK_cur, mV_cur = [cute.domain_offset((seqlen.offset_k, 0), t[None, head_idx_kv, None]) for t in (mK, mV)]

            # (m_block_size, head_dim, m_block)
            gQ = cute.local_tile(mQ_cur, blkQ_shape, (None, 0))
            # (n_block_size, head_dim)
            gK = cute.local_tile(mK_cur, blkK_shape, (n_block, 0))
            # (n_block_size, head_dim_v)
            gV = cute.local_tile(mV_cur, blkV_shape, (n_block, 0))
            # (m_block_size, head_dim_v, m_block)
            gdO = cute.local_tile(mdO_cur, blkdO_shape, (None, 0))
            gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (None,))
            gdPsum = cute.local_tile(mdPsum_cur, (self.m_block_size,), (None,))
            gdQaccum = cute.local_tile(mdQaccum_cur, (self.m_block_size * self.head_dim_padded,), (None,))

            # ///////////////////////////////////////////////////////////////////////////////
            # Get shared memory buffer
            # ///////////////////////////////////////////////////////////////////////////////
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sQ = storage.sQ.get_tensor(sQ_layout)
            sK = storage.sK.get_tensor(sK_layout)
            if cutlass.const_expr(not self.share_QV_smem):
                sV = storage.sV.get_tensor(sV_layout)
            else:
                sV = cute.make_tensor(cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout)
            sdO = storage.sdO.get_tensor(sdO_layout)
            sP = storage.sP.get_tensor(sPdS_layout)
            sdS = storage.sdS.get_tensor(sPdS_layout)
            sLSE = storage.sLSE.get_tensor(sLSE_layout)
            sdPsum = storage.sdPsum.get_tensor(sLSE_layout)
            sLSEMma = storage.sLSE.get_tensor(sLSEMma_layout)
            sdPsumMma = storage.sdPsum.get_tensor(sLSEMma_layout)

            # Transpose view of tensors for tiled mma
            sQt, sdOt, sKt, sPt, sdSt = [layout_utils.transpose_view(t) for t in (sQ, sdO, sK, sP, sdS)]

            gmem_thr_copy_QK = gmem_tiled_copy_QK.get_slice(tidx)
            gmem_thr_copy_VdO = gmem_tiled_copy_VdO.get_slice(tidx)
            gmem_thr_copy_lse = gmem_tiled_copy_LSE.get_slice(tidx)
            gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_slice(tidx)
            # (CPY_Atom, CPY_M, CPY_K, m_block)
            tQgQ = gmem_thr_copy_QK.partition_S(gQ)
            tQsQ = gmem_thr_copy_QK.partition_D(sQ)
            # (CPY_Atom, CPY_N, CPY_K)
            tKgK = gmem_thr_copy_QK.partition_S(gK)
            tKsK = gmem_thr_copy_QK.partition_D(sK)
            # (CPY_Atom, CPY_N, CPY_K)
            tVgV = gmem_thr_copy_VdO.partition_S(gV)
            tVsV = gmem_thr_copy_VdO.partition_D(sV)
            # (CPY_Atom, CPY_M, CPY_K, m_block)
            tdOgdO = gmem_thr_copy_VdO.partition_S(gdO)
            tdOsdO = gmem_thr_copy_VdO.partition_D(sdO)
            tLSEgLSE = gmem_thr_copy_lse.partition_S(gLSE)
            tLSEsLSE = gmem_thr_copy_lse.partition_D(sLSE)
            tLSEgdPsum = gmem_thr_copy_lse.partition_S(gdPsum)
            tLSEsdPsum = gmem_thr_copy_lse.partition_D(sdPsum)
            tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum)

            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            thr_mma_sdp = tiled_mma_sdp.get_slice(tidx)
            thr_mma_dkv = tiled_mma_dkv.get_slice(tidx)
            thr_mma_dq = tiled_mma_dq.get_slice(tidx)
            acc_shape_dK = thr_mma_dkv.partition_shape_C((self.n_block_size, self.head_dim_padded))
            acc_shape_dV = thr_mma_dkv.partition_shape_C((self.n_block_size, self.head_dim_v_padded))
            acc_dK = cute.make_rmem_tensor(acc_shape_dK, cutlass.Float32)
            acc_dV = cute.make_rmem_tensor(acc_shape_dV, cutlass.Float32)
            acc_dK.fill(0.0)
            acc_dV.fill(0.0)

            tSrQ = utils.mma_make_fragment_A(sQ[None, None, 0], thr_mma_sdp, swapAB=self.SdP_swapAB)
            tSrK = utils.mma_make_fragment_B(sK, thr_mma_sdp, swapAB=self.SdP_swapAB)
            tdPrdO = utils.mma_make_fragment_A(sdO[None, None, 0], thr_mma_sdp, swapAB=self.SdP_swapAB)
            tdPrV = utils.mma_make_fragment_B(sV, thr_mma_sdp, swapAB=self.SdP_swapAB)
            tdVrP = utils.mma_make_fragment_A(sPt, thr_mma_dkv, swapAB=self.dKV_swapAB)
            tdVrdO = utils.mma_make_fragment_B(sdOt[None, None, 0], thr_mma_dkv, swapAB=self.dKV_swapAB)
            tdKrdS = utils.mma_make_fragment_A(sdSt, thr_mma_dkv, swapAB=self.dKV_swapAB)
            tdKrQ = utils.mma_make_fragment_B(sQt[None, None, 0], thr_mma_dkv, swapAB=self.dKV_swapAB)
            tdQrdS = utils.mma_make_fragment_A(sdS, thr_mma_dq, swapAB=self.dQ_swapAB)
            tdQrK = utils.mma_make_fragment_B(sKt, thr_mma_dq, swapAB=self.dQ_swapAB)

            LSEslice = (None, 0, None) if cutlass.const_expr(not self.SdP_swapAB) else (0, None, None)
            tSsLSEMma = layout_utils.reshape_acc_to_mn(thr_mma_sdp.partition_C(sLSEMma))[LSEslice]
            tSsdPsumMma = layout_utils.reshape_acc_to_mn(thr_mma_sdp.partition_C(sdPsumMma))[LSEslice]

            # ///////////////////////////////////////////////////////////////////////////////
            # Smem copy atom tiling
            # ///////////////////////////////////////////////////////////////////////////////
            smem_copy_atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype,
            )
            smem_copy_atom_transposed = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype,
            )
            smem_thr_copy_QdO = utils.make_tiled_copy_A(
                smem_copy_atom, tiled_mma_sdp, swapAB=self.SdP_swapAB
            ).get_slice(tidx)
            smem_thr_copy_KV = utils.make_tiled_copy_B(
                smem_copy_atom, tiled_mma_sdp, swapAB=self.SdP_swapAB
            ).get_slice(tidx)
            # TODO: should this be smem_copy_atom_transposed?
            smem_thr_copy_PdSt = utils.make_tiled_copy_A(
                smem_copy_atom_transposed, tiled_mma_dkv, swapAB=self.dKV_swapAB
            ).get_slice(tidx)
            smem_thr_copy_QdOt = utils.make_tiled_copy_B(
                smem_copy_atom_transposed, tiled_mma_dkv, swapAB=self.dKV_swapAB
            ).get_slice(tidx)
            smem_thr_copy_dS = utils.make_tiled_copy_A(
                smem_copy_atom, tiled_mma_dq, swapAB=self.dQ_swapAB
            ).get_slice(tidx)
            smem_thr_copy_Kt = utils.make_tiled_copy_B(
                smem_copy_atom_transposed, tiled_mma_dq, swapAB=self.dQ_swapAB
            ).get_slice(tidx)
            # TODO: what's the number of bits? What if SdP_swapAB
            r2s_thr_copy_PdS = cute.make_tiled_copy_C(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=2 * self.dtype.width
                ),
                tiled_mma_sdp,
            ).get_slice(tidx)

            tSsQ = smem_thr_copy_QdO.partition_S(sQ)
            tdPsdO = smem_thr_copy_QdO.partition_S(sdO)
            tSsK = smem_thr_copy_KV.partition_S(sK)
            tdPsV = smem_thr_copy_KV.partition_S(sV)
            tdVsPt = smem_thr_copy_PdSt.partition_S(sPt)
            tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt)
            tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt)
            tdKsQt = smem_thr_copy_QdOt.partition_S(sQt)
            tdQsdS = smem_thr_copy_dS.partition_S(sdS)
            tdQsKt = smem_thr_copy_Kt.partition_S(sKt)
            tPsP = r2s_thr_copy_PdS.partition_D(sP)
            tdSsdS = r2s_thr_copy_PdS.partition_D(sdS)

            # ///////////////////////////////////////////////////////////////////////////////
            # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
            # of tile_shape
            # ///////////////////////////////////////////////////////////////////////////////
            # Construct identity layout for KV
            cQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
            tQcQ = gmem_thr_copy_QK.partition_S(cQ)
            t0QcQ = gmem_thr_copy_QK.get_slice(0).partition_S(cQ)
            if cutlass.const_expr(self.head_dim_padded == self.head_dim_v_padded):
                tdOcdO = tQcQ
                t0dOcdO = t0QcQ
            else:
                cdO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
                tdOcdO = gmem_thr_copy_VdO.partition_S(cdO)
                t0dOcdO = gmem_thr_copy_VdO.get_slice(0).partition_S(cdO)
            cLSE = cute.make_identity_tensor((self.m_block_size,))
            tLSEcLSE = gmem_thr_copy_lse.partition_S(cLSE)

            # Allocate predicate tensors for m and n, here we only allocate the tile of k, and
            # use "if" on the mn dimension.
            # This is to reduce register pressure and gets 2-3% performance gain.

            d_head = mQ.shape[cute.rank(mQ) - 1]
            d_head_v = mdO.shape[cute.rank(mdO) - 1]

            tQpQ = utils.predicate_k(tQcQ, limit=d_head)
            if cutlass.const_expr(self.same_hdim_kv):
                tdOpdO = tQpQ
            else:
                tdOpdO = utils.predicate_k(tdOcdO, limit=d_head_v)

            # group parameters for compute_one_m_block
            mma_params = SimpleNamespace(
                thr_mma_sdp=thr_mma_sdp, thr_mma_dkv=thr_mma_dkv, thr_mma_dq=thr_mma_dq,
                tSrQ=tSrQ, tSrK=tSrK, tdPrdO=tdPrdO, tdPrV=tdPrV,
                tdVrP=tdVrP, tdVrdO=tdVrdO, tdKrdS=tdKrdS, tdKrQ=tdKrQ,
                tdQrdS=tdQrdS, tdQrK=tdQrK,
                acc_dK=acc_dK, acc_dV=acc_dV,
            )
            smem_copy_params = SimpleNamespace(
                smem_thr_copy_QdO=smem_thr_copy_QdO,
                smem_thr_copy_KV=smem_thr_copy_KV,
                smem_thr_copy_PdSt=smem_thr_copy_PdSt,
                smem_thr_copy_QdOt=smem_thr_copy_QdOt,
                smem_thr_copy_dS=smem_thr_copy_dS,
                smem_thr_copy_Kt=smem_thr_copy_Kt,
                r2s_thr_copy_PdS=r2s_thr_copy_PdS,
                tSsQ=tSsQ, tSsK=tSsK, tdPsdO=tdPsdO, tdPsV=tdPsV,
                tSsLSEMma=tSsLSEMma, tSsdPsumMma=tSsdPsumMma,
                tPsP=tPsP, tdSsdS=tdSsdS,
                tdVsPt=tdVsPt, tdVsdOt=tdVsdOt, tdKsdSt=tdKsdSt, tdKsQt=tdKsQt,
                tdQsdS=tdQsdS, tdQsKt=tdQsKt,
            )
            gmem_copy_params = SimpleNamespace(
                gmem_thr_copy_dQaccum=gmem_thr_copy_dQaccum, tdQgdQaccum=tdQgdQaccum
            )
            load_Q_LSE = partial(
                self.load_Q_LSE, gmem_tiled_copy_QK, gmem_tiled_copy_LSE,
                tQgQ, tQsQ, tQcQ, t0QcQ, tQpQ,
                tLSEgLSE, tLSEsLSE, tLSEcLSE, seqlen=seqlen.seqlen_q
            )
            load_dO_dPsum = partial(
                self.load_dO_dPsum, gmem_tiled_copy_VdO, gmem_tiled_copy_LSE,
                tdOgdO, tdOsdO, tdOcdO, t0dOcdO, tdOpdO,
                tLSEgdPsum, tLSEsdPsum, tLSEcLSE, seqlen=seqlen.seqlen_q
            )
            compute_one_m_block = partial(
                self.compute_one_m_block, mma_params=mma_params,
                smem_copy_params=smem_copy_params, gmem_copy_params=gmem_copy_params,
                load_Q_LSE=load_Q_LSE, load_dO_dPsum=load_dO_dPsum,
                m_block_max=m_block_max,
                softmax_scale=softmax_scale,
                softmax_scale_log2=softmax_scale_log2,
                aux_data=aux_data,
            )

            if m_block_min < m_block_max:
                # ///////////////////////////////////////////////////////////////////////////////
                # Prologue
                # ///////////////////////////////////////////////////////////////////////////////
                # Start async loads of the last mn-tile, where we take care of the mn residue
                self.load_V(gmem_thr_copy_VdO, tVgV, tVsV, n_block, seqlen=seqlen.seqlen_k,
                            headdim=d_head_v)
                if cutlass.const_expr(self.V_in_regs):
                    cute.arch.cp_async_commit_group()
                self.load_K(gmem_thr_copy_QK, tKgK, tKsK, n_block, seqlen=seqlen.seqlen_k,
                            headdim=d_head)
                cute.arch.cp_async_commit_group()

                if cutlass.const_expr(self.V_in_regs):
                    cute.arch.cp_async_wait_group(1)
                    cute.arch.barrier()
                    tdPrV_copy_view = smem_thr_copy_KV.retile(tdPrV)
                    cute.copy(smem_thr_copy_KV, tdPsV, tdPrV_copy_view)
                    # Sync to avoid loading Q to smem_q, which overlaps with smem_v
                    cute.arch.barrier()

                m_block = m_block_min
                assert self.num_stages_Q >= self.num_stages_dO
                for stage in cutlass.range_constexpr(self.num_stages_Q):
                    if cutlass.const_expr(self.num_stages_Q == 1 or stage < self.num_stages_Q - 1):
                        if stage == 0 or m_block + stage < m_block_max:
                            load_Q_LSE(m_block + stage, smem_pipe_write_q=stage)
                        cute.arch.cp_async_commit_group()
                    if cutlass.const_expr(stage < self.num_stages_dO):
                        if stage == 0 or m_block + stage < m_block_max:
                            load_dO_dPsum(m_block + stage, smem_pipe_write_q=stage)
                        cute.arch.cp_async_commit_group()

                # ///////////////////////////////////////////////////////////////////////////////
                # Mainloop
                # ///////////////////////////////////////////////////////////////////////////////
                # Start processing of the first n-block.
                # == TELESCOPE CHANGE 2 of 2: the mask ====================
                # Same shape as AttentionMask's local branch -- one
                # [lo, hi) per row -- with the pair coming from the schedule
                # at this block's level instead of from an affine function
                # of the row index.
                mask_fn = partial(
                    self.apply_telescope_mask_bwd,
                    thr_mma=thr_mma_sdp,
                    level=level,
                    k_base=k_base,
                    seqlen_q=seqlen.seqlen_q,
                )
                smem_pipe_read_q = cutlass.Int32(0)
                smem_pipe_read_do = cutlass.Int32(0)
                smem_pipe_write_q = cutlass.Int32(self.num_stages_Q - 1)
                smem_pipe_write_do = cutlass.Int32(0)
                for m_tile in cutlass.range(m_block_min, m_block_max, unroll=1):
                    compute_one_m_block(
                        m_tile, smem_pipe_read_q, smem_pipe_read_do, smem_pipe_write_q, smem_pipe_write_do,
                        mask_fn=mask_fn,
                    )
                    smem_pipe_read_q = self.advance_pipeline(smem_pipe_read_q, self.num_stages_Q)
                    smem_pipe_read_do = self.advance_pipeline(smem_pipe_read_do, self.num_stages_dO)
                    smem_pipe_write_q = self.advance_pipeline(smem_pipe_write_q, self.num_stages_Q)
                    smem_pipe_write_do = self.advance_pipeline(smem_pipe_write_do, self.num_stages_dO)

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue
            # ///////////////////////////////////////////////////////////////////////////////
            # If GQA, we scale dK in the postprocessing kernel instead
            if cutlass.const_expr(self.qhead_per_kvhead == 1):
                acc_dK.store(acc_dK.load() * softmax_scale)
            # reuse sK and sV data iterator
            sdK = cute.make_tensor(sK.iterator, sK_layout)
            sdV = cute.make_tensor(sV.iterator, sV_layout)
            self.epilogue(
                acc_dK, acc_dV, mdK, mdV, sdK, sdV,
                gmem_tiled_copy_dK, gmem_tiled_copy_dV, tiled_mma_dkv,
                tidx, n_block, head_idx, batch_idx, seqlen, d_head, d_head_v
            )
