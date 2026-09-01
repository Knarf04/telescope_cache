"""
Host entry point for the telescoping attention kernels.

Mirrors the shape of `flash_attn.cute.interface._flash_attn_fwd`, minus the
features this kernel does not have (varlen, paged KV, split-KV, FP8, block
sparsity): build cute tensors, instantiate the kernel specialized to the
schedule, compile once per configuration, then call.

The compile cache key includes the schedule, because
`TelescopeAttentionForwardSm80` bakes the activation times, the cache size
and the padded level offsets in as compile-time constants -- that is what
turns every range bound into a shift and an add.
"""

import math
from typing import Dict, Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute

from flash_attn.cute import utils as fa_utils
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
from flash_attn.cute.utils import AuxData

from telescope_cache.cute.flash_fwd_telescope import TelescopeAttentionForwardSm80
from telescope_cache.cute.packing import AlignedPackedKV, pack_levels_aligned
from telescope_cache.range_spec import RangeSpec

__all__ = [
    "telescope_attn_bwd",
    "telescope_attn_packed",
    "telescope_attn_func",
    "DEFAULT_TILE_M",
    "DEFAULT_TILE_N",
    "DEFAULT_NUM_THREADS",
]

# Tuned on sm86 by test/sweep_tiles.py; predicted by test/analyze_tiles.py.
#
# These are much smaller than FA4's SM80 defaults (128 x 64 x 128 threads),
# and deliberately so. A query tile spans tile_m rows whose windows are
# staggered, so the hull a level presents is about tile_m + window wide while
# each row uses only `window` of it. With cache_size spread over L+1 levels
# the per-level windows are ~9 to ~64 entries, so a wide tile is mostly
# masked lanes: analyze_tiles.py puts occupancy at 0.272 for 128 x 64 against
# 0.610 for 32 x 16, and the measured speedup (2.19x) tracks that ratio.
#
# This is the one place where telescoping wants something a sliding-window
# kernel does not: SWA has a single window width, so FA4 can pick one tile
# and be right; here the levels have genuinely different widths.
DEFAULT_TILE_M = 32
DEFAULT_TILE_N = 16
DEFAULT_NUM_THREADS = 64

# Backward tiling: the SM120 BwdConfig, which drives the same SM80 MMA and is
# the only configuration FA4 exercises for it. Defined here, above its first
# use, because telescope_attn_func packs at BWD_N_BLOCK so one K/V buffer
# serves both directions.
# Tuned on sm86 by test/sweep_bwd.py (4.463 ms vs 4.959 for FA4's stock
# 64/64/128/4 at N=4096, D=128, GQA 8:1 -- a 1.11x win, far less than the
# forward's 2.2x). The reason the backward has less slack: its most
# promising axis, n_block, is pinned at 64 by the GQA atomic dK/dV epilogue
# (flash_bwd.py:1192 asserts the accumulator fragment matches
# AtomLayoutNdKV), and the Mma_dKV_is_RS fast path needs SdP_swapAB=True,
# which would require apply_telescope_mask_bwd to handle a transposed acc_S.
BWD_M_BLOCK = 32
BWD_N_BLOCK = 64
BWD_NUM_THREADS = 128
BWD_ATOM_LAYOUT = 2

_TORCH_TO_CUTLASS = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}

_compile_cache: Dict[tuple, object] = {}


def _get_compiled(
    dtype,
    head_dim,
    head_dim_v,
    qhead_per_kvhead,
    tile_m,
    tile_n,
    num_threads,
    softcap,
    activation_times,
    cache_size,
    pad_offsets,
    q_t,
    k_t,
    v_t,
    o_t,
    lse_t,
):
    key = (
        dtype, head_dim, head_dim_v, qhead_per_kvhead,
        tile_m, tile_n, num_threads, softcap,
        activation_times, cache_size, pad_offsets,
    )
    if key in _compile_cache:
        return _compile_cache[key]

    # softcap rides FA4's score_mod hook, exactly as `interface.py` does for
    # the stock kernels: scores are scaled first, then c*tanh(x/c).
    score_mod = (
        fa_utils.create_softcap_scoremod(softcap) if softcap else None
    )
    fa_fwd = TelescopeAttentionForwardSm80(
        _TORCH_TO_CUTLASS[dtype],
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        pack_gqa=False,
        tile_m=tile_m,
        tile_n=tile_n,
        num_stages=1,
        num_threads=num_threads,
        Q_in_regs=False,
        score_mod=score_mod,
        activation_times=activation_times,
        cache_size=cache_size,
        pad_offsets=pad_offsets,
    )
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        fa_fwd,
        q_t, k_t, v_t, o_t, lse_t,
        Float32_scale := 1.0,   # placeholder; softmax_scale is a runtime arg
        None, None, None, None,     # cu_seqlens_q/k, seqused_q/k
        None,                       # page_table
        None, None,                 # window_size_left/right
        None,                       # learnable_sink
        None,                       # blocksparse_tensors
        AuxData(None, None),
        None, None,                 # cu_total_m_blocks, cu_total_splits_m_blocks
        current_stream,
        options="--enable-tvm-ffi",
    )
    _compile_cache[key] = compiled
    return compiled


def telescope_attn_packed(
    q: torch.Tensor,
    packed: AlignedPackedKV,
    activation_times: Tuple[int, ...],
    cache_size: int,
    softcap: float = 20.0,
    softmax_scale: Optional[float] = None,
    tile_m: int = DEFAULT_TILE_M,
    tile_n: Optional[int] = None,
    num_threads: int = DEFAULT_NUM_THREADS,
):
    """
    q       [B, N, Hq, Dk]      fp16/bf16
    packed  block-aligned level-major K/V (packing.pack_levels_aligned)
    tile_n  key-block size to RUN at; defaults to the packing's own.

    Returns (out [B, N, Hq, Dv] same dtype as q, lse [B, N, Hq] float32).

    The kernel's tile_n need not equal the packing's. All it requires is that
    every level start on one of its block boundaries, and a packing aligned
    to 64 is also aligned to 16 or 32. That matters because the forward is
    fastest at tile_n=16 while the backward runs at 64: without this, a
    training step would need two copies of the K/V tree. Pack once at the
    backward's tile_n and let the forward run finer inside it.
    """
    if q.dtype not in _TORCH_TO_CUTLASS:
        raise ValueError(f"q dtype {q.dtype} must be float16 or bfloat16")
    if packed.k.dtype != q.dtype or packed.v.dtype != q.dtype:
        raise ValueError("q, k and v must share a dtype")

    B, N, Hq, Dk = q.shape
    Hkv, Dv = packed.k.shape[2], packed.v.shape[-1]
    if Hq % Hkv:
        raise ValueError(f"Hq={Hq} must be divisible by Hkv={Hkv}")
    tile_n = packed.tile_n if tile_n is None else tile_n
    if packed.pad_offsets[-1] != packed.k.shape[1]:
        raise ValueError("packed buffer length disagrees with pad_offsets")
    if any(off % tile_n for off in packed.pad_offsets):
        raise ValueError(
            f"cannot run at tile_n={tile_n} on a packing aligned to "
            f"{packed.tile_n}: every level must start on a block boundary, so "
            f"tile_n must divide {packed.tile_n}"
        )

    q = q.contiguous()
    out = torch.empty(B, N, Hq, Dv, dtype=q.dtype, device=q.device)
    lse = torch.empty(B, Hq, N, dtype=torch.float32, device=q.device)

    q_t, k_t, v_t, o_t = (to_cute_tensor(t) for t in (q, packed.k, packed.v, out))
    lse_t = to_cute_tensor(lse, assumed_align=4)

    compiled = _get_compiled(
        q.dtype, Dk, Dv, Hq // Hkv, tile_m, tile_n, num_threads,
        float(softcap) if softcap else 0.0,
        tuple(activation_times), int(cache_size), tuple(packed.pad_offsets),
        q_t, k_t, v_t, o_t, lse_t,
    )
    compiled(
        q, packed.k, packed.v, out, lse,
        softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(Dk),
        None, None, None, None,
        None,
        None, None,
        None,
        None,
        AuxData(None, None),
        None, None,
    )
    return out, lse.transpose(1, 2).contiguous()  # [B, Hq, N] -> [B, N, Hq]


def telescope_attn_func(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    fmap: Dict[int, int],
    cache_size: int,
    softcap: float = 20.0,
    softmax_scale: Optional[float] = None,
    tile_m: int = DEFAULT_TILE_M,
    tile_n: int = DEFAULT_TILE_N,
    num_threads: int = DEFAULT_NUM_THREADS,
):
    """
    Convenience wrapper taking the *tight* packing from `reference.pack_levels`
    and the schedule as an fmap, so it is a drop-in for
    `reference.multilevel_attention_forward` in the tests.

    Real callers should build the aligned packing once with
    `packing.pack_levels_aligned` and use `telescope_attn_packed`; the
    re-layout here is a copy of the whole K/V buffer.
    """
    from telescope_cache.cute.packing import repack_aligned
    from telescope_cache.reference import PackedKV

    N = q.shape[1]
    spec = RangeSpec.from_fmap(fmap, cache_size, N)
    tight = PackedKV(k_packed, v_packed, spec.level_offsets())
    # Pack at the backward's block size so the same buffer can be handed to
    # telescope_attn_bwd, and run the forward at its own finer tile_n.
    packed = repack_aligned(tight, max(tile_n, BWD_N_BLOCK))
    return telescope_attn_packed(
        q, packed, spec.activation_times, cache_size,
        softcap=softcap, softmax_scale=softmax_scale, tile_m=tile_m,
        tile_n=tile_n, num_threads=num_threads,
    )


# ===========================================================================
# Backward.
#
# FA4's own backward host path refuses this architecture -- `_flash_attn_bwd`
# asserts the compute capability is in {9, 10, 11, 12}, and past that assert
# the arch-8 branch raises UnboundLocalError on `dQ_single_wg`. The KERNEL is
# fine (verified against a float32 torch reference on sm86); what is missing
# is the plumbing, which is what this section supplies:
#
#   preprocess   dpsum = (O * dO).sum(-1), lse_log2 = lse * log2(e),
#                dq_accum zeroed          -- FA4's own, arch-agnostic
#   kernel       one KV-owned pass: dK/dV in registers, dQ into dq_accum
#   postprocess  dq_accum fp32 -> dq, and for GQA dk/dv_accum -> dk/dv
#
# The postprocess `atom_layout` and `swap_ab` MUST match the kernel's
# AtomLayoutMdQ / dQ_swapAB (resp. AtomLayoutNdKV / dKV_swapAB): they decide
# how the fp32 accumulator's lanes map back to rows. Passing the wrong one
# silently produces garbage dQ with correct dK/dV, which is exactly what it
# looked like the first time.
# ===========================================================================

_bwd_compile_cache: Dict[tuple, object] = {}


def _get_compiled_bwd(key, make_obj, tensors):
    if key in _bwd_compile_cache:
        return _bwd_compile_cache[key]
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        make_obj(), *tensors,
        None, None, None, None,       # cu_seqlens_q/k, seqused_q/k
        None, None,                   # window_size_left/right
        None, None, None,             # dQ/dK/dV semaphores
        AuxData(None, None),
        None,                         # blocksparse_tensors
        None,                         # cu_total_m_blocks
        stream,
        options="--enable-tvm-ffi",
    )
    _bwd_compile_cache[key] = compiled
    return compiled


def telescope_attn_bwd(
    q: torch.Tensor,
    packed: AlignedPackedKV,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    activation_times: Tuple[int, ...],
    cache_size: int,
    coarsest_span: int,
    softcap: float = 20.0,
    softmax_scale: Optional[float] = None,
    m_block: int = BWD_M_BLOCK,
    n_block: Optional[int] = None,
    num_threads: int = BWD_NUM_THREADS,
    atom_layout: int = BWD_ATOM_LAYOUT,
    num_stages: Optional[int] = None,
    atom_layout_msdp: Optional[int] = None,
    atom_layout_ndkv: Optional[int] = None,
    sdp_swap_ab: bool = False,
    dkv_swap_ab: bool = False,
):
    """
    Phase-2 backward at the packed boundary, matching
    `reference.multilevel_attention_backward`'s contract:

        q        [B, N, Hq, Dk]        out   [B, N, Hq, Dv]
        packed   block-aligned K/V     lse   [B, N, Hq] float32 (saved fwd)
                                       dout  [B, N, Hq, Dv]

        dq        [B, N, Hq, Dk]       q.dtype
        dk_packed [B, padded, Hkv, Dk] packed.k.dtype
        dv_packed [B, padded, Hkv, Dv] packed.v.dtype

    dk/dv come back in the PADDED packed layout; the padding rows are zero
    because every padded column is masked, so `P` and hence `dV`, `dS`, `dK`
    are zero there. Slice them per level with `packed.pad_offsets`.

    `lse` must be the forward's saved value, natural-log, exactly as the
    reference requires -- it is converted to log2 here, not recomputed.
    """
    from flash_attn.cute.interface import (
        _bwd_postprocess_convert,
        _bwd_preprocess,
    )

    from telescope_cache.cute.flash_bwd_telescope import (
        TelescopeAttentionBackwardSm80,
    )

    B, N, Hq, Dk = q.shape
    Hkv, Dv = packed.k.shape[2], packed.v.shape[-1]
    if Hq % Hkv:
        raise ValueError(f"Hq={Hq} must be divisible by Hkv={Hkv}")
    n_block = BWD_N_BLOCK if n_block is None else n_block
    if packed.tile_n != n_block:
        raise ValueError(
            f"the backward runs at n_block_size={n_block}, but the packing "
            f"was built with tile_n={packed.tile_n}; rebuild it with "
            f"pack_levels_aligned(..., tile_n={n_block})"
        )
    qhead_per_kvhead = Hq // Hkv
    device, dtype = q.device, q.dtype
    cute_dtype = _TORCH_TO_CUTLASS[dtype]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(Dk)

    q, dout = q.contiguous(), dout.contiguous()
    sq_r = -(-N // m_block) * m_block
    hd_r = -(-Dk // 32) * 32
    hdv_r = -(-Dv // 32) * 32

    # FA4's preprocess wants lse as [B, H, S]; the reference hands it as
    # [B, S, H]. It also writes lse_log2 and dpsum over the ROUNDED length,
    # so those are allocated at sq_r, not N.
    lse_bhs = lse.transpose(1, 2).contiguous()
    dq_accum = torch.empty(B, Hq, sq_r * hd_r, dtype=torch.float32, device=device)
    dpsum = torch.empty(B, Hq, sq_r, dtype=torch.float32, device=device)
    lse_log2 = torch.empty(B, Hq, sq_r, dtype=torch.float32, device=device)
    _bwd_preprocess(
        out.contiguous(), dout, dpsum, lse_bhs, lse_log2, dq_accum,
        None, None, None, cute_dtype, Dk, Dv, m_block, fake_mode=False,
    )

    dq = torch.empty_like(q)
    dk = torch.empty_like(packed.k)
    dv = torch.empty_like(packed.v)
    # With GQA several query heads scatter into the same dK/dV slot, so the
    # kernel accumulates in fp32 and a postprocess converts. Without GQA it
    # writes dK/dV directly.
    # A 2-deep Q/dO pipeline costs shared memory proportional to
    # m_block * head_dim. At m_block <= 32 it fits on sm_86 even at
    # head_dim 128, which the old m_block=64 default could not afford.
    if num_stages is not None:
        bwd_stages = num_stages
    elif max(Dk, Dv) <= 64 or m_block <= 32:
        bwd_stages = 2
    else:
        bwd_stages = 1
    gqa = qhead_per_kvhead > 1
    padded = packed.k.shape[1]
    if gqa:
        dk_accum = torch.zeros(
            B, Hkv, padded * hd_r, dtype=torch.float32, device=device
        )
        dv_accum = torch.zeros(
            B, Hkv, padded * hdv_r, dtype=torch.float32, device=device
        )
    else:
        dk_accum = dv_accum = None

    dk_arg = dk_accum if gqa else dk
    dv_arg = dv_accum if gqa else dv

    def make_obj():
        return TelescopeAttentionBackwardSm80(
            cute_dtype, Dk, Dv, qhead_per_kvhead,
            m_block, n_block,
            # Q/dO pipeline depth is bounded by shared memory, and sm_86 has
            # 99 KB per block against sm_80's 163 KB. At head_dim > 64 a
            # 2-deep pipeline needs 115712 bytes and the launch is rejected
            # outright. FA4's own SM120 BwdConfig makes the same call for the
            # same reason, so follow it rather than inventing a rule.
            num_stages_Q=bwd_stages, num_stages_dO=bwd_stages,
            num_threads=num_threads,
            pack_gqa=False,
            # Mma_dKV_is_RS (a register-source dK/dV MMA, which avoids a
            # round trip through shared memory) is enabled inside FA4 only
            # when AtomLayoutMSdP == 1, AtomLayoutNdKV == num_mma_warps,
            # SdP_swapAB and not dKV_swapAB. Exposing these lets the sweep
            # reach that path instead of being pinned to the slow one.
            SdP_swapAB=sdp_swap_ab, dKV_swapAB=dkv_swap_ab, dQ_swapAB=False,
            AtomLayoutMSdP=(atom_layout if atom_layout_msdp is None
                            else atom_layout_msdp),
            AtomLayoutNdKV=(atom_layout if atom_layout_ndkv is None
                            else atom_layout_ndkv),
            AtomLayoutMdQ=atom_layout,
            score_mod=(
                fa_utils.create_softcap_scoremod(softcap) if softcap else None
            ),
            score_mod_bwd=(
                fa_utils.create_softcap_scoremod_bwd(softcap) if softcap else None
            ),
            activation_times=tuple(activation_times),
            cache_size=int(cache_size),
            pad_offsets=tuple(packed.pad_offsets),
            level_lens=tuple(packed.level_lens),
            coarsest_span=int(coarsest_span),
        )

    tensors = [
        to_cute_tensor(t) for t in
        (q, packed.k, packed.v, dout, lse_log2, dpsum, dq_accum, dk_arg, dv_arg)
    ] + [scale]
    key = (
        dtype, Dk, Dv, qhead_per_kvhead, softcap, gqa, bwd_stages,
        m_block, n_block, num_threads, atom_layout,
        atom_layout_msdp, atom_layout_ndkv, sdp_swap_ab, dkv_swap_ab,
        tuple(activation_times), int(cache_size),
        tuple(packed.pad_offsets), tuple(packed.level_lens), int(coarsest_span),
    )
    compiled = _get_compiled_bwd(key, make_obj, tensors)
    compiled(
        q, packed.k, packed.v, dout, lse_log2, dpsum, dq_accum, dk_arg, dv_arg,
        scale,
        None, None, None, None,
        None, None,
        None, None, None,
        AuxData(None, None), None, None,
    )

    # atom_layout / swap_ab must match the kernel's; see the section comment.
    _bwd_postprocess_convert(
        dq_accum, dq, scale, None, None, 86, cute_dtype, Dk,
        m_block, 128, atom_layout, False, fake_mode=False,
    )
    if gqa:
        _bwd_postprocess_convert(
            dk_accum, dk, scale, None, None, 86, cute_dtype, Dk,
            n_block, 128,
            atom_layout if atom_layout_ndkv is None else atom_layout_ndkv,
            dkv_swap_ab, fake_mode=False,
        )
        _bwd_postprocess_convert(
            dv_accum, dv, 1.0, None, None, 86, cute_dtype, Dv,
            n_block, 128,
            atom_layout if atom_layout_ndkv is None else atom_layout_ndkv,
            dkv_swap_ab, fake_mode=False,
        )
    return dq, dk, dv
