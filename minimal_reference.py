"""
Minimal pure-PyTorch reference for telescoping (multiresolution) attention.

This is the SEMANTIC oracle: an executable mathematical definition of the
telescope attention block, optimized for clarity > compactness > efficiency.
It is deliberately structurally independent of reference.py (the
kernel-shaped reference) so the two can cross-check each other:

    minimal side                        reference.py side
    ------------                        -----------------
    per-query visible-entry list        packed level-major buffers,
    (chronological, oldest -> newest)   fwd_bounds/bwd_bounds tile hulls
    ordinary torch.softmax              level-by-level online softmax
    literal sink logit + zero value     out * sigmoid(lse - sinks) rescale
    relative distances arange(M-1..0)   analytic per-tile bin-rank formula
    autograd-only backward              explicit two-pass Phase-2 backward

The only shared code is the frozen integer schedule
(telescope_cache.range_spec: RangeSpec, range_bounds), which DEFINES which
entries each query sees. Nothing here imports from telescope_cache.reference.
Small primitive formulas (softcap c*tanh(x/c), the dyadic softmax merge,
rotate-half RoPE, right-endpoint summary positions (j+1)*2^l - 1, the FP32
conv-residual ordering, GQA mapping hkv = h // (Hq // Hkv)) are duplicated
on purpose.

Scope: full-sequence training semantics only (no decode / incremental
cache). This is a semantic FP32 oracle: all score math runs in FP32 and
exact mixed-precision operation ordering is intentionally NOT part of its
contract -- mixed-precision fidelity remains the responsibility of
reference.py. There is no manual backward anywhere in this file; gradients
come from torch.autograd through the ordinary tensor graph.
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from telescope_cache.range_spec import RangeSpec, range_bounds

__all__ = [
    "minimal_visible_entries",
    "minimal_rope_tables",
    "minimal_multilevel_attention",
    "minimal_attention_block",
]


# ============================================================
# Schedule enumeration
# ============================================================

def minimal_visible_entries(spec: RangeSpec, q: int) -> List[Tuple[int, int]]:
    """
    All entries visible to query q as (level, local_index) pairs, sorted
    chronologically by interval start local_index * 2**level (oldest ->
    newest). The entry (level, j) summarizes original tokens
    [j * 2**level, (j+1) * 2**level). The newest entry is always the
    query's own level-0 token (0, q).
    """
    entries = []
    for level in range(spec.num_levels):
        lo, hi = range_bounds(spec, q, level)
        for j in range(lo, hi):
            entries.append((level, j))
    entries.sort(key=lambda e: e[1] << e[0])
    return entries


def _check_summary_causality(spec: RangeSpec) -> None:
    """Every visible summary must be entirely past-or-current: a[l] >= 2^l - 1.
    (Duplicated from reference.validate_summary_causality on purpose.)"""
    for level in range(1, spec.num_levels):
        if spec.activation_times[level] < (1 << level) - 1:
            raise ValueError(
                f"schedule makes a level-{level} summary visible before its "
                f"interval completes (a[{level}]="
                f"{spec.activation_times[level]} < {(1 << level) - 1})"
            )


# ============================================================
# Small primitives
# ============================================================

def minimal_rope_tables(
    n_pos: int, dim: int, base: float = 10000.0, device=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin), each [n_pos, dim // 2] float32, theta_i = base**(-2i/dim)."""
    inv_freq = base ** (
        -torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
    )
    angles = torch.arange(
        n_pos, device=device, dtype=torch.float32
    )[:, None] * inv_freq[None, :]
    return torch.cos(angles), torch.sin(angles)


def _rope(x, positions, cos, sin):
    """Rotate-half RoPE (GPT-NeoX convention), FP32 compute, cast back.
    positions: integer tensor broadcastable to x.shape[:-1]."""
    D = x.shape[-1]
    c, s = cos[positions], sin[positions]
    x_fp32 = x.float()
    x1, x2 = x_fp32[..., : D // 2], x_fp32[..., D // 2:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)


def _short_conv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Causal depthwise short conv with mandatory residual, as an explicit
    tap loop:  y[b,t,c] = x[b,t,c] + sum_j weight[c,j] * x[b,t-K+1+j,c].
    FP32 taps and FP32 residual add, one cast back at the end.
    """
    B, N, C = x.shape
    K = weight.shape[1]
    x_fp32 = x.float()
    y = torch.zeros_like(x_fp32)
    for j in range(K):
        shift = K - 1 - j  # tap j reads the input `shift` steps in the past
        shifted = F.pad(x_fp32, (0, 0, shift, 0))[:, :N]  # left zero-pad in t
        y = y + weight[:, j].float() * shifted
    return (x_fp32 + y).to(x.dtype)


def _qk_summary_weights(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """w[b,t,hkv] = logsumexp over the KV head's query group of
    q_h . k / sqrt(Dk).  q: [B,N,Hq,Dk], k: [B,N,Hkv,Dk] -> [B,N,Hkv,1]."""
    B, N, Hq, Dk = q.shape
    Hkv = k.shape[2]
    q_grouped = q.reshape(B, N, Hkv, Hq // Hkv, Dk)
    scores = (q_grouped * k.unsqueeze(3)).sum(dim=-1) / math.sqrt(Dk)
    return torch.logsumexp(scores, dim=3, keepdim=True)


def _summary_tree(k, v, w, num_summary_levels):
    """
    Canonical dyadic K/V summary tree. For adjacent children with
    log-weights [w0, w1]: alpha = softmax([w0, w1]), parent K/V is the
    alpha-weighted sum, parent w = logsumexp([w0, w1]). Odd level lengths
    leave the trailing child unpaired. Returns (k_levels, v_levels,
    w_levels), lists of length num_summary_levels + 1.
    """
    k_levels, v_levels, w_levels = [k], [v], [w]
    for _ in range(num_summary_levels):
        k_prev, v_prev, w_prev = k_levels[-1], v_levels[-1], w_levels[-1]
        usable = (k_prev.shape[1] // 2) * 2
        if usable == 0:
            B = k_prev.shape[0]
            k_levels.append(k_prev.new_empty(B, 0, *k_prev.shape[2:]))
            v_levels.append(v_prev.new_empty(B, 0, *v_prev.shape[2:]))
            w_levels.append(w_prev.new_empty(B, 0, *w_prev.shape[2:]))
            continue
        k_parents, v_parents, w_parents = [], [], []
        for j in range(usable // 2):
            w_pair = torch.stack(
                [w_prev[:, 2 * j], w_prev[:, 2 * j + 1]], dim=1
            )  # [B, 2, Hkv, 1] -- the child axis is dim 1
            alpha = torch.softmax(w_pair, dim=1)
            k_parents.append(
                alpha[:, 0] * k_prev[:, 2 * j] + alpha[:, 1] * k_prev[:, 2 * j + 1]
            )
            v_parents.append(
                alpha[:, 0] * v_prev[:, 2 * j] + alpha[:, 1] * v_prev[:, 2 * j + 1]
            )
            w_parents.append(torch.logsumexp(w_pair, dim=1))
        k_levels.append(torch.stack(k_parents, dim=1))
        v_levels.append(torch.stack(v_parents, dim=1))
        w_levels.append(torch.stack(w_parents, dim=1))
    return k_levels, v_levels, w_levels


# ============================================================
# Attention
# ============================================================

def minimal_multilevel_attention(
    q: torch.Tensor,
    k_levels: List[torch.Tensor],
    v_levels: List[torch.Tensor],
    fmap: Dict[int, int],
    cache_size: int,
    *,
    softcap: float = 20.0,
    position_mode: str = "none",
    rope_cos=None,
    rope_sin=None,
    relative_states=None,
    relative_proj=None,
    sinks=None,
):
    """
    Ordinary softmax attention over each query's explicitly enumerated
    visible telescope entries.

    q: [B, N, Hq, Dk]; k_levels/v_levels: dyadic tree lists with
    k_levels[l]: [B, N >> l, Hkv, Dk]. Returns

        out [B, N, Hq, Dv] (V dtype), lse [B, N, Hq] float32.

    `lse` always means logsumexp of the ordinary visible attention scores
    BEFORE the sink logit is appended; enabling `sinks` changes `out` but
    never `lse`.

    Position modes:
      "none":     X = Q.K / sqrt(Dk).
      "rope":     Q rotated at its token position; each summary K rotated
                  at the RIGHT ENDPOINT of its interval, (j+1)*2^l - 1 --
                  a summary is a virtual token located at the newest raw
                  token it contains. This is R(p_s)(sum_i a_i K_i), NOT
                  sum_i a_i R(p_i) K_i. V is never rotated.
      "relative": learned bias in summary-bin space added BEFORE softcap.
                  With the M visible entries sorted oldest -> newest, the
                  bin distances are simply [M-1, ..., 1, 0] -- every entry
                  is one bin regardless of span. Requires
                  relative_states [B, N, Hq, d_rel] and
                  relative_proj [d_rel, max_relative_bins].

    sinks [Hq] (optional): one extra attention logit per query head with an
    identically-zero value row, appended literally before the softmax.
    """
    B, N, Hq, Dk = q.shape
    Hkv = k_levels[0].shape[2]
    Dv = v_levels[0].shape[-1]
    if Hq % Hkv != 0:
        raise ValueError(f"Hq={Hq} must be divisible by Hkv={Hkv}")
    expansion = Hq // Hkv
    spec = RangeSpec.from_fmap(fmap, cache_size, N)
    if len(k_levels) < spec.num_levels or len(v_levels) < spec.num_levels:
        raise ValueError(
            f"need {spec.num_levels} levels, got "
            f"{len(k_levels)}/{len(v_levels)}"
        )

    if position_mode not in ("none", "rope", "relative"):
        raise ValueError(f"unknown position_mode {position_mode!r}")
    rel_logits = None
    if position_mode == "none":
        if any(t is not None for t in (rope_cos, rope_sin,
                                       relative_states, relative_proj)):
            raise ValueError(
                "positional arguments provided with position_mode='none'"
            )
    elif position_mode == "rope":
        if rope_cos is None or rope_sin is None:
            raise ValueError("position_mode='rope' requires rope_cos/rope_sin")
        if relative_states is not None or relative_proj is not None:
            raise ValueError("relative_* arguments provided with rope mode")
        if Dk % 2 != 0 or rope_cos.shape != rope_sin.shape \
                or rope_cos.shape[-1] * 2 != Dk or rope_cos.shape[0] < N:
            raise ValueError("rope tables incompatible with Dk/N")
        _check_summary_causality(spec)
    else:  # "relative"
        if relative_states is None or relative_proj is None:
            raise ValueError(
                "position_mode='relative' requires relative_states and "
                "relative_proj"
            )
        if rope_cos is not None or rope_sin is not None:
            raise ValueError("rope_* arguments provided with relative mode")
        if tuple(relative_states.shape[:3]) != (B, N, Hq) \
                or relative_proj.shape[0] != relative_states.shape[-1]:
            raise ValueError("relative_states/relative_proj shape mismatch")
        _check_summary_causality(spec)
        rel_logits = torch.einsum(
            "bnhd,dr->bnhr", relative_states.float(), relative_proj.float()
        )  # [B, N, Hq, max_relative_bins]
    if sinks is not None and tuple(sinks.shape) != (Hq,):
        raise ValueError(f"sinks shape {tuple(sinks.shape)} != {(Hq,)}")

    out = torch.zeros(B, N, Hq, Dv, device=q.device, dtype=torch.float32)
    plain_lse = torch.zeros(B, N, Hq, device=q.device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(Dk)

    for b in range(B):
        for qi in range(N):
            entries = minimal_visible_entries(spec, qi)
            M = len(entries)
            K = torch.stack([k_levels[l][b, j] for l, j in entries])
            V = torch.stack([v_levels[l][b, j] for l, j in entries])
            # K: [M, Hkv, Dk], V: [M, Hkv, Dv], oldest -> newest.
            q_row = q[b, qi]  # [Hq, Dk]

            if position_mode == "rope":
                k_pos = torch.tensor(
                    [(j + 1) * (1 << l) - 1 for l, j in entries],
                    device=q.device,
                )
                K = _rope(K, k_pos[:, None], rope_cos, rope_sin)
                q_row = _rope(
                    q_row, torch.tensor(qi, device=q.device),
                    rope_cos, rope_sin,
                )
            elif position_mode == "relative":
                if M > relative_proj.shape[1]:
                    raise ValueError(
                        f"query {qi} sees {M} entries > max_relative_bins="
                        f"{relative_proj.shape[1]}"
                    )
                dist = torch.arange(M - 1, -1, -1, device=q.device)

            for h in range(Hq):
                hkv = h // expansion  # GQA: query head -> its KV head
                x = K[:, hkv].float().matmul(q_row[h].float()) * scale  # [M]
                if position_mode == "relative":
                    x = x + rel_logits[b, qi, h, dist]
                s = softcap * torch.tanh(x / softcap) if softcap is not None else x
                plain_lse[b, qi, h] = torch.logsumexp(s, dim=0)
                vals = V[:, hkv].float()
                if sinks is not None:
                    s = torch.cat([s, sinks[h].float().reshape(1)])
                    vals = torch.cat([vals, vals.new_zeros(1, Dv)])
                out[b, qi, h] = torch.softmax(s, dim=0).matmul(vals)

    return out.to(v_levels[0].dtype), plain_lse


# ============================================================
# Full block
# ============================================================

def minimal_attention_block(
    x: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    fmap: Dict[int, int],
    cache_size: int,
    *,
    num_summary_levels: int,
    k_conv_weight=None,
    v_conv_weight=None,
    weight_mode: str = "qk",
    w_proj=None,
    position_mode: str = "none",
    rope_cos=None,
    rope_sin=None,
    relative_weight=None,
    relative_proj=None,
    sinks=None,
    gate_weight=None,
    output_weight=None,
    softcap: float = 20.0,
) -> torch.Tensor:
    """
    The complete telescope attention block, starting AFTER the Q/K/V
    projections: callers that need projection gradients build q = x@Wq^T
    etc. immediately outside this function.

        (x, q, k, v)
          -> optional short causal depthwise conv on K/V   (flat channels
             c = h*D + d; QK summary weights use the CONVOLVED K)
          -> summary weights ("qk": logsumexp(q.k_conv);
                              "linear": x @ w_proj^T from the ORIGINAL x)
          -> dyadic summary tree
          -> minimal_multilevel_attention (positional mode + literal sink)
          -> optional SiLU(x @ gate_weight^T) * out        (ORIGINAL x)
          -> optional out @ output_weight^T

    Returns [B, N, Hq, Dv], or [B, N, emb] when output_weight is given.
    """
    B, N, Hq, Dk = q.shape
    Hkv = k.shape[2]
    Dv = v.shape[-1]

    if (k_conv_weight is None) != (v_conv_weight is None):
        raise ValueError("k_conv_weight/v_conv_weight: both or neither")
    if k_conv_weight is not None:
        k = _short_conv(
            k.reshape(B, N, Hkv * Dk), k_conv_weight
        ).reshape(B, N, Hkv, Dk)
        v = _short_conv(
            v.reshape(B, N, Hkv * Dv), v_conv_weight
        ).reshape(B, N, Hkv, Dv)

    if weight_mode == "qk":
        if w_proj is not None:
            raise ValueError("w_proj provided with weight_mode='qk'")
        w = _qk_summary_weights(q, k)
    elif weight_mode == "linear":
        if w_proj is None:
            raise ValueError("weight_mode='linear' requires w_proj")
        w = x.matmul(w_proj.t()).unsqueeze(-1)
    else:
        raise ValueError(f"unknown weight_mode {weight_mode!r}")

    k_levels, v_levels, _ = _summary_tree(k, v, w, num_summary_levels)

    relative_states = None
    if position_mode == "relative":
        if relative_weight is None or relative_proj is None:
            raise ValueError(
                "position_mode='relative' requires relative_weight and "
                "relative_proj"
            )
        d_rel = relative_weight.shape[0] // Hq
        relative_states = x.matmul(
            relative_weight.t()
        ).reshape(B, N, Hq, d_rel)

    out, _ = minimal_multilevel_attention(
        q, k_levels, v_levels, fmap, cache_size,
        softcap=softcap, position_mode=position_mode,
        rope_cos=rope_cos, rope_sin=rope_sin,
        relative_states=relative_states, relative_proj=relative_proj,
        sinks=sinks,
    )

    if gate_weight is not None:
        gate = F.silu(x.matmul(gate_weight.t()))
        out = gate.reshape(B, N, Hq, Dv) * out
    if output_weight is not None:
        out = out.reshape(B, N, Hq * Dv).matmul(output_weight.t())
    return out
