"""Summary-tree construction: merge weights, short conv, dyadic builder."""

import math
from typing import Dict, List, Optional, Tuple

import torch

from telescope_cache.telescoping_attn.range_spec import RangeSpec

def compute_summary_weights(
    q: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    q: [B, N, Hq, Dk]  k: [B, N, Hkv, Dk]  -> w: [B, N, Hkv, 1]
    logsumexp_h(q_h^T k / sqrt(Dk)). Phase-1 MERGE weight, not the attention LSE.
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

def compute_linear_weights(
    x: torch.Tensor,
    w_proj: torch.Tensor,
) -> torch.Tensor:
    """
    x: [B, N, emb_dim]  w_proj: [Hkv, emb_dim]  -> w: [B, N, Hkv, 1] = x @ w_proj^T
    """
    B, N, E = x.shape
    Hkv, Ew = w_proj.shape

    assert E == Ew

    return x.matmul(w_proj.t()).unsqueeze(-1)

def short_conv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    x: [B, N, C]  weight: [C, K]  -> [B, N, C], causal depthwise + residual.

    The residual is ADDED IN FP32 and only the sum is cast back -- contract, not
    x + conv(x).to(dtype).

    TODO(padding/packing): assumes one continuous sequence per batch row.
    """
    B, N, C = x.shape
    if weight.dim() != 2 or weight.shape[0] != C:
        raise ValueError(
            f"conv weight shape {tuple(weight.shape)} incompatible with "
            f"C={C}; expected [C, K]"
        )
    K = weight.shape[1]
    if K <= 0:
        raise ValueError(f"conv kernel size must be positive, got {K}")

    input_dtype = x.dtype
    x_fp32 = x.float()
    # conv1d is cross-correlation; padding=K-1 plus the [:N] crop gives the
    # causal sum above, no kernel flip.
    y = torch.nn.functional.conv1d(
        x_fp32.transpose(1, 2),
        weight.float().unsqueeze(1),
        padding=K - 1,
        groups=C,
    )[:, :, :N].transpose(1, 2)
    return (x_fp32 + y).to(input_dtype)

def build_dyadic_summaries(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_summary_levels: int,
    detach_weights: bool = False,
    x: Optional[torch.Tensor] = None,
    w_proj: Optional[torch.Tensor] = None,
    k_conv_weight: Optional[torch.Tensor] = None,
    v_conv_weight: Optional[torch.Tensor] = None,
):
    """
    -> (k_levels, v_levels, w_levels), length num_summary_levels + 1, with
    k_levels[l]: [B, N >> l, Hkv, Dk]. Only complete aligned pairs merge.

    Merge weights from q/k, or from x @ w_proj^T if both are given. The conv
    weights apply short_conv to k/v (flattened, c = h*D + d) BEFORE the weights
    and the tree. Each pair is both-or-neither. detach_weights cuts the gradient
    into the merge weights only.
    """
    if (k_conv_weight is None) != (v_conv_weight is None):
        raise ValueError(
            "k_conv_weight and v_conv_weight must both be provided or both "
            "be None"
        )
    if k_conv_weight is not None:
        B, N, Hkv, Dk = k.shape
        Dv = v.shape[-1]
        k = short_conv(
            k.reshape(B, N, Hkv * Dk), k_conv_weight
        ).reshape(B, N, Hkv, Dk)
        v = short_conv(
            v.reshape(B, N, Hkv * Dv), v_conv_weight
        ).reshape(B, N, Hkv, Dv)

    if (x is None) != (w_proj is None):
        raise ValueError(
            "x and w_proj must both be provided (linear weight mode) or "
            "both be None (QK weight mode)"
        )
    if x is not None:
        w = compute_linear_weights(x, w_proj)
        expected = (k.shape[0], k.shape[1], k.shape[2], 1)
        if tuple(w.shape) != expected:
            raise ValueError(
                f"linear weights shape {tuple(w.shape)} != {expected}"
            )
    else:
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
