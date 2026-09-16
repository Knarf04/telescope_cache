"""
Plumbing shared by flex_tree's prefill and decode paths: the positional
contract check, flex's head-dim minimum, and the GQA entry point.
"""

from typing import Dict, Optional, Tuple

import torch
from torch._inductor.exc import InductorError
from torch.nn.attention.flex_attention import flex_attention

from telescope_cache.telescoping_attn.range_spec import IntOps

# torch's flex_attention lowering rejects head dims below 16:
#   "NYI: embedding dimension of the query, key, and value must be at least 16"
FLEX_MIN_HEADDIM = 16

# Every schedule is a fresh shape under dynamic=False; past dynamo's default
# limit of 8 it falls back to an unfused eager path that OOMs at large N.
torch._dynamo.config.recompile_limit = 64

compiled_flex_attention = torch.compile(flex_attention, dynamic=False)

# enable_gqa lets the kernel broadcast Hkv heads over the query heads instead
# of materializing the expansion: bit-identical, and `expansion` times less KV.
_GQA_FALLBACK_BLOCK_M = 64
_gqa_needs_pin: Dict[tuple, bool] = {}


def _gqa_key(q, k, v):
    return (q.shape[1], q.shape[2], q.shape[3],
            k.shape[1], k.shape[2], v.shape[3], q.dtype)


def flex_attention_gqa(q, k, v, **kwargs):
    """
    flex_attention with native GQA: q may carry more heads than k/v.

    Falls back to a pinned BLOCK_M where enable_gqa leaves inductor no autotune
    choice. The failing window depends on (Q_LEN, head dim, ratio), so it is
    discovered per shape; pinning globally costs 3x on the Q_LEN=1 decode shape.
    """
    pinned = {"kernel_options": {"BLOCK_M": _GQA_FALLBACK_BLOCK_M}}
    key = _gqa_key(q, k, v)
    if _gqa_needs_pin.get(key):
        return compiled_flex_attention(q, k, v, enable_gqa=True, **pinned, **kwargs)
    try:
        return compiled_flex_attention(q, k, v, enable_gqa=True, **kwargs)
    except InductorError as exc:
        if "NoValidChoicesError" not in str(exc):
            raise
        _gqa_needs_pin[key] = True
        return compiled_flex_attention(q, k, v, enable_gqa=True, **pinned, **kwargs)


def _mx(a, b):
    """max() for the IntOps protocol, tolerant of a python-int operand."""
    if isinstance(b, int):
        return torch.clamp(a, min=b)
    return torch.maximum(a, b)


def _mn(a, b):
    if isinstance(b, int):
        return torch.clamp(a, max=b)
    return torch.minimum(a, b)


# The torch binding of range_spec's IntOps: every `max` in those closed forms
# is against a literal 0, so clamp works.
TORCH_OPS = IntOps(minimum=_mn, maximum=_mx)


def validate_position_args(
    position_mode: str,
    rope_cos: Optional[torch.Tensor],
    rope_sin: Optional[torch.Tensor],
    relative_states: Optional[torch.Tensor],
    relative_proj: Optional[torch.Tensor],
) -> None:
    """
    Raises unless the tables for `position_mode` are present and those for the
    other modes absent.
    """
    if position_mode not in ("none", "rope", "relative"):
        raise ValueError(f"unknown position_mode {position_mode!r}")
    if position_mode == "rope" and (rope_cos is None or rope_sin is None):
        raise ValueError("position_mode='rope' needs rope_cos and rope_sin")
    if position_mode != "rope" and (rope_cos is not None
                                    or rope_sin is not None):
        raise ValueError("rope tables given but position_mode is not 'rope'")
    if position_mode == "relative" and (relative_states is None
                                        or relative_proj is None):
        raise ValueError(
            "position_mode='relative' needs relative_states and relative_proj")
    if position_mode != "relative" and (relative_states is not None
                                        or relative_proj is not None):
        raise ValueError(
            "relative tensors given but position_mode is not 'relative'")


def pad_head_dims(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, Dk: int, Dv: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """
    Zero-pad the feature dim to flex's minimum. -> (q, k, v, dk_pad, dv_pad).

    Exact: padded lanes add 0 to every q.k. Callers must pin the scale to the
    TRUE Dk, and apply positional encoding BEFORE this -- rotate-half splits at
    D/2 and would mix real lanes with pad lanes.
    """
    dk_pad = max(0, FLEX_MIN_HEADDIM - Dk)
    dv_pad = max(0, FLEX_MIN_HEADDIM - Dv)
    if dk_pad:
        q = torch.nn.functional.pad(q, (0, dk_pad))
        k = torch.nn.functional.pad(k, (0, dk_pad))
    if dv_pad:
        v = torch.nn.functional.pad(v, (0, dv_pad))
    return q, k, v, dk_pad, dv_pad
