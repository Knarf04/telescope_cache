"""Learned attention sink and the output gate."""

from typing import Optional, Tuple

import torch

def apply_attention_sink(
    out: torch.Tensor,
    lse: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    """
    out: [B, N, Hq, Dv]  lse: [B, N, Hq] fp32  sinks: [Hq] per QUERY head
    -> out * sigmoid(lse - sinks[h]). A zero sink is NOT an identity.

    TODO(kernel): fuse into the online softmax and stop exposing lse.
    """
    B, N, Hq, Dv = out.shape
    if tuple(lse.shape) != (B, N, Hq):
        raise ValueError(
            f"lse shape {tuple(lse.shape)} != {(B, N, Hq)}"
        )
    if tuple(sinks.shape) != (Hq,):
        raise ValueError(
            f"sinks shape {tuple(sinks.shape)} != {(Hq,)} (one logit per "
            f"query head)"
        )
    scale = torch.sigmoid(lse.float() - sinks.float()[None, None, :])
    return out * scale.unsqueeze(-1).to(out.dtype)

def attention_sink_backward(
    out: torch.Tensor,
    lse: torch.Tensor,
    sinks: torch.Tensor,
    dout: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VJP of apply_attention_sink; dout is wrt the POST-sink out.
    -> (dout_pre [B, N, Hq, Dv], dlse [B, N, Hq] fp32, dsinks [Hq])
    Requires dout.dtype == out.dtype -- it mirrors the forward's mixed graph.
    """
    B, N, Hq, Dv = out.shape
    if tuple(lse.shape) != (B, N, Hq):
        raise ValueError(f"lse shape {tuple(lse.shape)} != {(B, N, Hq)}")
    if tuple(sinks.shape) != (Hq,):
        raise ValueError(
            f"sinks shape {tuple(sinks.shape)} != {(Hq,)} (one logit per "
            f"query head)"
        )
    if tuple(dout.shape) != (B, N, Hq, Dv):
        raise ValueError(
            f"dout shape {tuple(dout.shape)} != {(B, N, Hq, Dv)}"
        )
    if dout.dtype != out.dtype:
        raise ValueError(
            f"dout dtype {dout.dtype} must match out dtype {out.dtype}"
        )
    r = torch.sigmoid(lse.float() - sinks.float()[None, None, :])  # fp32
    r_out = r.to(out.dtype)                     # the forward's exact cast
    dout_pre = dout * r_out.unsqueeze(-1)       # same-dtype math as forward
    # grad wrt r_out is <dout, out> summed over Dv in the output dtype; the
    # cast-backward promotes to fp32.
    g_dot = (dout * out).sum(dim=-1).float()
    dlse = g_dot * r * (1.0 - r)                # sigmoid backward, fp32
    dsinks = (-dlse).sum(dim=(0, 1)).to(sinks.dtype)
    return dout_pre, dlse, dsinks

def apply_output_gate(
    out: torch.Tensor,
    x: torch.Tensor,
    gate_weight: torch.Tensor,
) -> torch.Tensor:
    """
    out: [B, N, Hq, Dv]  x: [B, N, emb_dim] ORIGINAL pre-projection hidden state
    gate_weight: [Hq*Dv, emb_dim]  -> SiLU(x @ gate_weight^T) * out
    """
    B, N, Hq, Dv = out.shape
    if x.dim() != 3 or x.shape[0] != B or x.shape[1] != N:
        raise ValueError(
            f"x shape {tuple(x.shape)} incompatible with out "
            f"{tuple(out.shape)}; expected [B, N, emb_dim]"
        )
    if tuple(gate_weight.shape) != (Hq * Dv, x.shape[-1]):
        raise ValueError(
            f"gate_weight shape {tuple(gate_weight.shape)} != "
            f"{(Hq * Dv, x.shape[-1])}"
        )
    gate = torch.nn.functional.silu(x.matmul(gate_weight.t()))
    return gate.reshape(B, N, Hq, Dv) * out
