import abc
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import nn

class QKV(nn.Module, metaclass=abc.ABCMeta):
    """Simple module for applying qkv in attention"""

    def __init__(
        self,
        emb_dim: int,
        nheads: int,
        kvheads: int,
        emb_kq_per_head: int,
        emb_v_per_head: int,
        use_bias: bool,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.emb_dim = emb_dim
        self.nheads = nheads
        self.kvheads = kvheads
        self.emb_kq_per_head = emb_kq_per_head
        self.emb_v_per_head = emb_v_per_head
        self.use_bias = use_bias

    @abc.abstractmethod
    def forward(
        self, q: torch.Tensor, k: Optional[torch.Tensor], v: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """applies query/key/value transformations on q, k, v inputs respectively and returns the resulting values

        Args:
            q: torch.Tensor
                the query tensor
            k: Optional[torch.Tensor]
                the optional key tensor
            v: Optional[torch.Tensor]
                the optional value tensor

        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            the query, key, and value computed
        """
        pass

    @abc.abstractmethod
    def reset_parameters(self):
        """resets the query, key, and value weights for training

        Args:
            gain: int
                gain for std in norm (default is 1)
        """
        pass


class UnfusedQKV(QKV):
    """
    Unfused Weights implementation of QKV
    """

    def __init__(
        self,
        emb_dim: int,
        nheads: int,
        kvheads: int,
        emb_kq_per_head: int,
        emb_v_per_head: int,
        use_bias: bool,
        *args,
        **kwargs,
    ):
        super().__init__(
            emb_dim,
            nheads,
            kvheads,
            emb_kq_per_head,
            emb_v_per_head,
            use_bias,
            *args,
            **kwargs,
        )
        self.query = nn.Linear(
            self.emb_dim, self.nheads * self.emb_kq_per_head, bias=use_bias
        )
        self.key = nn.Linear(
            self.emb_dim, self.kvheads * self.emb_kq_per_head, bias=use_bias
        )
        self.value = nn.Linear(
            self.emb_dim, self.kvheads * self.emb_v_per_head, bias=use_bias
        )

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if self.use_bias:
                    m.bias.data.zero_()

    def forward(
        self, q: torch.Tensor, k: Optional[torch.Tensor], v: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if k is None and v is None:
            k = q
            v = q
        elif k is None or v is None:
            raise ValueError(
                "both k and v must either be given as tensors or both None"
            )

        # b x h x qlen x ds
        queries = self.query(q)
        keys = self.key(k)
        values = self.value(v)
        return queries, keys, values


class FusedQKV(QKV):
    """
    Fused Weights implementation of QKV
    """

    def __init__(
        self,
        emb_dim: int,
        nheads: int,
        kvheads: int,
        emb_kq_per_head: int,
        emb_v_per_head: int,
        use_bias: bool,
        *args,
        **kwargs,
    ):
        super().__init__(
            emb_dim,
            nheads,
            kvheads,
            emb_kq_per_head,
            emb_v_per_head,
            use_bias,
            *args,
            **kwargs,
        )
        self.splits = [
            self.nheads * self.emb_kq_per_head,
            self.kvheads * self.emb_kq_per_head,
            self.kvheads * self.emb_v_per_head,
        ]

        self.qkv_fused = nn.Linear(
            self.emb_dim,
            sum(self.splits),
            bias=self.use_bias,
        )

    def reset_parameters(self):
        nn.init.trunc_normal_(self.qkv_fused.weight, mean=0.0, std=0.02)
        if self.use_bias:
            self.qkv_fused.bias.data.zero_()

    def forward(
        self, q: torch.Tensor, k: Optional[torch.Tensor], v: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (k is None and v is None) or (k is q and v is q):
            qkv = q
        else:
            raise ValueError("q, k, and v must be the same or k and v must be None")
        return self.qkv_fused(qkv).split(self.splits, dim=-1)

class LayerNormParameterized(nn.Module):
    """
    A generalized LayerNorm implementation. With all optional arguments set to True, equivalent to nn.LayerNorm up to epsilon stabilization term
    (this class divides inputs by min(norm, eps), while nn.LayerNorm divides by norm + eps).
    ...
    Args
    ----
    normalized_shape : int
        Dimensionality of input data (size of final tensor axis)
    eps : float
        Safety term to prevent division by zero. Make sure the chosen value fits in the range of your encoding scheme (i.e. fp16 requires eps >= 6e-8).
    elementwise_scale : bool
        Include a learned scaling term after normalization?
    elementwise_shift : bool
        Include a learned bias term after normalization?
    use_mean : bool
        Recenter inputs around zero before normalizing, or just rescale?
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-06,
        elementwise_scale=True,
        elementwise_shift=False,
        use_mean=False,
        use_high_precision_pow=False,
    ):
        super(LayerNormParameterized, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_scale = elementwise_scale
        self.elementwise_shift = elementwise_shift
        self.use_mean = use_mean
        self.use_high_precision_pow = use_high_precision_pow

        if self.elementwise_scale:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        # else:
        #     self.register_parameter("weight", None)
        if self.elementwise_shift:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        # else:
        #     self.register_parameter("bias", None)

    def reset_parameters(self):
        if self.elementwise_scale:
            self.weight.data.fill_(1)
        if self.elementwise_shift:
            self.bias.data.zero_()

    def forward(self, x):
        if self.use_mean:
            x = x - x.mean(-1, keepdim=True)
        # x = F.normalize(x, dim=-1)*math.sqrt(x.size(-1))
        xf = x
        if self.use_high_precision_pow:
            xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        x = xf.type_as(x)
        if self.elementwise_scale:
            x = self.weight * x
        if self.elementwise_shift:
            x = x + self.bias
        return x


class ShortConv1d(nn.Module):
    """
    Depthwise causal short convolution with a mandatory residual over
    [B, N, C]:  out = x + causal_depthwise_conv(x).

    Per channel c:  y[t, c] = sum_j weight[c, j] * x[t-K+1+j, c] -- only the
    current and past tokens contribute; no mixing across channels (groups=C),
    no bias, no activation. The input is cast to FP32, convolved, the
    residual is ADDED IN FP32, and only the final sum is cast back to the
    input dtype (same contract as telescope_cache.reference.short_conv, the
    independent oracle; equivalence is enforced by test/test_shortconv.py,
    not by shared code).

    TODO(decode): incremental decoding requires a rolling state of
    pre-convolution projected K/V values. Current implementation supports
    full-sequence training/prefill only. Missing pieces: persistent
    per-layer conv state (last K-1 pre-conv K/V rows), a one-token update
    path, prefill->decode state initialization, state reset/reorder on
    cache reset/beam reorder, interaction with the existing KV cache, and
    prefill-vs-incremental-decode equivalence tests.
    NOTE: the attention KV cache holds POST-conv K/V; the conv state is
    PRE-conv K/V. They are different objects -- do not conflate them.

    TODO(padding/packing): current implementation assumes one continuous
    causal sequence per batch row. It does not reset convolution state
    across padding or packed-example boundaries (would need seq_idx/segment
    boundaries).
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(
                f"kernel_size must be positive, got {kernel_size}"
            )
        self.channels = channels
        self.kernel_size = kernel_size
        # Zero initialization is intentional so enabling short-conv starts
        # exactly from the original attention model (the residual makes
        # W=0 an identity). Design choice for finetuning an existing stack,
        # not part of the Conv1D definition.
        self.weight = nn.Parameter(torch.zeros(channels, kernel_size))

    def reset_parameters(self):
        self.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        B, N, C = x.shape
        input_dtype = x.dtype
        x_fp32 = x.float()
        y = nn.functional.conv1d(
            x_fp32.transpose(1, 2),
            self.weight.float().unsqueeze(1),
            padding=self.kernel_size - 1,
            groups=C,
        )[:, :, :N].transpose(1, 2)
        return (x_fp32 + y).to(input_dtype)


class PositionEncoder:
    """
    Provides the ability to insert position-encoding logic into MHA.
    """

    # Override to adjust the mask e.g. for Alibi
    def adjusted_mask(
        self,
        mask: Optional[torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        past_kv_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        use_cache=False,
    ) -> Optional[torch.Tensor]:
        return mask

    # Override to adjust q/k's e.g. for rotary embeddings
    def adjusted_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        past_kv_state: Optional[Tuple[torch.Tensor | None, torch.Tensor | None]],
        use_cache=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return q, k