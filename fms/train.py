from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from telescope_cache.fms.fms_template import *

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
flex_attention = torch.compile(flex_attention, dynamic=False)

import functools

def get_scan_plan(x, fmap, h):
    # x: b n d
    # plan: for each level, which entries to avg from previous level ([l] n' 2)
    # inds: which level and entry to pull from in populating heads (n h 2) -> (n h)
    b, n, d = x.size()
    # Form ruler-tick progression sequence
    levels = sum(
        [
            torch.arange(n, device=x.device)
            .remainder(2**i)
            .sub(2**i - 1)
            .sign()
            .add(1)
            for i in range(n.bit_length())
        ]
    ).roll(1, 0)
    plan = [
        torch.zeros(0, 2, device=x.device, dtype=torch.int)
        for _ in range(len(fmap) + 2)
    ]  # [l] 0 2
    plan[1] = (
        torch.arange(x.size(1) + 1, device=x.device, dtype=torch.int)
        .unsqueeze(1)
        .expand(-1, 2)
    )
    inds = torch.zeros(n, h, 2, device=x.device, dtype=torch.long)  # n h 2
    inds[:, 0, 1] = torch.arange(n, device=inds.device, dtype=inds.dtype) + 1
    inds[:, :, 0] = 1
    for i in range(1, n):
        m = fmap.get(levels[i].item(), h)
        inds[i, 1:m] = inds[i - 1, : m - 1]
        if m < h:
            inds[i, m + 1 :] = inds[i - 1, m + 1 :]
            prev = inds[i - 1, m - 1 : m + 1].flip([0])  # 2 2
            assert prev[0, 0] == min(levels[i], len(fmap) + 1) or prev[0, 1] == 0, (
                levels[i],
                prev[0, 0],
            )
            assert prev[1, 0] == min(levels[i], len(fmap) + 1) or prev[1, 1] == 0, (
                levels[i],
                prev[1, 0],
            )
            level = plan[levels[i] + 1]
            inds[i, m, 0] = levels[i] + 1
            inds[i, m, 1] = level.size(0)
            plan[levels[i] + 1] = torch.cat(
                [plan[levels[i] + 1], prev[:, 1][None]], dim=0
            )
    # Flatten inds (indexing into flattened plan/cache) (n h)
    ls = [p.size(0) for p in plan]
    ls = [0] + ls[:-1]
    offset = torch.tensor(ls, device=inds.device).cumsum(0)
    offset = offset[inds[:, :, 0]]
    inds = offset + inds[:, :, 1]
    return plan + [inds]


class MultiHeadAttention(nn.Module):
    """
    Performs multi-headed self- or cross-attention, with optional attention masking.
    ...
    Args
    ----
    emb_dim : int
        Latent dimensionality of input and output tensors.
    emb_kq : int
        Latent dimensionality of each head in key and query projections (attention dimension).
    emb_v : int
        Latent dimensionality of each head in value projection (mixing dimension).
    nheads : int
        Number of attention heads.
    p_dropout : float|None
        Dropout probability. Must be in range [0,1]. If 0 or None, dropout will not be used.
    use_bias : bool
        Include bias terms in fully-connected sublayers?
    fused: bool
        if True, qkv weights will be fused, otherwise qkv weights will be unfused
    """

    def __init__(
        self,
        emb_dim,
        emb_kq,
        emb_v,
        nheads,
        kvheads,
        p_dropout=None,
        use_bias=False,
        position_encoder: Optional[PositionEncoder] = None,
        fused: bool = True,
    ):
        super(MultiHeadAttention, self).__init__()
        self.nheads = nheads
        self.kvheads = kvheads
        self.emb_dim = emb_dim
        self.emb_kq_per_head = emb_kq
        self.emb_v_per_head = emb_v
        self.p_dropout = p_dropout if p_dropout is not None else 0.0
        self.use_bias = use_bias
        self.fused = fused

        self.in_proj: QKV = (FusedQKV if self.fused else UnfusedQKV)(
            self.emb_dim,
            self.nheads,
            self.kvheads,
            self.emb_kq_per_head,
            self.emb_v_per_head,
            self.use_bias,
        )

        self.dense = nn.Linear(
            self.nheads * self.emb_v_per_head, self.emb_dim, bias=use_bias
        )
        if self.p_dropout:
            self.attn_dropout = nn.Dropout(self.p_dropout)
        self.position_encoder = position_encoder
        # self.ln_k = LayerNormParameterized(
        #     emb_kq, use_high_precision_pow=True
        # )
        # self.ln_v = LayerNormParameterized(emb_v, use_high_precision_pow=True)

        self.inp_len = 0
        self.cache_len = 0
        self.plan = None

        # fmap = {8 - i: 64 - (i) ** 2 for i in range(8)}
        # fmap.pop(8)
        # fmap.pop(7)
        # fmap = {
        #     1:7,
        #     2:13,
        #     3:18,
        #     4:22,
        #     5:25,
        #     6:27,
        #     7:29
        # }
        # fmap = {
        #     1:26,
        #     2:50,
        #     3:71,
        #     4:89,
        #     5:104,
        #     6:116,
        #     7:124
        # }
        # fmap = {
        #     1:16,
        #     2:30,
        #     3:44,
        #     4:58,
        #     5:72,
        #     6:86,
        #     7:100,
        #     8:114,
        # }
        # fmap = {1: 64, 2: 118, 3: 162, 4: 194, 5: 214, 6: 226, 7: 236, 8: 244, 9: 250}
        fmap = {1: 64, 2: 72, 3:80} #, 4:88}
        self.fmap = fmap
        self.cache_size = 512 #256

        self.weight_mode = "qk"  # "qk" | "linear" | None (unweighted scan)
        if self.weight_mode == "linear":
            self.w = nn.Linear(self.emb_dim, self.kvheads, bias=False)

        self.mask = None

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if self.use_bias:
                    m.bias.data.zero_()
            elif isinstance(m, LayerNormParameterized) or isinstance(m, QKV):
                m.reset_parameters()

    # def to_tp(self, group: ProcessGroup) -> "TPMultiHeadAttention":
    #     return TPMultiHeadAttention.import_module(self, group)

    def scan(self, x, plan, w=None):
        """
        Takes input x of shape [b n ...] and scan plan, computes recursive sums, and
        extracts cache values into the dimension specified by i > 1.
        Final output shape is therefore [b n ... c ...] with c the cache size.
        Applies specified LN to cache, so LN size should match x.size(-1).
        """
        s = x.size()
        weighted = w is not None
        if weighted:
            ws = w.size()
            weights = nn.functional.pad(w.view(s[0], s[1], -1), (0,0,1,0), value=-1000).view(
                s[0], s[1] + 1, *ws[2:]
            )
        plan = plan[:-1]
        # Plan and inds are formed, construct cache via recursive sums
        cache = [None for _ in plan]
        cache[1] = nn.functional.pad(x.view(s[0], s[1], -1), (0, 0, 1, 0)).view(
            s[0], s[1] + 1, *s[2:]
        )  # b n ...
        for j in range(2, len(cache)):
            if weighted:
                weights = weights.index_select(1, plan[j].view(-1)).view(s[0], -1, 2, *ws[2:])
                weights_ = weights.softmax(dim=2)
                weights = weights.logsumexp(2)
            cache[j] = (
                cache[j - 1]
                .index_select(1, plan[j].view(-1))
                .view(s[0], -1, 2, *s[2:])
            )
            if weighted:
                cache[j] = cache[j].mul(weights_).sum(2)
            else:
                cache[j] = cache[j].sum(2).div(2**0.5)
            
        cache = torch.cat(cache[1:], dim=1)  # b n' ...
        return cache
        # cache = cache.unsqueeze(i).expand(
        #     *[-1] * i, inds.size(-1), *[-1] * (len(s) - i)
        # )  # b n' ... h ...
        # inds = inds.view(
        #     1, inds.size(0), *[1] * (i - 2), inds.size(1), *[1] * (len(s) - i)
        # )  # b n 111 h 111
        # inds = inds.expand(s[0], -1, *s[2:i], -1, *s[i:])  # b n ... h ...
        # return cache.gather(1, inds)  # b n ... h ...

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        mask: Optional[Tensor] = None,
        position_ids=None,
        attn_algorithm=None,
        past_key_value_state: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache=False,
        is_self=True,
        is_causal_mask=False,
    ):
        """
        past_key_value_state: tuple
            the cache to be used in attention of the form (<self/cross>_key, <self/cross>_value)
        position_ids: Optional[torch.LongTensor]
            The position of each of the tokens encoded in q and k. Used for RoPE embeddings
        use_cache: bool
            if True, the kv states for self/cross attention will be saved, otherwise they will not be saved
        is_self: bool
            if True, this will perform self attention, otherwise this will perform cross attention. Note: This will
            only be used in the case that use_cache=True. This may be removed in future

        Returns
        -------
        tensor or tuple
            If use_cache=False, only the hidden state will be returned as a tensor. If use_cache=True, a tuple will be
            returned in the form (hidden_state, cache) where hidden_state is a tensor and cache is of the form specified
            in past_key_value_state
        """
        # q, k, v: batch_size x seq_len x emb_dim
        # mask: batch_size x seq_len x seq_len
        batch_size, q_len, _ = q.size()
        if is_self:
            k = q
            v = q
        kv_len = k.size(1)

        # if kv_len mismatch, build new scan plan
        if kv_len != self.inp_len:
            self.inp_len = kv_len
            with torch.no_grad():
                self.plan = get_scan_plan(k, self.fmap, self.cache_size)

        # if this is self attention, we always recompute
        # cross attention only gets computed when a cache does not exist
        # if we dont have the cache yet, we need to compute
        # d x (h x ds)
        # b x kvlen x d
        # b x kvlen x h x ds
        # b x h x kvlen x ds
        # todo: Cross attention (This always is true for now)
        if is_self or past_key_value_state is None:
            q_out, k_out, v_out = self.in_proj(q, k, v)

            # note: transposes will be moved in a later PR to fix dis-contiguous tensor issues
            queries = q_out.view(batch_size, q_len, self.nheads, self.emb_kq_per_head)
            keys = k_out.view(batch_size, q_len, self.kvheads, self.emb_kq_per_head)
            values = v_out.view(batch_size, q_len, self.kvheads, self.emb_v_per_head)
            # sink = queries.sum(3)  # b l he

            # You want to apply rotary embeddings pre-cache
            if self.position_encoder is not None:
                queries, keys = self.position_encoder.adjusted_qk(
                    queries, keys, position_ids, past_key_value_state, use_cache
                )

        # Build telescoping cache
        # k/v: b l h d
        expansion = self.nheads // self.kvheads
        queries = queries.unflatten(2, (self.kvheads, expansion))  # b l h e d
        w = None
        if self.weight_mode == "linear":
            # `k` is the PRE-projection hidden state [b, l, emb_dim]
            # (rebound to the raw q input above when is_self=True).
            # TODO: fms/inference.py still derives w from QK logits; a model
            # trained with linear weights mismatches at inference until it
            # grows the same mode.
            w = self.w(k).unsqueeze(-1)  # b l h 1
        elif self.weight_mode == "qk":
            w = queries.div(self.emb_kq_per_head**0.5).matmul(keys.unsqueeze(-1)).squeeze(-1).logsumexp(-1, True)  # b l h 1

        keys = self.scan(keys, self.plan, w)  # b n h d
        values = self.scan(values, self.plan, w)  # b n h d

        # if you built a new scan plan, invert the plan for use by backward kernels
        if keys.size(1) != self.cache_len:
            self.cache_len = keys.size(1)
            self.mask = torch.zeros(q_len, self.cache_len, device=q.device, dtype=torch.bool)
            with torch.no_grad():
                self.mask.scatter_(1, self.plan[-1], True)
                # zero out zero entries
                flags = torch.ones(1, q_len, device=q.device)
                flags = self.scan(flags, self.plan, flags)
                flags = flags[0].bool().logical_not()
                self.mask[:,flags] = False

        # q/k/v: b n h d
        # Expand kv so black-box attn will work
        if expansion != 1:
            keys_e = keys.transpose(1,2).unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
            values_e = (
                values.transpose(1,2).unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
            )
        else:
            keys_e = keys.transpose(1,2)
            values_e = values.transpose(1,2)
        queries = queries.view(batch_size, q_len, self.nheads, self.emb_kq_per_head).transpose(1,2)

        # q/k/v: b h n d
        def mask_index(mask, b, h, q_i, k_i):
            return mask[q_i.clamp(min=0, max=mask.size(0)-1), k_i.clamp(min=0, max=mask.size(1)-1)]
        block_mask = create_block_mask(functools.partial(mask_index, self.mask), 1, 1, q_len, self.cache_len)
        def soft_cap(score, b, h, q_i, kv_i):
            return 20 * score.div(20).tanh()
        attention = functools.partial(flex_attention, block_mask=block_mask, score_mod=soft_cap)
        attn = attention(queries, keys_e, values_e)
        attn = attn.transpose(1,2)  # b l h d
        attn = attn.reshape(batch_size, q_len, self.nheads * self.emb_v_per_head)
        out = self.dense(attn)

        # if use_cache=True, we return the hidden_state as well as the kv cache
        if use_cache:
            return out, (keys, values)
        else:
            return out
