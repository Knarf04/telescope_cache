from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from telescope_cache.fms.fms_template import *

def get_scan_plan(device, n, fmap, h):
    # x: b n d
    # plan: for each level, which entries to avg from previous level ([l] n' 2)
    # inds: which level and entry to pull from in populating heads (n h 2)

    # Form ruler-tick progression sequence
    levels = sum(
        [
            torch.arange(n, device=device)
            .remainder(2**i)
            .sub(2**i - 1)
            .sign()
            .add(1)
            for i in range(n.bit_length())
        ]
    ).roll(1, 0)
    plan = [
        torch.zeros(0, 2, device=device, dtype=torch.int)
        for _ in range(len(fmap) + 2)
    ]  # [l] 0 2
    plan[1] = (
        torch.arange(n + 1, device=device, dtype=torch.int)
        .unsqueeze(1)
        .expand(-1, 2)
    )
    inds = torch.zeros(n, h, 2, device=device, dtype=torch.long)  # n h 2
    inds[:, 0, 1] = torch.arange(n, device=inds.device, dtype=inds.dtype) + 1
    inds[:, :, 0] = 1
    for i in range(1, n):
        ran = levels[i].item()
        m = fmap.get(ran, h)
        inds[i, 1:m] = inds[i - 1, : m - 1]
        if m < h:
            inds[i, m + 1 :] = inds[i - 1, m + 1 :]
            prev = inds[i - 1, m - 1 : m + 1].flip([0])  # 2 2
            # assert prev[0, 0] == min(levels[i], len(fmap) + 1) or prev[0, 1] == 0, (
            #     levels[i],
            #     prev[0, 0],
            # )
            # assert prev[1, 0] == min(levels[i], len(fmap) + 1) or prev[1, 1] == 0, (
            #     levels[i],
            #     prev[1, 0],
            # )
            level = plan[levels[i] + 1]
            inds[i, m, 0] = levels[i] + 1
            inds[i, m, 1] = level.size(0)
            plan[levels[i] + 1] = torch.cat(
                [plan[levels[i] + 1], prev[:, 1][None]], dim=0
            )
    return plan, inds


def shrink_plan(plan, inds, l):
    # plan: for each level, which entries to avg from previous level ([h] n' 2)
    # inds: which level and entry to pull from in populating heads (n h 2)
    
    # Get modified recursive sum lens
    # First entry is empty, second is seq len plus one for the zero vector entry
    # Subsequent entries are 0 up to 2**(i-2), then increment every 2**(i-1), as seq len increases
    lens = [0,l+1] + [(l-1+2**(i-2))//2**(i-1) for i in range(2,len(plan))]
            
    # Slim down the plan and imap to desired l
    plan = [p[:l] for p,l in zip(plan,lens)]
    inds = inds[:l]

    # Flatten inds (indexing into flattened plan/cache) (n h)
    ls = [p.size(0) for p in plan]
    ls = [0] + ls[:-1]
    offset = torch.tensor(ls, device=inds.device).cumsum(0)
    offset = offset[inds[:, :, 0]]
    inds = offset + inds[:, :, 1]
    return plan, inds


class TelescopingAttention(nn.Module):
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
        super(TelescopingAttention, self).__init__()
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

        self.inp_len = 0
        self.plan = None
        self.imap = None

        # fmap = {8 - i: 64 - (i) ** 2 for i in range(8)}
        # fmap.pop(8)
        # fmap.pop(7)
        # fmap = {
        #     1:26,
        #     2:50,
        #     3:71,
        #     4:89,
        #     5:104,
        #     6:116,
        #     7:124
        # }
        fmap = {1: 64, 2: 72, 3:80}
        self.fmap = fmap
        self.cache_size = 512
        
        self.register_buffer("ringmap", torch.arange(self.cache_size).int())

        self.step = 0

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if self.use_bias:
                    m.bias.data.zero_()
            elif isinstance(m, LayerNormParameterized) or isinstance(m, QKV):
                m.reset_parameters()

    def scan(self, x, plan, inds, w):
        """
        Takes input x of shape [b n ...] and scan plan, computes recursive sums, and
        extracts cache values into the dimension specified by i > 1.
        Final output shape is therefore [b n ... c ...] with c the cache size.
        Applies specified LN to cache, so LN size should match x.size(-1).
        """
        s = x.size()
        ws = w.size()
        # Plan and inds are formed, construct cache via recursive sums
        cache = [None for _ in plan]
        cache[1] = nn.functional.pad(x.view(s[0], s[1], -1), (0, 0, 1, 0)).view(
            s[0], s[1] + 1, *s[2:]
        )  # b n ...
        weights = [None for _ in plan]
        weights[1] = nn.functional.pad(w.view(s[0], s[1], -1), (0,0,1,0), value=-1000).view(
            s[0], s[1] + 1, *ws[2:]
        )
        for j in range(2, len(cache)):
            weights[j] = weights[j-1].index_select(1, plan[j].view(-1)).view(s[0], -1, 2, *ws[2:])
            weights_ = weights[j].softmax(dim=2).unsqueeze(-1)
            weights[j] = weights[j].logsumexp(2)
            cache[j] = (
                cache[j - 1]
                .index_select(1, plan[j].view(-1))
                .view(s[0], -1, 2, *s[2:])
            )
            cache[j] = cache[j].mul(weights_).sum(2)

        # Gather cache    
        cache = torch.cat(cache[1:], dim=1)  # b n' ...
        inds_ = inds[-1]  # h
        state = cache.index_select(1, inds_)  # b h ...
        wt = state.shape
        state = state.view(wt[0], wt[1], -1).transpose(1,2)  # b -1 h
        state = state.reshape(wt[0], 1, *wt[2:], -1)  # b 1 ... h
        # cache = cache.unsqueeze(i).expand(
        #     *[-1] * i, inds.size(-1), *[-1] * (len(s) - i)
        # )  # b n' ... h ...
        # inds_ = inds.view(
        #     1, inds.size(0), *[1] * (i - 2), inds.size(1), *[1] * (len(s) - i)
        # )  # 1 n 111 h 111
        # inds_ = inds_.expand(s[0], -1, *s[2:i], -1, *s[i:])  # b n ... h ...
        # cache = cache.gather(1, inds_)  # b n ... h ...
        
        # Gather final weights
        weights = torch.cat(weights[1:], dim=1)  # b n' ...
        inds_ = inds[-1]  # h
        weights = weights.index_select(1, inds_)  # b h ...
        weights = weights.view(ws[0],weights.size(1),-1).transpose(1,2)  # b -1 h
        weights = weights.reshape(ws[0], 1, *ws[2:], -1)  # b 1 ... h

        return state.transpose(-1,-2), weights, cache
    
    def advance(self, cache, weights, x, w, update_ringmap=True):
        # cache: b h c d
        # weights: b h c
        # x: b h d
        # w: b h
        c = self.cache_size
        powers = (2**torch.arange(10, device=cache.device))
        ilevel = ((self.step-1)%powers).sub(powers-1).sign().add(1).sum().item()
        key = self.fmap.get(ilevel, c)
        
        if key == c:
            cache[:,:,self.ringmap[-1]] = x
            weights[:,:,self.ringmap[-1]] = w
            if update_ringmap:
                self.ringmap = self.ringmap.roll(1)
        else:
            w_ = weights[:,:,self.ringmap[key-1:key+1]]  # b h 2
            c_ = cache[:,:,self.ringmap[key-1:key+1]]  # b h 2 d
            cache[:,:,self.ringmap[key]] = c_.mul(w_.softmax(2).unsqueeze(3)).sum(2)
            weights[:,:,self.ringmap[key]] = w_.logsumexp(2)
            cache[:,:,self.ringmap[key-1]] = x
            weights[:,:,self.ringmap[key-1]] = w
            if update_ringmap:
                self.ringmap[:key] = self.ringmap[:key].roll(1)
        return cache, weights

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

        batch_size, q_len, _ = q.size()
        q_out, k_out, v_out = self.in_proj(q, k, v)
        
        # note: transposes will be moved in a later PR to fix dis-contiguous tensor issues
        queries = q_out.view(batch_size, q_len, self.nheads, self.emb_kq_per_head)
        keys = k_out.view(batch_size, q_len, self.kvheads, self.emb_kq_per_head)
        values = v_out.view(batch_size, q_len, self.kvheads, self.emb_v_per_head)

        # You want to apply rotary embeddings pre-cache
        if self.position_encoder is not None:
            if q_len == 1 and position_ids is None:
                position_ids = torch.ones(batch_size, q_len, device=q.device).mul(self.step).int()
            queries, keys = self.position_encoder.adjusted_qk(
                queries, keys, position_ids, past_key_value_state, use_cache
            )
        
        # Advance caches
        if q_len == 1:
            w = queries.div(self.emb_kq_per_head**0.5)
            w = w.view(batch_size, 1, self.kvheads, -1, self.emb_kq_per_head)  # b 1 h e d
            w = w.mul(keys.unsqueeze(-2)).sum(-1).logsumexp(-1).squeeze(1)  # b h
            past_key_value_state[0], past_key_value_state[2] = self.advance(
                past_key_value_state[0][:,0], 
                past_key_value_state[2][:,0], 
                keys.squeeze(1),  # b h d
                w.squeeze(1),  # b h 
                False
            )
            past_key_value_state[1], past_key_value_state[3] = self.advance(
                past_key_value_state[1][:,0],
                past_key_value_state[3][:,0],
                values.squeeze(1),  # b h d
                w.squeeze(1),  # b h
                True
            )
            past_key_value_state = [x.unsqueeze(1) for x in past_key_value_state]
            keys = past_key_value_state[0].squeeze(1).transpose(1,2)
            values = past_key_value_state[1].squeeze(1).transpose(1,2)
            mask_ = (past_key_value_state[2][0,0,0] > -100)  # c
        else:
            # Reset caches
            past_key_value_state = [None,] * 4
            self.step = 0
            self.ringmap = torch.ones_like(self.ringmap).cumsum(0).sub(1)
            # Generate plan by truncating master plan - generate new master if needed
            if q_len > self.inp_len:
                self.inp_len = 2**(q_len-1).bit_length()
                self.plan, self.imap = get_scan_plan(q.device, self.inp_len, self.fmap, self.cache_size)
            plan, imap = shrink_plan(self.plan, self.imap, q_len)
            # Get weights
            w = queries.div(self.emb_kq_per_head**0.5).view(
                batch_size, q_len, self.kvheads, -1, self.emb_kq_per_head
            ).matmul(keys.unsqueeze(-1)).squeeze(-1).logsumexp(-1)  # b l h
            # Scan
            past_key_value_state[0], past_key_value_state[2], keys = self.scan(keys, plan, imap, w)  # b 1 h c d, b 1 h c, b n h d
            past_key_value_state[1], past_key_value_state[3], values = self.scan(values, plan, imap, w)
            # Get mask
            mask_ = torch.zeros(q_len, keys.size(1), device=q.device, dtype=torch.bool)  # l n
            mask_.scatter_(1, imap, True)
            # Zero out zero entries
            flags = torch.ones(1, q_len, device=q.device)
            _,_,flags = self.scan(flags[:,:,None], plan, imap, flags)  # 1 n 1
            flags = flags.squeeze().bool().logical_not()
            mask_[:,flags] = False
            
        # Advance step counter
        self.step += q_len
        
        # Handle expansion
        expansion = self.nheads // self.kvheads
        if expansion != 1:
            keys_e = keys.transpose(1,2).unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
            values_e = (
                values.transpose(1,2).unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
            )
        else:
            keys_e = keys.transpose(1,2)
            values_e = values.transpose(1,2)
        queries = queries.view(batch_size, q_len, self.nheads, self.emb_kq_per_head).transpose(1,2)

        # Do attention against caches
        # q: b h l d
        # k: b h n d
        # v: b h n d
        # m: l n
        # attn = F.scaled_dot_product_attention(queries, keys_e, values_e, mask_)

        # Manual impl for soft capping
        attn = queries.div(self.emb_kq_per_head**.5).matmul(keys_e.transpose(2,3))
        attn = attn.div(20).tanh().mul(20)
        attn = attn + mask_.log()
        attn = attn.softmax(3)
        attn = attn.matmul(values_e)

        attn = attn.transpose(1,2)  # b l h d
        attn = attn.reshape(batch_size, q_len, self.nheads * self.emb_v_per_head)
        out = self.dense(attn)

        if use_cache:
            return out, past_key_value_state  #[x[:,-1:] for x in past_key_value_state]
        else:
            return out
   