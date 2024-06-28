import abc
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.distributed
from torch import Tensor, nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.nn import functional as F

from fms import distributed
from fms.distributed.tensorparallel import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import PositionEncoder
from fms.modules.tp import TPModule


def scan(state, g):
    # state: b n d h
    # g: 1/b n h h
    state = state.clone()
    g = g.clone()
    s = state.size()
    g0 = g.size(0)
    logl = s[1].bit_length() - 1
    # Up sweep: create ruler ticks
    for i in range(logl):
        span = 2**(i+1)
        state = state.view(s[0], -1, span, *s[2:])  # b -1 span d h
        g = g.view(g0, -1, span, s[3], s[3])  # 1 -1 span h h
        newstate = state[:,:,span//2-1].matmul(g[:,:,-1])
        newgate = g[:,:,span//2-1].matmul(g[:,:,-1])
        state[:,:,-1] += newstate
        g[:,:,-1] = newgate
        
    # Down sweep: fill in blanks
    state = state.view(*s)
    g = g.view(g0, s[1], s[3], s[3])
    state = nn.functional.pad(state, (0,0,0,0,1,0))
    g = nn.functional.pad(g, (0,0,0,0,1,0))[:,:-1]
    remainder = state[:,-1:]
    state = state[:,:-1]
    for i in range(logl-1):
        span = 2**(logl-i-1)
        state = state.view(s[0], -1, span, *s[2:])  # b -1 span d h
        g = g.view(g0, -1, span, s[3], s[3])  # b -1 span h h
        state[:,:,span//2] = state[:,:,span//2] + state[:,:,0].matmul(g[:,:,span//2])
        g[:,:,span//2] = g[:,:,0].matmul(g[:,:,span//2])
    state = torch.cat([state.view(*s)[:,1:], remainder], dim=1)
    return state


class MatScan(torch.autograd.Function):
    @staticmethod
    def forward(state, gate):
        return scan(state, gate)

    @staticmethod
    def setup_context(ctx, inputs, output):
        state, gate = inputs
        ctx.save_for_backward(gate)

    @staticmethod
    def backward(ctx, grad):
        gate = ctx.saved_tensors[0]

        # Gate-accumulate grads
        gflip = gate.flip([1]).transpose(2,3)
        gatesum = scan(grad.flip([1]), gflip.roll(1, dims=1)).flip([1])  # b n d h

        return gatesum, None


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
        fmap = {
            1:26,
            2:50,
            3:71,
            4:89,
            5:104,
            6:116,
            7:124
        }
        self.fmap = fmap
        self.cache_size = 128

        self.scan_impl = False
        if self.scan_impl:
            self.register_buffer("gates", self.make_gates())
            self.matscan = MatScan.apply

        self.weighted = False
        if self.weighted:
            self.w = nn.Linear(self.emb_dim, self.kvheads, bias=False)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if self.use_bias:
                    m.bias.data.zero_()
            elif isinstance(m, LayerNormParameterized) or isinstance(m, QKV):
                m.reset_parameters()

    def to_tp(self, group: ProcessGroup) -> "TPMultiHeadAttention":
        return TPMultiHeadAttention.import_module(self, group)
    
    def make_gates(self):
        n = 1024  # Roughly, total cache window length (actually somewhat smaller)
        d = self.cache_size
        interval = sum([torch.arange(n).remainder(2**x).sub(2**x-1).sign().add(1) 
                        for x in range(n.bit_length())])
        m = torch.zeros(n,d,d)
        for i in range(n):
            key = self.fmap.get(interval[i].item(), d)
            for j in range(key+1, d):
                m[i,j,j] = 1
            if key < d:
                m[i,key,key] = .5**.5
                m[i,key,key-1] = .5**.5
            for j in range(1,min(key,d)):
                m[i,j,j-1] = 1
        return m.transpose(1,2)  # 1k d d

    def scan(self, x, plan, ln, i, w=None):
        """
        Takes input x of shape [b n ...] and scan plan, computes recursive sums, and
        extracts cache values into the dimension specified by i > 1.
        Final output shape is therefore [b n ... c ...] with c the cache size.
        Applies specified LN to cache, so LN size should match x.size(-1).
        """
        assert i > 1, "Output must maintain batch and seqlen in first two dimensions"
        s = x.size()
        weighted = w is not None
        if weighted:
            ws = w.size()
            weights = nn.functional.pad(w.view(s[0], s[1], -1), (0,0,1,0), value=-1000).view(
                s[0], s[1] + 1, *ws[2:]
            )
        inds = plan[-1]  # n h
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
                cache[j] = cache[j][:,:,0]
            
        cache = torch.cat(cache[1:], dim=1)  # b n' ...
        # cache = ln(cache)
        cache = cache.unsqueeze(i).expand(
            *[-1] * i, inds.size(-1), *[-1] * (len(s) - i)
        )  # b n' ... h ...
        inds = inds.view(
            1, inds.size(0), *[1] * (i - 2), inds.size(1), *[1] * (len(s) - i)
        )  # b n 111 h 111
        inds = inds.expand(s[0], -1, *s[2:i], -1, *s[i:])  # b n ... h ...
        return cache.gather(1, inds)  # b n ... h ...

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
        if not self.scan_impl and kv_len != self.inp_len:
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

        queries = queries / (self.emb_kq_per_head**0.5)  # b l h d
        # sink = sink / (self.emb_kq_per_head**0.5)

        # Build telescoping cache
        # k/v: b l h d
        w = None
        if self.weighted:
            w = self.w(k).unsqueeze(-1)  # b l h 1
        if not self.scan_impl:
            # keys = keys.view(batch_size, kv_len, -1)
            keys = self.scan(keys, self.plan, None, 4, w)  # b l h d 64
            values = self.scan(values, self.plan, None, 3, w)  # b l h 64 d
        else:
            gate = self.gates.repeat(4,1,1)[None]  # 1 l 64 64
            keys = keys.view(batch_size, kv_len, -1, 1)
            keys = F.pad(keys, (0, self.cache_size-1))  # b l hd 64
            keys = self.matscan(keys, gate)
            # keys = keys / keys.pow(2).mean(2, True).sqrt().add(1e-6)
            keys = keys.unflatten(
                2, (self.kvheads, self.emb_kq_per_head)
            )  # b l h d 64
            values = values.view(batch_size, kv_len, -1, 1)
            values = F.pad(values, (0, self.cache_size-1))  # b l hd 64
            values = self.matscan(values, gate).unflatten(
                2, (self.kvheads, self.emb_v_per_head)
            )  # b l h d 64
            # values = values / values.pow(2).mean(3, True).sqrt().add(1e-6)
            values = values.transpose(3,4)  # b l h 64 d

        # if you want to use caching and past_key_value_state is not None meaning you have values in your cache
        if (
            use_cache
            and past_key_value_state is not None
            and past_key_value_state[0].numel() > 0
        ):
            if is_self:
                keys = torch.cat((past_key_value_state[0], keys), dim=2)
                values = torch.cat((past_key_value_state[1], values), dim=2)
            else:
                keys = past_key_value_state[0]
                values = past_key_value_state[1]

        # Expand kv so black-box attn will work
        expansion = self.nheads // self.kvheads
        queries = queries.unflatten(2, (self.kvheads, expansion))  # b l h e d

        # b l h e d, b l h d 64
        attn = queries.matmul(keys)  # b l h e 64
        # denom = torch.stack([
        #         sink.view(batch_size, q_len, self.kvheads, expansion),
        #         attn.logsumexp(4),
        #     ], 4)  # b l h e 2
        # denom = denom.logsumexp(4, True)  # b l h e 1
        # attn = attn.sub(denom).exp()
        attn = attn.softmax(4)
        # b l h e 64, b l h 64 d
        attn = attn.matmul(values)  # b l h e d

        attn = attn.reshape(batch_size, q_len, self.nheads * self.emb_v_per_head)
        out = self.dense(attn)

        # if use_cache=True, we return the hidden_state as well as the kv cache
        if use_cache:
            return out, (keys, values)
        else:
            return out


class TPMultiHeadAttention(MultiHeadAttention, TPModule):
    """
    Performs multi-headed self- or cross-attention, with optional attention masking.
    This subclass adds support for Tensor Parallel
    ...
    Args
    ----
    Check MultiHeadAttention for up-to-date docs

    world_size: int
        the number of processes running this model in TP
    rank: int
        the index of this process wrt to the rest running the model in TP
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
        group: Optional[ProcessGroup] = None,
    ):
        assert torch.distributed.is_initialized()

        rank, world_size = distributed.rank_and_world(group)
        assert (
            nheads % world_size == 0
        ), "The number of heads must be divisible by world size"
        assert (kvheads >= world_size and kvheads % world_size == 0) or (
            kvheads < world_size and world_size % kvheads == 0
        ), "the kv heads must be divisible by the world size or the world size must be divisible by kv heads"
        MultiHeadAttention.__init__(
            self,
            emb_dim,
            emb_kq,
            emb_v,
            nheads // world_size,
            (kvheads // world_size) if kvheads >= world_size else 1,
            p_dropout,
            use_bias,
            position_encoder,
            fused,
        )
        self.pre_tp_nheads = nheads
        self.pre_tp_kvheads = kvheads
        self.setup_tp(rank, world_size)

    def load_weights(
        self,
        tensor_values: Dict[str, torch.Tensor],
    ):
        used_keys: Set[str] = set()
        dense_weight = self._get_sd_weight(
            tensor_values, used_keys, ["dense", "weight"]
        )
        if self.use_bias:
            dense_bias = self._get_sd_weight(
                tensor_values, used_keys, ["dense", "bias"]
            )

        # 1. Grab the weights from tensor_values
        if self.fused:
            qkv_weight = self._get_sd_weight(
                tensor_values, used_keys, ["qkv_fused", "weight"]
            )
            if self.use_bias:
                qkv_bias = self._get_sd_weight(
                    tensor_values, used_keys, ["qkv_fused", "bias"]
                )

            # 2. Raise exceptions
            if len(tensor_values) > (4 if self.use_bias else 2):
                unused_keys = set(tensor_values.keys()).difference(used_keys)
                raise AttributeError(f"Unused weight(s): {', '.join(unused_keys)}")

            # 3. Load and shard the weights
            # The number in max_partition_sizes will signify the largest world size
            # til we need to duplicate.  For instance if we have nheads=16 and
            # world_size=32, then first 2 ranks will get first 1/16th of query
            self.sharded_copy(
                self.in_proj.qkv_fused.weight,
                qkv_weight,
                0,
                [self.pre_tp_nheads, self.pre_tp_kvheads, self.pre_tp_kvheads],
            )
            self.sharded_copy(self.dense.weight, dense_weight, 1, [self.world_size])
            if self.use_bias:
                self.sharded_copy(
                    self.in_proj.qkv_fused.bias,
                    qkv_bias,
                    0,
                    [self.pre_tp_nheads, self.pre_tp_kvheads, self.pre_tp_kvheads],
                )
                self.sharded_copy(
                    self.dense.bias, dense_bias, 1, [self.world_size], False
                )

        else:
            query_weight = self._get_sd_weight(
                tensor_values, used_keys, ["query", "weight"]
            )
            key_weight = self._get_sd_weight(
                tensor_values, used_keys, ["key", "weight"]
            )
            value_weight = self._get_sd_weight(
                tensor_values, used_keys, ["value", "weight"]
            )

            if self.use_bias:
                query_bias = self._get_sd_weight(
                    tensor_values, used_keys, ["query", "bias"]
                )
                key_bias = self._get_sd_weight(
                    tensor_values, used_keys, ["key", "bias"]
                )
                value_bias = self._get_sd_weight(
                    tensor_values, used_keys, ["value", "bias"]
                )

            # 2. Raise exceptions
            if len(tensor_values) > (8 if self.use_bias else 4):
                unused_keys = set(tensor_values.keys()).difference(used_keys)
                raise AttributeError(f"Unused weight(s): {', '.join(unused_keys)}")

            # 3. Load and shard the weights
            # The number in max_partition_sizes will signify the largest world size
            # til we need to duplicate.  For instance if we have nheads=16 and
            # world_size=32, then first 2 ranks will get first 1/16th of query
            self.sharded_copy(
                self.in_proj.query.weight, query_weight, 0, [self.pre_tp_nheads]
            )
            self.sharded_copy(
                self.in_proj.key.weight, key_weight, 0, [self.pre_tp_kvheads]
            )
            self.sharded_copy(
                self.in_proj.value.weight, value_weight, 0, [self.pre_tp_kvheads]
            )
            self.sharded_copy(self.dense.weight, dense_weight, 1, [self.world_size])
            if self.use_bias:
                self.sharded_copy(
                    self.in_proj.query.bias, query_bias, 0, [self.pre_tp_nheads]
                )
                self.sharded_copy(
                    self.in_proj.key.bias, key_bias, 0, [self.pre_tp_kvheads]
                )
                self.sharded_copy(
                    self.in_proj.value.bias, value_bias, 0, [self.pre_tp_kvheads]
                )
                self.sharded_copy(
                    self.dense.bias, dense_bias, 1, [self.world_size], False
                )

    @staticmethod
    def import_module(
        mha: MultiHeadAttention, group: ProcessGroup
    ) -> "TPMultiHeadAttention":
        tp_mha = TPMultiHeadAttention(
            emb_dim=mha.emb_dim,
            emb_kq=mha.emb_kq_per_head,
            emb_v=mha.emb_v_per_head,
            nheads=mha.nheads,
            kvheads=mha.kvheads,
            p_dropout=mha.p_dropout,
            use_bias=mha.use_bias,
            position_encoder=mha.position_encoder,
            group=group,
            fused=mha.fused,
        )
        return tp_mha

    def _copy_to_tp_region(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
    ):
        if (k is None and v is None) or (k is q and v is q):
            q_par = copy_to_tensor_model_parallel_region(q)
            if self.fused:
                k_par = None
                v_par = None
            else:
                k_par = copy_to_tensor_model_parallel_region(k)
                v_par = copy_to_tensor_model_parallel_region(v)
        else:
            raise ValueError(
                "both k and v must either be given as tensors or both None"
            )

        return q_par, k_par, v_par

    def forward(
        self,
        q,
        k=None,
        v=None,
        mask=None,
        position_ids=None,
        attn_algorithm=None,
        past_key_value_state=None,
        use_cache=False,
        is_self=True,
        is_causal_mask=False,
    ):
        """
        Check MultiHeadAttention for up-to-date arguments and docs
        """

        q_par, k_par, v_par = self._copy_to_tp_region(q, k, v)

        out_par = MultiHeadAttention.forward(
            self,
            q_par,
            k_par,
            v_par,
            mask,
            position_ids,
            attn_algorithm,
            past_key_value_state,
            use_cache,
            is_self,
            is_causal_mask,
        )

        # if use_cache=True, we return the hidden_state as well as the kv cache.
        # We only reduce the output, and keep the cache thread-local
        if use_cache:
            out = reduce_from_tensor_model_parallel_region(out_par[0])
            return out, out_par[1]
        else:
            out = reduce_from_tensor_model_parallel_region(out_par)
            return out
