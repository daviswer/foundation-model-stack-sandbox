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
from fms.modules.positions import PositionEncoder
from fms.modules.tp import TPModule


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
        self.w = nn.Linear(self.kvheads*self.emb_kq_per_head, self.emb_kq_per_head, bias=False)

        if self.p_dropout:
            self.attn_dropout = nn.Dropout(self.p_dropout)
        self.position_encoder = position_encoder
        # Avoiding graph breaks
        self.previous_flash: bool = torch.backends.cuda.flash_sdp_enabled()
        self.previous_mem_efficient: bool = (
            torch.backends.cuda.mem_efficient_sdp_enabled()
        )
        self.previous_math: bool = torch.backends.cuda.math_sdp_enabled()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if self.use_bias:
                    m.bias.data.zero_()
            elif isinstance(m, QKV):
                m.reset_parameters()

    def to_tp(self, group: ProcessGroup) -> "TPMultiHeadAttention":
        return TPMultiHeadAttention.import_module(self, group)

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

            # You want to apply rotary embeddings pre-cache
            if self.position_encoder is not None:
                queries, keys = self.position_encoder.adjusted_qk(
                    queries, keys, position_ids, past_key_value_state, use_cache
                )

        # q/k/v: b l h d
        w = self.w.weight.view(self.kvheads, self.emb_kq_per_head, self.emb_kq_per_head)  # h d d
        queries_unflat = queries.view(batch_size, q_len, self.kvheads, -1, self.emb_kq_per_head)  # b l h e d
        q_proj = queries_unflat.matmul(w)
        k_proj = keys.unsqueeze(-2).matmul(w)  # b l h 1 d

        kk_ = torch.cat([k_proj.exp(), k_proj.neg().exp()], dim=-1)  # b l h 1 d+d
        qq_ = torch.cat([q_proj.exp(), q_proj.neg().exp()], dim=-1)  # b l h e d+d
        kv = kk_.transpose(3,4).mul(values.unsqueeze(-2)).cumsum(1)  # b l h d+d d
        denom = kk_.cumsum(1).mul(qq_).sum(-1)  # b l h e
        qkv = qq_.unsqueeze(-1).mul(kv.unsqueeze(3)).sum(4).div(denom.unsqueeze(-1))  # b l h e d


        # v_pos = torch.where(values > 0, values.log(), float('-inf'))
        # v_neg = torch.where(values < 0, values.neg().log(), float('-inf'))
        # v_sep = torch.cat([v_pos, v_neg], dim=-1)  # b l h d+d
        
        # # Calculate denominators
        # qk_pos_denom = k_proj.logcumsumexp(1).add(q_proj).logsumexp(-1)  # b l h e
        # qk_neg_denom = k_proj.neg().logcumsumexp(1).sub(q_proj).logsumexp(-1)  # b l h e
        # qk_denom = torch.logsumexp(torch.stack([qk_pos_denom, qk_neg_denom], dim=-1), dim=-1)  # b l h e
        
        # # Calculate numerators
        # k_pos_v = k_proj.transpose(3,4).add(v_sep.unsqueeze(-2)).logcumsumexp(1)  # b l h d d+d
        # k_neg_v = v_sep.unsqueeze(-2).sub(k_proj.transpose(3,4)).logcumsumexp(1)  # b l h d d+d
        
        # # Perform querying
        # kv_pos = k_pos_v.unsqueeze(3).sub(qk_denom.unsqueeze(-1).unsqueeze(-1)).exp()  # b l h e d d+d
        # kv_neg = k_neg_v.unsqueeze(3).sub(qk_denom.unsqueeze(-1).unsqueeze(-1)).exp()  # b l h e d d+d
        # qkv_pos = q_proj.exp().unsqueeze(-1).mul(kv_pos).sum(4)  # b l h e d+d
        # qkv_neg = q_proj.neg().exp().unsqueeze(-1).mul(kv_neg).sum(4)  # b l h e d+d
        # qkv = qkv_pos.add(qkv_neg).view(batch_size, q_len, self.nheads, 2, self.emb_v_per_head)  # b l he 2 d
        # qkv = qkv[:,:,:,0] - qkv[:,:,:,1]  # b l he d

        attn = qkv.view(batch_size, q_len, -1)

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
