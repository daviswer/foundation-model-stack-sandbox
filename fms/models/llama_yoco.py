import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple
from typing_extensions import Unpack

import torch
import torch.nn as nn
from flash_attn import flash_attn_func

from fms import models
from fms.distributed.strategy import (
    DistributedStrategy,
    NoOpStrategy,
    TensorParallelStrategy,
)
from fms.modules.attention import (
    AttentionKwargs,
    MultiHeadAttention,
    get_attention_type,
)
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.head import LinearClassificationHead
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.linear import get_linear_type, get_linear
from fms.modules.positions import RotaryEmbedding
from fms.modules.yoco_rotary import apply_rotary_emb
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig
from fms.utils.headless import gather_outputs

from .llama import (
    LLaMAConfig,
    LLaMAHeadless,
)
# [CL] original YOCO import from fairseq.model_parallel.megatron.mpu, 
#       this package (pip install megatron-core) may be newer
# from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear


logger = logging.getLogger(__name__)
rank = int(os.environ["RANK"])

# params emb_dim heads layers lr
#  7B    4096    32    32     3.0E-04
# 13B    5120    40    40     3.0E-04
# 33B    6656    52    60     1.5.E-04
# 65B    8192    64    80     1.5.E-04


# --------- codes modified from YOCO repo
# main changes:
# 1. ROPE for rel_pos vs abs_pos
# 2.
# other minor changes:
# var names, TP compatibility check, ...

def init_method(tensor, **kwargs):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class SlidingWindowAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.emb_dim
        self.num_heads = cfg.nheads // getattr(cfg,"model_parallel_size", 1)  # TODO use world size or rank to replace this var
        self.window_size = cfg.sliding_window  # we set this in LlamaYOCO init already        
        self.head_dim = cfg.emb_dim // cfg.nheads

        # self.q_proj = ColumnParallelLinear(cfg.emb_dim, cfg.emb_dim, bias=False, gather_output=False, init_method=init_method)
        # self.k_proj = ColumnParallelLinear(cfg.emb_dim, cfg.emb_dim, bias=False, gather_output=False, init_method=init_method)
        # self.v_proj = ColumnParallelLinear(cfg.emb_dim, cfg.emb_dim, bias=False, gather_output=False, init_method=init_method)
        # self.out_proj = RowParallelLinear(cfg.emb_dim, cfg.emb_dim, bias=False, input_is_parallel=True, init_method=init_method)
        self.use_bias = False  # [CL] hard-coded for now.
        self.q_proj = get_linear(cfg.emb_dim, cfg.emb_dim, bias=False, linear_config=cfg.linear_config)
        self.k_proj = get_linear(cfg.emb_dim, cfg.emb_dim, bias=False, linear_config=cfg.linear_config)
        self.v_proj = get_linear(cfg.emb_dim, cfg.emb_dim, bias=False, linear_config=cfg.linear_config)
        self.out_proj = get_linear(cfg.emb_dim, cfg.emb_dim, bias=False, linear_config=cfg.linear_config)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if self.use_bias:
                    m.bias.data.zero_()
            # also assume non-Fused QKV

    def forward(
        self,
        x,
        rel_pos,
        start_pos=0,
        incremental_state=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim)

        q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        if incremental_state is not None:
            if "prev_key" not in incremental_state:
                incremental_state["prev_key"] = torch.empty(self.cfg.max_batch_size, self.window_size, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
                incremental_state["prev_value"] = torch.empty(self.cfg.max_batch_size, self.window_size, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)

            key = torch.cat([incremental_state["prev_key"][:bsz, :start_pos], k], dim=1)
            value = torch.cat([incremental_state["prev_value"][:bsz, :start_pos], v], dim=1)
            if key.shape[1] > self.window_size:
                incremental_state["prev_key"][:bsz] = key[:, -self.window_size:]
                incremental_state["prev_value"][:bsz] = value[:, -self.window_size:]
            else:
                incremental_state["prev_key"][:bsz, start_pos : start_pos + tgt_len] = k
                incremental_state["prev_value"][:bsz, start_pos : start_pos + tgt_len] = v
        else:
            key, value = k, v

        if rank==0:
            print("    Sliding window:", q.mean().item(), q.std().item())
        attn = flash_attn_func(q, key, value, causal=True, window_size=(self.window_size - 1, 0)) 
        attn = attn.reshape(bsz, tgt_len, self.head_dim * self.num_heads)

        attn = self.out_proj(attn)
        return attn


class CrossAttention(nn.Module):
    def __init__(self, cfg,):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.emb_dim
        self.num_heads = cfg.nheads // getattr(cfg,"model_parallel_size", 1)
        self.num_kv_heads = cfg.kvheads // getattr(cfg,"model_parallel_size", 1)
        
        self.head_dim = cfg.emb_dim // cfg.nheads
        # self.q_proj = ColumnParallelLinear(cfg.emb_dim, cfg.emb_dim, bias=False, gather_output=False, init_method=init_method)
        # self.out_proj = RowParallelLinear(cfg.emb_dim, cfg.emb_dim, bias=False, input_is_parallel=True, init_method=init_method)
        self.use_bias = False  # [CL] hard-coded for now.
        self.q_proj = get_linear(cfg.emb_dim, cfg.emb_dim, bias=False, linear_config=cfg.linear_config)
        self.out_proj = get_linear(cfg.emb_dim, cfg.emb_dim, bias=False, linear_config=cfg.linear_config)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if self.use_bias:
                    m.bias.data.zero_()

    def forward(
        self,
        x,
        key,
        value,
        rel_pos
    ):
        bsz, tgt_len, _ = x.size()
        
        q = self.q_proj(x)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
        q = apply_rotary_emb(q, *rel_pos, interleaved=True)

        if rank==0:
            print("    Cross attention:", q.mean().item(), q.std().item())
        attn = flash_attn_func(q, key, value, causal=True)
        attn = attn.view(bsz, tgt_len, self.head_dim * self.num_heads)

        attn = self.out_proj(attn)
        return attn


class LLaMABlockYOCO(nn.Module):
    """Modified from fms's LLaMABlock based on YOCO's DecoderLayer."""
    def __init__(
            self,
            config: LLaMAConfig,
            rotary_emb: RotaryEmbedding,
            attn_type:str = "mha",
        ):
        super().__init__()
        self.config = config
        emb_kq = self.config.emb_dim // self.config.nheads
        emb_v = self.config.emb_dim // self.config.nheads

        self.ln = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )
        self.ff_ln = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )

        if self.config.kvheads == 0:
            kvheads = self.config.nheads
        else:
            kvheads = self.config.kvheads
            assert self.config.nheads % self.config.kvheads == 0

        # modified based on YOCO's DecoderLayer
        self.attn_type = attn_type
        if attn_type == "cross_attn":
            self.attn = CrossAttention(config)
        elif attn_type in ["sliding_win_attn", "swa"]:
            self.attn = SlidingWindowAttention(config)
        else:
            self.attn = MultiHeadAttention(
                self.config.emb_dim,
                emb_kq,
                emb_v,
                self.config.nheads,
                kvheads,
                p_dropout=self.config.p_dropout,
                use_bias=self.config.attn_bias,
                position_encoder=rotary_emb,
                fused=self.config.fused_weights,
                linear_config=self.config.linear_config,
            )
        self.ff_sub_layer = GatedLinearUnit(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            multiple_of=self.config.multiple_of,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=self.config.mlp_bias,
            fused=self.config.fused_weights,
            linear_config=self.config.linear_config,
        )

        if self.config.p_dropout != 0:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def forward(
        self,
        x,
        *,
        position_ids=None,
        past_key_value_state=None,
        use_cache=False,
        # the following 6 are used by YOCO's DecoderLayer ---
        start_pos=0,
        key=None,
        value=None,
        rel_pos=None,
        incremental_state=None,
        is_prefilling=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        # if the cache is not empty, we need to get the kv cache for self and cross attention
        self_attn_past_key_value = past_key_value_state
        # if past_key_value_state is not None:
        #     self_attn_past_key_value = past_key_value_state[:2]
        # else:
        #     self_attn_past_key_value = None

        # first we do MHA and Add&Norm
        residual = x
        x = self.ln(x)
        # [CL] call signature changes with attn_type
        if self.attn_type == "cross_attn":
            x = self.attn(
                x,
                key,
                value,
                rel_pos=rel_pos,
            )
        elif self.attn_type in ["sliding_win_attn", "swa"]:
            x = self.attn(
                x,
                rel_pos=rel_pos,
                start_pos=start_pos,
                incremental_state=incremental_state,
            )
        else:
            # original MHA
            x = self.attn(
                q=x,
                position_ids=position_ids,
                past_key_value_state=self_attn_past_key_value,
                use_cache=use_cache,
                **attn_kwargs,
            )
        cache = None
        if use_cache:
            x, cache = x
        if self.config.p_dropout != 0:
            x = self.dropout(x)
        # residual connection
        x = x + residual

        # then we do FF and Add&Norm
        residual = x
        x = self.ff_ln(x)
        x = self.ff_sub_layer(x)
        if self.config.p_dropout != 0:
            x = self.dropout(x)
        # another residual
        x = x + residual

        if use_cache:
            return (x, cache)
        else:
            return x



class SelfDecoder(nn.Module):
    """Modified from YOCO's implementation"""
    def __init__(
        self,
        config: LLaMAConfig,
        rot_emb: RotaryEmbedding,
        checkpoint_activations: bool = False
    ):
        super().__init__()
        self.config = config 
        layers = [LLaMABlockYOCO(config, rot_emb, attn_type="sliding_win_attn",) for idx in range(config.nlayers // 2)]
        if checkpoint_activations:
            # layers = [checkpoint_wrapper(layer) for layer in layers]
            raise NotImplementedError
        self.layers = nn.ModuleList(layers)
        self.head_dim = config.emb_dim // config.nheads
        self.block_size = 256
        self._precomputed_freqs_cis = None

    def build_rel_pos(self, x, start_pos):
        if self._precomputed_freqs_cis is None:
            angle = 1.0 / (self.config.rope_theta ** torch.linspace(0, 1, self.head_dim // 2, dtype=torch.float, device=x.device))
            index = torch.arange(self.config. max_expected_seq_len).to(angle)
            self._precomputed_freqs_cis = index[:, None] * angle

        cos = torch.cos(self._precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        sin = torch.sin(self._precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        rel_pos = (cos.to(x.dtype), sin.to(x.dtype))
        return rel_pos

    def get_index_mask(self, x, length, pad_length):
        return torch.arange(pad_length, device=x.device) >= length

    def forward(
        self,
        x,
        incremental_state=None,
        is_prefilling=False,
        start_pos=0
    ):
        if is_prefilling and x.size(1) % self.block_size != 0 and self.config.sliding_window is None:
            padding_len = self.block_size - x.size(1) % self.block_size
            x = torch.nn.functional.pad(x, (0, 0, 0, padding_len), value=0)
        else:
            padding_len = 0

        if incremental_state is not None and is_prefilling:
            index_mask = self.get_index_mask(x, x.size(1) - padding_len, x.size(1))

        rel_pos = self.build_rel_pos(x, start_pos)
        for idx, layer in enumerate(self.layers):
            if incremental_state is not None:
                if idx not in incremental_state:
                    incremental_state[idx] = {}
                if is_prefilling:
                    incremental_state[idx]["index_mask"] = index_mask
            x = layer(
                x,
                start_pos=start_pos,
                rel_pos=rel_pos,
                incremental_state=incremental_state[idx] if incremental_state is not None else None,
                is_prefilling=is_prefilling,)

        x = x[:, :x.size(1) - padding_len, :]
        return x

class CrossDecoder(nn.Module):
    """Modified from YOCO's implementation"""
    def __init__(
        self,
        config: LLaMAConfig,
        rot_emb: RotaryEmbedding,
        checkpoint_activations: bool = False
    ):
        super().__init__()
        self.config = config
        self.num_heads = config.kvheads
        self.head_dim = config.emb_dim // config.nheads
        # self.k_proj = ColumnParallelLinear(config.emb_dim, self.head_dim * config.kvheads, bias=False, gather_output=False, init_method=init_method)
        # self.v_proj = ColumnParallelLinear(config.emb_dim, self.head_dim * config.kvheads, bias=False, gather_output=False, init_method=init_method)
        self.k_proj = get_linear(config.emb_dim, self.head_dim * config.kvheads, bias=False, linear_config=config.linear_config)
        self.v_proj = get_linear(config.emb_dim, self.head_dim * config.kvheads, bias=False, linear_config=config.linear_config)
        self.kv_layer_norm = RMSNorm(config.emb_dim, eps=config.norm_eps)
        layers = [LLaMABlockYOCO(config, rot_emb, attn_type="cross_attn") for idx in range(config.nlayers // 2)]
        # if checkpoint_activations:
        #     layers = [checkpoint_wrapper(layer) for layer in layers]
        self.layers = nn.ModuleList(layers)
        self._precomputed_freqs_cis = None

    def build_rel_pos(self, x, start_pos):
        if self._precomputed_freqs_cis is None:
            angle = 1.0 / (self.config.rope_theta ** torch.linspace(0, 1, self.head_dim // 2, dtype=torch.float, device=x.device))
            index = torch.arange(self.config.max_expected_seq_len).to(angle)
            self._precomputed_freqs_cis = index[:, None] * angle

        cos = torch.cos(self._precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        sin = torch.sin(self._precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        rel_pos = (cos.to(x.dtype), sin.to(x.dtype))
        return rel_pos

    def forward(
        self,
        x,
        incremental_state=None,
        start_pos=0,
        skip_cross_decoder=False,
    ):
        bsz, seqlen, embed_dim = x.size()
        x_norm = self.kv_layer_norm(x)
        key, value = self.k_proj(x_norm), self.v_proj(x_norm)
        key = key.view(bsz, seqlen, self.num_heads, self.head_dim)
        value = value.view(bsz, seqlen, self.num_heads, self.head_dim)
        rel_pos = self.build_rel_pos(x, start_pos)
        key = apply_rotary_emb(key, *rel_pos, interleaved=True)

        if incremental_state is not None:
            if "prev_key" not in incremental_state:
                incremental_state["prev_key"] = torch.empty(bsz, self.config.max_expected_seq_len, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
                incremental_state["prev_value"] = torch.empty(bsz, self.config.max_expected_seq_len, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
            incremental_state["prev_key"][:, start_pos : start_pos + seqlen] = key
            incremental_state["prev_value"][:, start_pos : start_pos + seqlen] = value
            key = incremental_state["prev_key"][:, : start_pos + seqlen]
            value = incremental_state["prev_value"][:, : start_pos + seqlen]
            if rank==0:
                print("    Got the persistent cache:", key.mean().item(), key.std().item())
        
        if skip_cross_decoder:
            return torch.zeros(bsz, 1, embed_dim, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x = layer(
                x,
                key=key,
                value=value,
                rel_pos=rel_pos)

        return x
# end of codes modified from YOCO --------- 


class LLaMAHeadlessYOCO(LLaMAHeadless):
    """Inherit fms LLaMAHeadless, override:
     1. init()
        to replace [LlamaBlock * n] with 2 new (borrowed) classes [SelfDecoder + CrossDecoder]
        where SelfDecoder has [(modifiedLlamaBlock)* n//2]
        where CrossDecoder has [(modifiedLlamaBlock)* n//2]
     2. forward()
        to invoke self_decoder and cross_decoder
     3. reset_parameters()
        to include the new Attention classes 
    """
    def __init__(
        self,
        config: Optional[LLaMAConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super().__init__()
        delattr(self, "layers")
        # NOTE super.init will have created all the Llama layers already. Among which, we will not
        # need the original .layers, i.e. the modList of transformer blocks.

        # [CL] make sure YOCO specific defaults exists, values based on YOCO paper/repo
        config.sliding_window = getattr(config, "sliding_window", 1024)  # 1st line on page 8 of the paper
        config.max_batch_size = getattr(config, "batch_size", 8)
        # print(config)

        if config is not None:
            self.config = config
        else:
            self.config = LLaMAConfig()

        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        embedding = nn.Embedding(
            self.config.src_vocab_size, self.config.emb_dim, self.config.pad_id
        )
        # TP does not work with tied weights
        if (
            not isinstance(self.distributed_strategy, TensorParallelStrategy)
            or not self.config.tie_heads
        ):
            self.embedding = self.distributed_strategy.distribute_module(embedding)
        else:
            logger.warning(
                "You're using TP on a model with tied weights between head and embedding. "
                "The tied weights won't be sharded, which can result in unexpected OOMs."
            )
            self.embedding = embedding

        self.rot_emb = RotaryEmbedding(
            dim=self.config.emb_dim // self.config.nheads,
            scaling=self.config.rope_scaling,
            max_seq_len=self.config.max_expected_seq_len,
            ratio=self.config.rope_theta,
        )
        # RoPE init
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

        # layers = []
        # for i in range(self.config.nlayers):
        #     block: nn.Module = LLaMABlock(self.config, self.rot_emb)
        #     block = self.distributed_strategy.distribute_layer(block, i)
        #     layers.append(block)
        # self.layers = nn.ModuleList(layers)
        self.self_decoder = SelfDecoder(config, self.rot_emb)
        self.cross_decoder = CrossDecoder(config, self.rot_emb)

        dec_norm = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )
        self.dec_norm = self.distributed_strategy.distribute_module(
            dec_norm, final_layers=True
        )

        if self.config.p_dropout:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def reset_parameters(self):
        """
        Call the original reset_parameters() and then the same func in the new attention
        classes in YOCOLlama.
        """
        super().reset_parameters()

        for m in self.modules():
            if isinstance(m, (SlidingWindowAttention, CrossAttention)):
                m.reset_parameters()

    def forward(
        self,
        x_in,
        position_ids=None,
        past_key_value_states=None,
        use_cache=False,
        start_pos=0,  # TODO calc start_pos from pos_ids?
        skip_cross_decoder=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        # Embed the given vocabulary indices using the given attention mask, with pre-/post-norm and dropout as specified
        # x_in: batch_size x seq_len
        # mask: batch_size x seq_len x seq_len
        # bias: nheads x seq_len x seq_len
        if past_key_value_states is None or len(past_key_value_states) == 0:
            # past_key_value_states = [None for _ in range(len(self.layers))]
            past_key_value_states = {}
            # NOTE 1. let SelfDecoder.SlidingWindowAttn do the init of the cache
            #      2. different data structure, i.e. list vs dict, need to reconcile
        x_in = self.embedding(x_in)

        # this is the output cache for all the decoder layers
        present_key_value_states = []

        # [CL] the following checks was in fms llama's MHA class, but could (should) be
        #       checked at upper level, hence, moved here
        attn_compute_dict = get_attention_type(**attn_kwargs)
        is_prefilling = attn_compute_dict["is_prefill"](**attn_kwargs)

        x_in = self.self_decoder(
            x_in,
            incremental_state=past_key_value_states,
            is_prefilling=is_prefilling,
            start_pos=start_pos,
        )

        x_in = self.cross_decoder(
            x_in,
            start_pos=start_pos,
            incremental_state=past_key_value_states,
            skip_cross_decoder=skip_cross_decoder,
        )

        dec_out = x_in
        dec_out = self.dec_norm(dec_out)
        if self.config.p_dropout:
            dec_out = self.dropout(dec_out)

        return dec_out, present_key_value_states


class LLaMA(nn.Module):
    """Only change self.base_model to LLaMAHeadlessYOCO, rest unchanged."""
    def __init__(
        self,
        config: Optional[LLaMAConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(LLaMA, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = LLaMAConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.base_model = LLaMAHeadlessYOCO(self.config, self.distributed_strategy)
        head = LinearClassificationHead(
            self.config.emb_dim, self.config.src_vocab_size, bias=False
        )
        # TP does not work with tied weights
        if (
            not isinstance(self.distributed_strategy, TensorParallelStrategy)
            or not self.config.tie_heads
        ):
            self.head = self.distributed_strategy.distribute_module(head)
        else:
            self.head = head

    def get_config(self) -> LLaMAConfig:
        return self.config

    @classmethod
    def from_config(cls, config: LLaMAConfig) -> "LLaMA":
        return cls(config)

    def reset_parameters(self):
        # Call reset_parameters for relevant sub-layers
        assert isinstance(self.head, torch.nn.Linear)
        self.head.weight.data.normal_(
            0,
            1 / math.sqrt(math.sqrt(self.config.emb_dim * self.config.src_vocab_size)),
        )
        self.base_model.reset_parameters()

    def validate_reset_parameters(self):
        # Verifies that the above self.reset_parameters() executed correctly.
        # This may not always be the case for distributed settings with sharded tensors,
        # such as FSDP or TP. Note that performing this check may require unsharding /
        # re-materializing the full model on a single rank to access the underlying tensors.
        tolerance = 1e-3

        def check_close(x):
            assert x.mean().abs() < tolerance
            assert x.std().sub(0.02).abs() < tolerance

        with torch.no_grad():
            for p in self.parameters():
                assert p.isnan().int().sum() == 0
                assert p.isinf().int().sum() == 0
            self.base_model.validate_reset_parameters()
            check_close(self.head.weight)

    def post_init(self):
        self.base_model.post_init()

        # if this model ties weights, they are tied here
        if self.config.tie_heads:
            # handle assignment of non-meta weights to meta parameters
            if self.head.weight.device == torch.device("meta"):
                self.head.weight = self.base_model.embedding.weight
            else:
                self.base_model.embedding.weight = self.head.weight

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value_states: Optional[Tuple[torch.FloatTensor,]] = None,
        use_cache: bool = False,
        last_n_tokens: int = 0,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        get_attention_type(**attn_kwargs)["validate_attn_kwargs"](
            input_ids=x,
            position_ids=position_ids,
            past_key_value_states=past_key_value_states,
            **attn_kwargs,
        )
        output, cache = self.base_model(
            x, position_ids, past_key_value_states, use_cache, **attn_kwargs
        )

        output = gather_outputs(output, last_n_tokens, **attn_kwargs)
        preds = self.head(output)
        if rank==0:
            print("Final output:", preds.mean().item(), preds.std().item(), preds.min().item(), preds.max().item())

        if use_cache:
            return preds, cache
        else:
            return preds


# Register common LLaMA variants with the model registration API

# a micro llama model to use with a char-level tokenizer
_micro_char_config = LLaMAConfig(
    emb_dim=192, nheads=4, nlayers=5, max_expected_seq_len=1024, src_vocab_size=256
)

_7b_config = LLaMAConfig()
_13b_config = LLaMAConfig(emb_dim=5120, nheads=40, nlayers=40)
# todo: add 35B config

_70b_config = LLaMAConfig(
    emb_dim=8192,
    multiple_of=4096,
    nheads=64,
    kvheads=8,
    nlayers=80,
    hidden_grow_factor=(1.3 * 8 / 3),
)

_8b_llama3_config = LLaMAConfig(
    src_vocab_size=128256,
    emb_dim=4096,
    norm_eps=1e-5,
    nheads=32,
    kvheads=8,
    nlayers=32,
    hidden_grow_factor=3.5,
    multiple_of=1024,
    max_expected_seq_len=8192,
    rope_theta=500_000.0,
)

# Granite configs
_granite_7b_config = LLaMAConfig(
    src_vocab_size=32008,
)

_granite_3b_code_config = LLaMAConfig(
    src_vocab_size=49152,
    emb_dim=2560,
    pad_id=0,
    hidden_grow_factor=10240 / 2560,
    multiple_of=1,
    p_dropout=0.1,
    max_expected_seq_len=2048,
    attn_bias=True,
    mlp_bias=True,
    tie_heads=True,
)

_granite_8b_code_config = LLaMAConfig(
    src_vocab_size=49152,
    emb_dim=4096,
    kvheads=8,
    nlayers=36,
    pad_id=0,
    hidden_grow_factor=14336 / 4096,
    multiple_of=1,
    p_dropout=0.1,
    max_expected_seq_len=4096,
    attn_bias=True,
    mlp_bias=True,
    tie_heads=True,
)

_architecture_name = "llama_yoco"


def _llama_factory_factory(config):
    def factory(**kwargs):
        return LLaMA(config, **kwargs)

    return factory


models.register_model(
    _architecture_name, "micro", _llama_factory_factory(_micro_char_config)
)
# Backwards compat
models.register_model(_architecture_name, "7b", _llama_factory_factory(_7b_config))
models.register_model(_architecture_name, "13b", _llama_factory_factory(_13b_config))
models.register_model(_architecture_name, "70b", _llama_factory_factory(_70b_config))

# LLama 2 family
models.register_model(_architecture_name, "2-7b", _llama_factory_factory(_7b_config))
models.register_model(_architecture_name, "2-13b", _llama_factory_factory(_13b_config))
models.register_model(_architecture_name, "2-70b", _llama_factory_factory(_70b_config))

# LLama 3 family
models.register_model(
    _architecture_name, "3-8b", _llama_factory_factory((_8b_llama3_config))
)

# Granite family
models.register_model(
    _architecture_name, "granite-7b", _llama_factory_factory((_granite_7b_config))
)
models.register_model(
    _architecture_name,
    "granite.code-3b",
    _llama_factory_factory((_granite_3b_code_config)),
)

models.register_model(
    _architecture_name,
    "granite.code-8b",
    _llama_factory_factory((_granite_8b_code_config)),
)

# Create all the pieces to generate adapters for different checkpoints
serialization.register_adapter_step(
    "llama_yoco", "pre0.0.6_attn_unfused_to_fused", serialization._pre006_attn_adapter_step
)

serialization.register_adapter_step(
    "llama_yoco",
    "swiglu_unfused_to_fused",
    serialization._mlp_glu_unfused_to_fused_adapter_step,
)


def _weight_fusion(
    input_sd: Mapping[str, Any], model_config: Optional[LLaMAConfig] = None, **kwargs
) -> Mapping[str, Any]:
    has_fused_weights = True
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False

    new_sd = input_sd
    if has_fused_weights:
        new_sd = serialization._mlp_glu_unfused_to_fused_adapter_step(
            serialization._attn_unfused_to_fused_step(new_sd)
        )
    return new_sd


serialization.register_adapter_step("llama_yoco", "weight_fusion", _weight_fusion)


def _hf_gptq_llama_check(
    input_sd: Mapping[str, Any], model_config: Optional[LLaMAConfig] = None, **kwargs
) -> Mapping[str, Any]:
    has_fused_weights = True
    linear_type = "torch_linear"
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False
        if model_config.linear_config:
            linear_type = model_config.linear_config["linear_type"]

    if not callable(linear_type) and "gptq" in linear_type and has_fused_weights:
        raise ValueError(
            "GPTQ HF llama checkpoints cannot be loaded into a model with fused weights"
        )

    return input_sd


serialization.register_adapter_step(
    "llama_yoco", "hf_gptq_fusion_check", _hf_gptq_llama_check
)


def _meta_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = [
        (r"^tok_embeddings", "base_model.embedding"),
        (r"^norm", "base_model.dec_norm"),
        (r"^output", "head"),
        (r"^layers", "base_model.layers"),
        (r"\.attention\.", ".attn."),
        (r"attn\.wq", "attn.in_proj.query"),
        (r"attn\.wk", "attn.in_proj.key"),
        (r"attn\.wv", "attn.in_proj.value"),
        (r"attn\.wo", "attn.dense"),
        (r"attention_norm", "ln"),
        (r"feed_forward\.w1", "ff_sub_layer.wg"),
        (r"feed_forward\.w2", "ff_sub_layer.w2"),
        (r"feed_forward\.w3", "ff_sub_layer.w1"),
        (r"ffn_norm", "ff_ln"),
    ]
    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

    return new_sd


serialization.register_adapter_step("llama_yoco", "meta_to_fms_names", _meta_to_fms_names)


def _hf_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = [
        (r"^lm_head.weight", "head.weight"),
        (r"^model.embed_tokens.weight", "base_model.embedding.weight"),
        (r"^model.norm", "base_model.dec_norm"),
        (r"^model.layers", "base_model.layers"),
        (r"self_attn\.k_proj", "attn.in_proj.key"),
        (r"self_attn\.v_proj", "attn.in_proj.value"),
        (r"self_attn\.q_proj", "attn.in_proj.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
    ]
    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param
    return new_sd


serialization.register_adapter_step("llama_yoco", "hf_to_fms_names", _hf_to_fms_names)


def _get_rope_params(linear_type: str) -> list[str]:
    if "gptq" in linear_type:
        return ["qweight", "scales", "qzeros", "bias"]
    elif "fp8" in linear_type:
        return ["weight", "weight_scale", "input_scale", "bias"]
    else:  # torch.nn.Linear
        return ["weight", "bias"]


def _hf_to_fms_rope(
    input_sd: Mapping[str, Any], model_config: Optional[LLaMAConfig] = None, **kwargs
) -> Mapping[str, Any]:
    new_sd = {}

    if model_config:
        head_size = model_config.emb_dim // model_config.nheads
    else:
        logger.warning("Missing model_config, assuming defaults for head_size")
        head_size = 128  # Good default for most models

    for name, param in input_sd.items():
        # Some checkpoints have weights in different precisions, which can have
        # auxiliary tensors (see _get_rope_params e.g. gptq, fp8).
        # Thus, we need to get rope_params per parameter.
        linear_type_str = "torch_linear"
        if model_config and model_config.linear_config:
            linear_type_str = get_linear_type(
                model_config.linear_config,
                module_name=name,
            )
        rope_params = _get_rope_params(linear_type_str)
        trans_required_pattern = re.compile(
            f"base_model.layers.[0-9]+.attn.in_proj.(query|key).({'|'.join(rope_params)})$"
        )

        # hf -> fms requires a transpose operation for the query and key
        # weight and bias parameters for Llama models
        # This transpose is due to the different implementation of RoPE in
        # HF and FMS. While FMS follows the original RoPE paper
        # (https://arxiv.org/abs/2104.09864), HF has its own implementation
        # that doesn't respect the order of outputs. This is OK as long as you
        # rearrange the weights of the query and key projections, as the
        # combination projection + RoPE ends up producing the same outputs.
        # Therefore, to make FMS produce the correct order of outputs when
        # loading from an HF checkpoint, we need to undo the transformation
        # that HF does from the original Meta weights:
        if bool(trans_required_pattern.match(name)) and param.size(0) > 1:
            temp = param
            if "gptq" in linear_type_str and temp.dim() == 2:
                # GPTQ qweights are [in_feat, out_feat] (unlike usual [out_feat, in_feat])
                # and are fully transposed before & after process
                temp = temp.transpose(0, 1)
            # num_heads is used in the transformation required for hf->fms
            # can't be precomputed because q and k might have different num_heads
            num_heads = temp.size(0) // head_size

            if temp.dim() == 2:  # weight
                temp_view = temp.view(num_heads, 2, -1, temp.size(1))
            else:  # bias
                temp_view = temp.view(num_heads, 2, -1)
            temp = temp_view.transpose(1, 2).reshape(*temp.size())

            if "gptq" in linear_type_str and temp.dim() == 2:
                temp = temp.transpose(0, 1)

            new_sd[name] = temp
        else:
            new_sd[name] = param

    return new_sd


serialization.register_adapter_step("llama_yoco", "hf_to_fms_rope", _hf_to_fms_rope)

serialization.register_adapter("llama_yoco", "meta", ["meta_to_fms_names", "weight_fusion"])
serialization.register_adapter(
    "llama_yoco",
    "hf",
    [
        "hf_to_fms_names",
        "hf_to_fms_rope",
        "hf_gptq_fusion_check",
        "weight_fusion",
    ],
)
serialization.register_adapter(
    "llama_yoco",
    "fms.pre0.0.6",
    ["pre0.0.6_attn_unfused_to_fused", "swiglu_unfused_to_fused", "weight_fusion"],
)
