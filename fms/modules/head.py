from typing import Dict, Optional, Set

import torch
import torch.distributed
import torch.nn as nn
from torch.distributed.distributed_c10d import ProcessGroup

from fms import distributed
from fms.distributed.tensorparallel import (
    all_gather_from_tensor_model_parallel_region,
    copy_to_tensor_model_parallel_region,
)
from fms.modules.tp import TPModule
from fms.modules.layernorm import LayerNormParameterized


class MLPClassificationHead(nn.Module):
    """
    A general purpose Classification Head. When applied on the output of a
    Headless model, will project from the embedding space to a space equal to
    the number of classes provided.
    """

    def __init__(
        self,
        emb_dim: int,
        num_classes: int,
        activation_fn: nn.Module,
        layer_norm: Optional[nn.Module] = None,
        dense_bias: bool = True,
        head_bias: bool = True,
        dropout: float = 0.0,
        do_pooling: bool = False,
    ):
        """
        Initialize a MLPClassificationHead

        Parameters
        ----------
        emb_dim: int
            the embedding dimension
        num_classes: int
            the output number of classes
        activation_fn: nn.Module
            the activation function to use prior to apply the dense layer
        layer_norm: nn.Module, optional
            the layer norm to apply prior to running the model head, (default is no layer_norm)
        dense_bias: bool
            the bias param in the dense layer (default is True)
        head_bias: bool
            the bias param in the head layer (default is True)
        dropout: float
            the dropout to use directly after activation (default is 0.0)
        do_pooling: bool
            if True, will take the first token for each sequence in the batch as input to the dense layer. Otherwise,
            use the entire sequence as input to the dense layer
        """
        super().__init__()
        self.dense = nn.Linear(emb_dim, emb_dim, bias=dense_bias)
        self.act = activation_fn
        self.dropout = nn.Dropout(dropout)
        self.ln = layer_norm
        self.head = nn.Linear(emb_dim, num_classes, bias=head_bias)
        self.do_pooling = do_pooling

    def forward(self, x: torch.Tensor):
        """Run the forward method of a classification head

        Parameters
        ----------
        x: torch.Tensor
            typically the output from a headless model

        Returns
        -------
        torch.Tensor
            a tensor projected to a space given by num_classes
        """
        if self.do_pooling:
            x = x[:, 0]
        x = self.dense(x)
        x = self.act(x)
        x = self.dropout(x)
        if self.ln is not None:
            x = self.ln(x)
        x = self.head(x)
        return x
    

class CFGHead(nn.Module):
    def __init__(self, emb_dim, vocab_size, mix_denom):
        super().__init__()
        self.d = emb_dim
        self.v = vocab_size
        self.mlp = nn.Sequential(
            nn.Linear(2*emb_dim, 2*emb_dim, bias=False),
            nn.SiLU(),
            nn.Linear(2*emb_dim, emb_dim, bias=False),
            LayerNormParameterized(emb_dim, use_high_precision_pow=True),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.mix_coeff = 0 if mix_denom==0 else 1/mix_denom
        self.inp_ln = LayerNormParameterized(emb_dim, use_high_precision_pow=True)
        self.zeroed = False

    def reset_parameters(self):
        for layer in [self.mlp[0], self.mlp[2]]:
            nn.init.trunc_normal_(
                layer.weight,
                mean=0.0,
                std=0.02,
            )
        self.inp_ln.reset_parameters()
        self.mlp[3].weight.data.fill_(1/self.d**.5)  # reset_parameters()  # This only works when low_cpu_fsdp is False!!!

    def forward(self, latent, embeds, targ, head, zl_coeff):
        # latent: b n d  (0...n-1)
        # embeds: b n d  (0...n-1)
        # targ: b n  (1...n)
        targ = targ.long()
        pred = head(latent)  # b n v  (0...n-1)
        prior_embeds = self.inp_ln(embeds.roll(1, dims=1))
        prior_embeds[:,0] = 0  # (_, 0...n-2)
        dumb_pred = torch.cat((latent, prior_embeds), dim=2)  # b n 2d
        dumb_pred = self.mlp(dumb_pred)  # b n d
        dumb_pred = head(dumb_pred)  # b n d  (_, 1...n-1)
        dumb_loss = self.criterion(dumb_pred.reshape(-1, self.v), targ.view(-1)) + zl_coeff * torch.logsumexp(dumb_pred, dim=-1).pow(2).mean()
        with torch.no_grad():
            loss = self.criterion(pred.reshape(-1, self.v), targ.view(-1)) + zl_coeff * torch.logsumexp(pred, dim=-1).pow(2).mean()
        train_pred = pred.mul(1-self.mix_coeff).add(dumb_pred.mul(self.mix_coeff))
        train_loss = self.criterion(train_pred.reshape(-1, self.v), targ.view(-1))
        train_loss = train_loss + zl_coeff * torch.logsumexp(train_pred, dim=-1).pow(2).mean()
        return dumb_loss, loss, train_loss


class LinearClassificationHead(nn.Linear):
    # To differentiate for TP
    def forward(self, input):
        return super().forward(input)
    
    def reset_parameters(self):
        self.weight.data.normal_(0, self.weight.data.numel()**-.25)

    def to_tp(self, group: ProcessGroup) -> "TPLinearClassificationHead":
        return TPLinearClassificationHead.import_module(self, group)


class TPLinearClassificationHead(LinearClassificationHead, TPModule):
    """
    Output embedding layer for language models.

    Args
    ----
    Check nn.Linear for up-to-date docs

    world_size: int
        the number of processes running this model in TP
    rank: int
        the index of this process wrt to the rest running the model in TP
    """

    def __init__(
        self,
        vocab_size,
        emb_dim,
        bias=False,
        device=None,
        dtype=None,
        group: Optional[ProcessGroup] = None,
    ):
        assert torch.distributed.is_initialized()
        rank, world_size = distributed.rank_and_world(group)
        assert vocab_size % world_size == 0, (
            "The number of tokens must be divisible by world size"
        )
        LinearClassificationHead.__init__(
            self,
            emb_dim,
            vocab_size // world_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.setup_tp(rank, group)

    @staticmethod
    def import_module(
        head: LinearClassificationHead, group: ProcessGroup
    ) -> "TPLinearClassificationHead":
        tp_lmh = TPLinearClassificationHead(
            vocab_size=head.out_features,
            emb_dim=head.in_features,
            bias=head.bias,
            device=head.weight.device,
            dtype=head.weight.dtype,
            group=group,
        )
        return tp_lmh

    def load_weights(
        self,
        tensor_values: Dict[str, torch.Tensor],
    ):
        # 1. Grab the weights from tensor_values
        used_keys: Set[str] = set()
        head_weight = self._get_sd_weight(tensor_values, used_keys, ["weight"])
        if self.bias is not None:
            head_bias = self._get_sd_weight(tensor_values, used_keys, ["bias"])

        # 2. Raise exceptions
        if len(tensor_values) > (2 if self.bias is not None else 1):
            unused_keys = set(tensor_values.keys()).difference(used_keys)
            raise AttributeError(f"Unused weight(s): {', '.join(unused_keys)}")

        # 3. Load and shard the weights
        self.sharded_copy(self.weight, head_weight, 0, [self.world_size])
        if self.bias is not None:
            self.sharded_copy(self.bias, head_bias, 0, [self.world_size])

    def forward(self, inp):
        # vocab_idx: b n d if reverse, else b n
        inp_par = copy_to_tensor_model_parallel_region(inp, self.group)
        out_par = LinearClassificationHead.forward(self, inp_par)
        return all_gather_from_tensor_model_parallel_region(
            out_par, self.rank, self.group
        )
