import torch
import torch.nn as nn


# The DEF of the fms_ops library has to happen ONLY ONCE
if not hasattr(torch, "_FMS_OPS_DEF_DONE"):
    lib = torch.library.Library("fms_ops", "DEF")
    lib.define(
        "universal_attention(Tensor q, Tensor k, Tensor v, bool causal, float sm_scale, "
        "Tensor? static_src=None, Tensor? static_dest=None, bool warp_specialize=True, bool ac=False) "
        "-> (Tensor, Tensor)"
    )
    torch._FMS_OPS_DEF_DONE = True


class UninitializedModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise RuntimeError("I haven't been initialized yet!")

    def initialize(self, name):
        raise RuntimeError("I have to be replaced by a child class!")
