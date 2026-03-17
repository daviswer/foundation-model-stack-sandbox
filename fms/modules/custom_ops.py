import torch
from torch import Tensor
import torch.utils.checkpoint as cp

from typing import Optional, Tuple

from ._universal_attention import _attention_ac_op as _universal_attention_func # where your torch.autograd.Function lives

if not hasattr(torch, "_FMS_UA_IMPL_DONE"):
    lib = torch.library.Library("fms_ops", "IMPL")

    def _universal_attention_impl(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool,
        sm_scale: float,
        static_src: Optional[Tensor] = None,
        static_dest: Optional[Tensor] = None,
        warp_specialize: bool = True,
        ac: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        return _universal_attention_func.apply(q, k, v, causal, sm_scale, static_src, static_dest, warp_specialize, ac)

    lib.impl("universal_attention", _universal_attention_impl, "CompositeExplicitAutograd")

    @torch.library.register_fake("fms_ops::universal_attention")
    def _universal_attention_fake(q, k, v, causal, sm_scale, static_src=None, static_dest=None, warp_specialize=True, ac=True):
        out = torch.empty_like(q)
        last_aff = q.new_empty((k.shape[0], k.shape[1], q.shape[2]))
        return out, last_aff

    torch._FMS_UA_IMPL_DONE = True

def universal_attention_op(*args, **kwargs):
    return torch.ops.fms_ops.universal_attention(*args, **kwargs)
