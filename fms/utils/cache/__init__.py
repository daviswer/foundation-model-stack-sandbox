import abc
import dataclasses
from typing import Tuple, List, Optional
import torch


@dataclasses.dataclass
class CacheDataLayer(metaclass=abc.ABCMeta):
    data_layer: Tuple[torch.Tensor, torch.Tensor]

    @abc.abstractmethod
    def store(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


@dataclasses.dataclass
class CacheData(metaclass=abc.ABCMeta):
    data: List[Tuple[torch.Tensor, torch.Tensor]]

    @abc.abstractmethod
    def get_layer(self, layer_index: int) -> CacheDataLayer:
        pass

    @abc.abstractmethod
    def is_filled(self) -> bool:
        pass


@dataclasses.dataclass
class CacheDataWithMetadata(CacheData):
    data: List[Tuple[torch.Tensor, torch.Tensor]]
    sequence_ids: List[int]
    max_sequence_length: int
    context_lengths: torch.Tensor


class KVCacheManager(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def allocate_prompt_tokens(
        self, num_tokens_per_sequence: List[int]
    ) -> CacheDataWithMetadata:
        pass

    @abc.abstractmethod
    def allocate_generated_tokens(
        self, sequence_ids: List[int], num_tokens_per_sequence: List[int]
    ) -> CacheDataWithMetadata:
        pass


KVCache = Tuple[torch.Tensor, torch.Tensor]  # (key cache, value cache)


@dataclasses.dataclass
class OutOfPlaceCacheDataLayer(CacheDataLayer):
    data_layer: Tuple[torch.Tensor, torch.Tensor]

    def store(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.data_layer is not None:
            self.data_layer = (
                torch.cat((self.data_layer[0], keys), dim=2),
                torch.cat((self.data_layer[1], values), dim=2),
            )
            keys, values = self.data_layer
        return keys, values


class OutOfPlaceCacheData(CacheData):
    def __init__(self, data: List[Tuple[torch.Tensor, torch.Tensor]]):
        self.data = data

    def get_layer(self, layer_index: int) -> OutOfPlaceCacheDataLayer:
        return OutOfPlaceCacheDataLayer(data_layer=self.data[layer_index])

    def is_filled(self) -> bool:
        return self.data[0] is not None




def flatten_batch(inp):
    # Takes a bsize x n_candidates x candidate_len rectangular batch of input indices
    # Returns 1) a flattened set of indices with all redundant tokens removed,
    # and 2) a tensor, sized as input, mapping each input token to its slot in output,
    # and 3) a tensor, sized as output, mapping each output token to slot in flattened input
    ind_out = torch.zeros_like(inp)
    inp = inp.tolist()
    out = []
    ind_flat = []
    batch_offset = 0
    for b,candidate_set in enumerate(inp):
        lineages = []
        for k,candidate in enumerate(candidate_set):
            for n in range(len(candidate)):
                lineage = tuple(candidate[:n+1])
                if lineage in lineages:
                    # Token is redundant
                    ind_out[b,k,n] = lineages.index(lineage)+batch_offset
                else:
                    # Token is not redundant
                    ind_out[b,k,n] = len(lineages)+batch_offset
                    lineages.append(lineage)
                    ind_flat.append(b*len(inp[0])*len(inp[0][0]) + k*len(inp[0][0]) + n)
        out.append(torch.Tensor([lineage[-1] for lineage in lineages], 
                                device=ind_out.device, dtype=torch.int32))
        batch_offset += len(lineages)
    return torch.cat(out), ind_out, torch.Tensor(ind_flat, device=ind_out.device, dtype=torch.int32)

def select_inflate_dim(inp, inds, dim=0):
    # Takes a flattened input of size ([...] x n x [...]) with n in slot dim, and token mappings of size (a x ... x z)
    # and over/under-samples on n to create output tensor with size ([...] x a x ... x z x [...])
    inds_shape = inds.size()
    inp_shape = inp.size()
    out = inp.index_select(dim, inds.view(-1))
    return out.view(*inp_shape[:dim],*inds_shape,*inp_shape[dim+1:])