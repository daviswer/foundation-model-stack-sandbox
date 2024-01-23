import time
from typing import Any, Callable, List, MutableMapping, Union, Optional

import torch
import torch.nn.functional as F
from torch import distributed as dist

from fms.modules.positions import compute_position_ids
from fms.modules.speculator import Speculator
from fms.utils.cache import KVCacheManager, CacheDataWithMetadata, flatten_batch, select_inflate_dim
from fms.utils.cache.expandable import ExpandableKVCacheManager
from fms.utils.cache.paged import PagedKVCacheManager


def _make_cache_contiguous(past_key_value_states):
    # kv updates are required for torch.compile with
    # mode='reduce-overhead'
    n_kv_s: List[List[torch.Tensor]] = []
    for layer_idx in range(len(past_key_value_states)):
        n_kv_s.append([])
        for tensor_idx in range(len(past_key_value_states[layer_idx])):
            n_kv_s[layer_idx].append(
                past_key_value_states[layer_idx][tensor_idx]
                .clone(memory_format=torch.contiguous_format)
                .detach()
            )
            # torch._dynamo.mark_dynamic(n_kv_s[layer_idx][tensor_idx], 2)
    return n_kv_s


def generate(
    model: Union[Callable, torch.nn.Module],
    input_ids: torch.Tensor,
    max_seq_len: int = 2048,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 10,
    do_sample: bool = True,
    num_beams: int = 1,
    use_cache: bool = False,
    kv_cache_manager: Optional[KVCacheManager] = None,
    contiguous_cache: bool = False,
    expand: bool = False
):
    """
    A trivial generate function that can be used for validation/testing in
    cases where HF is not available.
    We could add implementations for other types of generation, but this is
    enough for making sure a model is working.
    Does not implement batching nor beam search, but those could be added.

    Args:
        model: A function or nn.Module that takes a batch of input_ids and
            returns logits
        prefix: A tensor of token IDs.
        max_seq_len: the sequence length of the model
        max_new_tokens: max tokens to generate
        temperature: temperature of softmax when sampling
        top_k: only search among top k tokens
        do_sample: multinomial sampling. False for greedy.
        num_beams: TODO: support beam search
        use_cache: requires that the model accept use_cache and
            past_key_value_states args in forward method.
    """

    # Build padded batched input tensor
    max_len = max([seq.size(0) for seq in input_ids])
    n_pads_init = [max_len - seq.size(0) for seq in input_ids]
    input_ids = torch.stack(
        [F.pad(input_ids[i], (n_pads_init[i], 0)) for i in range(len(input_ids))]
    )

    batched = False
    if num_beams != 1:
        raise NotImplementedError("generate() does yet not support beam search")
    if type(input_ids) == torch.Tensor:
        if input_ids.dim() != 1:
            batched = True
    else:
        raise RuntimeError("generate() requires a tensor of token ids as the prefix")

    if not batched:
        input_ids = input_ids.unsqueeze(0)

    result = input_ids
    next_input = input_ids
    kwargs: MutableMapping[str, Any] = dict()
    kwargs["use_cache"] = use_cache

    if use_cache:
        kwargs["cache_data"] = None
        if kv_cache_manager is None:
            # TODO: standardized way of getting nlayers, nheads, emb_dim
            kv_cache_manager = ExpandableKVCacheManager(
                model.config.nlayers,  # type: ignore
                model.config.nheads,  # type: ignore
                model.config.emb_dim,  # type: ignore
                tensor_parallel_size=dist.get_world_size(),
                dtype=torch.get_default_dtype(),
                device=model.device,  # type: ignore
            )
    def _time():
        torch.cuda.synchronize()
        return time.time()
    times = {"forward_pass":0}
    for i in range(max_new_tokens):

        input_ids = next_input[:, -max_seq_len:]

        if i == 1:
            start_time = time.time()
            if expand:
                # Inflate cache
                parent_sequence_ids = cache_data.sequence_ids
                child_sequence_ids_list = []
                child_sequence_ids_flattened = []
                # each parent will have top_k*n_adds child sequences
                for parent_sequence_id in parent_sequence_ids:
                    child_sequence_ids = kv_cache_manager.add_child_sequences(parent_sequence_id, top_k*4)
                    child_sequence_ids_list.append(child_sequence_ids)
                    child_sequence_ids_flattened.extend(child_sequence_ids)
                sequence_ids = child_sequence_ids_flattened
                input_ids = torch.cat([input_ids]*4*top_k, dim=0)
                result = torch.cat([result]*4*top_k, dim=0)
        

        # compute the mask
        if not use_cache or i == 0:
            is_pad = input_ids == 0
            mask = is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)
            kwargs["mask"] = mask.tril(diagonal=0)
        else:
            kwargs["mask"] = None
            # is_pad = result == 0
            # mask = is_pad.unsqueeze(-1)
            # kwargs["mask"] = mask

        # get the cache data and position ids if using cache
        if use_cache and kv_cache_manager:
            if i == 0:
                num_tokens_per_sequence = torch.count_nonzero(
                    input_ids.T, dim=0
                ).tolist()
                cache_data: CacheDataWithMetadata = (
                    kv_cache_manager.allocate_prompt_tokens(num_tokens_per_sequence)
                )
                # context lengths here actually have the real lengths, but we want to start at 0 for first iteration
                # might want to have 2 variables for this, but for now, just keep as is
                context_lengths: Optional[List[int]] = None
            else:
                num_tokens_per_sequence = [1 for _ in range(input_ids.size(0))]
                cache_data = kv_cache_manager.allocate_generated_tokens(
                    sequence_ids, num_tokens_per_sequence
                )
                context_lengths = cache_data.context_lengths.tolist()

                # todo: is this supported?
                # if contiguous_cache:
            sequence_ids: List[int] = cache_data.sequence_ids
            position_ids = compute_position_ids(
                num_tokens_per_sequence, context_lengths
            )

            kwargs["cache_data"] = cache_data
            kwargs["position_ids"] = torch.tensor(position_ids, device=input_ids.device)

        if i > 0:
            _start = _time()
        output = model(input_ids, **kwargs)
        if use_cache:
            logits, _ = output
        else:
            logits = output
        logits = logits[:, -1, :]
        if i > 0:
            times["forward_pass"] += _time() - _start

        if do_sample:
            # get logits from last value in sequence nad scale
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_val = torch.multinomial(probs, num_samples=1)
        else:
            next_val = torch.argmax(logits, dim=-1).unsqueeze(0).t()

        result = torch.cat((result, next_val), dim=-1)

        if use_cache:
            next_input = next_val
        else:
            next_input = result

    if not batched:
        result = result[0]

    if (
        use_cache
        and kv_cache_manager
        and callable(getattr(kv_cache_manager, "free_sequences", None))
    ):
        if expand:
            for child_sequence_id in child_sequence_ids_flattened:
                kv_cache_manager.free(child_sequence_id)
            kv_cache_manager.free_sequences(parent_sequence_ids)
        else:
            kv_cache_manager.free_sequences(sequence_ids)  # type: ignore

    end_time = time.time()
    return result, max_new_tokens, (end_time - start_time), times


def truncate_after_eos(result, eos_token_id):
    """
    Helper function to return a truncated sequence of token IDs stopping at
    (and including) the 'end of sentence' token.
    Currently only handles unbatched sequences.
    """
    if eos_token_id is None:
        return result

    eos_idx = torch.where(result == eos_token_id)
    eos_idx = eos_idx[0]
    if eos_idx.shape[0] >= 1:
        eos_idx = eos_idx[0].item()
        result = result[: eos_idx + 1]
    return result

 
def speculative_generate(
    model: Union[Callable, torch.nn.Module],
    input_ids: Union[torch.Tensor, List[torch.Tensor]],
    speculator: Speculator,
    max_seq_len: int = 2048,
    new_tokens: int = 256,
    top_k: int = 5,
    threshes=[5, 3, 2],
    verbose_dict=None,
    kv_cache_manager: PagedKVCacheManager = None,
):
    """
    A reference implementation of speculative decoding generation.
    Returns at least the specified number of tokens - the speculator may return a
    few extra in the final step.
    If input is batched, continues generating until EVERY sequence has produced AT LEAST the required number of tokens.
    Input (and output) tokens beyond max_seq_len are simply dropped for a sliding-window approach.
    Currently reproduces behavior of greedy decoding only.
    Args:
        model: A function or nn.Module that takes a batch of input_ids and
            returns logits
        input_ids: A length n tensor of token IDs, or list of such tensors
        speculator: A function or nn.Module that takes a state vector and sampled token
            and returns a set of candidate suffixes
        max_seq_len: the sequence length of the base model
        new_tokens: number of tokens to generate
        top_k: only score the top k candidates from the speculator
        threshes: use top k predictions from each head to generate speculator candidate pool
        verbose_dict: Optional HF tokenizer vocab dict. If provided, runs verbosely and prints
            speculator behavior and scoring for each step
    Returns:
        result: List of id tensors, possibly different lengths if batching.
        n_steps: Number of foward passes used to generate provided tokens.
    """

    verbose = False
    if verbose_dict is not None:
        verbose = True
        vinv = {v: k for k, v in verbose_dict.items()}

    def decode_obo(x, vinv):
        return [vinv[z] for z in x.squeeze().tolist()]

    # Construct batch(es) and initial inputs
    bsize = len(input_ids)
    result = input_ids  # [b] n
    # Build padded batched input tensor
    max_len = max([seq.size(0) for seq in input_ids])
    n_pads_init = [max_len - seq.size(0) for seq in input_ids]
    n_pads = torch.Tensor(n_pads_init).to(device=input_ids[0].device, dtype=torch.int)
    inputs = torch.stack(
        [F.pad(input_ids[i], (n_pads_init[i], 0)) for i in range(bsize)]
    )
    num_tokens_per_sequence = torch.count_nonzero(
        inputs[:, :-1].T, dim=0
    ).tolist()
    cache_data = kv_cache_manager.allocate_prompt_tokens(num_tokens_per_sequence)
    parent_sequence_ids = cache_data.sequence_ids
    # Build padded causal mask
    mask = torch.ones(
        bsize,
        1,
        inputs.size(1) - 1,
        inputs.size(1) - 1,
        device=inputs.device,
    )
    mask = mask.tril()  # b 1 n-1 n-1
    # Mask off any left-pads
    pad_mask = torch.arange(mask.size(3), device=mask.device).view(
        1, 1, 1, -1
    )  # 1 1 1 n-1
    pad_mask = pad_mask.expand(bsize, 1, 1, -1)  # b 1 1 n-1
    pad_mask = pad_mask.sub(n_pads.sub(1).view(-1, 1, 1, 1)).clamp(0, 1)
    eye = torch.eye(mask.size(3), device=mask.device)[None, None, :, :]  # 1 1 n-1 n-1
    mask = mask.mul(pad_mask).logical_or(eye).log()  # b 1 n-1 n-1
    # Handle position_ids
    pos_ids = torch.arange(mask.size(3), device=inputs.device).repeat(bsize, 1)  # b n-1
    pos_ids -= n_pads[:, None]

    kwargs: MutableMapping[str, Any] = dict()
    kwargs["use_cache"] = True

    # Build kv cache and get initial state vector
    n_adds = speculator.n_predict + 1
    inputs = inputs[:, -max_seq_len + n_adds :]
    position_ids = torch.tensor(compute_position_ids(num_tokens_per_sequence), dtype=torch.int64, device=inputs.device)
    def _time():
        torch.cuda.synchronize()
        return time.time()
    fields = ["step0", "child_sequencing", "create_candidates", "forward_pass", "score_candidates", "best_guess", "toss_children", "update_inputs"]
    times = {k:0 for k in fields}
    _start = _time()
    output = model(
        inputs[:, :-1],
        include_embeds=True,
        position_ids=position_ids,
        mask=mask,
        cache_data=cache_data,
        **kwargs
    )
    times["step0"] = _time() - _start
    _, _, embeds = output
    embeds = embeds[:, -1:]

    n_gen = torch.zeros(bsize, device=inputs.device, dtype=torch.int)
    n_steps = 0
    inputs = inputs[:, -1:]
    start_time = _time()
    # while min(n_gen) < new_tokens:
    for _ in range(new_tokens):
        n_steps += 1

        _start = _time()
        # create candidate sequences
        child_sequence_ids_list = []
        child_sequence_ids_flattened = []
        num_tokens_per_sequence = [n_adds for _ in range(inputs.size(0) * top_k)]
        # each parent will have top_k child sequences
        for parent_sequence_id in parent_sequence_ids:
            child_sequence_ids = kv_cache_manager.add_child_sequences(parent_sequence_id, top_k)
            child_sequence_ids_list.append(child_sequence_ids)
            child_sequence_ids_flattened.extend(child_sequence_ids)

        # add n_adds tokens to each candidate
        cache_data = kv_cache_manager.allocate_generated_tokens(child_sequence_ids_flattened, num_tokens_per_sequence)
        position_ids = torch.tensor(compute_position_ids(num_tokens_per_sequence, cache_data.context_lengths.tolist()), 
                                    dtype=torch.int64, device=inputs.device) # bk 1+h
        times["child_sequencing"] += _time()-_start

        # Get candidate set of speculations
        _start = _time()
        adds = speculator.generate_suffixes(embeds, inputs, threshes, top_k)  # b k h
        inputs = torch.cat(
            [inputs.unsqueeze(1).expand(bsize, top_k, 1), adds], dim=-1
        ).int()  # b k 1+h
        flat_inputs, unflat_indices, flat_indices = flatten_batch(inputs) # b', b k 1+h
        flat_inputs = flat_inputs[None,] # 1 b'
        cache_data.unflatten_indices = unflat_indices
        cache_data.flatten_indices = flat_indices
        position_ids = select_inflate_dim(position_ids.view(-1), flat_indices)[None,]
        times["create_candidates"] += _time()-_start

        # Base model forward pass
        _start = _time()
        output = model(
            flat_inputs, include_embeds=True, position_ids=position_ids, cache_data=cache_data, **kwargs
        ) # 1 b' v
        logits, _, embeds = output # 1 n' v, 1 n' d
        next_vals = torch.argmax(logits, dim=-1)  # 1 n'
        unflat_indices = unflat_indices.view(-1, unflat_indices.size(2))
        next_vals = select_inflate_dim(next_vals[0], unflat_indices) # bk 1+h
        embeds = select_inflate_dim(embeds[0], unflat_indices) # bk 1+h d
        times["forward_pass"] += _time()-_start

        # Check correctness of speculator predictions
        _start = _time()
        test = inputs.view(-1, n_adds).roll(-1, 1).eq(next_vals).cumprod(1)
        n_correct = (
            test.sum(1).clamp(0, n_adds - 1).view(bsize, top_k)
        )  # clamp in case pred[0]==targ[-1]
        print(n_correct)
        best_guess = n_correct.argmax(1)  # b
        best_guess_unflat = (
            best_guess.unsqueeze(1).expand(bsize, n_adds).unsqueeze(1)
        )  # b 1 1+h
        times["score_candidates"] += _time()-_start

        # Set global values to those of best guess
        _start = _time()
        next_vals = next_vals.view(bsize, top_k, n_adds).gather(1, best_guess_unflat).squeeze(1)  # b 1+h
        n_correct = n_correct.gather(1, best_guess.unsqueeze(1)).squeeze(1)  # b
        embeds = embeds.view(bsize, top_k, *embeds.size()[1:]).gather(
            1, best_guess_unflat.unsqueeze(3).expand(-1, -1, -1, embeds.size(2))
        ).squeeze(1)  # b 1+h d
        times["best_guess"] += _time()-_start

        if verbose:
            test = inputs.view(bsize, top_k, n_adds).gather(1, best_guess_unflat).squeeze(1)
            for i, line in enumerate(test):
                print(
                    "Speculation:",
                    decode_obo(line, vinv),
                    "n_correct:",
                    n_correct[i].item(),
                )

        # free all worst candidates and keep best candidates as parents
        _start = _time()
        parent_sequence_ids = []
        for parent_index, child_sequence_ids in enumerate(child_sequence_ids_list):
            best_index = best_guess[parent_index].item()

            # free all bad candidates
            kv_cache_manager.free_sequences(child_sequence_ids[:best_index] + child_sequence_ids[best_index + 1:])

            # decrease the context length of the sequence which used to be sequence length + n_adds by the number of incorrect tokens
            # for the correct candidate
            best_sequence_id = child_sequence_ids[best_index]
            parent_sequence_ids.append(best_sequence_id)
            kv_cache_manager.remove_tokens(best_sequence_id, n_adds - n_correct[parent_index].item() - 1)
        times["toss_children"] += _time()-_start

        # Toss any wrong speculator tokens
        _start = _time()
        next_vals_split = list(next_vals)
        next_vals_split = [
            next_vals_split[i][: n_correct[i] + 1] for i in range(len(next_vals_split))
        ]  # [b] h'
        n_gen += n_correct + 1
        embeds = embeds.gather(
            1, n_correct.view(-1, 1, 1).expand(-1, -1, embeds.size(2))
        )  # Grab last correct embed

        # Update results
        result = [
            torch.cat((result[i], next_vals_split[i]), dim=0) for i in range(bsize)
        ]
        inputs = torch.stack([line[-1:] for line in next_vals_split], dim=0)  # b 1
        times["update_inputs"] += _time()-_start

        if verbose:
            for line in result:
                print("Updated output:", decode_obo(line, vinv))
            print()

    for parent_sequence_id in parent_sequence_ids:
        prefix = kv_cache_manager.cbg_map[parent_sequence_id].prefix
        kv_cache_manager.free(parent_sequence_id)
        while prefix is not None:
            kv_cache_manager.free(prefix.sequence_id)
            prefix = prefix.prefix

    end_time = _time()
    return result, n_steps, (end_time - start_time), times
