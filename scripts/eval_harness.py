import argparse
import os

import lm_eval
import torch
import torch._inductor.config
from lm_eval.utils import make_table
from torch import distributed as dist
from torch.distributed._shard.checkpoint import FileSystemReader, load_state_dict

from fms.models.llama import LLaMA, LLaMAConfig
from fms.utils import evaluation, tokenizers


"""
Example use:
```
srun -N 1 --gres=gpu:1 --cpus-per-task=12 --mem=128G --unbuffered --gres-flags=enforce-binding  python scripts/eval_harness.py --model_path=~/models/7B-F/ --tokenizer=~/models/tokenizer.model --model_source=meta --tasks=hellaswag --num_fewshot=10

|  Tasks  |Version|Filter|n-shot| Metric |Value |   |Stderr|
|---------|-------|------|-----:|--------|-----:|---|-----:|
|hellaswag|Yaml   |none  |    10|acc     |0.5915|±  |0.0049|
|         |       |none  |    10|acc_norm|0.7713|±  |0.0042|
```
"""


parser = argparse.ArgumentParser(description="Script to evaluate a causal model")
parser.add_argument("--device_type", type=str, default="cuda")
parser.add_argument(
    "--architecture",
    type=str,
    default="llama",
    help="The model architecture to benchmark",
)
parser.add_argument(
    "--variant",
    type=str,
    default="7b",
    help="The model variant (configuration) to benchmark. E.g. 7b, 13b, 70b.",
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the directory containing LLaMa weights (.pth files sharded by tensor parallel rank, not HF weights)",
)
parser.add_argument(
    "--model_source",
    type=str,
    help="Source of the checkpoint. E.g. 'meta', 'hf', None",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    required=True,
    help="Path to the tokenizer (e.g. ~/tokenizer.model)",
)
parser.add_argument(
    "--no_use_cache",
    action="store_false",
    help="Disable the kv-cache (on by default)",
)
parser.add_argument(
    "--compile",
    action="store_true",
    help="Use torch.compile (slow for first inference pass)",
)
parser.add_argument(
    "--compile_mode",
    type=str,
    help="Mode for compilation",
    default="default",
    choices=["default", "reduce-overhead"],
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Set torch.use_deterministic_algorithms? Requires env variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
)
parser.add_argument("--tasks", type=str, help="Task names to pass to lm_eval")
parser.add_argument(
    "--num_fewshot",
    type=int,
    default=None,
    help="Number of examples in few-shot context",
)

args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
if args.device_type == "cuda":
    device = torch.device(args.device_type, local_rank)
    torch.cuda.set_device(device)
else:
    device = torch.device(args.device_type)

torch.set_default_dtype(torch.half)

# requires setting environment variable: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
if args.deterministic:
    torch.use_deterministic_algorithms(True)

if args.distributed:
    dist.init_process_group()
    # Fix until PT 2.3
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

print("loading model")
if args.distributed:
    distr_param = "tp"
else:
    if torch.cuda.device_count() > 1 and world_size == 1:
        distr_param = "mp"
    else:
        distr_param = None

# model = get_model(
#     args.architecture,
#     args.variant,
#     model_path=args.model_path,
#     device_type=args.device_type,
#     source=args.model_source,
#     distributed_strategy=distr_param,
#     group=dist.group.WORLD,
# )
# c = LLaMAConfig(
#     nlayers=24,
#     nheads=16,
#     kvheads=8,
#     emb_dim=2048,
#     # hidden_grow_factor=3
# )
c = LLaMAConfig(
    src_vocab_size=128256,
    emb_dim=3072,
    nheads=24,
    kvheads=8,
    nlayers=24,
    hidden_grow_factor=8/3,
    max_expected_seq_len=4096,
    rope_theta=10000.0,
)
model = LLaMA(c)
d = {"model_state": {"_orig_mod": model.state_dict()}}
load_state_dict(
    state_dict=d, 
    storage_reader=FileSystemReader(args.model_path), 
    no_dist=True
)
model.load_state_dict(d["model_state"]["_orig_mod"])

# d = torch.load(args.model_path)['model_state']
# d = {k[10:]:v for k,v in d.items()}
# # d = {k[10:]:q for k,q in d.items()}
# # for i in range(24):
# #     x = d.pop(f"layers.{i}.ff_sub_layer.wg1_fused.weight")
# #     d[f"layers.{i}.ff_sub_layer.wg.weight"] = x[:x.size(0)//2]
# #     d[f"layers.{i}.ff_sub_layer.w1.weight"] = x[x.size(0)//2:]
# model.load_state_dict(d)

model = model.to(device)
tokenizer = tokenizers.get_tokenizer(args.tokenizer)
model.eval()
torch.set_grad_enabled(False)
print("loading complete on rank", local_rank)

if args.compile:
    print("compiling model")
    # Bug with kv-cache in PT2.1
    torch._inductor.config.joint_graph_constant_folding = False
    # compiling can make first inference pass slow
    model = torch.compile(model, mode=args.compile_mode)

lm_obj = evaluation.FMSEvalHarnessLM(
    model=model, 
    tokenizer=tokenizer, 
    device=device, 
    rank=local_rank, 
    world_size=world_size,
)

results = lm_eval.simple_evaluate(
    model=lm_obj,
    tasks=args.tasks.split(","),
    num_fewshot=args.num_fewshot,
)
print(make_table(results))
if "groups" in results:
    print(make_table(results, "groups"))
