import torch
import time
import torch.nn.functional as f

def _time():
    torch.cuda.synchronize()
    return time.time()

bsizes = [1,2,4,8,16]
times = {b:0 for b in bsizes}
for bsize in bsizes:
    print(f"Benchmarking bsize {bsize}...")
    for _ in range(100):
        # inp = torch.randn(bsize, 40, 2048, 128).bfloat16().cuda()
        inp = torch.randn(bsize, 2048, 5120).bfloat16().cuda()
        m = torch.randn(5120,5120).bfloat16().cuda()
        _start = _time()
        inp.matmul(m)
        # with torch.backends.cuda.sdp_kernel(
        #     enable_flash=True,
        #     enable_math=False,
        #     enable_mem_efficient=False,
        # ):
        #     f.scaled_dot_product_attention(inp, inp, inp, is_causal=True)
        times[bsize] += _time() - _start
for b in times:
    print(f"Bsize {b}, time:"+"{:.2f}".format(times[b]))
