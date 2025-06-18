

import torch


from flops.utils.add import triton_block_add
from flops.utils.util import output_check
from flops.utils.benchmark import benchmark_func

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def torch_add(x, outputs, accum=True):
    if accum:
        x += outputs
        return x
    else:
        return x.float()

M, N = 8192, 8192

dtype = torch.bfloat16
device = 'cuda:0'

# outputs = torch.randn(M, N, dtype=torch.float32, device=device)
outputs = torch.randn(M, N, dtype=torch.float16, device=device)
x = torch.randn(M, N, dtype=dtype, device=device)

out = outputs.clone()
triton_block_add(out, x)
out_ref = outputs+x
output_check(out_ref,out,'sum')

n_repeat = 100

ref_time = benchmark_func(torch_add, x, out, accum=False, n_repeat=n_repeat)
benchmark_func(triton_block_add,out, x, accum=False,  n_repeat=n_repeat,ref_time=ref_time)

ref_time = benchmark_func(torch_add, x, out,accum=True,  n_repeat=n_repeat)
benchmark_func(triton_block_add,out, x, accum=True,  n_repeat=n_repeat,ref_time=ref_time)

