

import torch

import time
import os
import random
from flops.utils.util import *
from flops.utils.scatter import *
from flops.utils.benchmark import benchmark_func

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def torch_fp16_scatter(x, outputs, indices, weights):
    x = x*weights[:,None]
    dim = x.size(1)
    outputs.scatter_add_(0, indices.unsqueeze(1).expand(-1, dim), x)
    return outputs

M, N = 8192, 8192
K = 8

dtype = torch.bfloat16
device = 'cuda:0'

n_repeat = 100

x = torch.randn(M*K, N, dtype=dtype, device=device)
outputs = torch.zeros(M, N, dtype=dtype, device=device)
tmp = torch.argsort(torch.randn(M*K,dtype=torch.float64,device=device))
indices = (torch.arange(M*K, dtype=torch.int64, device=device)//K)[tmp]
weights = torch.randn(M*K, dtype=dtype,device=device)

sums = triton_scatter(x, outputs.clone(), indices, weights=weights)

sums_ref = torch_fp16_scatter(x, outputs.clone(), indices, weights)
output_check(sums_ref,sums,'sum')

ref_time = benchmark_func(torch_fp16_scatter,x, outputs, indices, weights,n_repeat=n_repeat)
benchmark_func(triton_scatter,x, outputs, indices, weights=weights, n_repeat=n_repeat,ref_time=ref_time)