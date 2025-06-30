

import torch

import time
import os
import random
from flops.utils.util import *
from flops.utils.scatter import *
from flops.utils.benchmark import benchmark_func

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def torch_fp16_scatter_add(x, outputs, indices, weights):
    if weights is not None:
        x = x*weights[:,None]
    dim = x.size(1)
    outputs.scatter_add_(0, indices.unsqueeze(1).expand(-1, dim), x)
    return outputs

M, N = 8192, 8192
topk = 2

dtype = torch.bfloat16
device = 'cuda:0'

n_expert = 4

x = torch.randn(M*topk, N, dtype=dtype, device=device)
outputs = torch.zeros(M, N, dtype=dtype, device=device)
logits = torch.randn((M,n_expert),dtype=torch.float32,device=device)
double_logits = logits.to(torch.float64)*(1+torch.arange(n_expert, device=device, dtype=torch.float64)*1e-12)
double_top_values, top_indices = torch.topk(double_logits, topk, dim=-1, largest=True, sorted=True)
routing_map = double_logits>=double_top_values[:,-1:]
counts = routing_map.sum(-1)

routing_map = routing_map.bool().T.contiguous()
dummy_indices = torch.arange(M, device=routing_map.device).unsqueeze(0).expand(n_expert, -1)
indices = dummy_indices.masked_select(routing_map)
weights = torch.randn(M*topk, dtype=dtype,device=device)

# sums = triton_aligned_scatter_add(x, outputs.clone(), indices, weights=weights)
# sums_ref = torch_fp16_scatter_add(x, outputs.clone(), indices, weights)
# output_check(sums_ref,sums,'aligned_scatter_add')

sums_ref = torch_fp16_scatter_add(x, outputs.clone(), indices, None)
# sums = triton_scatter_add(x, outputs.clone(), indices)
sums_split = triton_split_scatter_add(x, outputs.clone(), indices, counts)

# output_check(sums_ref,sums,'scatter_add')
output_check(sums_ref, sums_split,'split_scatter_add')


n_repeat = 100
ref_time = benchmark_func(torch_fp16_scatter_add,x, outputs, indices, weights,n_repeat=n_repeat)
benchmark_func(triton_aligned_scatter_add,x, outputs, indices, weights=weights, n_repeat=n_repeat,ref_time=ref_time)
benchmark_func(triton_scatter_add,x, outputs, indices, n_repeat=n_repeat,ref_time=ref_time)
benchmark_func(triton_split_scatter_add,x, outputs, indices, counts, n_repeat=n_repeat,ref_time=ref_time)
