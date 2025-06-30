

import torch

import time
import os
import random
from flops.utils.util import *
from flops.utils.gather import *
from flops.utils.benchmark import benchmark_func

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def torch_fp16_index_select(x, scales, indices):
    return x.index_select(0, indices), scales.index_select(0, indices)

M, N = 8192, 8192

dtype = torch.bfloat16
device = 'cuda:0'

n_expert = 256

x = torch.randn(M, N, dtype=dtype, device=device)
scales = torch.randn(M, dtype=dtype, device=device)
indices = torch.argsort(torch.randn(M*2, dtype=dtype, device=device))%M

x_out, scale_out = triton_index_select(x, indices, scale=scales)
x_out_ref, scale_out_ref = torch_fp16_index_select(x, scales, indices)
output_check(x_out_ref,x_out,'x_out')
output_check(scale_out_ref,scale_out,'scale_out')


n_repeat = 100
ref_time = benchmark_func(torch_fp16_index_select,x, scales, indices, n_repeat=n_repeat)
benchmark_func(triton_index_select,x, indices, scale=scales, n_repeat=n_repeat, ref_time=ref_time)

# def torch_scatter(logits,routing_map,weights):
#     logits[routing_map] = weights

# benchmark_func(torch_scatter,logits.to(dtype), routing_map.T.contiguous(), weights, n_repeat=n_repeat)
