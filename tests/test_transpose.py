

from flops.utils.benchmark import benchmark_func
import torch

import time
import os
import random
from flops.utils.util import *
from flops.utils.transpose import *
from torch.profiler import profile, record_function, ProfilerActivity


# M, N, K = 8192, 4096, 13312
# M, N, K = 4096, 4096, 6144
# M, N, K = 4096, 4096, 4096

M, N, K = 2048, 8192, 8192

dtype = torch.bfloat16
device = 'cuda:0'

n_repeat = 100

x = torch.randn(M, K, dtype=dtype, device=device)
w = torch.randn(N, K, dtype=dtype, device=device)

x_q = x.to(torch.float8_e4m3fn)
w_q = w.to(torch.float8_e4m3fn)

mode = 'batch_transpose'
if mode == 'block':
    ref_output = x_q.t().contiguous()
    opt_output = triton_transpose(x_q)
    # opt_output = triton_transpose_and_pad(x_q, pad=True)
    output_check(ref_output.float(),opt_output[:,:M].float(),'transpose')
    
    benchmark_func(triton_transpose, x_q, n_repeat=n_repeat, ref_bytes=M*K*2)

if mode == 'batch_transpose':
    xs = [torch.randn((M,K), dtype=dtype,device=device).to(torch.float8_e4m3fn) for _ in range(8)]
    xts = triton_batch_transpose(xs)

    def triton_split_transpose(xs):
        outputs = []
        for x in xs:
            output = triton_transpose(x)
            outputs.append(output)
        return outputs 

    ref_time = benchmark_func(triton_split_transpose, xs, n_repeat=n_repeat, ref_bytes=M*K*2*8)

    benchmark_func(triton_batch_transpose, xs, n_repeat=n_repeat, ref_bytes=M*K*2*8, ref_time=ref_time)


if mode == 'batch_pad_transpose':
    count_list = [random.randint(1500,2600) for x in range(32)]
    counts = torch.tensor(count_list, device=device)
    xs = torch.randn((sum(count_list),K), dtype=dtype,device=device).to(torch.float8_e4m3fn)
    x_t = triton_batch_transpose_and_pad(xs, count_list, x_t=None, pad=True)

    def triton_split_transpose(xs, count_list):
        s = 0
        outputs = []
        for i, c in enumerate(count_list):
            x = xs[s:s+c]
            output = triton_transpose_and_pad(x, pad=True)
            outputs.append(output)
            s += c
        return outputs 

    x_t_ref = triton_split_transpose(xs, count_list)

    # for i in range(len(count_list)):
    #     output_check(x_t_ref[i].float(),x_t[i].float(),'transpose')

    n_repeat = 100
    ref_time = benchmark_func(triton_split_transpose, xs, count_list, n_repeat=n_repeat)
    benchmark_func(triton_batch_transpose_and_pad, xs, count_list, x_t=None, pad=True, n_repeat=n_repeat, ref_time=ref_time)
