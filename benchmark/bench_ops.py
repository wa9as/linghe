import torch

import time
import os
import random
from flops.utils.benchmark import benchmark_func
from flops.gemm.fp8_gemm import *
from flops.gemm.blockwise_fp8_gemm import *
from flops.utils.util import *
from flops.utils.transpose import *
from flops.quant.channel.channel import *
from flops.quant.block.block import *
from flops.quant.block.group import *
from flops.quant.smooth.seperate_smooth import *

from torch.profiler import profile, record_function, ProfilerActivity




# M, N, K = 8192, 10240, 8192  # max qkv
# M, N, K = 8192, 8192, 8192  # max out
M, N, K = 1024, 4096, 8192  # max gate_up
# M, N, K = 8192, 2048, 8192  # max down

# M, N, K = M-1, N-1, K-1

dtype = torch.bfloat16
device = 'cuda:0'
n_repeat = 100

x = torch.randn(M, K, dtype=dtype, device=device)
w = torch.randn(N, K, dtype=dtype, device=device)
y = torch.randn(M, N, dtype=dtype, device=device)

x_q = x.to(torch.float8_e4m3fn)
w_q = w.to(torch.float8_e4m3fn)
y_q = y.to(torch.float8_e4m3fn)

mode = 'megatron'
if mode == 'gemm':

    benchmark_func(trival_fp8_gemm, x_q, w_q, torch.bfloat16, n_repeat=n_repeat)
    benchmark_func(persistent_fp8_gemm, x_q, w_q.t(), torch.bfloat16, n_repeat=n_repeat)
    benchmark_func(fp8_gemm_nn, x_q,w_q,torch.bfloat16, n_repeat=n_repeat)

elif mode == 'bb':
    B = 64
    ref_flops = M * N * K * 2
    x_scales = torch.randn((M//B, K//B),dtype=torch.float32, device=device)
    w_scales = torch.randn((N//B, K//B),dtype=torch.float32, device=device)
    fp8_gemm(x_q, x_scales, w_q, w_scales, dtype)
    benchmark_func(fp8_gemm, x_q, w_q, x_scales, w_scales, out_dtype=dtype, n_repeat=n_repeat, ref_flops=ref_flops)

elif mode == 'quant':
    benchmark_func(block_quant, x,n_repeat=n_repeat)
    benchmark_func(stupid_group_quant, x, n_repeat=n_repeat)

    y_ref, s_ref = stupid_group_quant(x)
    y_opt, s_opt = group_quant(x)

    torch.testing.assert_close(y_opt.float(), y_ref.float(),
                                    rtol=0.02, atol=0.02)
    torch.testing.assert_close(s_ref.float(), s_opt.float(),
                                    rtol=0.02, atol=0.02)
    benchmark_func(stupid_group_quant, x, n_repeat=n_repeat)
    benchmark_func(group_quant, x, n_repeat=n_repeat)
    benchmark_func(persist_group_quant, x, n_repeat=n_repeat)

elif mode == 'transpose':
    benchmark_func(fp16_transpose, x, n_repeat=n_repeat)
    benchmark_func(triton_transpose,x, n_repeat=n_repeat)
    benchmark_func(triton_block_transpose,x, n_repeat=n_repeat)

    # benchmark_func(triton_opt_transpose,x, n_repeat=n_repeat)

    benchmark_func(fp8_transpose, x_q, n_repeat=n_repeat)
    benchmark_func(triton_transpose,x_q, n_repeat=n_repeat)
    benchmark_func(triton_block_transpose,x_q, n_repeat=n_repeat)
    # benchmark_func(triton_opt_transpose,x_q, n_repeat=n_repeat)

elif mode == 'megatron':
    benchmark_func(triton_calc_smooth_scale, x, n_repeat=n_repeat)
    smooth_scale = torch.ones((K,),device=device,dtype=torch.float32)
    benchmark_func(triton_smooth_quant_x, x, smooth_scale, transpose=True, pad=True, n_repeat=n_repeat)

    smooth_scale = torch.ones((N,),device=device,dtype=torch.float32)
    transpose_smooth_scale = torch.ones((M,),device=device,dtype=torch.float32)
    benchmark_func(triton_smooth_quant_y, y, smooth_scale, transpose_smooth_scale, reverse=True, pad=True, n_repeat=n_repeat)

elif mode == 'profile':
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]) as prof:
        benchmark_func(group_quant, x, n_repeat=n_repeat)
    print(prof.key_averages().table(sort_by=None, top_level_events_only=True, row_limit=2000))
    print(prof.key_averages(group_by_stack_n=5).table(sort_by=None, row_limit=100))
    prof.export_chrome_trace("trace.json")









