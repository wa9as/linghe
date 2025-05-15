import torch

import time
import os
import random
from flops.utils.benchmark import benchmark_func
from flops.gemm.fp8_gemm import *
from flops.gemm.tile_fp8_gemm import *
from flops.utils.util import *
from flops.utils.transpose import *
from flops.quant.channel import *
from flops.quant.tile import *
from torch.profiler import profile, record_function, ProfilerActivity


M, N, K = 8192, 4096, 13312
# M, N, K = 4096, 4096, 6144
# M, N, K = 4096, 4096, 4096

dtype = torch.bfloat16
device = 'cuda:0'

n_repeat = 100

x = torch.randn(M, K, dtype=dtype, device=device)
w = torch.randn(N, K, dtype=dtype, device=device)

x_f8 = x.to(torch.float8_e4m3fn)
w_f8 = w.to(torch.float8_e4m3fn)
mode = 'bb'

if mode == 'gemm':

    benchmark_func(trival_fp8_gemm, x_f8, w_f8, torch.bfloat16, n_repeat=n_repeat)
    benchmark_func(persistent_fp8_gemm, x_f8, w_f8.t(), torch.bfloat16, n_repeat=n_repeat)
    benchmark_func(fp8_gemm_nn, x_f8,w_f8,torch.bfloat16, n_repeat=n_repeat)

elif mode == 'bb':
    B = 64
    ref_flops = M * N * K * 2
    x_scales = torch.randn((M//B, K//B),dtype=torch.float32, device=device)
    w_scales = torch.randn((N//B, K//B),dtype=torch.float32, device=device)
    fp8_gemm(x_f8, x_scales, w_f8, w_scales, dtype)
    benchmark_func(fp8_gemm, x_f8, x_scales, w_f8, w_scales, dtype, n_repeat=n_repeat, ref_flops=ref_flops)

elif mode == 'quant':
    benchmark_func(block_quant, x,n_repeat=n_repeat)
    benchmark_func(stupid_tile_quant, x, n_repeat=n_repeat)

    y_ref, s_ref = stupid_tile_quant(x)
    y_opt, s_opt = tile_quant(x)

    torch.testing.assert_close(y_opt.float(), y_ref.float(),
                                    rtol=0.02, atol=0.02)
    torch.testing.assert_close(s_ref.float(), s_opt.float(),
                                    rtol=0.02, atol=0.02)
    benchmark_func(stupid_tile_quant, x, n_repeat=n_repeat)
    benchmark_func(tile_quant, x, n_repeat=n_repeat)
    benchmark_func(persist_tile_quant, x, n_repeat=n_repeat)

elif mode == 'transpose':
    benchmark_func(fp16_transpose, x, n_repeat=n_repeat)
    benchmark_func(triton_transpose,x, n_repeat=n_repeat)
    benchmark_func(triton_opt_transpose,x, n_repeat=n_repeat)


    benchmark_func(triton_row_quant, x, n_repeat=n_repeat)
    benchmark_func(triton_transpose_row_quant, x, n_repeat=n_repeat)


    benchmark_func(fp8_transpose, x_f8, n_repeat=n_repeat)
    benchmark_func(triton_transpose,x_f8, n_repeat=n_repeat)
    benchmark_func(triton_opt_transpose,x_f8, n_repeat=n_repeat)

else:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]) as prof:
        benchmark_func(tile_quant, x, n_repeat=n_repeat)
    print(prof.key_averages().table(sort_by=None, top_level_events_only=True, row_limit=2000))
    print(prof.key_averages(group_by_stack_n=5).table(sort_by=None, row_limit=100))
    prof.export_chrome_trace("trace.json")









