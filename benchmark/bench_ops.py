import torch

import time
import os
import random
from flops.utils.benchmark import benchmark_func
from flops.gemm.fp8_gemm import *
from flops.gemm.blockwise_fp8_gemm import *
from flops.gemm.channelwise_fp8_gemm import *
from flops.utils.util import *
from flops.utils.transpose import *
from flops.utils.add import *
from flops.quant.channel.channel import *
from flops.quant.block.block import *
from flops.quant.block.group import *
from flops.quant.smooth.seperate_smooth import *
from flops.utils.rearange import *




# M, N, K = 8192, 10240, 8192  # max qkv
# M, N, K = 8192, 8192, 8192  # max out
# M, N, K = 2048, 4096, 8192  # max gate_up
# M, N, K = 2048, 8192, 2048  # max down
M, N, K = 8192, 8192, 8192
# M, N, K = 2048, 8192, 8192

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

mode = 'channelwise'
if mode == 'gemm':

    ref_flops = M*N*K*2
    benchmark_func(trival_fp8_gemm, x_q, w_q, torch.bfloat16, n_repeat=n_repeat, ref_flops=ref_flops)
    benchmark_func(persistent_fp8_gemm, x_q, w_q.t(), torch.bfloat16, n_repeat=n_repeat, ref_flops=ref_flops)


elif mode == 'bb':
    B = 64
    ref_flops = M * N * K * 2
    x_scales = torch.randn((M//B, K//B),dtype=torch.float32, device=device)
    w_scales = torch.randn((N//B, K//B),dtype=torch.float32, device=device)
    fp8_gemm(x_q, x_scales, w_q, w_scales, dtype)
    benchmark_func(fp8_gemm, x_q, w_q, x_scales, w_scales, out_dtype=dtype, n_repeat=n_repeat, ref_flops=ref_flops)

elif mode == 'channelwise':
    ref_flops = M*N*K*2
    x_scales = torch.randn((M, ),dtype=torch.float32, device=device)
    w_scales = torch.randn((N, ),dtype=torch.float32, device=device)

    y_fp32 = torch.zeros(M, N, dtype=torch.float32, device=device)
    y_fp16 = torch.zeros(M, N, dtype=torch.float16, device=device)

    def separate_gemm(x_q,w_q,x_scales,w_scales,c=None, accum=True):
        bf16_out = torch._scaled_mm(x_q, 
                                w_q.t(),
                                scale_a=x_scales.view(-1,1),
                                scale_b=w_scales.view(1,-1),
                                out_dtype=torch.bfloat16,
                                use_fast_accum=True,
                                )
        triton_block_add(c, bf16_out, accum=accum)
        return c

    benchmark_func(triton_scaled_mm, x_q, w_q, x_scales, w_scales, c=y, accum=True, n_repeat=n_repeat, ref_flops=ref_flops)
    benchmark_func(triton_scaled_mm, x_q, w_q, x_scales, w_scales, c=y_fp16, accum=True, n_repeat=n_repeat, ref_flops=ref_flops)
    benchmark_func(triton_scaled_mm, x_q, w_q, x_scales, w_scales, c=y_fp32, accum=True, n_repeat=n_repeat, ref_flops=ref_flops)

    benchmark_func(separate_gemm, x_q, w_q, x_scales, w_scales, c=y, accum=True, n_repeat=n_repeat, ref_flops=ref_flops)
    benchmark_func(separate_gemm, x_q, w_q, x_scales, w_scales, c=y_fp16, accum=True, n_repeat=n_repeat, ref_flops=ref_flops)
    benchmark_func(separate_gemm, x_q, w_q, x_scales, w_scales, c=y_fp32, accum=True, n_repeat=n_repeat, ref_flops=ref_flops)


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
    # benchmark_func(fp16_transpose, x, n_repeat=n_repeat)
    # benchmark_func(triton_transpose,x, n_repeat=n_repeat)
    # benchmark_func(triton_block_transpose,x, n_repeat=n_repeat)
    # benchmark_func(triton_opt_transpose,x, n_repeat=n_repeat)

    benchmark_func(fp8_transpose, x_q, n_repeat=n_repeat)
    # benchmark_func(triton_transpose,x_q, n_repeat=n_repeat)
    benchmark_func(triton_block_transpose,x_q, n_repeat=n_repeat)
    # benchmark_func(triton_block_pad_transpose,x_q,pad=True, n_repeat=n_repeat)
    # benchmark_func(triton_opt_transpose,x_q, n_repeat=n_repeat)

elif mode == 'zero':
    benchmark_func(torch.zeros,(M,K),dtype=dtype,device=device, n_repeat=n_repeat, ref_bytes=M*K*2)

elif mode == 'd2h':

    def d2h(x,device):
        return x.to(device)
    counts = torch.tensor([0]*32,dtype=torch.int64)
    benchmark_func(d2h,counts,device=device, n_repeat=n_repeat, ref_bytes=M*K*2)








