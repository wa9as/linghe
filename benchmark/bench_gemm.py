import torch

import time
import os
import random
from flops.utils.benchmark import benchmark_func
from flops.gemm.fp8_gemm import *   # noqa: F403
from flops.gemm.blockwise_fp8_gemm import *   # noqa: F403
from flops.gemm.channelwise_fp8_gemm import *   # noqa: F403
from flops.utils.util import *   # noqa: F403
from flops.utils.transpose import *   # noqa: F403
from flops.utils.add import *   # noqa: F403
from flops.quant.channel.channel import *   # noqa: F403
from flops.quant.block.block import *   # noqa: F403
from flops.quant.block.group import *   # noqa: F403
from flops.quant.smooth.seperate_smooth import *   # noqa: F403
from flops.utils.rearange import *   # noqa: F403




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







