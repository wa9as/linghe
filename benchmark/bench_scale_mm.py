import torch

import time
import os
import random
from flops.utils.benchmark import benchmark_func
from flops.gemm.channelwise_fp8_gemm import triton_scaled_mm 

from flops.utils.util import fp16_forward


def triton_accum_weight(x,w,out,x_scale,w_scale):
    output = torch._scaled_mm(
        x,
        w,
        scale_a=x_scale,
        scale_b=w_scale,
        out_dtype=torch.bfloat16,
        use_fast_accum=True
    )
    triton_block_add(out, output)
    return out

def torch_accum_weight(x,w,out,x_scale,w_scale):
    output = torch._scaled_mm(
        x,
        w,
        scale_a=x_scale,
        scale_b=w_scale,
        out_dtype=torch.bfloat16,
        use_fast_accum=True
    )
    out.add_(output)
    return out
    

M, N, K = 8192, 8192, 4096
dtype = torch.bfloat16
device = 'cuda:0'
n_repeat = 100

x = torch.randn(M, K, dtype=dtype, device=device)
w = torch.randn(N, K, dtype=dtype, device=device)

xrs = x.abs().float().amax(dim=1,keepdim=True)
wcs = w.abs().float().amax(dim=1,keepdim=True)
x_q = (448*x/xrs).to(torch.float8_e4m3fn)
w_q = (448*w/wcs).to(torch.float8_e4m3fn)
ref_flops = M*N*K*2
ones = torch.ones((1,),dtype=torch.float32,device=device)

out = torch.zeros((M,N), dtype=torch.float32,device=device)
o = torch.empty((M,N), dtype=dtype,device=device)
# benchmark_func(triton_block_add, out, o, n_repeat=n_repeat, name=f'M:{M}')

ref_time=benchmark_func(fp16_forward, x, w.t(), n_repeat=n_repeat, ref_flops=ref_flops, name=f'M:{M}')
# benchmark_func(torch_fp16_vector_scaled_mm, x_q, w_q.t(), xrs, wcs.view(1,-1), n_repeat=n_repeat,  ref_flops=ref_flops, ref_time=ref_time, name=f'M:{M}')
# benchmark_func(torch_fp32_vector_scaled_mm, x_q, w_q.t(), xrs, wcs.view(1,-1), ones, out=out, n_repeat=n_repeat,  ref_flops=ref_flops, ref_time=ref_time, name=f'M:{M}')
# benchmark_func(torch_fp16_scaler_scaled_mm, x_q, w_q.t(), xrs[0,0], wcs[0,0], n_repeat=n_repeat,  ref_flops=ref_flops, ref_time=ref_time, name=f'M:{M}')
# benchmark_func(torch_fp32_scaler_scaled_mm, x_q, w_q.t(), xrs[0,0], wcs[0,0], n_repeat=n_repeat,  ref_flops=ref_flops, ref_time=ref_time, name=f'M:{M}')
benchmark_func(torch_accum_weight, x_q, w_q.t(), out, xrs, wcs.view(1,-1), n_repeat=n_repeat,  ref_flops=ref_flops, ref_time=ref_time, name=f'M:{M}')
benchmark_func(triton_accum_weight, x_q, w_q.t(), out, xrs, wcs.view(1,-1), n_repeat=n_repeat,  ref_flops=ref_flops, ref_time=ref_time, name=f'M:{M}')
benchmark_func(triton_scaled_mm, x_q, w_q, xrs, wcs, c=out, accum=True, n_repeat=n_repeat,  ref_flops=ref_flops, ref_time=ref_time, name=f'M:{M}')
benchmark_func(triton_scaled_mm, x_q, w_q, xrs, wcs, c=out, accum=False, n_repeat=n_repeat,  ref_flops=ref_flops, ref_time=ref_time, name=f'M:{M}')
