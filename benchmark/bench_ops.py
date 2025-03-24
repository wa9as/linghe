import torch

import time
import os
import random
from flops.utils.benchmark import benchmark_func
from flops.gemm.fp8_gemm import *
from flops.utils.util import *


def torch_fp16_post_vector_scaled_mm(x, weight, x_scale, weight_scale, one):
    output = torch._scaled_mm(x,
                                    weight,
                                    scale_a=one,
                                    scale_b=one,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    output = output * x_scale * weight_scale
    return output

def torch_fp32_post_vector_scaled_mm(x, weight, x_scale, weight_scale, one):
    output = torch._scaled_mm(x,
                                    weight,
                                    scale_a=one,
                                    scale_b=one,
                                    out_dtype=torch.float32,
                                    use_fast_accum=True)
    output = output * x_scale * weight_scale
    output = output.to(dtype=torch.bfloat16)
    return output

def torch_fp16_vector_scaled_mm(x, weight, x_scale, weight_scale):
    output = torch._scaled_mm(x,
                                    weight,
                                    scale_a=x_scale,
                                    scale_b=weight_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output

def torch_fp32_vector_scaled_mm(x, weight, x_scale, weight_scale):
    output = torch._scaled_mm(x,
                                    weight,
                                    scale_a=x_scale,
                                    scale_b=weight_scale,
                                    out_dtype=torch.float32,
                                    use_fast_accum=True)
    return output

def torch_fp16_scaler_scaled_mm(x, weight, x_scale, weight_scale):
    output = torch._scaled_mm(x,
                                weight,
                                scale_a=x_scale,
                                scale_b=weight_scale,
                                out_dtype=torch.bfloat16,
                                use_fast_accum=True)
    return output

for i in range(0, 1):
    batch_size, out_dim, in_dim = 4096, 4096, 4096
    dtype = torch.bfloat16
    n_repeat = 1000

    x = torch.randn(batch_size, in_dim, dtype=dtype, device='cuda:0')
    w = torch.randn(out_dim, in_dim, dtype=dtype, device='cuda:0')

    org_out = x @ w.t()

    xrs = x.abs().max(dim=1,keepdim=True)[0]
    wcs = w.abs().max(dim=1,keepdim=True)[0]
    x_f8 = (448*x/xrs).to(torch.float8_e4m3fn)
    w_f8 = (448*w/wcs).to(torch.float8_e4m3fn)
    one = torch.tensor([1.0],dtype=torch.float32,device='cuda:0')
    opt_out = torch_fp16_post_vector_scaled_mm(x_f8, w_f8.t(), xrs/448, wcs.t()/448, one)
    abs_error = (opt_out.float() - org_out.float()).abs().mean().item()
    rel_error = abs_error/org_out.float().abs().mean().item()
    print(f'\ntorch_fp16_post_vector_scaled_mm abs_error:{abs_error:.3f} rel_error:{rel_error:.3f}')
    benchmark_func(torch_fp16_post_vector_scaled_mm, x_f8, w_f8.t(), xrs, wcs.t(), one, n_repeat=n_repeat, name=f'bs:{batch_size}')


    xrs = x.abs().float().max(dim=1,keepdim=True)[0]
    wcs = w.abs().float().max(dim=1,keepdim=True)[0]
    x_f8 = (448*x/xrs).to(torch.float8_e4m3fn)
    w_f8 = (448*w/wcs).to(torch.float8_e4m3fn)
    opt_out = torch_fp32_post_vector_scaled_mm(x_f8, w_f8.t(), xrs/448, wcs.t()/448, one)
    # print(org_out.shape, org_out[0,:4])
    # print(opt_out.shape, opt_out[0,:4])
    abs_error = (opt_out.float() - org_out.float()).abs().mean().item()
    rel_error = abs_error/org_out.float().abs().mean().item()
    print(f'\ntorch_fp32_post_vector_scaled_mm abs_error:{abs_error:.3f} rel_error:{rel_error:.3f}')
    benchmark_func(torch_fp32_post_vector_scaled_mm, x_f8, w_f8.t(), xrs, wcs.t(), one, n_repeat=n_repeat, name=f'bs:{batch_size}')


    xrs = x.abs().float().max(dim=1,keepdim=True)[0]
    wcs = w.abs().float().max(dim=1,keepdim=True)[0]
    x_f8 = (448*x/xrs).to(torch.float8_e4m3fn)
    w_f8 = (448*w/wcs).to(torch.float8_e4m3fn)
    opt_out = torch_fp16_vector_scaled_mm(x_f8, w_f8.t(), xrs/448, wcs.t()/448)
    # print(org_out.shape, org_out[0,:4])
    # print(opt_out.shape, opt_out[0,:4])
    abs_error = (opt_out.float() - org_out.float()).abs().mean().item()
    rel_error = abs_error/org_out.float().abs().mean().item()
    print(f'\ntorch_fp16_vector_scaled_mm abs_error:{abs_error:.3f} rel_error:{rel_error:.3f}')
    benchmark_func(torch_fp16_vector_scaled_mm, x_f8, w_f8.t(), xrs, wcs.t(), n_repeat=n_repeat, name=f'bs:{batch_size}')

    xrs = x.abs().float().max()
    wcs = w.abs().float().max()
    x_f8 = (448*x/xrs).to(torch.float8_e4m3fn)
    w_f8 = (448*w/wcs).to(torch.float8_e4m3fn)
    opt_out = torch_fp16_scaler_scaled_mm(x_f8, w_f8.t(), xrs/448, wcs/448)
    abs_error = (opt_out.float() - org_out.float()).abs().mean().item()
    rel_error = abs_error/org_out.float().abs().mean().item()
    print(f'\ntorch_fp16_scaler_scaled_mm abs_error:{abs_error:.3f} rel_error:{rel_error:.3f}')
    benchmark_func(torch_fp16_scaler_scaled_mm, x_f8, w_f8.t(), xrs, wcs, n_repeat=n_repeat, name=f'bs:{batch_size}')

    benchmark_func(trival_fp8_gemm, x_f8, w_f8, torch.bfloat16, n_repeat=n_repeat)
    benchmark_func(persistent_fp8_gemm, x_f8, w_f8.t(), torch.bfloat16, n_repeat=n_repeat)
    benchmark_func(nt_fp8_gemm, x_f8,w_f8,torch.bfloat16, n_repeat=n_repeat)

    benchmark_func(fp8_transpose, x_f8, n_repeat=n_repeat)
    benchmark_func(fp16_transpose, x, n_repeat=n_repeat)
