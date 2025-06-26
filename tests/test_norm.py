

import torch

import time
import os
import random
from flops.utils.util import *
from flops.utils.norm import *
from flops.utils.benchmark import benchmark_func


# M, N, K = 8192, 4096, 13312
# M, N, K = 4096, 4096, 6144
# M, N, K = 4096, 4096, 4096
M, N, K = 4096, 8192, 4096

dtype = torch.bfloat16
device = 'cuda:0'
n_repeat = 100

x = torch.randn(M, K, dtype=dtype, device=device)
weight = torch.randn(K, dtype=dtype, device=device)
dy = torch.randn(M, K, dtype=dtype, device=device)

mode = 'rms_backward'

if mode == 'rms_forward':
    def torch_rms_norm(x, weight, eps=1e-6):
        input_dtype = x.dtype
        hidden_states = x.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return weight * hidden_states.to(input_dtype)


    ref_output = torch_rms_norm(x, weight)
    output, norm = triton_rms_norm_forward(x, weight, 1e-6)
    output_check(ref_output.float(), output.float(),'rms')
    
    ref_time = benchmark_func(torch_rms_norm, x, weight, n_repeat=n_repeat, ref_bytes=M*K*4)
    benchmark_func(triton_rms_norm_forward, x, weight, n_repeat=n_repeat, ref_bytes=M*K*4, ref_time=ref_time)

if mode == 'rms_backward':
    def torch_rms_norm_forward(x, weight, eps=1e-6):
        input_dtype = x.dtype
        hidden_states = x.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance + eps)
        hidden_states = hidden_states * rstd
        return weight * hidden_states.to(input_dtype), rstd.squeeze(-1)

    def torch_rms_norm_backward(dy, x, norm, weight):
        N = x.shape[-1]
        rstd = norm.unsqueeze(-1)
        x_hat = x * rstd
        dl_dx_hat = dy * weight
        
        mean = torch.sum(dl_dx_hat * x_hat, dim=-1, keepdim=True) / N
        dx = rstd * (dl_dx_hat - x_hat * mean)
        dweight = torch.sum(dy * x_hat, dim=0)
        return dx, dweight

    ref_out, ref_norm = torch_rms_norm_forward(x, weight)
    triton_out, triton_norm = triton_rms_norm_forward(x, weight)
    
    ref_dx, ref_dweight = torch_rms_norm_backward(dy, x, ref_norm, weight)
    triton_dx, triton_dweight = triton_rms_norm_backward(dy, x, triton_norm, weight)

    output_check(ref_dx, triton_dx, 'triton_dx')
    output_check(ref_dweight, triton_dweight, 'triton_dweight')


    ref_time = benchmark_func(
        torch_rms_norm_backward,
        dy,
        x,
        ref_norm,
        weight, 
        n_repeat=n_repeat
    )
    
    benchmark_func(
        triton_rms_norm_backward,
        dy,
        x,
        triton_norm,
        weight, 
        n_repeat=n_repeat,
        ref_time=ref_time
    )