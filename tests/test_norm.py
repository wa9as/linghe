

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

x = torch.randn(M, K, dtype=dtype, device=device, requires_grad=True)
weight = torch.randn(K, dtype=dtype, device=device, requires_grad=True)
dy = torch.randn(M, K, dtype=dtype, device=device)
norm = torch.randn(M, dtype=torch.float32, device=device)
eps = 1e-6

def torch_rms_norm(x, weight, eps=1e-6):
    input_dtype = x.dtype
    hidden_states = x.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


if False:

    ref_output = torch_rms_norm(x, weight)
    output, norm = triton_rms_norm_forward(x, weight, eps=eps, output_norm=True)
    output_check(ref_output.float(), output.float(),'rms')
    
    ref_time = benchmark_func(torch_rms_norm, x, weight, n_repeat=n_repeat, ref_bytes=M*K*4)
    benchmark_func(triton_rms_norm_forward, x, weight, n_repeat=n_repeat, ref_bytes=M*K*4, ref_time=ref_time)

if True:
    y = torch_rms_norm(x, weight, eps=eps)
    dx, dw = triton_rms_norm_backward(dy, x, weight, eps=eps)
    y.backward(dy)
    dx_ref = x.grad 
    dw_ref = weight.grad 
    output_check(dx_ref.float(), dx.float(),'dx')
    output_check(dw_ref.float(), dw.float(),'dw')

    ref_time = benchmark_func(triton_depracated_rms_norm_backward, dy, x, weight, norm, n_repeat=n_repeat, ref_bytes=M*K*6)
    benchmark_func(triton_rms_norm_backward, dy, x, weight, n_repeat=n_repeat, ref_bytes=M*K*6, ref_time=ref_time)

