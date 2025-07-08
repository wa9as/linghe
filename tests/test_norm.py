

import torch

import time
import os
import random
import numpy as np
from flops.utils.util import *
from flops.utils.norm import *
from flops.facade.rmsnorm import RMSNormtriton
from flops.utils.benchmark import benchmark_func
import transformer_engine as te

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(20)


# M, N, K = 8192, 4096, 13312
# M, N, K = 4096, 4096, 6144
# M, N, K = 4096, 4096, 4096
M, N, K = 4096, 8192, 4096

dtype = torch.bfloat16
device = 'cuda:0'
n_repeat = 100

x = torch.randn(M, K, dtype=dtype, requires_grad=True, device=device)
weight = torch.randn(K, dtype=dtype,requires_grad=True,  device=device)
dy = torch.randn(M, K, dtype=dtype, device=device)

mode = 'rms_backward'
# mode = 'rms_forward'

rmsnorm_torch = torch.nn.RMSNorm(
    normalized_shape=K,
    eps=1e-6,
    dtype=torch.bfloat16,
    device='cuda'
)

with torch.no_grad():
    rmsnorm_torch.weight.copy_(weight)

rmsnorm_torch = torch.compile(rmsnorm_torch)

te_norm = te.pytorch.RMSNorm(
                hidden_size=K,
                eps=1e-6)

    
if mode == 'rms_forward':

    out_torch = rmsnorm_torch(x)
    output_te = te_norm(x)
    output_triton = RMSNormtriton.apply(x, weight, 1e-6)
    output_check(out_torch.float(), output_triton.float(),'rms')
    
    ref_time = benchmark_func(rmsnorm_torch, x, n_repeat=n_repeat, name="rms_torch", ref_bytes=M*K*4)
    benchmark_func(te_norm, x, n_repeat=n_repeat, ref_bytes=M*K*4, name="rms_te", ref_time=ref_time)
    benchmark_func(RMSNormtriton.apply, x, weight, n_repeat=n_repeat, ref_bytes=M*K*4, name="rms_triton", ref_time=ref_time)

if mode == 'rms_backward':

    def torch_forward_backward(x_torch_back, dy):
        y_torch_back = rmsnorm_torch(x_torch_back)
        y_torch_back.backward(gradient=dy)
        return  x_torch_back.grad, rmsnorm_torch.weight.grad

    def te_forward_backward(x_te_back, dy):
        y_te_back = te_norm(x_te_back)
        y_te_back.backward(gradient=dy)
        return  x_te_back.grad, te_norm.weight.grad

    def triton_forward_backward(x_triton_back, g_triton_back, dy):
        y_triton_back = RMSNormtriton.apply(x_triton_back, g_triton_back)
        y_triton_back.backward(gradient=dy)
        return x_triton_back.grad, g_triton_back.grad
    
    x_torch_back = x.detach().clone().to(torch.float32).requires_grad_()

    x_triton_back = x.detach().clone().to(torch.float32).requires_grad_()
    g_triton_back = weight.detach().clone().requires_grad_()
    
    rmsnorm_torch.zero_grad()
    
    dx_torch, dg_torch = torch_forward_backward(x_torch_back, dy)
    dx_triton, dg_triton = triton_forward_backward(x_triton_back, g_triton_back, dy)
    
    output_check(dx_torch, dx_triton, mode="dx")
    output_check(dg_torch, dg_triton, mode='dg')
    
    ref_time = benchmark_func(
        torch_forward_backward,
        x_torch_back,
        dy,
        n_repeat=n_repeat
    )
    
    ref_time = benchmark_func(
        te_forward_backward,
        x_torch_back,
        dy,
        n_repeat=n_repeat
    )
    
    benchmark_func(
        triton_forward_backward,
        x_triton_back,
        g_triton_back,
        dy,
        n_repeat=n_repeat,
        ref_time=ref_time
    )
