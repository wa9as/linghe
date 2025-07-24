

import torch

import time
import os
import random
import numpy as np
from flops.utils.util import *   # noqa: F403
from flops.utils.norm import *   # noqa: F403
from flops.facade.rmsnorm import RMSNormFunction
from flops.utils.benchmark import benchmark_func
import transformer_engine as te



def test_rmsnorm(M=4096, K=4096):

    dtype = torch.bfloat16
    device = 'cuda:0'
    n_repeat = 100

    x = torch.randn(M, K, dtype=dtype, requires_grad=True, device=device)
    weight = torch.randn(K, dtype=dtype,requires_grad=True,  device=device)
    dy = torch.randn(M, K, dtype=dtype, device=device)

    x_torch = x.detach().clone().to(torch.float32).requires_grad_()
    x_triton = x.detach().clone().to(torch.float32).requires_grad_()
    w_triton = weight.detach().clone().requires_grad_()
    
    rmsnorm_torch = torch.nn.RMSNorm(
        normalized_shape=K,
        eps=1e-6,
        dtype=torch.bfloat16,
        device='cuda'
    )

    with torch.no_grad():
        rmsnorm_torch.weight.copy_(weight)

    rmsnorm_torch.zero_grad()
    
    y_torch = rmsnorm_torch(x_torch)
    y_torch.backward(gradient=dy)
    dx_ref = x_torch.grad
    dw_ref = rmsnorm_torch.weight.grad

    y_triton = RMSNormFunction.apply(x_triton, w_triton, 1e-6)
    y_triton.backward(gradient=dy)
    dx = x_torch.grad
    dw = w_triton.grad

    output_check(y_torch.float(), y_triton.float(),'rms')
    output_check(dx_ref, dx, mode="dx")
    output_check(dw_ref, dw, mode='dg')
    

if __name__ == '__main__':
    test_rmsnorm(M=4096, K=4096)