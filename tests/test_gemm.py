import torch

import time
import os
import random
from flops.utils.benchmark import benchmark_func
from flops.gemm.fp8_gemm import *
from flops.gemm.channelwise_fp8_gemm import *
from flops.utils.util import *
from flops.utils.add import *
from flops.gemm.fp32_gemm import *



def test_scaled_mm(M=2048,N=8192,K=8192):
    # M, N, K = 2048, 8192, 8192-16
    dtype = torch.bfloat16
    device = 'cuda:0'
    n_repeat = 100

    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)

    x_scale = x.abs().float().amax(dim=1)/448
    w_scale = w.abs().float().amax(dim=1)/448
    x_q = (x/x_scale[:,None]).to(torch.float8_e4m3fn)
    w_q = (w/w_scale[:,None]).to(torch.float8_e4m3fn)
    ref_flops = M*N*K*2

    o = 10*torch.ones((M,N), dtype=torch.float32, device=device)
    ref_out = (x_q.float()*x_scale[:,None])@(w_q.float()*w_scale[:,None]).t() + o

    out = triton_scaled_mm(x_q,w_q,x_scale,w_scale, c=o, accum=True)
    output_check(ref_out, out.float(), mode='gemm')



def test_fp32_matmul(M=2048,N=256,K=8192):
    # M, N, K = 4096, 256, 8192
    dtype = torch.bfloat16
    device = 'cuda:0'
    n_repeat = 100

    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)
    scale = torch.randn(M, dtype=torch.float32, device=device)
    dy = torch.randn(M, N, dtype=torch.float32, device=device)

    def torch_fp32_matmul(x,w):
        return torch.nn.functional.linear(x.float(),w.float())

    def torch_fp32_matmul_backward(dy, w):
        return (dy@w).to(torch.bfloat16)

    def torch_fp32_matmul_update(y, x):
        return (y.t()@x).to(torch.bfloat16)

    y_ref = torch_fp32_matmul(x,w)
    y = triton_fp32_gemm(x,w)
    output_check(y_ref, y.float(), mode='fp32_gemm')

    y_ref = torch_fp32_matmul(x*scale[:,None],w)
    y = triton_scaled_fp32_gemm(x,w,scale)
    output_check(y_ref, y.float(), mode='scaled_fp32_gemm')

    dx = torch.zeros(M, K, dtype=dtype, device=device)
    dx_clone = dx.clone()
    triton_fp32_gemm_for_backward(dy, w, dx_clone, accum=True)
    dx_ref = dy@w.float() + dx.float()
    output_check(dx_ref, dx_clone.float(), mode='backward')


    main_grad = triton_fp32_gemm_for_update(y,  x)
    main_grad_ref = y.t()@(x.float())
    output_check(main_grad_ref, main_grad.float(), mode='update')

    main_grad = triton_scaled_fp32_gemm_for_update(y,  x, scale)
    main_grad_ref = y.t()@(x.float()*scale[:,None])
    output_check(main_grad_ref, main_grad.float(), mode='scaled_update')

    print('\nbenchmark\n')

    ref_time = benchmark_func(torch_fp32_matmul, x,w, n_repeat=n_repeat, ref_bytes=M*K*6+N*K*6+M*N*4, ref_flops=2*M*N*K)
    benchmark_func(triton_fp32_gemm, x, w, n_repeat=n_repeat, ref_bytes=M*K*6+N*K*6+M*N*4, ref_flops=2*M*N*K, ref_time=ref_time)
    benchmark_func(triton_scaled_fp32_gemm, x, w, scale, n_repeat=n_repeat, ref_bytes=M*K*6+N*K*6+M*N*4, ref_flops=2*M*N*K, ref_time=ref_time)

    ref_time = benchmark_func(torch_fp32_matmul_backward, dy, w.float(), n_repeat=n_repeat, ref_bytes=M*K*10+N*K*4+M*N*4, ref_flops=2*M*N*K)
    benchmark_func(triton_fp32_gemm_for_backward, dy, w, dx_clone, accum=True, n_repeat=n_repeat, ref_bytes=M*K*2+N*K*2+M*N*4, ref_flops=2*M*N*K, ref_time=ref_time)

    ref_time = benchmark_func(torch_fp32_matmul_update, dy, x.float(), n_repeat=n_repeat, ref_bytes=M*K*4+N*K*12+M*N*4, ref_flops=2*M*N*K)
    benchmark_func(triton_fp32_gemm_for_update, dy, x, n_repeat=n_repeat, ref_bytes=M*K*2+N*K*8+M*N*4, ref_flops=2*M*N*K, ref_time=ref_time)
    benchmark_func(triton_scaled_fp32_gemm_for_update, dy, x, scale, n_repeat=n_repeat, ref_bytes=M*K*2+N*K*8+M*N*4, ref_flops=2*M*N*K, ref_time=ref_time)



if __name__ == '__main__':
    test_scaled_mm(M=2048,N=8192,K=8192)
    test_fp32_matmul(M=2048,N=256,K=8192)