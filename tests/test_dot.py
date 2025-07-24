

import torch

import time
import os
import random
from flops.utils.util import *   # noqa: F403
from flops.utils.dot import *   # noqa: F403
from flops.utils.benchmark import benchmark_func


def torch_fp16_dot(x,y):
  return (x*y).sum(1).float()


def test_dot(M=4096,N=4096):
    dtype = torch.bfloat16
    device = 'cuda:0'

    n_repeat = 100

    x = torch.randn(M, N, dtype=dtype, device=device)
    y = torch.randn(M, N, dtype=dtype, device=device)
    q = torch.randn(M, N, dtype=dtype, device=device).to(torch.float8_e4m3fn)
    quant_scale = torch.randn(M, dtype=torch.float32, device=device).abs()
    smooth_scale = torch.randn(N, dtype=torch.float32, device=device).abs()

    sums = triton_dot(x,q)
    sums_ref = torch_fp16_dot(x,y)
    output_check(sums_ref,sums,'sum')

    sums_ref = (x.float()*(q.to(torch.float32)*quant_scale[:,None]*smooth_scale[None,:])).sum(dim=1)
    sums = triton_mix_precise_dot(x,q,smooth_scale,quant_scale,reverse=True)
    output_check(sums_ref,sums,'sum')

    ref_time = benchmark_func(torch_fp16_dot,x,y,n_repeat=n_repeat)
    benchmark_func(triton_mix_precise_dot,x,q,smooth_scale,quant_scale,reverse=True,n_repeat=n_repeat,ref_time=ref_time)


if __name__ == '__main__':
    test_dot(M=4096, N=4096)