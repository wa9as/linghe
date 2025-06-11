

import torch

import time
import os
import random
from flops.utils.util import *
from flops.utils.dot import *
from flops.utils.benchmark import benchmark_func


def torch_fp16_dot(x,y):
  return (x*y).sum(1).float()

M, N = 8192*8, 8192


dtype = torch.bfloat16
device = 'cuda:0'

n_repeat = 100

x = torch.randn(M, N, dtype=dtype, device=device)
y = torch.randn(M, N, dtype=dtype, device=device)
q = torch.randn(M, N, dtype=dtype, device=device).to(torch.float8_e4m3fn)
quant_scale = torch.randn(M, dtype=torch.float32, device=device).abs()
smooth_scale = torch.randn(N, dtype=torch.float32, device=device).abs()

sums = triton_dot(x,q,smooth_scale,quant_scale,reverse=True)
sums_ref = (x.float()*(q.to(torch.float32)*quant_scale[:,None]*smooth_scale[None,:])).sum(dim=1)
output_check(sums_ref,sums,'sum')

ref_time = benchmark_func(torch_fp16_dot,x,y,n_repeat=n_repeat)
benchmark_func(triton_dot,x,q,smooth_scale,quant_scale,reverse=True,n_repeat=n_repeat,ref_time=ref_time)