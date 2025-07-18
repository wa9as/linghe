

import torch

import time
import os
import random
from flops.utils.util import *
from flops.utils.benchmark import benchmark_func
from flops.quant.smooth.seperate_smooth import triton_update_weight_smooth_scale

M, N, K = 8192, 4096, 8192
# M, N, K = 4096, 4096, 6144
# M, N, K = 4096, 4096, 4096
# M, N, K = 4096, 256, 8192

# dtype = torch.float32
dtype = torch.bfloat16
device = 'cuda:0'
n_repeat = 100


if False:
    x = torch.randn(M, K, dtype=dtype, device=device)
    benchmark_func(torch.amax, x, n_repeat=n_repeat, ref_bytes=M*K*8)

if False:
    xs = [torch.randn(N, dtype=dtype, device=device) for x in range(32)]
    benchmark_func(torch.stack, xs, 0, n_repeat=n_repeat, ref_bytes=N*32*8)

if False:
    x = torch.randn(M, K, dtype=dtype, device=device)
    benchmark_func(lambda x:x.zero_(), x, n_repeat=n_repeat, ref_bytes=M*K*2)


if False:
    benchmark_func(torch.zeros,(M,K),dtype=dtype,device=device, n_repeat=n_repeat, ref_bytes=M*K*2)

if False:
    def d2h(x,device):
        return x.to(device)
    counts = torch.tensor([0]*32,dtype=torch.int64)
    benchmark_func(d2h,counts,device=device, n_repeat=n_repeat, ref_bytes=M*K*2)

if True:
    x = torch.randn(M, K, dtype=dtype, device=device)
    scales = torch.sqrt(x.abs().float().amax(0))
    scales = torch.where(scales<4,1,scales)
    smooth_scale_ref = torch.exp2(torch.ceil(torch.log2(scales)))
    smooth_scale = triton_update_weight_smooth_scale(x, round_scale=True)
    output_check(smooth_scale_ref, smooth_scale)
    benchmark_func(triton_update_weight_smooth_scale, x, round_scale=True, n_repeat=n_repeat, ref_bytes=M*K*2)
