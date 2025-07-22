


import torch

import time
import os
import random
from flops.utils.util import *
from flops.utils.benchmark import benchmark_func

M, N = 4096, 8192
# dtype = torch.float32
dtype = torch.bfloat16
device = 'cuda:0'
n_repeat = 100
from flops.utils.reduce import *



if False:
    x = 100*torch.randn(M, 1, N, dtype=dtype, device=device)

    scales = 1.0/torch.sqrt(torch.maximum(x[:,0].abs().float().amax(0), torch.ones(M,N,dtype=dtype,device=device)) )
    smooth_scale_ref = torch.exp2(torch.ceil(torch.log2(scales)))

    smooth_scale = triton_update_weight_smooth_scale(x, round_scale=True)
    output_check(smooth_scale_ref, smooth_scale)
    benchmark_func(triton_update_weight_smooth_scale, x, round_scale=True, n_repeat=n_repeat, ref_bytes=M*N*2)

if True:
    xs = [torch.randn(4096, 8192, dtype=torch.float32, device=device).to(torch.float8_e4m3fn).to(torch.float32) for i in range(64)]

    ref_bytes =sum([x.numel() for x in xs])*4
    def torch_count_zero(xs):
        count = torch.tensor([0], dtype=torch.int64, device='cuda')
        for x in xs:
            count += x.numel()-torch.count_nonzero(x)
        return count
    
    def torch_sum(xs):
        return sum([x.square().sum() for x in xs])

    count_ref =  torch_count_zero(xs)
    count = triton_batch_count_zero(xs)
    print(f'{count_ref=} {count=}')
    assert count_ref.item()-count.item() == 0

    sum_ref = torch_sum(xs)
    sums = triton_batch_sum_with_ord(xs)
    output_check(sum_ref, sums)

    ref_time = benchmark_func(torch_count_zero, xs,  n_repeat=n_repeat, ref_bytes=ref_bytes)
    benchmark_func(triton_batch_count_zero, xs, n_repeat=n_repeat, ref_bytes=ref_bytes, ref_time=ref_time)
