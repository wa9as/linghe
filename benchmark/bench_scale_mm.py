import torch

import time
import os
import random
from flops.utils.benchmark import benchmark_func
from flops.gemm.fp8_gemm import *
from flops.utils.util import *


for i in range(0, 1):
    M, N, K = 8192, 8192, 8192
    dtype = torch.bfloat16
    n_repeat = 1000

    x = torch.randn(M, K, dtype=dtype, device='cuda:0')
    w = torch.randn(N, K, dtype=dtype, device='cuda:0')

    xrs = x.abs().float().amax(dim=1,keepdim=True)
    wcs = w.abs().float().amax(dim=1,keepdim=True)
    x_f8 = (448*x/xrs).to(torch.float8_e4m3fn)
    w_f8 = (448*w/wcs).to(torch.float8_e4m3fn)

    ref_time=benchmark_func(fp16_forward, x, w.t(), n_repeat=n_repeat, name=f'M:{M}')
    benchmark_func(torch_fp16_vector_scaled_mm, x_f8, w_f8.t(), xrs, wcs.t(), n_repeat=n_repeat, ref_time=ref_time, name=f'M:{M}')
