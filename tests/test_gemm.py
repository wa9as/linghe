import torch

import time
import os
import random
from flops.utils.benchmark import benchmark_func
from flops.gemm.fp8_gemm import *
from flops.gemm.channelwise_fp8_gemm import *
from flops.utils.util import *
from flops.utils.add import *



for i in range(0, 1):
    M, N, K = 2048, 8192, 8192-32
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
    ref_out = (x_q.float()*x_scale[:,None])@(w_q.float()*w_scale[:,None]).t()

    o = torch.zeros((M,N), dtype=dtype,device=device)
    out = triton_scaled_mm(x_q,w_q,x_scale,w_scale, c=o, accum=True)
    output_check(ref_out, out.float(), mode='gemm')
