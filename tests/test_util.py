

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
M, N, K = 128, 8192, 4096

dtype = torch.float32
device = 'cuda:0'
n_repeat = 100

x = torch.randn(M, K, dtype=dtype, device=device)

mode = 'torch_max'
if mode == 'torch_max':

    benchmark_func(torch.amax, x, n_repeat=n_repeat, ref_bytes=M*K*8)

    xs = [torch.randn(N, dtype=dtype, device=device) for x in range(32)]
    benchmark_func(torch.stack, xs, 0, n_repeat=n_repeat, ref_bytes=N*32*8)

