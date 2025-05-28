

import torch

import time
import os
import random
from flops.utils.util import *
from flops.utils.transpose import *
from torch.profiler import profile, record_function, ProfilerActivity


# M, N, K = 8192, 4096, 13312
# M, N, K = 4096, 4096, 6144
# M, N, K = 4096, 4096, 4096

M, N, K = 4095, 768, 704


dtype = torch.bfloat16
device = 'cuda:0'

n_repeat = 100

x = torch.randn(M, K, dtype=dtype, device=device)
w = torch.randn(N, K, dtype=dtype, device=device)

x_f8 = x.to(torch.float8_e4m3fn)
w_f8 = w.to(torch.float8_e4m3fn)

ref_output = x_f8.t().contiguous()
# opt_output = triton_block_transpose(x_f8)
opt_output = triton_block_pad_transpose(x_f8, pad=4096)
torch.cuda.synchronize()
output_check(ref_output.float(),opt_output[:,:M].float(),'transpose')