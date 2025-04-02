import torch

import time
import os
import random
from flops.utils.benchmark import benchmark_func
from flops.gemm.fp8_gemm import *
from flops.utils.util import *
from flops.utils.transpose import *
from flops.quant.channel import *
from flops.quant.tile import *


batch_size, out_dim, in_dim = 8192, 4096, 13312
# batch_size, out_dim, in_dim = 4096, 4096, 4096

dtype = torch.bfloat16
n_repeat = 1000

x = torch.randn(batch_size, in_dim, dtype=dtype, device='cuda:0')
w = torch.randn(out_dim, in_dim, dtype=dtype, device='cuda:0')

x_f8 = x.to(torch.float8_e4m3fn)
w_f8 = w.to(torch.float8_e4m3fn)

# benchmark_func(trival_fp8_gemm, x_f8, w_f8, torch.bfloat16, n_repeat=n_repeat)
# benchmark_func(persistent_fp8_gemm, x_f8, w_f8.t(), torch.bfloat16, n_repeat=n_repeat)
# benchmark_func(fp8_gemm_nn, x_f8,w_f8,torch.bfloat16, n_repeat=n_repeat)

benchmark_func(block_quant, x,n_repeat=n_repeat)

benchmark_func(stupid_tile_quant, x, n_repeat=n_repeat)
benchmark_func(tile_quant, x, n_repeat=n_repeat)



# benchmark_func(fp16_transpose, x, n_repeat=n_repeat)
# benchmark_func(triton_transpose,x, n_repeat=n_repeat)
# benchmark_func(triton_opt_transpose,x, n_repeat=n_repeat)


# benchmark_func(triton_row_quant, x, n_repeat=n_repeat)
# benchmark_func(triton_transpose_row_quant, x, n_repeat=n_repeat)


# benchmark_func(fp8_transpose, x_f8, n_repeat=n_repeat)
# benchmark_func(triton_transpose,x_f8, n_repeat=n_repeat)
# benchmark_func(triton_opt_transpose,x_f8, n_repeat=n_repeat)





