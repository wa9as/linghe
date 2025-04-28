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
from torch.profiler import profile, record_function, ProfilerActivity


batch_size, out_dim, in_dim = 4096, 4096, 13312
# batch_size, out_dim, in_dim = 4096, 4096, 6144
# batch_size, out_dim, in_dim = 4096, 4096, 4096

dtype = torch.bfloat16
n_repeat = 100

x = torch.randn(batch_size, in_dim, dtype=dtype, device='cuda:0')
w = torch.randn(out_dim, in_dim, dtype=dtype, device='cuda:0')

x_f8 = x.to(torch.float8_e4m3fn)
w_f8 = w.to(torch.float8_e4m3fn)

# benchmark_func(trival_fp8_gemm, x_f8, w_f8, torch.bfloat16, n_repeat=n_repeat)
# benchmark_func(persistent_fp8_gemm, x_f8, w_f8.t(), torch.bfloat16, n_repeat=n_repeat)
# benchmark_func(fp8_gemm_nn, x_f8,w_f8,torch.bfloat16, n_repeat=n_repeat)

# benchmark_func(block_quant, x,n_repeat=n_repeat)
# benchmark_func(stupid_tile_quant, x, n_repeat=n_repeat)



print(f'x:[{batch_size},{in_dim}]')

y_ref, s_ref = stupid_tile_quant(x)
y_opt, s_opt = tile_quant(x)

torch.testing.assert_close(y_opt.float(), y_ref.float(),
                                   rtol=0.02, atol=0.02)
torch.testing.assert_close(s_ref.float(), s_opt.float(),
                                   rtol=0.02, atol=0.02)
benchmark_func(stupid_tile_quant, x, n_repeat=n_repeat)
benchmark_func(tile_quant, x, n_repeat=n_repeat)
# benchmark_func(persist_tile_quant, x, n_repeat=n_repeat)


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]) as prof:
    benchmark_func(tile_quant, x, n_repeat=n_repeat)
print(prof.key_averages().table(sort_by=None, top_level_events_only=True, row_limit=2000))
# print(prof.key_averages(group_by_stack_n=5).table(sort_by=None, row_limit=100))
# prof.export_chrome_trace("trace.json")





# benchmark_func(fp16_transpose, x, n_repeat=n_repeat)
# benchmark_func(triton_transpose,x, n_repeat=n_repeat)
# benchmark_func(triton_opt_transpose,x, n_repeat=n_repeat)


# benchmark_func(triton_row_quant, x, n_repeat=n_repeat)
# benchmark_func(triton_transpose_row_quant, x, n_repeat=n_repeat)


# benchmark_func(fp8_transpose, x_f8, n_repeat=n_repeat)
# benchmark_func(triton_transpose,x_f8, n_repeat=n_repeat)
# benchmark_func(triton_opt_transpose,x_f8, n_repeat=n_repeat)





