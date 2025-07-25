import torch
from torch.profiler import profile, ProfilerActivity

from flops.quant.block.group import (triton_group_quant)

# M, N, K = 8192, 10240, 8192  # max qkv
# M, N, K = 8192, 8192, 8192  # max out
M, N, K = 1024, 4096, 8192  # max gate_up
# M, N, K = 8192, 2048, 8192  # max down

# M, N, K = M-1, N-1, K-1

dtype = torch.bfloat16
device = 'cuda:0'
n_repeat = 100

x = torch.randn(M, K, dtype=dtype, device=device)
w = torch.randn(N, K, dtype=dtype, device=device)
y = torch.randn(M, N, dtype=dtype, device=device)

x_q = x.to(torch.float8_e4m3fn)
w_q = w.to(torch.float8_e4m3fn)
y_q = y.to(torch.float8_e4m3fn)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA,
                         ProfilerActivity.XPU]) as prof:
    for i in range(100):
        triton_group_quant(x)
print(prof.key_averages().table(sort_by=None, top_level_events_only=True,
                                row_limit=2000))
print(prof.key_averages(group_by_stack_n=5).table(sort_by=None, row_limit=100))
prof.export_chrome_trace("trace.json")
