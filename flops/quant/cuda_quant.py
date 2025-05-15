import time
from typing import Optional

import torch
from torch.utils.cpp_extension import load
from channel import triton_row_quant
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = "9.0"

# from flops.utils.benchmark import benchmark_func

def benchmark_func(fn, *args, n_repeat=1000, ref_flops=None, ref_time=None, name='', **kwargs):
    func_name = fn.__name__

    for i in range(100):
        fn(*args,**kwargs)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    
    ts = time.time()
    for i in range(n_repeat):
        start_events[i].record()
        fn(*args,**kwargs)
        end_events[i].record()
    
    torch.cuda.synchronize() 
    te = time.time()
    
    # times = sum([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
    # average_event_time = times * 1000 / n_repeat

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times = sorted(times)
    clip = max(1,n_repeat//100)
    times = sum(times[clip:-clip])
    
    average_event_time = times * 1000 / (n_repeat - 2*clip)
    
    fs = ''
    if ref_flops is not None:
        flops = ref_flops/1e12/(average_event_time/1e6)
        fs = f'FLOPS:{flops:.2f}T'
    ss = ''
    if ref_time is not None:
        ss = f'speedup:{ref_time/average_event_time:.3f}'
    print(f'{func_name:<30} {name} time:{average_event_time:.1f} us {fs} {ss}')
    return average_event_time



torch.set_grad_enabled(False)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
    #  np.random.seed(seed)
    #  random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(100)


# Load the CUDA kernel as a python module
lib = load(
    name="row_quant_lib",
    sources=["quant.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)


# un-fused naive layer norm
def naive_layer_norm(x: torch.Tensor, g: float, b: float):
    s_mean = torch.mean(x, dim=1, keepdim=True)  # m
    s_variance = 1 / torch.std(x, dim=1, keepdim=True)  # 1/std(x)
    y = ((x - s_mean) * s_variance) * g + b
    return y


# def run_benchmark(
#     perf_func: callable,
#     x: torch.Tensor,
#     tag: str,
#     out: Optional[torch.Tensor] = None,
#     warmup: int = 10,
#     iters: int = 1000,
#     show_all: bool = False,
# ):
#     g = 1.0
#     b = 0.0
#     if out is not None:
#         out.fill_(0)
#     if out is not None:
#         for i in range(warmup):
#             perf_func(x, out, g, b)
#     else:
#         for i in range(warmup):
#             _ = perf_func(x, g, b)
#     torch.cuda.synchronize()
#     start = time.time()
#     # iters
#     if out is not None:
#         for i in range(iters):
#             perf_func(x, out, g, b)
#     else:
#         for i in range(iters):
#             out = perf_func(x, g, b)
#     torch.cuda.synchronize()
#     end = time.time()
#     total_time = (end - start) * 1000  # ms
#     mean_time = total_time / iters
#     out_info = f"out_{tag}"
#     out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
#     out_val = [round(v, 8) for v in out_val]
#     out_val = [f"{v:<12}" for v in out_val]
#     print(f"{out_info:>17}: {out_val}, time:{mean_time:.8f}ms")
#     if show_all:
#         print(out)
#     return out, mean_time


qtype = torch.float8_e4m3fn
device = 'cuda:0'
dtype = torch.bfloat16

batch_size = 4096
in_dim = 4096
# batch_size = 8192
# in_dim = 4096
device = 'cuda:0'


x = torch.randn(batch_size, in_dim, dtype=dtype, device=device)
y = torch.empty((batch_size, in_dim), device=device, dtype=torch.float8_e4m3fn)
x_scale = torch.empty((batch_size,1), device=device, dtype=torch.float32)

# # print(torch.amax(torch.abs(x).float()/448.0, dim=1, keepdim=True))

# print("-" * 85)
# # run_benchmark(lib.row_quant_bf16, x_f16, "f16x8packf16", out_f16)
# print(x[0, :])
# indices = torch.nonzero(torch.eq(x[0, :], -0.8281))
# print(indices)
# print("-" * 85)

# lib.row_quant_bf16(x, y, x_scale)
# # print(x[0, :8])
# # print(x[0,2048:2048+8])
# print(y[0, 256:256+8])
# # print(y[0, 2048:2048+256])
# # print(y[0, 2048+256:2048+256+8])
# # print(x_scale)
# print("-" * 85)


# triton_xq, triton_scale = triton_row_quant(x)
# # print(1/triton_scale[0])
# print(x[0, 256:256+8])
# # print(x[0,2048:2048+8])
# print(triton_xq[0, 256:256+8])
# # print(triton_xq[0, 2048:2048+8])
# # print(triton_xq[0, 2048:2048+256])
# # print(triton_xq[0, 2048+256:2048+256+8])
# # print(triton_scale)

# print(torch.equal(x_scale, triton_scale))

n_repeat = 100
benchmark_func(triton_row_quant, x, n_repeat=n_repeat)
benchmark_func(lib.row_quant_bf16, x, y, x_scale, n_repeat=n_repeat)


