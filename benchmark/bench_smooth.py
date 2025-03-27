
import torch 
from flops.quant.smooth import *
from flops.utils.util import *
from flops.utils.benchmark import benchmark_func


batch_size, out_dim, in_dim = 8192, 4096, 13312

device = 'cuda:0'
dtype = torch.bfloat16
qtype = torch.float8_e4m3fn  # torch.float8_e5m2
n_repeat = 1000
gpu = torch.cuda.get_device_properties(0).name.split(' ')[-1]


x = torch.randn(batch_size, in_dim, dtype=dtype, device=device)
w = torch.randn(out_dim, in_dim, dtype=dtype, device=device)
y = torch.randn(batch_size, out_dim, dtype=dtype, device=device)
xw_smooth_scale = torch.randn(in_dim, dtype=dtype, device=device)
yw_smooth_scale = torch.randn(out_dim, dtype=dtype, device=device)
yx_smooth_scale = torch.randn(batch_size, dtype=dtype, device=device)

x_f8 = x.to(qtype)
w_f8 = w.to(qtype)
y_f8 = y.to(qtype)

print(f'\ndevice:{gpu} M:{batch_size} N:{out_dim} K:{in_dim}')


benchmark_func(triton_smooth_quant_nt, x,w)
benchmark_func(triton_smooth_quant_nn, y,w)
benchmark_func(triton_smooth_quant_tn, y,x)


ref_time = benchmark_func(fp16_forward, x, w.t(), n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
benchmark_func(smooth_quant_forward, x, w, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)
ref_time = benchmark_func(fp16_backward, y, w, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
benchmark_func(smooth_quant_backward, y, w, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)
ref_time = benchmark_func(fp16_update, y, x, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
benchmark_func(smooth_quant_update, y, x, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)


benchmark_func(triton_slide_smooth_quant_nt,x,w,xw_smooth_scale)
benchmark_func(triton_slide_smooth_quant_nn,y,w,yw_smooth_scale)
benchmark_func(triton_slide_smooth_quant_tn,y,x,yx_smooth_scale)

ref_time = benchmark_func(fp16_forward, x, w.t(), n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
benchmark_func(slide_smooth_quant_forward, x, w, xw_smooth_scale, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)
ref_time = benchmark_func(fp16_backward, y, w, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
benchmark_func(slide_smooth_quant_backward, y, w, yw_smooth_scale, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)
ref_time = benchmark_func(fp16_update, y, x, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
benchmark_func(slide_smooth_quant_update, y, x, yx_smooth_scale, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)

