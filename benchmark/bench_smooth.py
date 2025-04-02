
import torch 
from flops.quant.smooth import *
from flops.utils.util import *
from flops.utils.benchmark import benchmark_func



def benchmark_with_shape(shape):

    batch_size, out_dim, in_dim = shape

    device = 'cuda:0'
    dtype = torch.bfloat16
    qtype = torch.float8_e4m3fn  # torch.float8_e5m2
    n_repeat = 100
    gpu = torch.cuda.get_device_properties(0).name.split(' ')[-1]

    x = torch.randn(batch_size, in_dim, dtype=dtype, device=device)
    w = torch.randn(out_dim, in_dim, dtype=dtype, device=device)
    y = torch.randn(batch_size, out_dim, dtype=dtype, device=device)
    xw_smooth_scale = torch.randn(in_dim, dtype=torch.float32, device=device)
    yw_smooth_scale = torch.randn(out_dim, dtype=torch.float32, device=device)
    yx_smooth_scale = torch.randn(batch_size, dtype=torch.float32, device=device)
    x_s = torch.randn((batch_size,1), dtype=torch.float32, device=device)
    w_s = torch.randn((1,), dtype=torch.float32, device=device)

    x_f8 = x.to(qtype)
    w_f8 = w.to(qtype)
    y_f8 = y.to(qtype)

    print(f'\ndevice:{gpu}  M:{batch_size}  N:{out_dim}  K:{in_dim}')


    benchmark_func(triton_smooth_quant_nt, x,w)
    # benchmark_func(triton_smooth_quant_nn, y,w)
    # benchmark_func(triton_smooth_quant_tn, y,x)


    # ref_time = benchmark_func(fp16_forward, x, w.t(), n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
    # benchmark_func(smooth_quant_forward, x, w, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)
    # ref_time = benchmark_func(fp16_backward, y, w, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
    # benchmark_func(smooth_quant_backward, y, w, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)
    # ref_time = benchmark_func(fp16_update, y, x, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
    # benchmark_func(smooth_quant_update, y, x, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)

    # benchmark_func(triton_slide_smooth_quant,x,xw_smooth_scale)
    # benchmark_func(triton_slide_smooth_quant,w,1/xw_smooth_scale)

    # benchmark_func(triton_slide_smooth_quant_nt,x,w,xw_smooth_scale)
    # benchmark_func(triton_slide_smooth_quant_nn,y,w,yw_smooth_scale)
    # benchmark_func(triton_slide_smooth_quant_tn,y,x,yx_smooth_scale)

    # ref_time = benchmark_func(fp16_forward, x, w.t(), n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
    # benchmark_func(slide_smooth_quant_forward, x, w, xw_smooth_scale, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)
    # ref_time = benchmark_func(fp16_backward, y, w, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
    # benchmark_func(slide_smooth_quant_backward, y, w, yw_smooth_scale, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)
    # ref_time = benchmark_func(fp16_update, y, x, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
    # benchmark_func(slide_smooth_quant_update, y, x, yx_smooth_scale, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)


    ref_time = benchmark_func(fp16_forward, x, w.t(), n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
    benchmark_func(reused_smooth_quant_forward, x, w, smooth_scale=xw_smooth_scale, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)
    ref_time = benchmark_func(fp16_backward, y, w, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
    benchmark_func(reused_smooth_quant_backward, y, w_f8, xw_smooth_scale, w_s, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)
    ref_time = benchmark_func(fp16_update, y, x, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
    benchmark_func(reused_smooth_quant_update, y, x_f8, xw_smooth_scale, x_s, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)

    ref_time = benchmark_func(fp16_f_and_b, x, w, y, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*6)
    benchmark_func(reused_smooth_quant_f_and_b, x, w, y, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*6, ref_time=ref_time)



# 5b: hidden_size:4k  seq_length:8K shape:(M,N,K)
# qkv: 8192, 6144, 4096
# out: 8192, 4096, 4096
# up/gate: 8192, 13312, 4096
# down: 8192, 4096, 13312   # benchmark setting


# 80b: hidden_size:8k  seq_length:8K shape:(M,N,K)
# qkv: 8192, 10240, 8192
# out: 8192, 8192, 8192
# up/gate: 8192, 34048, 8192
# down: 8192, 4096, 34048


# benchmark_with_shape([8192, 8192, 34048 ])

for shape in [[8192, 6144, 4096], [8192, 4096, 4096], [8192, 13312, 4096], [8192, 4096, 13312],
            [8192, 10240, 8192],[8192, 8192, 8192],[8192, 34048, 8192],[8192, 4096, 34048]]:
    benchmark_with_shape(shape)
