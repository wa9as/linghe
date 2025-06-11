
import torch
from triton.language.standard import xor_sum 
from flops.quant.smooth.naive_smooth import *
from flops.quant.smooth.reused_smooth import *
from flops.quant.smooth.seperate_smooth import *

from flops.utils.util import *
from flops.utils.benchmark import benchmark_func



def benchmark_with_shape(shape):

    M, N, K = shape

    device = 'cuda:0'
    dtype = torch.bfloat16
    qtype = torch.float8_e4m3fn  # torch.float8_e5m2
    n_repeat = 100
    gpu = torch.cuda.get_device_properties(0).name.split(' ')[-1]

    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)
    y = torch.randn(M, N, dtype=dtype, device=device)
    xw_smooth_scale = torch.randn(K, dtype=torch.float32, device=device)
    yw_smooth_scale = torch.randn(N, dtype=torch.float32, device=device)
    yx_smooth_scale = torch.randn(M, dtype=torch.float32, device=device)
    x_s = torch.randn((M,1), dtype=torch.float32, device=device)
    w_s = torch.randn((1,), dtype=torch.float32, device=device)

    x_q = x.to(qtype)
    w_q = w.to(qtype)
    y_q = y.to(qtype)

    print(f'\ndevice:{gpu}  M:{M}  N:{N}  K:{K}')


    # benchmark_func(triton_smooth_quant_y, y, yw_smooth_scale, yx_smooth_scale, reverse=True, transpose=True, pad=False, n_repeat=n_repeat)
    # benchmark_func(triton_reused_smooth_quant, y, yw_smooth_scale, reverse=True, pad_scale=False, round_scale=True, n_repeat=n_repeat)
    # benchmark_func(triton_reused_transpose_pad_rescale_smooth_quant, y_q, yw_smooth_scale, yx_smooth_scale, yx_smooth_scale, reverse=True, pad=False, n_repeat=n_repeat)


    # benchmark_func(triton_smooth_quant_nt, x,w)
    # benchmark_func(triton_smooth_quant_nn, y,w)
    # benchmark_func(triton_smooth_quant_tn, y,x)


    # ref_time = benchmark_func(fp16_forward, x, w.t(), n_repeat=n_repeat, ref_flops=M*K*N*2)
    # benchmark_func(smooth_quant_forward, x, w, n_repeat=n_repeat, ref_flops=M*K*N*2, ref_time=ref_time)
    # ref_time = benchmark_func(fp16_backward, y, w, n_repeat=n_repeat, ref_flops=M*K*N*2)
    # benchmark_func(smooth_quant_backward, y, w, n_repeat=n_repeat, ref_flops=M*K*N*2, ref_time=ref_time)
    # ref_time = benchmark_func(fp16_update, y, x, n_repeat=n_repeat, ref_flops=M*K*N*2)
    # benchmark_func(smooth_quant_update, y, x, n_repeat=n_repeat, ref_flops=M*K*N*2, ref_time=ref_time)

    # ref_time = benchmark_func(fp16_forward, x, w.t(), n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
    # benchmark_func(seperate_smooth_quant_forward, x, w, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)
    # ref_time = benchmark_func(fp16_forward, y, w, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
    # benchmark_func(seperate_smooth_quant_backward, y, w, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)
    # ref_time = benchmark_func(fp16_update, y, x, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
    # benchmark_func(seperate_smooth_quant_update, y, x, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)

    ref_time = benchmark_func(fp16_f_and_b, x, w, y, n_repeat=n_repeat, ref_flops=M*K*N*6)
    benchmark_func(seperate_smooth_quant_f_and_b, x, w, y, xw_smooth_scale, n_repeat=n_repeat, ref_flops=M*K*N*6, ref_time=ref_time)

    # benchmark_func(triton_slide_smooth_quant,x,xw_smooth_scale)
    # benchmark_func(triton_slide_smooth_quant,w,1/xw_smooth_scale)

    # benchmark_func(triton_slide_smooth_quant_nt,x,w,xw_smooth_scale)
    # benchmark_func(triton_slide_smooth_quant_nn,y,w,yw_smooth_scale)
    # benchmark_func(triton_slide_smooth_quant_tn,y,x,yx_smooth_scale)

    # ref_time = benchmark_func(fp16_forward, x, w.t(), n_repeat=n_repeat, ref_flops=M*K*N*2)
    # benchmark_func(slide_smooth_quant_forward, x, w, xw_smooth_scale, n_repeat=n_repeat, ref_flops=M*K*N*2, ref_time=ref_time)
    # ref_time = benchmark_func(fp16_backward, y, w, n_repeat=n_repeat, ref_flops=M*K*N*2)
    # benchmark_func(slide_smooth_quant_backward, y, w, yw_smooth_scale, n_repeat=n_repeat, ref_flops=M*K*N*2, ref_time=ref_time)
    # ref_time = benchmark_func(fp16_update, y, x, n_repeat=n_repeat, ref_flops=M*K*N*2)
    # benchmark_func(slide_smooth_quant_update, y, x, yx_smooth_scale, n_repeat=n_repeat, ref_flops=M*K*N*2, ref_time=ref_time)


    # ref_time = benchmark_func(fp16_forward, x, w.t(), n_repeat=n_repeat, ref_flops=M*K*N*2)
    # benchmark_func(reused_smooth_quant_forward, x, w, smooth_scale=xw_smooth_scale, n_repeat=n_repeat, ref_flops=M*K*N*2, ref_time=ref_time)
    # ref_time = benchmark_func(fp16_backward, y, w, n_repeat=n_repeat, ref_flops=M*K*N*2)
    # benchmark_func(reused_smooth_quant_backward, y, w_f8, xw_smooth_scale, w_s, n_repeat=n_repeat, ref_flops=M*K*N*2, ref_time=ref_time)
    # ref_time = benchmark_func(fp16_update, y, x, n_repeat=n_repeat, ref_flops=M*K*N*2)
    # benchmark_func(reused_smooth_quant_update, y, x_f8, xw_smooth_scale, x_s, n_repeat=n_repeat, ref_flops=M*K*N*2, ref_time=ref_time)

    ref_time = benchmark_func(fp16_f_and_b, x, w, y, n_repeat=n_repeat, ref_flops=M*K*N*6)
    benchmark_func(reused_smooth_quant_f_and_b, x, w, y, xw_smooth_scale, n_repeat=n_repeat, ref_flops=M*K*N*6, ref_time=ref_time)




benchmark_with_shape([2048, 8192, 2048])

# for shape in [[8192, 6144, 4096], [8192, 4096, 4096], [8192, 13312, 4096], [8192, 4096, 13312],
#             [8192, 10240, 8192],[8192, 8192, 8192],[8192, 34048, 8192],[8192, 4096, 34048]]:
#     benchmark_with_shape(shape)
