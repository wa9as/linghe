
import torch
from flops.utils.util import output_check
from flops.utils.benchmark import benchmark_func
from flops.utils.silu import *
from flops.quant.smooth.reused_smooth import *

from megatron.core.fusions.fused_bias_swiglu import weighted_swiglu, weighted_swiglu_back


if True:
    M,N = 4096,4096
    x = torch.randn((M,N),dtype=torch.bfloat16,device='cuda:0')
    weight = torch.randn((M,1),dtype=torch.bfloat16,device='cuda:0')
    grad_output = torch.randn((M,N//2),dtype=torch.bfloat16,device='cuda:0')
    smooth_scale = 1+torch.rand((N//2,),dtype=torch.float32,device='cuda:0')
    grad_smooth_scale = 1+torch.rand((N,),dtype=torch.float32,device='cuda:0')

    if False:
        ref_y = weighted_swiglu(x,weight)
        y = triton_weighted_silu_forward(x,weight)
        output_check(ref_y, y, 'y')

        dx_ref, dw_ref = weighted_swiglu_back(grad_output, x, weight)
        dx, dw = triton_weighted_silu_backward(grad_output, x, weight)
        output_check(dx_ref, dx, 'dx')
        output_check(dw_ref, dw, 'dw')

        benchmark_func(weighted_swiglu,x,weight, n_repeat=100, ref_bytes=M*N*3)

        benchmark_func(triton_weighted_silu_forward,x,weight, n_repeat=100, ref_bytes=M*N*3)

        benchmark_func(weighted_swiglu_back, grad_output, x, weight, n_repeat=100, ref_bytes=M*N*5)
        benchmark_func(triton_weighted_silu_backward, grad_output, x, weight, n_repeat=100, ref_bytes=M*N*5)

    if True:
        x_q, x_scale = triton_weighted_silu_and_quant_forward(x,weight,smooth_scale)    
        y = weighted_swiglu(x,weight)
        y_smooth = y/smooth_scale
        x_scale_ref = y_smooth.float().abs().amax(1)/448
        x_q_ref = (y_smooth/x_scale_ref[:,None]).to(torch.float8_e4m3fn)

        output_check(x_q_ref.float(), x_q.float(), 'data')
        output_check(x_scale_ref, x_scale, 'scale')

        def split_silu_and_quant_forward(x,weight,smooth_scale):
            y = triton_weighted_silu_forward(x,weight)
            return triton_tokenwise_reused_smooth_quant(y,smooth_scale)

        def split_silu_and_quant_backward(grad_output, x,weight,smooth_scale):
            dx, dw = triton_weighted_silu_backward(grad_output, x, weight)
            return triton_tokenwise_reused_smooth_quant(dx,smooth_scale)


        ref_time = benchmark_func(split_silu_and_quant_forward,x,weight,smooth_scale, n_repeat=100, ref_bytes=M*N*4.5)
        benchmark_func(triton_weighted_silu_and_quant_forward,x,weight,smooth_scale, n_repeat=100, ref_bytes=M*N*2.5, ref_time=ref_time)
        benchmark_func(triton_weighted_silu_and_quant_and_calibrate_forward,x,weight,smooth_scale, n_repeat=100, ref_bytes=M*N*2.5, ref_time=ref_time)

        ref_time = benchmark_func(split_silu_and_quant_backward,grad_output,x,weight,smooth_scale, n_repeat=100, ref_bytes=M*N*8)
        benchmark_func(triton_weighted_silu_and_quant_backward,grad_output,x,weight,grad_smooth_scale, n_repeat=100, ref_bytes=M*N*4, ref_time=ref_time)



if True:
    M,N = 2048,4096
    n_experts = 32
    x = torch.randn((M*n_experts,N),dtype=torch.bfloat16,device='cuda:0')
    weight = torch.randn((M*n_experts,1),dtype=torch.bfloat16,device='cuda:0')
    grad_output = torch.randn((M*n_experts,N//2),dtype=torch.bfloat16,device='cuda:0')
    smooth_scales = 1+torch.rand((n_experts,N//2),dtype=torch.float32,device='cuda:0')
    grad_smooth_scales = 1+torch.rand((n_experts,N),dtype=torch.float32,device='cuda:0')
    count_list = [M]*n_experts
    counts = torch.tensor(count_list,device='cuda:0',dtype=torch.int32)

    # x_q, x_scale = triton_weighted_silu_and_quant_forward(x,weight,smooth_scale)    
    # y = weighted_swiglu(x,weight)
    # y_smooth = y/smooth_scale
    # x_scale_ref = y_smooth.float().abs().amax(1)/448
    # x_q_ref = (y_smooth/x_scale_ref[:,None]).to(torch.float8_e4m3fn)

    # output_check(x_q_ref.float(), x_q.float(), 'data')
    # output_check(x_scale_ref, x_scale, 'scale')

    def split_batch_silu_and_quant_forward(x,weight,smooth_scales,counts):
        y = triton_weighted_silu_forward(x,weight)
        return triton_batch_smooth_quant(y,smooth_scales,counts)


    def split_batch_silu_and_quant_backward(grad_output,x,weight,grad_smooth_scales,counts):
        dx, dw = triton_weighted_silu_backward(grad_output, x, weight)
        return triton_batch_smooth_quant(dx,grad_smooth_scales,counts)


    ref_time = benchmark_func(split_batch_silu_and_quant_forward,x,weight,smooth_scales,counts, n_repeat=100, ref_bytes=n_experts*M*N*4.5)
    benchmark_func(triton_batch_weighted_silu_and_quant_forward,x,weight,smooth_scales,counts, n_repeat=100, ref_bytes=n_experts*M*N*2.5, ref_time=ref_time)
    benchmark_func(triton_batch_weighted_silu_and_quant_and_calibrate_forward,x,weight,smooth_scales,counts, n_repeat=100, ref_bytes=n_experts*M*N*2.5, ref_time=ref_time)

    ref_time = benchmark_func(split_batch_silu_and_quant_backward,grad_output,x,weight,grad_smooth_scales,counts, n_repeat=100, ref_bytes=n_experts*M*N*8)
    benchmark_func(triton_batch_weighted_silu_and_quant_backward,grad_output,x,weight,smooth_scales,counts, n_repeat=100, ref_bytes=n_experts*M*N*4, ref_time=ref_time)


