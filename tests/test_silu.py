
import torch
from flops.utils.util import output_check
from flops.utils.benchmark import benchmark_func
from flops.utils.silu import *
# from flops.quant.smooth.reused_smooth import *

from megatron.core.fusions.fused_bias_swiglu import weighted_swiglu, weighted_swiglu_back



def torch_silu(x):
    M, N = x.shape 
    x1, x2 = torch.split(x, N//2, dim=1)
    y = torch.sigmoid(x1)*x1*x2
    return y 


def torch_silu_and_quant_forward(x, smooth_scale):
    M, N = x.shape 
    x1, x2 = torch.split(x, N//2, dim=1)
    y = torch.sigmoid(x1)*x1*x2
    y_smooth = y/smooth_scale 
    y_scale = y_smooth.abs().amax(1)/448
    y_scale = torch.exp2(torch.ceil(torch.log2(y_scale)))
    q = (y_smooth/y_scale[:,None]).to(torch.float8_e4m3fn)
    return q, y_scale

def torch_silu_and_quant_backward(grad, x, smooth_scale):
    y = torch_silu(x)
    y.backward(gradient=grad)
    dx = x.grad 
    dx_smooth = dx/smooth_scale 
    dx_scale = dx_smooth.abs().amax(1)/448
    dx_scale = torch.exp2(torch.ceil(torch.log2(dx_scale)))
    q = (dx_smooth/dx_scale[:,None]).to(torch.float8_e4m3fn)
    return q, dx_scale

if True:
    M,N = 8192,5120
    x = torch.randn((M,N),dtype=torch.bfloat16,device='cuda:0',requires_grad=True)
    weight = torch.randn((M,1),dtype=torch.bfloat16,device='cuda:0')
    grad_output = torch.randn((M,N//2),dtype=torch.bfloat16,device='cuda:0')
    smooth_scale = 1+torch.rand((N//2,),dtype=torch.float32,device='cuda:0')
    grad_smooth_scale = 1+torch.rand((N,),dtype=torch.float32,device='cuda:0')

    
    if True:
        # y = torch_silu(x)
        # y_ref = torch.nn.functional.silu(x[:,:N//2])*x[:,N//2:]
        # output_check(y_ref.float(), y.float(), 'data')

        y_q_ref, y_scale_ref = torch_silu_and_quant_forward(x, smooth_scale)
        y_q, y_scale, _ = triton_silu_and_quant_forward(x, smooth_scale, round_scale=True)
        output_check(y_q_ref.float(), y_q.float(), 'y_data')
        output_check(y_scale_ref, y_scale, 'y_scale')

        dx_q_ref, dx_scale_ref = torch_silu_and_quant_backward(grad_output, x, grad_smooth_scale)
        dx_q, dx_scale = triton_silu_and_quant_backward(grad_output, x, grad_smooth_scale, reverse=False, round_scale=True)

        output_check(dx_q_ref.float(), dx_q.float(), 'dx_data')
        output_check(dx_scale_ref, dx_scale, 'dx_scale')


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

    if False:
        x_q, x_scale = triton_weighted_silu_and_quant_forward(x,weight,smooth_scale,round_scale=True)    
        y = weighted_swiglu(x,weight)
        y_smooth = y/smooth_scale
        x_scale_ref = y_smooth.float().abs().amax(1)/448
        x_scale_ref = torch.exp2(torch.ceil(torch.log2(x_scale_ref)))
        x_q_ref = (y_smooth/x_scale_ref[:,None]).to(torch.float8_e4m3fn)

        output_check(x_q_ref.float(), x_q.float(), 'data')
        output_check(x_scale_ref, x_scale, 'scale')

        def split_silu_and_quant_forward(x,weight,smooth_scale):
            y = triton_weighted_silu_forward(x,weight)
            return triton_tokenwise_reused_smooth_quant(y,smooth_scale,round_scale=True)

        def split_silu_and_quant_backward(grad_output, x,weight,smooth_scale):
            dx, dw = triton_weighted_silu_backward(grad_output, x, weight)
            return triton_tokenwise_reused_smooth_quant(dx,smooth_scale,round_scale=True)


        ref_time = benchmark_func(split_silu_and_quant_forward,x,weight,smooth_scale, n_repeat=100, ref_bytes=M*N*4.5)
        benchmark_func(triton_weighted_silu_and_quant_forward,x,weight,smooth_scale,round_scale=True, n_repeat=100, ref_bytes=M*N*2.5, ref_time=ref_time)
        benchmark_func(triton_weighted_silu_and_quant_and_calibrate_forward,x,weight,smooth_scale, round_scale=True, n_repeat=100, ref_bytes=M*N*2.5, ref_time=ref_time)

        ref_time = benchmark_func(split_silu_and_quant_backward,grad_output,x,weight,smooth_scale, n_repeat=100, ref_bytes=M*N*8)
        benchmark_func(triton_weighted_silu_and_quant_backward,grad_output,x,weight,grad_smooth_scale,round_scale=True, n_repeat=100, ref_bytes=M*N*4, ref_time=ref_time)



if False:
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
        return triton_batch_smooth_quant(y,smooth_scales,counts, round_scale=True)


    def split_batch_silu_and_quant_backward(grad_output,x,weight,grad_smooth_scales,counts):
        dx, dw = triton_weighted_silu_backward(grad_output, x, weight)
        return triton_batch_smooth_quant(dx,grad_smooth_scales,counts, round_scale=True)


    ref_time = benchmark_func(split_batch_silu_and_quant_forward,x,weight,smooth_scales,counts, n_repeat=100, ref_bytes=n_experts*M*N*4.5)
    benchmark_func(triton_batch_weighted_silu_and_quant_forward,x,weight,smooth_scales,counts, round_scale=True, n_repeat=100, ref_bytes=n_experts*M*N*2.5, ref_time=ref_time)
    benchmark_func(triton_batch_weighted_silu_and_quant_and_calibrate_forward,x,weight,smooth_scales,counts, round_scale=True, n_repeat=100, ref_bytes=n_experts*M*N*2.5, ref_time=ref_time)

    ref_time = benchmark_func(split_batch_silu_and_quant_backward,grad_output,x,weight,grad_smooth_scales,counts, n_repeat=100, ref_bytes=n_experts*M*N*8)
    benchmark_func(triton_batch_weighted_silu_and_quant_backward,grad_output,x,weight,smooth_scales,counts, round_scale=True, n_repeat=100, ref_bytes=n_experts*M*N*4, ref_time=ref_time)


