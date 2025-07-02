
import torch
from flops.utils.util import output_check
from flops.utils.benchmark import benchmark_func
from flops.utils.silu import triton_weighted_silu_forward, triton_weighted_silu_backward

from megatron.core.fusions.fused_bias_swiglu import weighted_swiglu, weighted_swiglu_back

M,N = 4096*8,4096
x = torch.randn((M,N),dtype=torch.bfloat16,device='cuda:0')
weight = torch.randn((M,1),dtype=torch.bfloat16,device='cuda:0')
grad_output = torch.randn((M,N//2),dtype=torch.bfloat16,device='cuda:0')

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