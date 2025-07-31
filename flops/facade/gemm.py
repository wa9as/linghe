	
import torch 
from flops.gemm.fp32_gemm import triton_fp32_gemm, triton_fp32_gemm_for_backward, triton_fp32_gemm_for_update


class FusedFp32GEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        shape = input.shape 
        assert len(shape) == 3
        input = input.view(shape[0]*shape[1], shape[2])
        logits = triton_fp32_gemm(input, weight.data)
        ctx.input_requires_grad = input.requires_grad
        ctx.weight_requires_grad = weight.requires_grad
        ctx.shape = shape 
        ctx.save_for_backward(input, weight.data)
        return logits.view(shape[0], shape[1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        shape = grad_output.shape 
        grad_output = grad_output.view(shape[0]*shape[1], shape[2])
        input, weight = ctx.saved_tensors
        dx = triton_fp32_gemm_for_backward(grad_output, weight, accum=False)
        dx = dx.view(*ctx.shape)
        dw = triton_fp32_gemm_for_update(grad_output, input)
        
        return dx, dw