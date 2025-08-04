import torch

from flops.utils.transpose import triton_transpose


class TransposeDim01Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return triton_transpose(x, dim0=0, dim1=1)

    @staticmethod
    def backward(ctx, grad_output):
        return triton_transpose(grad_output, dim0=0, dim1=1)
