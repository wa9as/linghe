
import torch
from flops.utils.add import triton_block_add

class InplaceAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        return triton_block_add(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output