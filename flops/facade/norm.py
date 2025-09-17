import torch
import triton

from flops.utils.norm import triton_rms_norm_forward, triton_rms_norm_backward, triton_group_norm_gate_forward, triton_group_norm_gate_backward


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        
        output = triton_rms_norm_forward(
            x,
            weight,
            eps
        )
        # ctx.save_for_backward(x, weight, norm)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps

        return output

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors

        dx, dw = triton_rms_norm_backward(
            dy,
            x,
            weight,
            ctx.eps
        )

        return dx, dw, None




class GroupNormGateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, weight, eps=1e-6, group_size=4):
        
        output = triton_group_norm_gate_forward(
            x,
            gate,
            weight.data,
            eps=eps,
            group_size=group_size
        )
        ctx.save_for_backward(x, gate, weight.data)
        ctx.eps = eps
        ctx.group_size = group_size

        return output

    @staticmethod
    def backward(ctx, dy):
        x, gate, weight = ctx.saved_tensors

        dx, dg, dw = triton_group_norm_gate_backward(
            dy,
            x,
            gate,
            weight,
            ctx.eps,
            ctx.group_size
        )

        return dx, dg, dw, None, None
