import math
from typing import Optional, Union

import torch
from flops.quant.smooth import smooth_quant_forward, smooth_quant_update, smooth_quant_backward
from flops.quant.smooth import reused_smooth_quant_forward, reused_smooth_quant_update, reused_smooth_quant_backward


# https://code.alipay.com/Arc/atorch/blob/master/atorch/modules/fp8/scaled_linear.py#L45

class _SmoothQuantExperts(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor]
    ):
        ctx.input_requires_grad = input.requires_grad
        ctx.weight_requires_grad = weight.requires_grad
        ctx.bias_requires_grad = bias is not None and bias.requires_grad

        ctx.out_dtype = input.dtype
        ctx.input_shape = input.shape
        input = input.view(-1, input.shape[-1])

        output,x_q,w_q,x_s,w_s,smooth_scale = smooth_quant_forward(input, weight)
        if bias is not None:
            output += bias
        
        saved_tensors = [
            input if ctx.weight_requires_grad else None,
            weight if ctx.input_requires_grad else None
        ]

        ctx.save_for_backward(*saved_tensors)
        out_shape = (*ctx.input_shape[0:-1], -1)
        return output.view(out_shape)

    @staticmethod
    def backward(
        ctx,
        output_grad: torch.Tensor,
    ):
        x,w = ctx.saved_tensors
        results = [None, None, None, None]

        output_grad = output_grad.view(-1, output_grad.shape[-1])

        # calculate input grad and assign to results[0]
        dx=smooth_quant_backward(output_grad, w)
        results[0] = dx.view(ctx.input_shape)

        # calculate weight grad and assign to results[1]
        dw=smooth_quant_update(output_grad, x)
        results[1] = dw

        if ctx.bias_requires_grad:
            # calculate bias grad and assign to results[2]
            results[2] = torch.sum(output_grad, dim=0)

        return tuple(results)




class _MixSmoothQuantExperts(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ):
        ctx.input_requires_grad = input.requires_grad
        ctx.weight_requires_grad = weight.requires_grad
        ctx.bias_requires_grad = bias is not None and bias.requires_grad

        ctx.out_dtype = input.dtype
        ctx.input_shape = input.shape
        input = input.view(-1, input.shape[-1])

        output,x_q,w_q,x_s,w_s,smooth_scale = smooth_quant_forward(input, weight)

        if bias is not None:
            output += bias
        
        saved_tensors = [
            x_q if ctx.weight_requires_grad else None,
            x_s if ctx.weight_requires_grad else None,
            w_q if ctx.input_requires_grad else None,
            w_s if ctx.input_requires_grad else None,
            smooth_scale if ctx.weight_requires_grad or ctx.weight_requires_grad else None
        ]

        ctx.save_for_backward(*saved_tensors)
        out_shape = (*ctx.input_shape[0:-1], -1)
        return output.view(out_shape), smooth_scale

    @staticmethod
    def backward(
        ctx,
        output_grad: torch.Tensor,
        smooth_scale_grad: Optional[torch.Tensor]
    ):
        x_q,x_s,w_q,w_s,smooth_scale = ctx.saved_tensors
        results = [None, None, None, None]

        output_grad = output_grad.view(-1, output_grad.shape[-1])

        # calculate input grad and assign to results[0]
        dx=reused_smooth_quant_backward(output_grad, w_q, smooth_scale, w_s)
        results[0] = dx.view(ctx.input_shape)

        # calculate weight grad and assign to results[1]
        dw=reused_smooth_quant_update(output_grad, x_q, smooth_scale, x_s)
        results[1] = dw

        if ctx.bias_requires_grad:
            # calculate bias grad and assign to results[2]
            results[2] = torch.sum(output_grad, dim=0)

        return tuple(results)




class _ReusedSmoothQuantExperts(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        smooth_scale: Optional[torch.Tensor]
    ):
        ctx.input_requires_grad = input.requires_grad
        ctx.weight_requires_grad = weight.requires_grad
        ctx.bias_requires_grad = bias is not None and bias.requires_grad

        ctx.out_dtype = input.dtype
        ctx.input_shape = input.shape
        input = input.view(-1, input.shape[-1])

        if smooth_scale is None:
            output,x_q,w_q,x_s,w_s,smooth_scale = smooth_quant_forward(input, weight)
        else:
            output,x_q,w_q,x_s,w_s,smooth_scale = reused_smooth_quant_forward(input, weight,smooth_scale)

        if bias is not None:
            output += bias
        
        saved_tensors = [
            x_q if ctx.weight_requires_grad else None,
            x_s if ctx.weight_requires_grad else None,
            w_q if ctx.input_requires_grad else None,
            w_s if ctx.input_requires_grad else None,
            smooth_scale if ctx.weight_requires_grad or ctx.weight_requires_grad else None
        ]

        ctx.save_for_backward(*saved_tensors)
        out_shape = (*ctx.input_shape[0:-1], -1)
        return output.view(out_shape), smooth_scale

    @staticmethod
    def backward(
        ctx,
        output_grad: torch.Tensor,
        smooth_scale_grad: Optional[torch.Tensor]
    ):
        x_q,x_s,w_q,w_s,smooth_scale = ctx.saved_tensors
        results = [None, None, None, None]

        output_grad = output_grad.view(-1, output_grad.shape[-1])

        # calculate input grad and assign to results[0]
        dx=reused_smooth_quant_backward(output_grad, w_q, smooth_scale, w_s)
        results[0] = dx.view(ctx.input_shape)

        # calculate weight grad and assign to results[1]
        dw=reused_smooth_quant_update(output_grad, x_q, smooth_scale, x_s)
        results[1] = dw

        if ctx.bias_requires_grad:
            # calculate bias grad and assign to results[2]
            results[2] = torch.sum(output_grad, dim=0)

        return tuple(results)


class QuantExperts(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        ims: int,
        experts: int,
        device=None,
        dtype=None,
        impl='mix'
    ):
        super().__init__()
        self.dim = dim
        self.ims = ims

        self.gate_up_weight = torch.nn.parameter.Parameter(torch.empty((experts, 2*ims, dim), device=device, dtype=dtype))
        self.down_weight = torch.nn.parameter.Parameter(torch.empty((experts, dim, ims), device=device, dtype=dtype))

        assert impl in ('naive', 'mix', 'reused')
        self.impl = impl

        self.gap_step = 16
        self.decay_coef = 0.9
        self.smooth_scale = None
        self.smooth_update_step = 0

        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            if self.impl == 'naive':
                return _SmoothQuantExperts.apply(input, self.weight, self.bias)
            elif self.impl == 'mix':
                output, _ =  _MixSmoothQuantExperts.apply(input, self.weight, self.bias)
                return output
            elif self.impl == 'reused':
                if self.smooth_update_step % self.gap_step == 0:
                    output, smooth_scale = _ReusedSmoothQuantExperts.apply(input, self.weight, self.bias, None)
                else:
                    output, smooth_scale = _ReusedSmoothQuantExperts.apply(input, self.weight, self.bias, self.smooth_scale.detach())
                if self.smooth_update_step == 0:
                    self.smooth_scale = smooth_scale
                elif self.smooth_update_step % self.gap_step == 0:
                    self.smooth_scale = self.decay_coef*self.smooth_scale + (1.0-self.decay_coef)*smooth_scale
                self.smooth_update_step += 1
        else:
            output = input@self.weight.t()
            if self.bias is not None:
                output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def reset_parameters(self):
        self.weight.data.normal_(mean=0.0, std=0.02)
        if self.bias is not None:
            self.bias.data.zero_()
