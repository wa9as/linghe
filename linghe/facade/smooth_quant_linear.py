# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from typing import Optional

import torch


from linghe.quant.smooth import triton_smooth_quant, \
    triton_transpose_smooth_quant
from linghe.utils.transpose import triton_transpose_and_pad
from linghe.utils.reduce import triton_abs_max

class _SmoothQuantLinear(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            input: torch.Tensor,
            weight: torch.Tensor,
            smooth_scale: torch.Tensor,
            bias: Optional[torch.Tensor]
    ):
        ctx.input_requires_grad = input.requires_grad
        ctx.weight_requires_grad = weight.requires_grad
        ctx.bias_requires_grad = bias is not None and bias.requires_grad

        ctx.out_dtype = input.dtype
        ctx.input_shape = input.shape
        input = input.view(-1, input.shape[-1])

        x_q, x_scale, x_maxs = triton_smooth_quant(input, 1 / smooth_scale)
        w_q, w_scale, w_maxs = triton_smooth_quant(weight, smooth_scale)

        output = torch._scaled_mm(x_q,
                                  w_q.t(),
                                  scale_a=x_scale.view(-1, 1),
                                  scale_b=w_scale.view(1, -1),
                                  out_dtype=ctx.out_dtype,
                                  use_fast_accum=True)

        if bias is not None:
            output += bias

        saved_tensors = [
            x_q if ctx.weight_requires_grad else None,
            x_scale if ctx.weight_requires_grad else None,
            w_q if ctx.input_requires_grad else None,
            w_scale if ctx.input_requires_grad else None,
            smooth_scale if ctx.weight_requires_grad or ctx.weight_requires_grad else None
        ]

        ctx.save_for_backward(*saved_tensors)
        out_shape = (*ctx.input_shape[0:-1], -1)
        return output.view(out_shape)

    @staticmethod
    def backward(
            ctx,
            output_grad: torch.Tensor
    ):
        x_q, x_s, w_q, w_s, smooth_scale = ctx.saved_tensors
        results = [None, None, None, None]

        output_grad = output_grad.view(-1, output_grad.shape[-1])

        y_q, y_scale, y_maxs = triton_smooth_quant(output_grad, w_s)

        wt_q = triton_transpose_and_pad(w_q, pad=True)
        dx = torch._scaled_mm(y_q,
                                  wt_q.t(),
                                  scale_a=y_scale.view(-1, 1),
                                  scale_b=smooth_scale.view(1, -1),
                                  out_dtype=ctx.out_dtype,
                                  use_fast_accum=True)

        # calculate input grad and assign to results[0]
        results[0] = dx.view(ctx.input_shape)

        # calculate weight grad and assign to results[1]
        yt_q, yt_scale, yt_maxs = triton_transpose_smooth_quant(output_grad, x_s)

        xt_q = triton_transpose_and_pad(x_q, pad=True)
        dw = torch._scaled_mm(yt_q,
                                  xt_q.t(),
                                  scale_a=yt_scale.view(-1, 1),
                                  scale_b=1/smooth_scale.view(1, -1),
                                  out_dtype=ctx.out_dtype,
                                  use_fast_accum=True)

        results[1] = dw

        if ctx.bias_requires_grad:
            # calculate bias grad and assign to results[2]
            results[2] = torch.sum(output_grad, dim=0)

        return tuple(results)


class QuantLinear(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.parameter.Parameter(
            torch.empty((out_features, in_features), device=device,
                        dtype=dtype))
        if bias:
            self.bias = torch.nn.parameter.Parameter(
                torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.bias = None

        self.gap_step = 16
        self.decay_coef = 0.9
        self.smooth_scale = None
        self.smooth_update_step = 0

        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:

            if self.smooth_update_step % self.gap_step == 0:
                input_maxs = triton_abs_max(input)
                weight_maxs = triton_abs_max(self.weight.data)
                self.smooth_scale = torch.sqrt(input_maxs * weight_maxs)

            output, smooth_scale = _SmoothQuantLinear.apply(input,
                                                                  self.weight,
                                                                  self.bias,
                                                                  self.smooth_scale)
            self.smooth_update_step += 1
        else:
            output = input @ self.weight.t()
            if self.bias is not None:
                output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def reset_parameters(self):
        self.weight.data.normal_(mean=0.0, std=0.02)
        if self.bias is not None:
            self.bias.data.zero_()
