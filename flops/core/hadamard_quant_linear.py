import math
from typing import Optional, Union

import torch
from flops.quant.hadamard import hadamard_quant_forward, hadamard_quant_update, hadamard_quant_backward
from flops.quant.hadamard import fused_hadamard_quant_forward, fused_hadamard_quant_update, fused_hadamard_quant_backward
from flops.quant.hadamard import bit_hadamard_quant_forward, bit_hadamard_quant_update, bit_hadamard_quant_backward


# https://code.alipay.com/Arc/atorch/blob/master/atorch/modules/fp8/scaled_linear.py#L45

class _HadamardQuantLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        hadamard_matrix: torch.Tensor
    ):
        ctx.input_requires_grad = input.requires_grad
        ctx.weight_requires_grad = weight.requires_grad
        ctx.bias_requires_grad = bias is not None and bias.requires_grad

        ctx.out_dtype = input.dtype
        ctx.input_shape = input.shape
        input = input.view(-1, input.shape[-1])

        output = hadamard_quant_forward(input, weight, hadamard_matrix)
        if bias is not None:
            output += bias
        
        saved_tensors = [
            input if ctx.input_requires_grad else None,
            weight if ctx.weight_requires_grad else None,
            hadamard_matrix if ctx.weight_requires_grad or ctx.weight_requires_grad else None
        ]

        ctx.save_for_backward(*saved_tensors)
        out_shape = (*ctx.input_shape[0:-1], -1)
        return output.view(out_shape)

    @staticmethod
    def backward(
        ctx,
        output_grad: torch.Tensor,
    ):
        x,w,hadamard_matrix = ctx.saved_tensors
        results = [None, None, None, None]

        output_grad = output_grad.view(-1, output_grad.shape[-1])

        # calculate input grad and assign to results[0]
        dx=hadamard_quant_backward(output_grad, w, hadamard_matrix)
        results[0] = dx.view(ctx.input_shape)

        # calculate weight grad and assign to results[1]
        dw=hadamard_quant_update(output_grad, x, hadamard_matrix)
        results[1] = dw

        if ctx.bias_requires_grad:
            # calculate bias grad and assign to results[2]
            results[2] = torch.sum(output_grad, dim=0)

        return tuple(results)



class _FusedHadamardQuantLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        hadamard_matrix: torch.Tensor
    ):
        ctx.input_requires_grad = input.requires_grad
        ctx.weight_requires_grad = weight.requires_grad
        ctx.bias_requires_grad = bias is not None and bias.requires_grad

        ctx.out_dtype = input.dtype
        ctx.input_shape = input.shape
        input = input.view(-1, input.shape[-1])

        output = fused_hadamard_quant_forward(input, weight, hadamard_matrix)
        if bias is not None:
            output += bias
        
        saved_tensors = [
            input if ctx.input_requires_grad else None,
            weight if ctx.weight_requires_grad else None,
            hadamard_matrix if ctx.weight_requires_grad or ctx.weight_requires_grad else None
        ]

        ctx.save_for_backward(*saved_tensors)
        out_shape = (*ctx.input_shape[0:-1], -1)
        return output.view(out_shape)

    @staticmethod
    def backward(
        ctx,
        output_grad: torch.Tensor,
    ):
        x,w,hadamard_matrix = ctx.saved_tensors
        results = [None, None, None, None]

        output_grad = output_grad.view(-1, output_grad.shape[-1])

        # calculate input grad and assign to results[0]
        dx=fused_hadamard_quant_backward(output_grad, w, hadamard_matrix)
        results[0] = dx.view(ctx.input_shape)

        # calculate weight grad and assign to results[1]
        dw=fused_hadamard_quant_update(output_grad, x, hadamard_matrix)
        results[1] = dw

        if ctx.bias_requires_grad:
            # calculate bias grad and assign to results[2]
            results[2] = torch.sum(output_grad, dim=0)

        return tuple(results)



class _BitHadamardQuantLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        hadamard_matrix: torch.Tensor
    ):
        ctx.input_requires_grad = input.requires_grad
        ctx.weight_requires_grad = weight.requires_grad
        ctx.bias_requires_grad = bias is not None and bias.requires_grad

        ctx.out_dtype = input.dtype
        ctx.input_shape = input.shape
        input = input.view(-1, input.shape[-1])

        output,x_bt,w_bt = bit_hadamard_quant_forward(input, weight, hadamard_matrix)
        if bias is not None:
            output += bias
        
        saved_tensors = [
            x_bt if ctx.input_requires_grad else None,
            w_bt if ctx.weight_requires_grad else None,
            hadamard_matrix if ctx.weight_requires_grad or ctx.weight_requires_grad else None
        ]

        ctx.save_for_backward(*saved_tensors)
        out_shape = (*ctx.input_shape[0:-1], -1)
        return output.view(out_shape)

    @staticmethod
    def backward(
        ctx,
        output_grad: torch.Tensor,
    ):
        x,w,hadamard_matrix = ctx.saved_tensors
        results = [None, None, None, None]

        output_grad = output_grad.view(-1, output_grad.shape[-1])

        # calculate input grad and assign to results[0]
        dx,y_bt=bit_hadamard_quant_backward(output_grad, w, hadamard_matrix)
        results[0] = dx.view(ctx.input_shape)

        # calculate weight grad and assign to results[1]
        dw=bit_hadamard_quant_update(y_bt, x, hadamard_matrix)
        results[1] = dw

        if ctx.bias_requires_grad:
            # calculate bias grad and assign to results[2]
            results[2] = torch.sum(output_grad, dim=0)

        return tuple(results)



class HadamardQuantLinear(torch.nn.Module):
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
        self.weight = torch.nn.parameter.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.bias = None
    
        size = 32 if 'H20' in torch.cuda.get_device_properties(0).name else 64
        self.hadamard_matrix = torch.nn.parameter.Parameter(self._hadamard_matrix(size, device=device, dtype=dtype, norm=True), requires_grad=False)
        # self.register_buffer("hadamard_matrix", hadamard_matrix, persistent=False)
        self.reset_parameters()

    def _hadamard_matrix(self, size, device=None, dtype=None, norm=False):
        assert 2**int(math.log2(size)) == size
        m2 = torch.tensor([[1,1],[1,-1]], device=device, dtype=torch.float32)
        m = m2
        for i in range(int(math.log2(size))-1):
            m = torch.kron(m,m2)
        if norm:
            m = m / size**0.5
        if dtype is not None:
            m = m.to(dtype)
        return m

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # if self.hadamard_matrix.device != input.device or self.hadamard_matrix.dtype != input.dtype:
        #     self.hadamard_matrix = self.hadamard_matrix.to(device=input.device,dtype=input.dtype)
        return _HadamardQuantLinear.apply(input, self.weight, self.bias, self.hadamard_matrix)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def reset_parameters(self):
        self.weight.data.normal_(mean=0.0, std=0.02)
        if self.bias is not None:
            self.bias.data.zero_()
