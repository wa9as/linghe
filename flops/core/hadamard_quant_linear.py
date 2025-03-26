import math
from typing import Optional, Union

import torch
# from torch.nn.parameter import Parameter
from flops.quant.hadamard import bit_hadamard_quant_forward, bit_hadamard_quant_update, bit_hadamard_quant_backward


class _HadamardQuantLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        hadamard_matrix: torch.Tensor,
    ):
        assert input.requires_grad
        assert weight.requires_grad
        if bias is not None:
            assert input.requires_grad

        ctx.out_dtype = input.dtype
        ctx.input_shape = input.shape
        ctx.bias = bias is not None
        input = input.view(-1, input.shape[-1])

        output,x_bt,w_bt,x_q,w_q,x_scale,w_scale = bit_hadamard_quant_forward(input, weight, hadamard_matrix)
        if bias is not None:
            output += bias
        
        saved_tensors = [
            x_bt,
            w_bt,
            hadamard_matrix
        ]

        ctx.save_for_backward(*saved_tensors)
        out_shape = (*ctx.input_shape[0:-1], -1)
        return output.view(out_shape)

    @staticmethod
    def backward(
        ctx,
        output_grad: torch.Tensor,
    ):
        x_bt,w_bt,hadamard_matrix = ctx.saved_tensors
        results = [None, None, None, None]

        output_grad = output_grad.view(-1, output_grad.shape[-1])

        # calculate input grad and assign to results[0]
        dx,y_bt,y_q,w_q,y_scale,w_scale=bit_hadamard_quant_backward(output_grad, w_bt, hadamard_matrix)
        results[0] = dx.view(ctx.input_shape)

        # calculate weight grad and assign to results[1]
        dw,y_q,x_q,y_scale,x_scale=bit_hadamard_quant_update(y_bt, x_bt, hadamard_matrix)
        results[1] = dw

        if ctx.bias:
            # calculate bias grad and assign to results[2]
            # db = sum(dy)
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
        self.reset_parameters()
        self.hadamard_matrix_size = 64
        self.hadamard_matrix = self._hadamard_matrix(self.hadamard_matrix_size, device, dtype)

    def _hadamard_matrix(self, size, device, dtype):
        m2 = torch.tensor([[1,1],[1,-1]],device=device,dtype=dtype)
        m = m2
        for i in range(int(round(math.log2(size)-1))):
            m = torch.kron(m,m2)
        return m.to(dtype)

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.is_autocast_enabled():
            input = input.to(torch.get_autocast_gpu_dtype())
        return _HadamardQuantLinear.apply(input, self.weight, self.bias, self.hadamard_matrix)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
