import torch

from flops.utils.norm import (triton_rms_norm_and_quant_forward,
                              triton_rms_norm_backward,
                              triton_rms_norm_forward)
from flops.utils.util import (output_check,
                              torch_smooth_quant)


def torch_rms_forward(x, weight):
    N = x.shape[-1]
    rmsnorm = torch.nn.RMSNorm(
        normalized_shape=N,
        eps=1e-6,
        dtype=torch.bfloat16,
        device=x.device
    )
    with torch.no_grad():
        rmsnorm.weight.copy_(weight)
    return rmsnorm(x)


def torch_rms_backward(x, weight, dy):
    N = x.shape[-1]
    rmsnorm = torch.nn.RMSNorm(
        normalized_shape=N,
        eps=1e-6,
        dtype=torch.bfloat16,
        device=x.device
    )
    with torch.no_grad():
        rmsnorm.weight.copy_(weight)
    x = x.clone().detach().requires_grad_()
    y = rmsnorm(x)
    y.backward(gradient=dy)
    return x.grad, rmsnorm.weight.grad


def test_rmsnorm(M=4096, N=4096):
    dtype = torch.bfloat16
    device = 'cuda:0'

    x = torch.randn(M, N, dtype=dtype, requires_grad=True, device=device) ** 3
    weight = torch.randn(N, dtype=dtype, requires_grad=True, device=device)
    dy = torch.randn(M, N, dtype=dtype, device=device)

    y_ref = torch_rms_forward(x, weight)
    y = triton_rms_norm_forward(x, weight)
    output_check(y_ref.float(), y.float(), 'y')

    dx_ref, dw_ref = torch_rms_backward(x, weight, dy)
    dx, dw = triton_rms_norm_backward(dy, x, weight)
    output_check(dx_ref, dx, mode="dx")
    output_check(dw_ref, dw, mode='dw')

    smooth_scale = torch.rand(N, dtype=torch.float32, requires_grad=False,
                              device=device) + 0.1
    q, scale, maxs, rms = triton_rms_norm_and_quant_forward(x, weight,
                                                            smooth_scale,
                                                            round_scale=True)
    q_ref, scale_ref = torch_smooth_quant(y_ref, smooth_scale, reverse=False,
                                          round_scale=True)
    output_check(q_ref, q, mode="data")
    output_check(scale_ref, scale, mode='scale')


if __name__ == '__main__':
    test_rmsnorm(M=4096, N=4096)
    test_rmsnorm(M=4096, N=8192)
    test_rmsnorm(M=8192, N=2048)
