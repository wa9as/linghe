import torch

from flops.utils.norm import (triton_rms_norm_and_quant_forward,
                              triton_rms_norm_backward,
                              triton_rms_norm_forward)
from flops.utils.util import (output_check,
                              torch_smooth_quant,
                              torch_group_quant)
from flops.utils.benchmark import benchmark_func


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




def torch_rms_and_quant_forward(x, weight, smooth_scale=None, round_scale=False):
    N = x.shape[-1]
    rmsnorm = torch.nn.RMSNorm(
        normalized_shape=N,
        eps=1e-6,
        dtype=torch.bfloat16,
        device=x.device
    )
    with torch.no_grad():
        rmsnorm.weight.copy_(weight)
    y = rmsnorm(x)
    if smooth_scale is None:
        # blockwise
        y_q, y_scale = torch_group_quant(y, round_scale=round_scale)
        yt_q, yt_scale = torch_group_quant(y.t(), round_scale=round_scale)
        y_maxs = None
    else:
        # smooth
        y_q, y_scale, y_maxs = torch_smooth_quant(y, smooth_scale, reverse=False, round_scale=round_scale)
        yt_q = None 
        yt_scale = None 
    return y_q, y_scale, y_maxs, yt_q, yt_scale

# backward of rms is bf16, do not need quant
def torch_rms_and_quant_backward(x, weight, dy, smooth_scale=None, round_scale=False):
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
    dx = x.grad
    dw = rmsnorm.weight.grad
    if smooth_scale is None:
        dx_q, dx_scale = torch_group_quant(dx, round_scale=round_scale)
        dxt_q, dxt_scale = torch_group_quant(dx.t(), round_scale=round_scale)
        dx_maxs = None
    else:
        # smooth
        dx_q, dx_scale, dx_maxs = torch_smooth_quant(y, smooth_scale, reverse=False, round_scale=round_scale)
        dxt_q = None 
        dxt_scale = None 
    return dx_q, dx_scale, dw, dxt_q, dxt_scale


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


def test_rmsnorm_and_quant(M=4096, N=4096, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    x = torch.randn(M, N, dtype=dtype, requires_grad=True, device=device) ** 3
    weight = torch.randn(N, dtype=dtype, requires_grad=True, device=device)
    smooth_scale = torch.rand(N, dtype=torch.float32, requires_grad=False,
                              device=device) + 0.1

    # smooth
    q_ref, scale_ref, maxs_ref, qt_ref, scale_t_ref = torch_rms_and_quant_forward(x, weight, 
                                          smooth_scale=smooth_scale,
                                          round_scale=True)
    q, scale, maxs, rms, q_t, scale_t  = triton_rms_norm_and_quant_forward(x, weight,
                                                            smooth_scale=smooth_scale,
                                                            calibrate=True,
                                                            output_rms=True,
                                                            round_scale=True)
    output_check(q_ref, q, mode="smooth.data")
    output_check(scale_ref, scale, mode='smooth.scale')
    output_check(maxs_ref, maxs, mode="smooth.maxs")


    # blockwise
    q_ref, scale_ref, _, qt_ref, scale_t_ref = torch_rms_and_quant_forward(x, weight, smooth_scale=None,
                                          round_scale=True)
    q, scale, _, rms, q_t, scale_t  = triton_rms_norm_and_quant_forward(x, weight,
                                                            smooth_scale=None,
                                                            round_scale=True,
                                                            output_rms=True,
                                                            calibrate=True,
                                                            output_mode=2)
    output_check(q_ref, q, mode="block.data")
    output_check(scale_ref.t(), scale, mode='block.scale')
    output_check(qt_ref, q_t, mode='block.t_data')
    output_check(scale_t_ref.t(), scale_t, mode="block.t_scale")

    q, scale, _, _, _, _  = triton_rms_norm_and_quant_forward(x, weight,
                                                            smooth_scale=None,
                                                            round_scale=True,
                                                            output_rms=True,
                                                            calibrate=True,
                                                            output_mode=0)
    output_check(q_ref, q, mode="block.data")
    output_check(scale_ref.t(), scale, mode='block.scale')



    _, _, _, _, q_t, scale_t  = triton_rms_norm_and_quant_forward(x, weight,
                                                            smooth_scale=None,
                                                            round_scale=True,
                                                            rms=rms,
                                                            output_rms=True,
                                                            calibrate=True,
                                                            output_mode=1)
    output_check(qt_ref, q_t, mode='block.t_data')
    output_check(scale_t_ref.t(), scale_t, mode="block.t_scale")


    if bench:
        benchmark_func(triton_rms_norm_and_quant_forward, x, weight,
                                                                smooth_scale=None,
                                                                round_scale=True,
                                                                output_rms=True,
                                                                output_mode=0,
                                                                ref_bytes=M*N*3)

        benchmark_func(triton_rms_norm_and_quant_forward, x, weight,
                                                            smooth_scale=None,
                                                            round_scale=True,
                                                            output_rms=True,
                                                            output_mode=1,
                                                            ref_bytes=M*N*3)

if __name__ == '__main__':
    # test_rmsnorm(M=4096, N=4096)
    # test_rmsnorm(M=4096, N=8192)
    # test_rmsnorm(M=8192, N=2048)
    test_rmsnorm_and_quant(M=8192, N=2048, bench=True)

