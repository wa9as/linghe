import torch
import transformer_engine as te

from flops.facade.rmsnorm import RMSNormFunction
from flops.utils.benchmark import benchmark_func


def bench_rmsnorm(M=4096, N=4096):
    # M, N, K = 8192, 4096, 13312
    # M, N, K = 4096, 4096, 6144
    # M, N, K = 4096, 4096, 4096
    # M, N, K = 4096, 8192, 4096

    dtype = torch.bfloat16
    device = 'cuda:0'
    n_repeat = 100

    x = torch.randn(M, N, dtype=dtype, requires_grad=True, device=device)
    weight = torch.randn(N, dtype=dtype, requires_grad=True, device=device)
    dy = torch.randn(M, N, dtype=dtype, device=device)

    rmsnorm_torch = torch.nn.RMSNorm(
        normalized_shape=N,
        eps=1e-6,
        dtype=torch.bfloat16,
        device='cuda'
    )

    rmsnorm_torch = torch.compile(rmsnorm_torch)

    te_norm = te.pytorch.RMSNorm(hidden_size=N, eps=1e-6)

    def torch_forward_backward(x_torch_back, dy):
        y_torch_back = rmsnorm_torch(x_torch_back)
        y_torch_back.backward(gradient=dy)
        return x_torch_back.grad, rmsnorm_torch.weight.grad

    def te_forward_backward(x_te_back, dy):
        y_te_back = te_norm(x_te_back)
        y_te_back.backward(gradient=dy)
        return x_te_back.grad, te_norm.weight.grad

    def triton_forward_backward(x_triton_back, g_triton_back, dy):
        y_triton_back = RMSNormFunction.apply(x_triton_back, g_triton_back)
        y_triton_back.backward(gradient=dy)
        return x_triton_back.grad, g_triton_back.grad

    ref_time = benchmark_func(rmsnorm_torch, x, n_repeat=n_repeat,
                              name="rms_torch", ref_bytes=M * N * 4)
    benchmark_func(te_norm, x, n_repeat=n_repeat, ref_bytes=M * N * 4,
                   name="rms_te", ref_time=ref_time)
    benchmark_func(RMSNormFunction.apply, x, weight, n_repeat=n_repeat,
                   ref_bytes=M * N * 4, name="rms_triton", ref_time=ref_time)

    ref_time = benchmark_func(torch_forward_backward, x, dy, n_repeat=n_repeat)

    ref_time = benchmark_func(te_forward_backward, x, dy, n_repeat=n_repeat)

    benchmark_func(triton_forward_backward, x, weight, dy, n_repeat=n_repeat,
                   ref_time=ref_time)


bench_rmsnorm(4096, 4096)
