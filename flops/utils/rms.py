import os
import torch

import triton
import triton.language as tl
import numpy as np
import random
import torch.nn as nn
from flops.utils.benchmark import benchmark_func

def output_check(org_out, opt_out, mode=''):
    abs_error = (opt_out.float() - org_out.float()).abs().mean().item()
    rel_error = abs_error/org_out.float().abs().mean().item()
    print(f'\nmode:{mode} abs_error:{abs_error:.3f} rel_error:{rel_error:.3f} ' \
            f'org:{org_out.abs().max():.3f}/{org_out.abs().mean():.3f} ' \
            f'opt:{opt_out.abs().max():.3f}/{opt_out.abs().mean():.3f} ')

# def setup_seed(seed):
#      torch.manual_seed(seed)
#      torch.cuda.manual_seed_all(seed)
#      np.random.seed(seed)
#      random.seed(seed)
#      torch.backends.cudnn.deterministic = True

# setup_seed(20)


class RMSNormtriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, eps=1e-5):
        M, N = x.shape
        y = torch.empty_like(x, dtype=torch.float32, device=x.device)
        rsigma = torch.empty(M, dtype=torch.float32, device=x.device)

        grid = (M, )
        rms_norm_forward_kernel[grid](
            x, y, g, rsigma, x.stride(0), N, eps, BLOCK_SIZE=256
        )

        ctx.save_for_backward(x, g)
        ctx.eps = eps
        ctx.N = N

        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, g = ctx.saved_tensors
        eps = ctx.eps

        n_rows = x.shape[0]
        N = x.shape[-1]
        x_shape = x.shape

        grad_x = torch.empty_like(x)
        grad_g = torch.empty_like(x)

        rms_norm_backward_kernel[(n_rows,)](
            x,
            g,
            grad_output,
            x.stride(0),
            grad_x,
            grad_g,
            N,
            eps,
            num_warps=16,
            block_size=triton.next_power_of_2(N),
        )
        return grad_x.view(*x_shape), grad_g.sum(dim=0)


@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr,
    y_ptr,
    g_ptr,
    rsigma,
    stride_x,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    idx = row_idx * stride_x
    y_ptr += idx
    x_ptr += idx

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols_offsets = off + tl.arange(0, BLOCK_SIZE)
        tmp = tl.load(x_ptr + cols_offsets, mask=cols_offsets < N, other=0.0).to(tl.float32)
        acc += tmp * tmp
    rms = tl.sqrt(tl.sum(acc) / N + eps)

    # tl.store(rsigma + row_idx, rms)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        g = tl.load(g_ptr + cols, mask=mask)
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x / rms
        y = x_hat * g
        tl.store(y_ptr + cols, y, mask=mask)


@triton.jit
def rmsnorm_bwd_kernel(
    x_ptr,
    g_ptr,
    grad_output_ptr,
    x_row_stride,
    grad_x_ptr,
    grad_gamma_ptr,
    N,
    eps,
    block_size: tl.constexpr,
):
    row_idx = tl.program_id(0)

    offsets = tl.arange(0, block_size)

    x_offsets = row_idx * x_row_stride + offsets
    x_ptrs = x_ptr + x_offsets
    grad_output_ptrs = grad_output_ptr + x_offsets
    gamma_ptrs = g_ptr + offsets

    valid_elements_mask = offsets < N

    x = tl.load(x_ptrs, mask=valid_elements_mask, other=0).to(tl.float32)
    g = tl.load(gamma_ptrs, mask=valid_elements_mask, other=0)
    grad_outputs = tl.load(grad_output_ptrs, mask=valid_elements_mask, other=0)

    rms = tl.sqrt(tl.sum(x * x) / N + eps)

    gamma_grad = x * grad_outputs / rms
    tl.store(
        grad_gamma_ptr + x_offsets, gamma_grad, mask=valid_elements_mask,
    )

    grad_x_term_1 = grad_outputs * g / rms
    grad_input_term_2 = (
        tl.sum(x * grad_outputs * g) * x
        / (N * rms * rms * rms)
    )
    grad_input_values = grad_x_term_1 - grad_input_term_2
    tl.store(
        grad_x_ptr + x_offsets, grad_input_values, mask=valid_elements_mask
    )


def triton_rms_norm_forward_new(x, g, eps=1e-6):
    M, N = x.shape
    y = torch.empty_like(x, dtype=torch.float32, device=x.device)
    rsigma = torch.empty(M, dtype=torch.float32, device=x.device)

    grid = (M, )
    rmsnorm_fwd_kernel[grid](
        x, y, g, rsigma, x.stride(0), N, eps, BLOCK_SIZE=256
    )
    return y

def triton_rms_norm_backward_new(grad_output, x, g, norm=None, weight=None):
    M, N = x.shape
    grad_x = torch.empty(M, N, dtype=torch.float32, device=x.device)
    grad_g = torch.empty(M, N, dtype=torch.float32, device=x.device)

    n_rows = M
    x_shape = x.shape
    eps = 1e-6

    rmsnorm_bwd_kernel[(n_rows,)](
            x,
            g,
            grad_output,
            x.stride(0),
            grad_x,
            grad_g,
            N,
            eps,
            num_warps=16,
            block_size=triton.next_power_of_2(N),
        )

    return grad_x.view(*x_shape), grad_g.sum(dim=0)
    

M = 4096
N = 4096
x = torch.randn(M, N, dtype=torch.bfloat16, requires_grad=True).cuda()
g = torch.randn(N, dtype=torch.bfloat16, requires_grad=True).cuda()
dy = torch.randn(M, N, dtype=torch.bfloat16).cuda()

rmsnorm_torch = nn.RMSNorm(
    normalized_shape=N,
    eps=1e-5,
    dtype=torch.bfloat16,
    device='cuda'
)

with torch.no_grad():
    rmsnorm_torch.weight.copy_(g)

rmsnorm_torch = torch.compile(rmsnorm_torch)

# ### forward ####

# # y_triton = RMSNormtriton.apply(x, g)
# # x_torch = x.detach().clone().to(torch.float32).requires_grad_()
# # y_torch = rmsnorm_torch(x_torch)

# # output_check(y_torch, y_triton)

# ### backward ####
# x_torch_back = x.detach().clone().to(torch.float32).requires_grad_()
# rmsnorm_torch = torch.compile(rmsnorm_torch)
# rmsnorm_torch.zero_grad()
# y_torch_back = rmsnorm_torch(x_torch_back)
# y_torch_back.backward(gradient=dy)
# grad_x_torch = x_torch_back.grad
# grad_g_torch = rmsnorm_torch.weight.grad
# print(grad_x_torch.size())
# print(grad_g_torch.size())

# x_triton_back = x.detach().clone().requires_grad_()
# g_triton_back = g.detach().clone().requires_grad_()
# y_triton_back = RMSNormtriton.apply(x_triton_back, g_triton_back)
# y_triton_back.backward(gradient=dy)
# grad_x_triton = x_triton_back.grad
# grad_g_triton = g_triton_back.grad
# print(grad_x_triton.size())
# print(grad_g_triton.size())

# output_check(grad_x_torch, grad_x_triton)
# output_check(grad_g_torch, grad_g_triton)

def troch_forward_backward(x_torch_back, dy):
    y_torch_back = rmsnorm_torch(x_torch_back)
    y_torch_back.backward(gradient=dy)
    
def triton_forward_backward(x_triton_back, g_triton_back, dy):
    y_triton_back = RMSNormtriton.apply(x_triton_back, g_triton_back)
    y_triton_back.backward(gradient=dy)

x_torch_back = x.detach().clone().to(torch.float32).requires_grad_()

x_triton_back = x.detach().clone().requires_grad_()
g_triton_back = g.detach().clone().requires_grad_()

benchmark_func(troch_forward_backward, x_torch_back, dy, n_repeat=100)
benchmark_func(triton_forward_backward, x_triton_back, g_triton_back, dy, n_repeat=100)