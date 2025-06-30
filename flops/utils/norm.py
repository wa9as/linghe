import math
import os
import torch
import triton
import triton.language as tl
from triton import Config


# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

@triton.jit
def rms_norm_forward_kernel(x_ptr, weight_ptr, out_ptr, norm_ptr, eps, M, T, N: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    weight = tl.load(weight_ptr+tl.arange(0, N))[None,:]

    offs = pid*W*T*N+tl.arange(0, W)[:,None]*N+tl.arange(0, N)[None,:]
    for i in range(T):
        x = tl.load(x_ptr+offs, mask=pid*W*T+i*W+tl.arange(0, W)[:,None]<M).to(tl.float32)
        norm = tl.sqrt(tl.sum(x*x, axis=1)/N+eps)
        tl.store(norm_ptr+pid*W*T+i*W+tl.arange(0, W), norm, mask=pid*W*T+i*W+tl.arange(0, W)<M)

        x = (x/norm[:,None])*weight

        tl.store(out_ptr+offs, x, mask=pid*W*T+i*W+tl.arange(0, W)[:,None]<M)
        offs += N*W


def triton_rms_norm_forward(x, weight, eps=1e-6, out=None, norm=None):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device 
    if out is None:
        out = torch.empty((M, N), device=device, dtype=x.dtype)
    if norm is None:
        norm = torch.empty((M,), device=device, dtype=torch.float32)
    W = 8192//N 
    T = triton.cdiv(M, 132*W)
    grid = lambda META: (132, )
    rms_norm_forward_kernel[grid](
        x,
        weight,
        out,
        norm,
        eps,
        M, T,
        N, 
        W,
        num_stages=3,
        num_warps=16
    )
    return out, norm


@triton.jit
def rms_norm_backward_kernel(
    dx_ptr, dy_ptr, x_ptr, weight_ptr, norm_ptr, M, N: tl.constexpr, BLOCK_N: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return
        
    row_start = row_idx * N
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N
    
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
    norm = tl.load(norm_ptr + row_idx).to(tl.float32)
    
    inv_norm = 1.0 / norm
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    weighted_dy = dy * weight
    
    norm_cubed = norm * norm * norm
    dot = tl.sum(weighted_dy * x)
    mean_dy_x = dot / (N * norm_cubed)
    
    dx = (weighted_dy - x * mean_dy_x) * inv_norm
    tl.store(dx_ptr + row_start + col_offsets, dx, mask=mask)


def triton_rms_norm_backward(dy, x, norm=None, weight=None):
    M, N = x.shape
    assert N <= 8192
    dx = torch.empty_like(x)
    BLOCK_N = min(triton.next_power_of_2(N), 1024)
    
    grid = lambda META: (M,)
    rms_norm_backward_kernel[grid](
        dx,
        dy,
        x,
        weight,
        norm,
        M, N,
        BLOCK_N,
        num_stages=3,
        num_warps=8
    )
    
    return dx, torch.sum(dy * (x / norm.unsqueeze(1)), dim=0)
