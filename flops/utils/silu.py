import math
import os
import torch
import triton
import triton.language as tl
from triton import Config





# @triton.jit
# def weighted_silu_forward_kernel(x_ptr, weight_ptr, out_ptr, M, T, N: tl.constexpr, n:tl.constexpr):
#     pid = tl.program_id(axis=0)

#     offs = pid*T*N+tl.arange(0, n)
#     out_offs = pid*T*n+tl.arange(0, n)
#     for i in range(T):
#         mask = pid*T+i<M
#         x1 = tl.load(x_ptr+offs, mask=mask).to(tl.float32)
#         x2 = tl.load(x_ptr+n+offs, mask=mask).to(tl.float32)
#         w = tl.load(weight_ptr+pid*T+i, mask=mask).to(tl.float32)
#         x = x1/(1+tl.exp(-x1))*x2*w
#         tl.store(out_ptr+out_offs, x, mask=mask)
#         offs += N
#         out_offs += n

# def triton_weighted_silu_forward(x, weight, out=None):
#     # row-wise read, row-wise write
#     M, N = x.shape
#     device = x.device 
#     if out is None:
#         out = torch.empty((M, N//2), device=device, dtype=x.dtype)
#     T = triton.cdiv(M, 132)
#     grid = lambda META: (132, )
#     weighted_silu_forward_kernel[grid](
#         x,
#         weight,
#         out,
#         M, T,
#         N, 
#         N//2,
#         num_stages=5,
#         num_warps=16
#     )
#     return out


@triton.jit
def weighted_silu_forward_kernel(x_ptr, weight_ptr, out_ptr, M, T, N: tl.constexpr, n:tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)

    row_offs = pid*W*T*n+tl.arange(0, W)[:,None]*n
    col_offs = tl.arange(0, n)[None,:]
    
    for i in range(T):
        indices = pid*W*T+i*W+tl.arange(0, W)
        mask = indices[:,None]<M
        x1 = tl.load(x_ptr+row_offs*2+col_offs, mask=mask).to(tl.float32)
        x2 = tl.load(x_ptr+n+row_offs*2+col_offs, mask=mask).to(tl.float32)
        w = tl.load(weight_ptr+indices, mask=indices<M).to(tl.float32)[:,None]
        x = x1/(1+tl.exp(-x1))*x2*w
        tl.store(out_ptr+row_offs+col_offs, x, mask=mask)
        row_offs += n*W

def triton_weighted_silu_forward(x, weight, out=None):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device 
    if out is None:
        out = torch.empty((M, N//2), device=device, dtype=x.dtype)
    W = 8192//N 
    T = triton.cdiv(M, 132*W)
    grid = lambda META: (132, )
    weighted_silu_forward_kernel[grid](
        x,
        weight,
        out,
        M, T,
        N, 
        N//2,
        W,
        num_stages=3,
        num_warps=16
    )
    return out




@triton.jit
def weighted_silu_backward_kernel(x_ptr, weight_ptr, g_ptr, dx_ptr, dw_ptr, M, T, N: tl.constexpr, n:tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)

    offs = pid*W*T*N+tl.arange(0, W)[:,None]*N+tl.arange(0, n)[None,:]
    hoffs = pid*W*T*n+tl.arange(0, W)[:,None]*n+tl.arange(0, n)[None,:]
    for i in range(T):
        mask = pid*W*T+i*W+tl.arange(0, W)
        x1 = tl.load(x_ptr+offs, mask=mask[:,None]<M).to(tl.float32)
        x2 = tl.load(x_ptr+offs+n, mask=mask[:,None]<M).to(tl.float32)
        g = tl.load(g_ptr+hoffs, mask=mask[:,None]<M).to(tl.float32)
        w = tl.load(weight_ptr+mask, mask=mask<M).to(tl.float32)[:,None]
        sigmoid = 1/(1+tl.exp(-x1))
        dw = tl.sum(x1*sigmoid*x2*g,1)
        tl.store(dw_ptr+mask, dw, mask=mask<M)
        dx1 = g*x2*w*sigmoid*(1+x1*tl.exp(-x1)* sigmoid)
        tl.store(dx_ptr+offs, dx1, mask=mask[:,None]<M)

        dx2 = g*x1*sigmoid*w
        tl.store(dx_ptr+offs+n, dx2, mask=mask[:,None]<M)

        offs += N*W
        hoffs += n*W


def triton_weighted_silu_backward(g, x, weight):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device 
    dw = torch.empty((M, 1), device=device, dtype=x.dtype)
    dx = torch.empty((M, N), device=device, dtype=x.dtype)
    W = 8192//N 
    T = triton.cdiv(M, 132*W)
    grid = lambda META: (132, )
    weighted_silu_backward_kernel[grid](
        x,
        weight,
        g,
        dx,
        dw,
        M, T,
        N, 
        N//2,
        W,
        num_stages=3,
        num_warps=16
    )
    return dx,dw
