
import math
import os
import torch
import triton
import triton.language as tl
from triton import Config


# for megatron 0.11 scatter_add

@triton.jit
def aligned_scatter_add_kernel(x_ptr, o_ptr, indices_ptr, weights_ptr, M, N: tl.constexpr, K: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, N)

    sums = tl.zeros((N,),dtype=tl.float32)
    for i in range(K):
        idx = tl.load(indices_ptr+pid*K+i)
        x = tl.load(x_ptr+idx*N+offs)
        if SCALE == 1:
            weight = tl.load(weights_ptr+idx)
            sums += x*weight
        else:
            sums += x

    tl.store(o_ptr+pid*N+offs,sums)


def triton_aligned_scatter_add(x, outputs, indices, weights=None):
    M, N = x.shape
    m = outputs.size(0)

    indices = torch.argsort(indices)
    K = M//m 
    assert K*m == M
    SCALE = 1 if weights is not None else 0

    num_stages = 5
    num_warps = 8

    grid = lambda META: (m, )
    aligned_scatter_add_kernel[grid](
        x, outputs,
        indices,
        weights,
        M, N, K,
        SCALE ,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return outputs




# for deepep scatter_add

@triton.jit
def scatter_add_kernel(x_ptr, o_ptr, indices_ptr, M, T, N: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, N)

    for i in range(T):
        src_idx = pid*T+i
        dst_idx = tl.load(indices_ptr+src_idx, mask=src_idx<M)
        x = tl.load(x_ptr+src_idx*N+offs, mask=src_idx<M)
        tl.atomic_add(o_ptr+pid*N+offs,x)


def triton_scatter_add(x, outputs, indices):
    M, N = x.shape
    m = outputs.size(0)

    T = triton.cdiv(M, 132)

    num_stages = 5
    num_warps = 8

    grid = lambda META: (132, )
    scatter_add_kernel[grid](
        x, outputs,
        indices,
        M, T, N, 
        num_stages=num_stages,
        num_warps=num_warps
    )
    return outputs
