
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
# atomic_add supports fp16 and fp32, but not bf16 

@triton.jit
def scatter_add_kernel(x_ptr, o_ptr, indices_ptr, M, T, N: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, N)

    for i in range(T):
        src_idx = pid*T+i
        dst_idx = tl.load(indices_ptr+src_idx, mask=src_idx<M)
        x = tl.load(x_ptr+src_idx*N+offs, mask=src_idx<M).to(tl.float32)
        tl.atomic_add(o_ptr+dst_idx*N+offs,x)

@triton.jit
def fp32_to_bf16_kernel(x_ptr, o_ptr, M, T, N: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, N)

    for i in range(T):
        idx = pid*T+i
        x = tl.load(x_ptr+idx*N+offs, mask=idx<M)
        tl.store(o_ptr+idx*N+offs,x, mask=idx<M)


def triton_scatter_add(x, outputs, indices):
    M, N = x.shape

    float_outputs = torch.zeros(outputs.shape, dtype=torch.float32, device=outputs.device)

    T = triton.cdiv(M, 132)

    num_stages = 5
    num_warps = 8

    grid = lambda META: (132, )
    scatter_add_kernel[grid](
        x, float_outputs,
        indices,
        M, T, N, 
        num_stages=num_stages,
        num_warps=num_warps
    )

    m = outputs.shape[0]    
    T = triton.cdiv(m, 132)
    grid = lambda META: (132, )
    fp32_to_bf16_kernel[grid](
        float_outputs, outputs,
        m, T, N, 
        num_stages=num_stages,
        num_warps=num_warps
    )

    return outputs




@triton.jit
def scatter_add_with_count_kernel(x_ptr, o_ptr, indices_ptr, counts_ptr, accum_ptr, M, m, T, N: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, N)
    for i in range(T):
        count = tl.load(counts_ptr + pid*T+i, mask=pid*T+i<m)
        ei = tl.load(accum_ptr + pid*T+i, mask=pid*T+i<m)
        si = ei - count
        sums = tl.zeros((N,),dtype=tl.float32)
        for j in range(si, ei):
            idx = tl.load(indices_ptr+j, mask=pid*T+i<m)
            x = tl.load(x_ptr+idx*N+offs, mask=pid*T+i<m).to(tl.float32)
            sums += x 
        tl.store(o_ptr+pid*T*N+i*N+offs, sums, mask=pid*T+i<m)



def triton_scatter_add_with_count(x, outputs, indices, counts):
    M, N = x.shape
    m = outputs.size(0)

    num_stages = 3
    num_warps = 16

    indices = torch.argsort(indices)
    accum = torch.cumsum(counts, 0)

    T = triton.cdiv(m, 132)
    grid = lambda META: (132, )
    scatter_add_with_count_kernel[grid](
        x, outputs,
        indices,
        counts,
        accum,
        M, m, T, N,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return outputs
