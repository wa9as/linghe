
import math
import os
import torch
import triton
import triton.language as tl
from triton import Config



@triton.jit
def scatter_kernel(x_ptr, o_ptr, indices_ptr, weights_ptr, M, N: tl.constexpr, K: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, N)

    sums = tl.zeros((N,),dtype=tl.float32)
    for i in range(K):
        idx = tl.load(indices_ptr+pid*K+i)
        x = tl.load(x_ptr+idx*N+offs)
        if SCALE:
            weight = tl.load(weights_ptr+idx)
            sums += x*weight
        else:
            sums += x

    tl.store(o_ptr+pid*N+offs,sums)


def triton_scatter(x, outputs, indices, weights=None):
    M, N = x.shape
    m = outputs.size(0)

    indices = torch.argsort(indices)
    K = M//m 
    assert K*m == M
    SCALE = 1 if weights is not None else 0

    num_stages = 5
    num_warps = 8

    grid = lambda META: (m, )
    scatter_kernel[grid](
        x, outputs,
        indices,
        weights,
        M, N, K,
        SCALE ,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return outputs
