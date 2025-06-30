import math

import torch
import triton
import triton.language as tl
from triton import Config




@triton.jit
def index_select_kernel(x_ptr, out_ptr, scale_ptr, scale_out_ptr, index_ptr, M, T, N: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    for i in range(T):
        dst_idx = pid*T+i
        src_idx = tl.load(index_ptr+dst_idx, mask=dst_idx < M)
        x = tl.load(x_ptr+ src_idx*N+tl.arange(0, N), mask=dst_idx < M)
        tl.store(out_ptr+dst_idx*N+tl.arange(0, N), x, mask=dst_idx < M)

        if SCALE:
            scale = tl.load(scale_ptr+ src_idx, mask=dst_idx < M)
            tl.store(scale_out_ptr+dst_idx, scale,  mask=dst_idx < M)

"""
index select for quantized tensor
x: [bs, dim]
x_scale: [bs]
indices: [K]
"""
def triton_index_select(x, indices, scale=None, out=None, scale_out=None):
    # row-wise read, row-wise write
    M, N = x.shape
    E = indices.shape[0]
    device = x.device 
    if out is None:
        out = torch.empty((E, N), device=device, dtype=x.dtype)
    if scale is not None and scale_out is None:
        scale_out = torch.empty((E,), device=device, dtype=scale.dtype)
    T = triton.cdiv(E, 132)
    SCALE = scale is not None
    grid = lambda META: (132, )
    index_select_kernel[grid](
        x,
        out,
        scale,
        scale_out,
        indices,
        E, T, N, 
        SCALE,
        num_stages=3,
        num_warps=8
    )
    return out,scale_out
