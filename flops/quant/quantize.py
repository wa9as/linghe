from enum import IntEnum
from typing import Tuple
import math
import torch
import triton
import triton.language as tl
from triton import Config




@triton.jit
def row_quant_kernel(x_ptr, q_ptr, s_ptr,  M, N,  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    n_block = tl.cdiv(N, BLOCK_SIZE)
    indices = tl.arange(0, BLOCK_SIZE)
    max_val = 1e-6
    for j in range(n_block):
        offs = pid*N + j*BLOCK_SIZE + indices
        x = tl.load(x_ptr + offs, mask=j*BLOCK_SIZE + indices<N, other=0)
        max_val = tl.maximum(tl.max(tl.abs(x.to(tl.float32))), max_val)
    scale = max_val/448.0
    tl.store(s_ptr + pid, scale)
    for j in range(n_block):
        offs = pid*N + j*BLOCK_SIZE + indices
        x = tl.load(x_ptr + offs, mask=j*BLOCK_SIZE + indices<N, other=0)
        y = x.to(tl.float32) / scale
        y = y.to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + offs, y, mask=j*BLOCK_SIZE + indices<N)


def triton_row_quant(x):
    M, N = x.shape 
    BLOCK_SIZE = 4096
    x_q = torch.empty((M,N),dtype=torch.float8_e4m3fn,device=x.device)
    x_scale = torch.empty((M,1),dtype=torch.float32,device=x.device)
    grid = lambda META: (M, )
    row_quant_kernel[grid](
        x, x_q, x_scale,
        M,N,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=4
    )
    return x_q, x_scale