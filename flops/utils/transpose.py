import math
import torch
import triton
import triton.language as tl
from triton import Config



@triton.jit
def transpose_kernel(x_ptr, t_ptr, M, N, H: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, col-wise write
    offs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:] 
    toffs = pid*W*M + tl.arange(0, W)[:,None]*M + tl.arange(0, H)[None,:]
    m = tl.cdiv(M, H)
    for i in range(m):
        y = tl.trans(tl.load(x_ptr+offs))
        tl.store(t_ptr+toffs, y)
        offs += H*N
        toffs += H


def triton_transpose(x):
    M, N = x.shape
    device = x.device

    t = torch.empty((N, M),device=device,dtype=x.dtype)
    H = 1024
    W = 16
    grid = lambda META: (N//W, )
    transpose_kernel[grid](
        x, 
        t,
        M,N,
        H,
        W,
        num_stages=6,
        num_warps=4
    )
