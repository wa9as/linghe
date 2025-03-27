import math
import os
import torch
import triton
import triton.language as tl
from triton import Config


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

@triton.jit
def transpose_kernel(x_ptr, t_ptr, M, N, H: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)
    # col-wise read, row-wise write
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
    if x.dtype == torch.float8_e4m3fn:
        H = 512 if M%512==0 else 256
        W = 32
        num_stages = 6
        num_warps = 2
    else:
        H = 256
        W = 16
        num_stages = 5
        num_warps = 8
    grid = lambda META: (N//W, )
    transpose_kernel[grid](
        x, t,
        M, N,
        H, W,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return t



configs = [
    Config({"H": H, "W": W}, num_stages=num_stages, num_warps=num_warps)
    for H in [32, 64, 128, 256, 512]
    for W in [16, 32, 64]
    for num_stages in [3, 4, 5, 6]
    for num_warps in [2, 4, 8]
]

@triton.autotune(configs=configs, key=["M", "N", "D"])
@triton.jit
def opt_transpose_kernel(x_ptr, t_ptr, M, N, D, H: tl.constexpr, W: tl.constexpr):
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

def triton_opt_transpose(x):
    M, N = x.shape
    device = x.device
    D = 0 if x.dtype == torch.float8_e4m3fn else 1
    t = torch.empty((N, M),device=device,dtype=x.dtype)
    grid = lambda META: (N//META["W"], )
    opt_transpose_kernel[grid](
        x, t,
        M, N, D
    )
    return t


