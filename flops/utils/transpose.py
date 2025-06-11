import math
import os
import torch
import triton
import triton.language as tl
from triton import Config
from flops.utils.util import round_up


# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

@triton.jit
def transpose_kernel(x_ptr, t_ptr, M, N, H: tl.constexpr, W: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(axis=0)
    # col-wise read, row-wise write
    offs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:] 
    toffs = pid*W*M + tl.arange(0, W)[:,None]*M + tl.arange(0, H)[None,:]
    m = tl.cdiv(M, H)
    for i in range(m):
        if EVEN:
            y = tl.trans(tl.load(x_ptr+offs))
            tl.store(t_ptr+toffs, y)
        else:
            y = tl.trans(tl.load(x_ptr+offs, mask=(pid*W+tl.arange(0, W)[None,:] < N) & (i*H+tl.arange(0, H)[:,None] < M) ))
            tl.store(t_ptr+toffs, y, mask=(pid*W+tl.arange(0, W)[:,None] < N) & (i*H+tl.arange(0, H)[None,:] < M))
        offs += H*N
        toffs += H




def triton_transpose(x):
    M, N = x.shape
    device = x.device
    t = torch.empty((N, M),device=device,dtype=x.dtype) 

    H = max([x for x in [1,64,128,256,512] if M%x == 0])
    if H > 1:
        EVEN = True 
        if x.dtype.itemsize == 1:
            W = 32
            num_stages = 5
            num_warps = 8
        else:
            W = 16
            num_stages = 5
            num_warps = 8 
    else:
        EVEN = False 
        if x.dtype.itemsize == 1:
            H = 64
            W = 32
            num_stages = 5
            num_warps = 4
        else:
            H = 128
            W = 16
            num_stages = 5
            num_warps = 8 


    grid = lambda META: ((N-1)//W+1, )
    transpose_kernel[grid](
        x, t,
        M, N,
        H, W,
        EVEN,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return t





@triton.jit
def block_transpose_kernel(x_ptr, t_ptr, M, N, H: tl.constexpr, W: tl.constexpr, EVEN: tl.constexpr):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)
    offs = rid*H*N + cid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:] 
    toffs = rid*H + cid*M*W + tl.arange(0, W)[:,None]*M + tl.arange(0, H)[None,:]
    if EVEN:
        y = tl.trans(tl.load(x_ptr+offs))
        tl.store(t_ptr+toffs, y)
    else:
        y = tl.trans(tl.load(x_ptr+offs, mask=(cid*W+tl.arange(0, W)[None,:] < N) & (rid*H+tl.arange(0, H)[:,None] < M) ))
        tl.store(t_ptr+toffs, y, mask=(cid*W+tl.arange(0, W)[:,None] < N) & (rid*H+tl.arange(0, H)[None,:] < M))




def triton_block_transpose(x):
    M, N = x.shape
    device = x.device
    t = torch.empty((N, M),device=device,dtype=x.dtype) 
    H = max([x for x in [1,64,128,256,512] if M%x == 0])
    if H > 1:
        EVEN = True 
        if x.dtype.itemsize == 1:
            W = 32
            num_stages = 5
            num_warps = 8
        else:
            W = 16
            num_stages = 5
            num_warps = 8 
    else:
        EVEN = False 
        if x.dtype.itemsize == 1:
            H = 64
            W = 32
            num_stages = 5
            num_warps = 4
        else:
            H = 128
            W = 16
            num_stages = 5
            num_warps = 8 


    grid = lambda META: ((M-1)//H+1, (N-1)//W+1)
    block_transpose_kernel[grid](
        x, t,
        M, N,
        H, W,
        EVEN,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return t


@triton.jit
def block_pad_transpose_kernel(x_ptr, t_ptr, M, N, P, H: tl.constexpr, W: tl.constexpr, EVEN: tl.constexpr):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)
    offs = rid*H*N + cid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:] 
    toffs = rid*H + cid*P*W + tl.arange(0, W)[:,None]*P + tl.arange(0, H)[None,:]
    if EVEN:
        y = tl.trans(tl.load(x_ptr+offs))
        tl.store(t_ptr+toffs, y)
    else:
        y = tl.trans(tl.load(x_ptr+offs, mask=(cid*W+tl.arange(0, W)[None,:] < N) & (rid*H+tl.arange(0, H)[:,None] < M) ))
        tl.store(t_ptr+toffs, y, mask=(cid*W+tl.arange(0, W)[:,None] < N) & (rid*H+tl.arange(0, H)[None,:] < M))


"""
pad x for scaled_mm
M of x should be mutiplier of 16
"""
def triton_block_pad_transpose(x, pad=True):
    M, N = x.shape
    P = round_up(M) if pad else M 
    device = x.device
    t = torch.empty((N, P),device=device,dtype=x.dtype) 
    
    H = max([x for x in [1,64,128,256,512] if M%x == 0])
    if H > 1:
        EVEN = True
        W = 32 if x.dtype.itemsize == 1 else 16
    else:
        EVEN = False
        H = 64 if x.dtype.itemsize == 1 else 128
        W = 32 if x.dtype.itemsize == 1 else 16
    
    if not EVEN or (pad and P > M):
        with torch.no_grad():
            t.fill_(0)
    
    grid = lambda META: ((M-1)//H+1, (N-1)//W+1)
    block_pad_transpose_kernel[grid](
        x, t,
        M, N, P,
        H, W,
        EVEN,
        num_stages=3,
        num_warps=8
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
    D = 0 if x.dtype.itemsize == 1 else 1
    t = torch.empty((N, M),device=device,dtype=x.dtype)
    grid = lambda META: (N//META["W"], )
    opt_transpose_kernel[grid](
        x, t,
        M, N, D
    )
    return t


