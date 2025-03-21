
from enum import IntEnum
from typing import Tuple
import math
import torch
import triton
import triton.language as tl
from triton import Config






@triton.jit
def smooth_quant_v0_nt_kernel(x_ptr, xq_ptr, w_ptr, wq_ptr, s_ptr, M, N, K, BLOCK_SIZE: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid*BLOCK_K + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_K)[None,:]

    m = tl.cdiv(M, BLOCK_SIZE)
    x_max = tl.zeros((BLOCK_K,),dtype=tl.float32)
    x_ptrs = x_ptr + offs
    for i in range(m):
        x = tl.load(x_ptrs)
        x_max = tl.maximum(tl.max(tl.abs(x), axis=0),x_max)
        # if pid==0:
        #     if i==0:
        #         tl.device_print('x_max',tl.max(tl.abs(x), axis=0))
        x_ptrs += BLOCK_SIZE*K


    n = tl.cdiv(N, BLOCK_SIZE)
    w_max = tl.zeros((BLOCK_K,),dtype=tl.float32)
    w_ptrs = w_ptr + offs
    for i in range(n):
        w = tl.load(w_ptrs)
        w_max = tl.maximum(tl.max(tl.abs(w), axis=0),w_max)
        w_ptrs += BLOCK_SIZE*K

    scale = tl.sqrt(x_max/w_max)

    tl.store(s_ptr+pid*BLOCK_K+tl.arange(0,BLOCK_K), scale)

    x_max = x_max/scale
    w_max = w_max*scale
    x_scale = x_max/448.0
    w_scale = w_max/448.0
    xs = (1.0/scale/x_scale).to(x_ptr.dtype.element_ty)[None]
    ws = (scale/w_scale).to(x_ptr.dtype.element_ty)[None]

    x_ptrs = x_ptr + offs
    xq_ptrs = xq_ptr + offs
    for i in range(m):
        x = tl.load(x_ptrs)
        x = (x*xs).to(xq_ptr.dtype.element_ty)
        tl.store(xq_ptrs, x)
        x_ptrs += BLOCK_SIZE*K
        xq_ptrs += BLOCK_SIZE*K

    w_ptrs = w_ptr + offs
    wq_ptrs = wq_ptr + offs
    for i in range(n):
        w = tl.load(w_ptrs)
        w = (w*ws).to(wq_ptr.dtype.element_ty)
        tl.store(wq_ptrs, w)
        w_ptrs += BLOCK_SIZE*K
        wq_ptrs += BLOCK_SIZE*K



def triton_smooth_quant_v0_nt(a, b):
    M, K = a.shape
    N, K = b.shape
    a_q = torch.empty((M, K), device=a.device, dtype=torch.float8_e4m3fn)
    b_q = torch.empty((N, K), device=a.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((K,), device=a.device, dtype=torch.float32)

    BLOCK_SIZE = 512
    BLOCK_K = 32
    grid = lambda META: (K//BLOCK_K, )
    smooth_quant_v0_nt_kernel[grid](
        a, a_q,
        b, b_q, scale,
        M,N,K,
        BLOCK_SIZE,
        BLOCK_K,
        num_stages=5,
        num_warps=16
    )
    return a_q,b_q,scale



# grid K//BLOCK_K
@triton.jit
def sm_nt_kernel(x_ptr, xs_ptr, w_ptr, ws_ptr, M, N, K, BLOCK_SIZE: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)

    offs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_SIZE)[None,:]
    m = tl.cdiv(M, BLOCK_SIZE)
    x_max = tl.zeros((BLOCK_K,),dtype=tl.float32)
    x_ptrs = x_ptr + offs
    for i in range(m):
        x = tl.load(x_ptr)
        x_max = tl.maximum(tl.max(tl.abs(x), axis=0),x_max)
        x_ptrs += BLOCK_SIZE*K

    # offs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_SIZE)[None,:]
    n = tl.cdiv(N, BLOCK_SIZE)
    w_max = tl.zeros((BLOCK_K,),dtype=tl.float32)
    w_ptrs = w_ptr + offs
    for i in range(n):
        w = tl.load(w_ptr)
        w_max = tl.maximum(tl.max(tl.abs(x), axis=0),x_max)
        w_ptr += BLOCK_SIZE*K

    scale = tl.sqrt(x_max*w_max)
    x_scale = x_max/scale
    w_scale = w_max/scale

    x_ptrs = x_ptr + offs
    xs_ptrs = xs_ptr + offs
    for i in range(m):
        x = tl.load(x_ptrs)
        x = x / x_scale
        tl.store(xs_ptrs, x)
        x_ptrs += BLOCK_SIZE*K
        xs_ptrs += BLOCK_SIZE*K

    w_ptrs = w_ptr + offs
    ws_ptrs = ws_ptr + offs
    for i in range(n):
        w = tl.load(w_ptrs)
        w = w / w_scale
        tl.store(ws_ptrs, w)
        w_ptrs += BLOCK_SIZE*K
        ws_ptrs += BLOCK_SIZE*K

# grid K//BLOCK_K
@triton.jit
def smooth_v3_kernel_nt(x_ptr, xs_ptr, xs_max_ptr, w_ptr, ws_ptr, ws_max_ptr, M, N, K, eps, BLOCK_SIZE: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)

    offs = pid*BLOCK_K + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_K)[None,:]
    m = tl.cdiv(M, BLOCK_SIZE)
    x_max = tl.zeros((BLOCK_K,),dtype=tl.float32)
    x_ptrs = x_ptr + offs
    for i in range(m):
        x = tl.load(x_ptrs)
        x_max = tl.maximum(tl.max(tl.abs(x), axis=0),x_max)
        x_ptrs += BLOCK_SIZE*K

    # offs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_SIZE)[None,:]
    n = tl.cdiv(N, BLOCK_SIZE)
    w_max = tl.zeros((BLOCK_K,),dtype=tl.float32)
    w_ptrs = w_ptr + offs
    for i in range(n):
        w = tl.load(w_ptrs)
        w_max = tl.maximum(tl.max(tl.abs(w), axis=0),x_max)
        w_ptrs += BLOCK_SIZE*K

    scale = tl.sqrt(x_max*w_max)
    x_scale = x_max/scale
    w_scale = w_max/scale

    x_ptrs = x_ptr + offs
    xs_ptrs = xs_ptr + offs
    xs_offs = tl.arange(0, BLOCK_SIZE)
    xs_max_ptrs =  xs_max_ptr + pid*M + xs_offs
    for i in range(m):
        x = tl.load(x_ptrs)
        x = x / x_scale
        xs_max = tl.maximum(tl.max(tl.abs(x), axis=1), eps)
        tl.store(xs_ptrs, x)
        tl.store(xs_max_ptrs, xs_max)
        x_ptrs += BLOCK_SIZE*K
        xs_ptrs += BLOCK_SIZE*K
        xs_max_ptrs += BLOCK_SIZE

    w_ptrs = w_ptr + offs
    ws_ptrs = ws_ptr + offs
    ws_offs = tl.arange(0, BLOCK_SIZE)
    ws_max_ptrs = ws_max_ptr + pid*N + ws_offs
    for i in range(n):
        w = tl.load(w_ptrs)
        ws = w / w_scale
        ws_max = tl.maximum(tl.max(tl.abs(ws), axis=1),eps)
        tl.store(ws_ptrs, ws)
        tl.store(ws_max_ptrs, ws_max)
        w_ptrs += BLOCK_SIZE*K
        ws_ptrs += BLOCK_SIZE*K
        ws_max_ptrs += BLOCK_SIZE

@triton.jit
def row_quant_sm_kernel(x_ptr, q_ptr, s_ptr,  M, N,  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    n_block = (N-1) // BLOCK_SIZE + 1 
    indices = tl.arange(0, BLOCK_SIZE)
    
    scale = tl.load(s_ptr+pid)
    for j in range(n_block):
        offs = pid*N + j*BLOCK_SIZE + indices
        x = tl.load(x_ptr + offs, mask=j*BLOCK_SIZE + indices<N, other=0)
        y = x.to(tl.float32) / scale
        y = y.to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + offs, y, mask=j*BLOCK_SIZE + indices<N)



"""
used for g@w^T, w is [out, in] and row_major, should be quantized and transposed to [in, out]
"""
@triton.jit
def smooth_quant_v0_nn_kernel(x_ptr, xq_ptr, w_ptr, wq_ptr, s_ptr, M, N, K, BLOCK_SIZE: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid*BLOCK_K + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_K)[None,:]

    m = tl.cdiv(M, BLOCK_SIZE)
    x_max = tl.zeros((BLOCK_K,),dtype=tl.float32)
    x_ptrs = x_ptr + offs
    for i in range(m):
        x = tl.load(x_ptrs)
        x_max = tl.maximum(tl.max(tl.abs(x), axis=0),x_max)
        # if pid==0:
        #     if i==0:
        #         tl.device_print('x_max',tl.max(tl.abs(x), axis=0))
        x_ptrs += BLOCK_SIZE*K


    n = tl.cdiv(N, BLOCK_SIZE)
    offs_ = pid*BLOCK_K + tl.arange(0, BLOCK_SIZE)[:,None]*N+tl.arange(0, BLOCK_K)[None,:]
    w_max = tl.zeros((BLOCK_K,),dtype=tl.float32)
    w_ptrs = w_ptr + offs_
    for i in range(n):
        w = tl.load(w_ptrs)
        w_max = tl.maximum(tl.max(tl.abs(w), axis=0),w_max)
        w_ptrs += BLOCK_SIZE

    scale = tl.sqrt(x_max/w_max)

    tl.store(s_ptr+pid*BLOCK_K+tl.arange(0,BLOCK_K), scale)

    x_max = x_max/scale
    w_max = w_max*scale
    x_scale = x_max/448.0
    w_scale = w_max/448.0
    xs = (1.0/scale/x_scale).to(x_ptr.dtype.element_ty)[None]
    ws = (scale/w_scale).to(x_ptr.dtype.element_ty)[None]

    x_ptrs = x_ptr + offs
    xq_ptrs = xq_ptr + offs
    for i in range(m):
        x = tl.load(x_ptrs)
        x = (x*xs).to(xq_ptr.dtype.element_ty)
        tl.store(xq_ptrs, x)
        x_ptrs += BLOCK_SIZE*K
        xq_ptrs += BLOCK_SIZE*K

    w_ptrs = w_ptr + offs_
    wq_ptrs = wq_ptr + pid*BLOCK_K + tl.arange(0, BLOCK_SIZE)[:,None]*K+tl.arange(0, BLOCK_K)[None,None]
    for i in range(n):
        w = tl.load(w_ptrs)
        w = (w*ws[:,None]).to(wq_ptr.dtype.element_ty)
        tl.store(wq_ptrs, w)
        w_ptrs += BLOCK_SIZE
        wq_ptrs += BLOCK_SIZE*K



def triton_smooth_quant_v0_nn(a, b):
    M, K = a.shape
    K, N = b.shape
    a_q = torch.empty((M, K), device=a.device, dtype=torch.float8_e4m3fn)
    b_q = torch.empty((N, K), device=a.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((K,), device=a.device, dtype=torch.float32)

    BLOCK_SIZE = 512
    BLOCK_K = 32
    grid = lambda META: (K//BLOCK_K, )
    smooth_quant_v0_nn_kernel[grid](
        a, a_q,
        b, b_q, scale,
        M,N,K,
        BLOCK_SIZE,
        BLOCK_K,
        num_stages=5,
        num_warps=16
    )
    return a_q,b_q,scale

