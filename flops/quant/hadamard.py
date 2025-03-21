from enum import IntEnum
from typing import Tuple
import math
import torch
import triton
import triton.language as tl
from triton import Config
from flops.quant.quantize import row_quant_kernel


@triton.jit
def ht_nt_kernel(x_ptr, xb_ptr, w_ptr, wb_ptr, hm_ptr, M, N, K, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    hm = tl.load(hm_ptr + tl.arange(0, BLOCK_SIZE)[:,None]*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None,:])
    offs = pid*BLOCK_SIZE + tl.arange(0, 2*BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_SIZE)[None,:]
    n = tl.cdiv(N, 2*BLOCK_SIZE)
    for i in range(n):
        w = tl.load(w_ptr+offs)
        tl.store(wb_ptr+offs, tl.dot(w, hm))
        offs += 2*BLOCK_SIZE*K

    # norm hm in x
    hm = (hm/BLOCK_SIZE).to(x_ptr.dtype.element_ty)
    offs = pid*BLOCK_SIZE + tl.arange(0, 2*BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_SIZE)[None,:]
    m = tl.cdiv(M, 2*BLOCK_SIZE)
    for i in range(m):
        x = tl.load(x_ptr+offs)
        tl.store(xb_ptr+offs, tl.dot(x, hm) )
        offs += 2*BLOCK_SIZE*K


def triton_ht_nt(x, w, hm):
    assert w.size(1) == w.size(1)
    M, K = x.shape
    N, K = w.shape
    x_b = torch.empty_like(x)
    w_b = torch.empty_like(w)
    BLOCK_SIZE = hm.size(0)
    grid = lambda META: (K//BLOCK_SIZE, )
    ht_nt_kernel[grid](
        x, x_b,
        w, w_b, 
        hm,
        M,N,K,
        BLOCK_SIZE,
        num_stages=6,
        num_warps=4
    )
    return x_b,w_b


def triton_ht_quant_nt(x, w, hm):
    M, K = x.shape
    N, K = w.shape
    device = x.device 
    x_b = torch.empty_like(x)
    w_b = torch.empty_like(w)
    x_q = torch.empty((M, K), device=device, dtype=torch.float8_e4m3fn)
    w_q = torch.empty((N, K), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((M,1), device=device, dtype=torch.float32)
    w_scale = torch.empty((1,N), device=device, dtype=torch.float32)

    BLOCK_SIZE = hm.size(0)
    grid = lambda META: (K//BLOCK_SIZE, )
    ht_nt_kernel[grid](
        x, x_b,
        w, w_b, 
        hm,
        M,N,K,
        BLOCK_SIZE,
        num_stages=6,
        num_warps=4
    )

    BLOCK_SIZE = 4096
    grid = lambda META: (M, )
    row_quant_kernel[grid](
        x_b, x_q, x_scale,
        M,K,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=4
    )

    BLOCK_SIZE = 4096
    grid = lambda META: (N, )
    row_quant_kernel[grid](
        w_b, w_q, w_scale,
        N,K,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=4
    )

    return x_q,w_q,x_scale,w_scale



@triton.jit
def ht_tn_kernel(y_ptr, yb_ptr, x_ptr, xb_ptr, hm_ptr, M, N, K, BLOCK_SIZE: tl.constexpr):
    # both need transpose
    # dwT = yT @ x
    # y: [M,N] -> [N,M]
    # x: [M,K] -> [K,M]
    pid = tl.program_id(axis=0)

    hm = tl.load(hm_ptr + tl.arange(0, BLOCK_SIZE)[:,None]*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None,:])

    offs = pid*BLOCK_SIZE*N + tl.arange(0, BLOCK_SIZE)[:,None]*N + tl.arange(0, 2*BLOCK_SIZE)[None,:] 
    toffs = pid*BLOCK_SIZE + tl.arange(0, 2*BLOCK_SIZE)[:,None]*M + tl.arange(0, BLOCK_SIZE)[None,:]
    n = tl.cdiv(N, 2*BLOCK_SIZE)
    for i in range(n):
        y = tl.trans(tl.load(y_ptr+offs))
        o = tl.dot(y, hm)
        tl.store(yb_ptr+toffs, o)
        offs += 2*BLOCK_SIZE
        toffs += 2*M*BLOCK_SIZE
        
    # # norm hm in x
    hm = (hm/BLOCK_SIZE).to(x_ptr.dtype.element_ty)
    offs = pid*BLOCK_SIZE*K + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, 2*BLOCK_SIZE)[None,:]
    toffs = pid*BLOCK_SIZE + tl.arange(0, 2*BLOCK_SIZE)[:,None]*M + tl.arange(0, BLOCK_SIZE)[None,:]
    k = tl.cdiv(K, 2*BLOCK_SIZE)
    for i in range(k):
        x = tl.trans(tl.load(x_ptr+offs))
        x = tl.dot(x, hm)
        tl.store(xb_ptr+toffs, x)
        offs += 2*BLOCK_SIZE
        toffs += 2*M*BLOCK_SIZE


# v1: ht+token/channelx quant
def triton_ht_quant_tn(y, x, hm):
    # both need transpose
    # dwT = yT @ x
    # y: [M,N] -> [N,M]
    # x: [M,K] -> [K,M]
    assert y.size(0) == x.size(0)
    M, N = y.shape
    M, K = x.shape
    device = x.device

    y_b = torch.empty((N, M),device=device,dtype=y.dtype)
    y_q = torch.empty((N, M), device=device, dtype=torch.float8_e4m3fn)
    y_scale = torch.empty((N,1), device=device, dtype=torch.float32)

    x_b = torch.empty((K, M),device=device,dtype=x.dtype)
    x_q = torch.empty((K, M), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((1,K), device=device, dtype=torch.float32)

    BLOCK_SIZE = hm.size(0)
    grid = lambda META: (M//BLOCK_SIZE, )
    ht_tn_kernel[grid](
        y, y_b, 
        x, x_b,
        hm,
        M,N,K,
        BLOCK_SIZE,
        num_stages=6,
        num_warps=4
    )

    BLOCK_SIZE = 4096
    grid = lambda META: (K, )
    row_quant_kernel[grid](
        x_b, x_q, x_scale,
        K,M,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=4
    )

    BLOCK_SIZE = 4096
    grid = lambda META: (N, )
    row_quant_kernel[grid](
        y_b, y_q, y_scale,
        N,M,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=4
    )

    return y_q,x_q,y_scale,x_scale



@triton.jit
def ht_nn_kernel(y_ptr, yb_ptr, w_ptr, wb_ptr, hm_ptr, M, N, K, BLOCK_SIZE: tl.constexpr):
    # w need transpose
    # dx = y @ w
    # y: [M,N]
    # w: [N,K] -> [K,N]
    pid = tl.program_id(axis=0)

    hm = tl.load(hm_ptr + tl.arange(0, BLOCK_SIZE)[:,None]*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None,:])

    offs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE*2)[:,None]*N + tl.arange(0, BLOCK_SIZE)[None,:]
    m = tl.cdiv(M, 2*BLOCK_SIZE)
    for i in range(m):
        y = tl.load(y_ptr+offs)
        o = tl.dot(y, hm)
        tl.store(yb_ptr+offs, o)
        offs += 2*BLOCK_SIZE*N

    # norm hm in y 
    hm = (hm/BLOCK_SIZE).to(w_ptr.dtype.element_ty)
    offs = pid*BLOCK_SIZE*K + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, 2*BLOCK_SIZE)[None,:] 
    toffs = pid*BLOCK_SIZE + tl.arange(0, 2*BLOCK_SIZE)[:,None]*N + tl.arange(0, BLOCK_SIZE)[None,:]
    k = tl.cdiv(K, 2*BLOCK_SIZE)
    for i in range(k):
        w = tl.trans(tl.load(w_ptr+offs))
        o = tl.dot(w, hm)
        tl.store(wb_ptr+toffs, o)
        offs += 2*BLOCK_SIZE
        toffs += 2*N*BLOCK_SIZE


def triton_ht_quant_nn(y, w, hm):
    # w need transpose
    # dx = y @ w
    # y: [M,N]
    # w: [N,K] -> [K,N]
    M, N = y.shape
    N, K = w.shape
    device = y.device
    y_b = torch.empty((M, N),device=device,dtype=y.dtype)
    y_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    y_scale = torch.empty((M,1), device=device, dtype=torch.float32)

    w_b = torch.empty((K, N),device=device,dtype=y.dtype)
    w_q = torch.empty((K, N), device=device, dtype=torch.float8_e4m3fn)
    w_scale = torch.empty((1,K), device=device, dtype=torch.float32)

    BLOCK_SIZE = hm.size(0)
    grid = lambda META: (N//BLOCK_SIZE, )
    ht_nn_kernel[grid](
        y, y_b,
        w, w_b, 
        hm,
        M,N,K,
        BLOCK_SIZE,
        num_stages=6,
        num_warps=4
    )

    BLOCK_SIZE = 4096
    grid = lambda META: (M, )
    row_quant_kernel[grid](
        y_b, y_q, y_scale,
        M,N,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=4
    )

    BLOCK_SIZE = 4096
    grid = lambda META: (K, )
    row_quant_kernel[grid](
        w_b, w_q, w_scale,
        K,N,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=4
    )

    return y_q,w_q,y_scale,w_scale


def ht_quant_forward(x,w,hm):
    x_q,w_q,x_scale,w_scale = triton_ht_quant_nt(x, w, hm)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_scale,
                                    scale_b=w_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,x_q,w_q,x_scale,w_scale

def ht_quant_update(y,x,hm):
    y_q,x_q,y_scale,x_scale = triton_ht_quant_tn(y, x, hm)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_scale,
                                    scale_b=x_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_q,x_q,y_scale,x_scale

def ht_quant_backward(y,w,hm):

    y_q,w_q,y_scale,w_scale = triton_ht_quant_nn(y, w, hm)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_scale,
                                    scale_b=w_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_q,w_q,y_scale,w_scale


def fp8_ht_f_and_b(x,w,y,hm):
    ht_quant_forward(x, w, hm)
    ht_quant_update(y,x, hm)
    ht_quant_backward(y, w, hm)
