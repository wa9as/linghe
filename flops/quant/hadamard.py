from enum import IntEnum
from typing import Tuple
import math
import torch
import triton
import triton.language as tl
from triton import Config
from flops.quant.quantize import row_quant_kernel


@triton.jit
def hadamard_nt_kernel(x_ptr, xb_ptr, w_ptr, wb_ptr, hm_ptr, M, N, K, BLOCK_SIZE: tl.constexpr, R: tl.constexpr):
    pid = tl.program_id(axis=0)

    hm = tl.load(hm_ptr + tl.arange(0, BLOCK_SIZE)[:,None]*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None,:])
    offs = pid*BLOCK_SIZE + tl.arange(0, R*BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_SIZE)[None,:]
    n = tl.cdiv(N, R*BLOCK_SIZE)
    for i in range(n):
        w = tl.load(w_ptr+offs)
        tl.store(wb_ptr+offs, tl.dot(w, hm))
        offs += R*BLOCK_SIZE*K

    # norm hm in x
    hm = (hm/BLOCK_SIZE).to(x_ptr.dtype.element_ty)
    offs = pid*BLOCK_SIZE + tl.arange(0, R*BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_SIZE)[None,:]
    m = tl.cdiv(M, R*BLOCK_SIZE)
    for i in range(m):
        x = tl.load(x_ptr+offs)
        tl.store(xb_ptr+offs, tl.dot(x, hm) )
        offs += R*BLOCK_SIZE*K


def triton_hadamard_nt(x, w, hm):
    assert w.size(1) == w.size(1)
    M, K = x.shape
    N, K = w.shape
    x_b = torch.empty_like(x)
    w_b = torch.empty_like(w)
    BLOCK_SIZE = hm.size(0)
    R = 2
    grid = lambda META: (K//BLOCK_SIZE, )
    hadamard_nt_kernel[grid](
        x, x_b,
        w, w_b, 
        hm,
        M,N,K,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=8
    )
    return x_b,w_b




def triton_hadamard_quant_nt(x, w, hm):
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
    R = 2
    grid = lambda META: (K//BLOCK_SIZE, )
    hadamard_nt_kernel[grid](
        x, x_b,
        w, w_b, 
        hm,
        M,N,K,
        BLOCK_SIZE,
        R,
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
def hadamard_tn_kernel(y_ptr, yb_ptr, x_ptr, xb_ptr, hm_ptr, M, N, K, BLOCK_SIZE: tl.constexpr, R: tl.constexpr):
    # dwT = yT @ x
    # both need transpose
    # y: [M,N] -> [N,M]
    # x: [M,K] -> [K,M]
    pid = tl.program_id(axis=0)

    hm = tl.load(hm_ptr + tl.arange(0, BLOCK_SIZE)[:,None]*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None,:])

    offs = pid*BLOCK_SIZE*N + tl.arange(0, BLOCK_SIZE)[:,None]*N + tl.arange(0, R*BLOCK_SIZE)[None,:] 
    toffs = pid*BLOCK_SIZE + tl.arange(0, R*BLOCK_SIZE)[:,None]*M + tl.arange(0, BLOCK_SIZE)[None,:]
    n = tl.cdiv(N, R*BLOCK_SIZE)
    for i in range(n):
        y = tl.trans(tl.load(y_ptr+offs))
        o = tl.dot(y, hm)
        tl.store(yb_ptr+toffs, o)
        offs += R*BLOCK_SIZE
        toffs += R*M*BLOCK_SIZE
        
    # # norm hm in x
    hm = (hm/BLOCK_SIZE).to(x_ptr.dtype.element_ty)
    offs = pid*BLOCK_SIZE*K + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, R*BLOCK_SIZE)[None,:]
    toffs = pid*BLOCK_SIZE + tl.arange(0, R*BLOCK_SIZE)[:,None]*M + tl.arange(0, BLOCK_SIZE)[None,:]
    k = tl.cdiv(K, R*BLOCK_SIZE)
    for i in range(k):
        x = tl.trans(tl.load(x_ptr+offs))
        x = tl.dot(x, hm)
        tl.store(xb_ptr+toffs, x)
        offs += R*BLOCK_SIZE
        toffs += R*M*BLOCK_SIZE



# v1: hadamard+token/channelx quant
def triton_hadamard_quant_tn(y, x, hm):
    # dwT = yT @ x
    # both need transpose
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
    R = 2
    grid = lambda META: (M//BLOCK_SIZE, )
    hadamard_tn_kernel[grid](
        y, y_b, 
        x, x_b,
        hm,
        M,N,K,
        BLOCK_SIZE,
        R,
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
def hadamard_nn_kernel(y_ptr, yb_ptr, w_ptr, wb_ptr, hm_ptr, M, N, K, BLOCK_SIZE: tl.constexpr, R: tl.constexpr):
    # w need transpose
    # dx = y @ w
    # y: [M,N]
    # w: [N,K] -> [K,N]
    pid = tl.program_id(axis=0)

    hm = tl.load(hm_ptr + tl.arange(0, BLOCK_SIZE)[:,None]*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None,:])

    offs = pid*BLOCK_SIZE + tl.arange(0, R*BLOCK_SIZE)[:,None]*N + tl.arange(0, BLOCK_SIZE)[None,:]
    m = tl.cdiv(M, R*BLOCK_SIZE)
    for i in range(m):
        y = tl.load(y_ptr+offs)
        o = tl.dot(y, hm)
        tl.store(yb_ptr+offs, o)
        offs += R*BLOCK_SIZE*N

    # norm hm in y 
    hm = (hm/BLOCK_SIZE).to(w_ptr.dtype.element_ty)
    offs = pid*BLOCK_SIZE*K + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, R*BLOCK_SIZE)[None,:] 
    toffs = pid*BLOCK_SIZE + tl.arange(0, R*BLOCK_SIZE)[:,None]*N + tl.arange(0, BLOCK_SIZE)[None,:]
    k = tl.cdiv(K, R*BLOCK_SIZE)
    for i in range(k):
        w = tl.trans(tl.load(w_ptr+offs))
        o = tl.dot(w, hm)
        tl.store(wb_ptr+toffs, o)
        offs += R*BLOCK_SIZE
        toffs += R*BLOCK_SIZE*N


def triton_hadamard_quant_nn(y, w, hm):
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
    R = 2
    grid = lambda META: (N//BLOCK_SIZE, )
    hadamard_nn_kernel[grid](
        y, y_b,
        w, w_b, 
        hm,
        M,N,K,
        BLOCK_SIZE,
        R,
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





@triton.jit
def fused_hadamard_kernel(x_ptr, b_ptr, s_ptr, q_ptr, hm_ptr, M, N, BLOCK_SIZE: tl.constexpr, R: tl.constexpr):
    # apply hadamard transform and row-wise quant
    # SIDE=0: hadamard@block  
    # SIDE=1: block@hadamard
    pid = tl.program_id(axis=0)
    hm = tl.load(hm_ptr + tl.arange(0, BLOCK_SIZE)[:,None]*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None,:])
    # row-wise read, row-wise write
    offs = pid*BLOCK_SIZE*N + tl.arange(0, R*BLOCK_SIZE)[:,None]*N + tl.arange(0, BLOCK_SIZE)[None,:]
    n = tl.cdiv(N, BLOCK_SIZE)
    maxs = tl.zeros((R*BLOCK_SIZE,),dtype=tl.float32)
    for i in range(n):
        x = tl.load(x_ptr+offs)
        # if SIDE == 0:
        #     x = tl.dot(hm, x)
        # else:
        #     x = tl.dot(x, hm)
        o = tl.dot(x, hm)
        tl.store(b_ptr+offs, o)
        maxs = tl.maximum(maxs, tl.max(o,1))
        offs += BLOCK_SIZE

    scales = maxs/448.0

    tl.store(s_ptr + pid*R*BLOCK_SIZE + tl.arange(0, R*BLOCK_SIZE), scales)
    rs = (448.0/maxs)[:,None]

    offs = pid*R*BLOCK_SIZE*N + tl.arange(0, R*BLOCK_SIZE)[:,None]*N + tl.arange(0, 2*BLOCK_SIZE)[None,:]
    n = tl.cdiv(N, 2*BLOCK_SIZE)
    for i in range(n):
        x = tl.load(b_ptr+offs)
        y = (x.to(tl.float32)*rs).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr+offs, y)
        offs += 2*BLOCK_SIZE



def triton_fused_hadamard(x, hm, hm_side=1, op_side=0):
    # for x in y = x @ w, op_side = 0, hm_side=1
    # for wT in dx = y @ wT, hm_side=1: hm_side = 0 in logical format, but hm_side will be 1 with transpose format
    # for yT in dwT = yT @ x, hm_side=1, op_side = 0
    # for x in dwT = yT @ x, hm_side=1, op_side = 1
    M, N = x.shape
    x_b = torch.empty((M,N),dtype=x.dtype,device=x.device)
    if op_side == 0:
        x_s = torch.empty((M,1),dtype=torch.float32,device=x.device)
    else:
        x_s = torch.empty((1,N),dtype=torch.float32,device=x.device)
    x_q = torch.empty((M,N),dtype=torch.float8_e4m3fn,device=x.device)
    BLOCK_SIZE = hm.size(0)
    R = 2
    grid = lambda META: (M//BLOCK_SIZE//R, )
    fused_hadamard_kernel[grid](
        x, 
        x_b,
        x_s,
        x_q, 
        hm,
        M,N,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=4
    )
    return x_q,x_s


@triton.jit
def fused_transpose_hadamard_kernel(x_ptr, b_ptr, s_ptr, q_ptr, hm_ptr, M, N, BLOCK_SIZE: tl.constexpr, SIDE: tl.constexpr):
    # transpose x: [M, N] -> [N, M] 
    # and then apply hadamard transform
    # SIDE=0: hadamard@block  
    # SIDE=1: block@hadamard
    pid = tl.program_id(axis=0)
    hm = tl.load(hm_ptr + tl.arange(0, BLOCK_SIZE)[:,None]*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None,:])
    # row-wise read, col-wise write
    offs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:,None]*N + tl.arange(0, BLOCK_SIZE)[None,:]
    toffs = pid*BLOCK_SIZE*M + tl.arange(0, BLOCK_SIZE)[:,None]*M + tl.arange(0, BLOCK_SIZE)[None,:]
    m = tl.cdiv(M, BLOCK_SIZE)
    maxs = tl.zeros((BLOCK_SIZE,),dtype=tl.float32)
    for i in range(m):
        x = tl.trans(tl.load(x_ptr+offs))
        if SIDE == 0:
            x = tl.dot(hm, x)
        else:
            x = tl.dot(x, hm)
        maxs = tl.maximum(maxs, tl.max(x,1))
        tl.store(b_ptr+toffs, x)
        offs += BLOCK_SIZE*N
        toffs += BLOCK_SIZE

    scales = maxs/448.0
    rs = 448.0/maxs

    tl.store(s_ptr + pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), scales)

    toffs = pid*BLOCK_SIZE*M + tl.arange(0, BLOCK_SIZE)[:,None]*M + tl.arange(0, BLOCK_SIZE)[None,:]
    for i in range(m):
        x = tl.load(b_ptr+toffs).to(tl.float32)
        y = (x*rs).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr+toffs, y)
        toffs += BLOCK_SIZE


def triton_fused_transpose_hadamard(x, hm, hm_side=1, op_side=0):
    # for wT in dx = y @ wT, op_side = 0, hm_side = 1: hm_side = 0 in logical format, but hm_side will be 1 with transpose format
    # for yT in dwT = yT @ x, op_side = 0, hm_side = 1
    # for x in dwT = yT @ x, op_side = 1, hm_side = 1
    M, N = x.shape
    x_b = torch.empty((N,M),dtype=x.dtype,device=x.device)
    if op_side == 0:
        x_s = torch.empty((N,1),dtype=torch.float32,device=x.device)
    else:
        x_s = torch.empty((1,M),dtype=torch.float32,device=x.device)
    x_q = torch.empty((N,M),dtype=torch.float8_e4m3fn,device=x.device)
    BLOCK_SIZE = hm.size(0)
    grid = lambda META: (N//BLOCK_SIZE, )
    fused_transpose_hadamard_kernel[grid](
        x, 
        x_b,
        x_s,
        x_q, 
        hm,
        M,N,
        BLOCK_SIZE,
        hm_side,
        num_stages=6,
        num_warps=4
    )
    return x_q,x_s




def hadamard_quant_forward(x,w,hm):
    x_q,w_q,x_scale,w_scale = triton_hadamard_quant_nt(x, w, hm)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_scale,
                                    scale_b=w_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,x_q,w_q,x_scale,w_scale

def hadamard_quant_update(y,x,hm):
    y_q,x_q,y_scale,x_scale = triton_hadamard_quant_tn(y, x, hm)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_scale,
                                    scale_b=x_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_q,x_q,y_scale,x_scale

def hadamard_quant_backward(y,w,hm):

    y_q,w_q,y_scale,w_scale = triton_hadamard_quant_nn(y, w, hm)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_scale,
                                    scale_b=w_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_q,w_q,y_scale,w_scale


def fp8_hadamard_f_and_b(x,w,y,hm):
    hadamard_quant_forward(x, w, hm)
    hadamard_quant_update(y,x, hm)
    hadamard_quant_backward(y, w, hm)


def triton_fused_hadamard_quant_nt(x, w, hm):
    x_q,x_s = triton_fused_hadamard(x, hm, hm_side=1, op_side=0)
    w_q,w_s = triton_fused_hadamard(w, hm, hm_side=1, op_side=1)
    return x_q,x_s,w_q,w_s


