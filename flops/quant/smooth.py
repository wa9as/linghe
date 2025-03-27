
from enum import IntEnum
from typing import Tuple
import math
import torch
import triton
import triton.language as tl
from triton import Config
from flops.quant.channel import row_quant_kernel


@triton.jit
def smooth_direct_quant_nt_kernel(x_ptr, xq_ptr, w_ptr, wq_ptr, s_ptr, M, N, K, H: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid*W + tl.arange(0, H)[:,None]*K + tl.arange(0, W)[None,:]

    m = tl.cdiv(M, H)
    x_max = tl.zeros((W,),dtype=tl.float32)
    x_ptrs = x_ptr + offs
    for i in range(m):
        x = tl.load(x_ptrs)
        x_max = tl.maximum(tl.max(tl.abs(x), axis=0),x_max)
        x_ptrs += H*K

    n = tl.cdiv(N, H)
    w_max = tl.zeros((W,),dtype=tl.float32)
    w_ptrs = w_ptr + offs
    for i in range(n):
        w = tl.load(w_ptrs)
        w_max = tl.maximum(tl.max(tl.abs(w), axis=0),w_max)
        w_ptrs += H*K

    scale = tl.sqrt(x_max/w_max)

    tl.store(s_ptr+pid*W+tl.arange(0,W), scale)

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
        x_ptrs += H*K
        xq_ptrs += H*K

    w_ptrs = w_ptr + offs
    wq_ptrs = wq_ptr + offs
    for i in range(n):
        w = tl.load(w_ptrs)
        w = (w*ws).to(wq_ptr.dtype.element_ty)
        tl.store(wq_ptrs, w)
        w_ptrs += H*K
        wq_ptrs += H*K



def triton_smooth_direct_quant_nt(x, w):
    M, K = x.shape
    N, K = w.shape
    device = x.device
    x_q = torch.empty((M, K), device=device, dtype=torch.float8_e4m3fn)
    w_q = torch.empty((N, K), device=device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((K,), device=device, dtype=torch.float32)

    BLOCK_SIZE = 512
    BLOCK_K = 32
    grid = lambda META: (K//BLOCK_K, )
    smooth_direct_quant_nt_kernel[grid](
        x, x_q,
        w, w_q, scale,
        M,N,K,
        BLOCK_SIZE,
        BLOCK_K,
        num_stages=5,
        num_warps=16
    )
    return x_q,w_q,scale



@triton.jit
def smooth_nt_kernel(x_ptr, xs_ptr, w_ptr, ws_ptr, smooth_scale_ptr, M, N, K, H: tl.constexpr, W: tl.constexpr, P: tl.constexpr):
    pid = tl.program_id(axis=0)
    # col-wise read, col-wise write
    offs = pid*W + tl.arange(0, H)[:,None]*K + tl.arange(0, W)[None,:]
    x_max = tl.zeros((W,),dtype=tl.float32) + 1e-9
    m = tl.cdiv(M, H)
    for i in range(m):
        x = tl.load(x_ptr+offs)
        x_max = tl.maximum(tl.max(tl.abs(x), axis=0),x_max)
        offs += H*K

    n = tl.cdiv(N, H)
    offs = pid*W + tl.arange(0, H)[:,None]*K + tl.arange(0, W)[None,:]
    w_max = tl.zeros((W,),dtype=tl.float32) + 1e-9
    for i in range(n):
        w = tl.load(w_ptr+offs)
        w_max = tl.maximum(tl.max(tl.abs(w), axis=0),x_max)
        offs += H*K

    scale = tl.sqrt(x_max*w_max)
    w_scale = w_max/scale  # reciprocal of x_scale

    if P == 1:
        tl.store(smooth_scale_ptr+pid*W + tl.arange(0, W), w_scale)

    w_scale_ = w_scale.to(x_ptr.dtype.element_ty)  # reciprocal of x_scale
    offs = pid*W + tl.arange(0, H)[:,None]*K + tl.arange(0, W)[None,:]
    for i in range(m):
        x = tl.load(x_ptr+offs)
        x = x*w_scale_   # x / x_scale
        tl.store(xs_ptr+offs, x)
        offs += H*K

    x_scale_ = (1/w_scale).to(x_ptr.dtype.element_ty)  # reciprocal of w_scale
    offs = pid*W + tl.arange(0, H)[:,None]*K + tl.arange(0, W)[None,:]
    for i in range(n):
        w = tl.load(w_ptr+offs)
        ws = w*x_scale_  # w / w_scale
        tl.store(ws_ptr+offs, ws)
        offs += H*K



def triton_smooth_quant_nt(x, w, persist=0):
    M, K = x.shape
    N, K = w.shape
    device = x.device 
    x_b = torch.empty((M, K), device=device, dtype=x.dtype)
    w_b = torch.empty((N, K), device=device, dtype=x.dtype)
    x_q = torch.empty((M, K), device=device, dtype=torch.float8_e4m3fn)
    w_q = torch.empty((N, K), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((M,1), device=device, dtype=torch.float32)
    w_scale = torch.empty((1,N), device=device, dtype=torch.float32)
    
    smooth_scale = torch.empty((K,), device=device, dtype=torch.float32) if persist == 1 else None

    H = 1024
    W = 32
    P = persist
    grid = lambda META: (K//W, )
    smooth_nt_kernel[grid](
        x, x_b,
        w, w_b,
        smooth_scale,
        M,N,K,
        H, W, P,
        num_stages=5,
        num_warps=4
    )

    BLOCK_SIZE = 4096
    grid = lambda META: (M, )
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
        w_b, w_q, w_scale,
        N,M,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=4
    )


    return x_q,w_q,x_scale,w_scale



# dx = y @wT
@triton.jit
def smooth_nn_kernel(y_ptr, ys_ptr, w_ptr, ws_ptr, M, N, K, H: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)
    # y: col-wise read, col-wise write
    # w: row-wise read, col-wise write
    offs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:]
    y_max = tl.zeros((W,),dtype=tl.float32) + 1e-9
    m = tl.cdiv(M, H)
    for i in range(m):
        y = tl.load(y_ptr+offs)
        y_max = tl.maximum(tl.max(tl.abs(y), axis=0),y_max)
        offs += H*N

    k = tl.cdiv(K, H)
    w_max = tl.zeros((W,),dtype=tl.float32) + 1e-9
    offs = pid*W*K + tl.arange(0, W)[:,None]*K + tl.arange(0, H)[None,:]
    for i in range(k):
        w = tl.load(w_ptr+offs)
        w_max = tl.maximum(tl.max(tl.abs(w), axis=1),w_max)
        offs += H

    scale = tl.sqrt(y_max*w_max)
    w_scale = (w_max/scale).to(y_ptr.dtype.element_ty)  # reciprocal of x_scale
    offs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:]
    for i in range(m):
        y = tl.load(y_ptr+offs)
        y = y*w_scale   # y / y_scale
        tl.store(ys_ptr+offs, y)
        offs += H*N

    y_scale = (y_max/scale).to(y_ptr.dtype.element_ty)  # reciprocal of w_scale
    offs = pid*W*K + tl.arange(0, W)[:,None]*K + tl.arange(0, H)[None,:]
    toffs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:]
    for i in range(k):
        w = tl.trans(tl.load(w_ptr+offs))
        ws = w*y_scale  # w / w_scale
        tl.store(ws_ptr+toffs, ws)
        offs += H
        toffs += H*N


# dx = y @ wT
# w should be transposed
def triton_smooth_quant_nn(y, w):
    M, N = y.shape
    N, K = w.shape
    device = y.device 
    y_b = torch.empty((M, N), device=device, dtype=y.dtype)
    y_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    w_b = torch.empty((K, N), device=device, dtype=y.dtype)
    w_q = torch.empty((K, N), device=device, dtype=torch.float8_e4m3fn)
    y_scale = torch.empty((M,1), device=device, dtype=torch.float32)
    w_scale = torch.empty((1,K), device=device, dtype=torch.float32)

    H = 1024
    W = 32
    grid = lambda META: (N//W, )
    smooth_nn_kernel[grid](
        y, y_b,
        w, w_b,
        M,N,K,
        H,
        W,
        num_stages=5,
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



# dwT = yT @ x
@triton.jit
def smooth_tn_kernel(y_ptr, ys_ptr, x_ptr, xs_ptr, M, N, K, H: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)
    # y: row-wise read, col-wise write
    # x: row-wise read, col-wise write
    n = tl.cdiv(N, H)
    y_max = tl.zeros((W,),dtype=tl.float32) + 1e-9
    offs = pid*W*N + tl.arange(0, W)[:,None]*N + tl.arange(0, H)[None,:]
    for i in range(n):
        y = tl.load(y_ptr+offs)
        y_max = tl.maximum(tl.max(tl.abs(y), axis=1),y_max)
        offs += H

    k = tl.cdiv(K, H)
    x_max = tl.zeros((W,),dtype=tl.float32) + 1e-9
    offs = pid*W*K + tl.arange(0, W)[:,None]*K + tl.arange(0, H)[None,:]
    for i in range(k):
        x = tl.load(x_ptr+offs)
        x_max = tl.maximum(tl.max(tl.abs(x), axis=1),x_max)
        offs += H

    scale = tl.sqrt(y_max*x_max)

    y_scale = (x_max/scale).to(y_ptr.dtype.element_ty)  # reciprocal of x_scale
    offs = pid*W*N + tl.arange(0, W)[:,None]*N + tl.arange(0, H)[None,:]
    toffs = pid*W + tl.arange(0, H)[:,None]*M + tl.arange(0, W)[None,:]
    for i in range(n):
        y = tl.trans(tl.load(y_ptr+offs))
        ys = y*y_scale  
        tl.store(ys_ptr+toffs, ys)
        offs += H
        toffs += H*M

    x_scale = (y_max/scale).to(y_ptr.dtype.element_ty)  # reciprocal of w_scale
    offs = pid*W*K + tl.arange(0, W)[:,None]*K + tl.arange(0, H)[None,:]
    toffs = pid*W + tl.arange(0, H)[:,None]*M + tl.arange(0, W)[None,:]
    for i in range(k):
        x = tl.trans(tl.load(x_ptr+offs))
        xs = x*x_scale  
        tl.store(xs_ptr+toffs, xs)
        offs += H
        toffs += H*M




# dwT = yT @ x
# y & w should be transposed
def triton_smooth_quant_tn(y, x):
    M, N = y.shape
    M, K = x.shape
    device = y.device 
    y_b = torch.empty((N, M), device=device, dtype=y.dtype)
    y_q = torch.empty((N, M), device=device, dtype=torch.float8_e4m3fn)
    x_b = torch.empty((K, M), device=device, dtype=y.dtype)
    x_q = torch.empty((K, M), device=device, dtype=torch.float8_e4m3fn)
    y_scale = torch.empty((N,1), device=device, dtype=torch.float32)
    x_scale = torch.empty((1,K), device=device, dtype=torch.float32)

    H = 1024
    W = 16
    grid = lambda META: (M//W, )
    smooth_tn_kernel[grid](
        y, y_b,
        x, x_b,
        M,N,K,
        H,
        W,
        num_stages=6,
        num_warps=2
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

    BLOCK_SIZE = 4096
    grid = lambda META: (K, )
    row_quant_kernel[grid](
        x_b, x_q, x_scale,
        K,M,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=4
    )


    return y_q,x_q,y_scale,x_scale





def smooth_quant_forward(x,w):

    x_q,w_q,x_s,w_s = triton_smooth_quant_nt(x, w)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_s,
                                    scale_b=w_s,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output

def smooth_quant_backward(y,w):

    y_q,w_q,y_s,w_s = triton_smooth_quant_nn(y, w)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_s,
                                    scale_b=w_s,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output



def smooth_quant_update(y,x):
    y_q,x_q,y_s,x_s = triton_smooth_quant_tn(y, x)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_s,
                                    scale_b=x_s,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output



@triton.jit
def slide_smooth_quant_kernel(x_ptr, q_ptr, ss_ptr, qs_ptr, M, N, H: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    offs = pid*W*N + tl.arange(0, W)[:,None]*N + tl.arange(0, H)[None,:]
    soffs = tl.arange(0, H)
    x_max = tl.zeros((W,),dtype=tl.float32) + 1e-9
    m = tl.cdiv(N, H)
    for i in range(m):
        x = tl.load(x_ptr+offs)
        scale = tl.load(ss_ptr+soffs)
        x = x * scale
        x_max = tl.maximum(tl.max(tl.abs(x), axis=1),x_max)
        offs += H 
        soffs += H

    scale = x_max/448.0
    tl.store(qs_ptr+pid*W+tl.arange(0, W), scale)

    s = (1.0/scale)[:,None]
    offs = pid*W*N + tl.arange(0, W)[:,None]*N + tl.arange(0, H)[None,:]
    for i in range(m):
        x = tl.load(x_ptr+offs)
        x = (x*s).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr+offs, x)
        offs += H 


# smooth_scale: w_max/tl.sqrt(x_max*w_max)
def triton_slide_smooth_quant(x, smooth_scale):
    M, N = x.shape
    device = x.device 
    x_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((M,1), device=device, dtype=torch.float32)
    H = 1024
    W = 16
    grid = lambda META: (M//W, )
    slide_smooth_quant_kernel[grid](
        x,
        x_q,
        smooth_scale,
        x_scale,
        M, N,
        H, W,
        num_stages=5,
        num_warps=4
    )

    return x_q,x_scale



@triton.jit
def slide_transpose_smooth_quant_kernel(x_ptr, q_ptr, ss_ptr, qs_ptr, M, N, H: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)
    # col-wise read, row-wise write
    offs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:]
    soffs = tl.arange(0, H)
    x_max = tl.zeros((W,),dtype=tl.float32) + 1e-9
    m = tl.cdiv(M, H)
    for i in range(m):
        x = tl.load(x_ptr+offs)
        scale = tl.load(ss_ptr+soffs)[:,None]
        x = x * scale
        x_max = tl.maximum(tl.max(tl.abs(x), axis=0),x_max)
        offs += H*N 
        soffs += H

    scale = (x_max/448.0)
    tl.store(qs_ptr+pid*W+tl.arange(0, W), scale)

    s = (1.0/scale)[:,None]
    offs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:]
    toffs = pid*W*M + tl.arange(0, W)[:,None]*M + tl.arange(0, H)[None,:]
    for i in range(m):
        x = tl.trans(tl.load(x_ptr+offs))
        x = (x*s).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr+toffs, x)
        offs += H*N
        toffs += H


# smooth_scale: w_max/tl.sqrt(x_max*w_max)
def triton_slide_tranpose_smooth_quant(x, smooth_scale):
    M, N = x.shape
    device = x.device 
    x_q = torch.empty((N, M), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((1,N), device=device, dtype=torch.float32)
    H = 512
    W = 16
    grid = lambda META: (N//W, )
    slide_transpose_smooth_quant_kernel[grid](
        x,
        x_q,
        smooth_scale,
        x_scale,
        M, N,
        H, W,
        num_stages=5,
        num_warps=4
    )

    return x_q,x_scale



def triton_slide_smooth_quant_nt(x, w, smooth_scale):
    x_q,x_scale = triton_slide_smooth_quant(x, smooth_scale)
    w_q,w_scale = triton_slide_smooth_quant(w, 1/smooth_scale)
    return x_q,x_scale,w_q,w_scale


def triton_slide_smooth_quant_nn(y, w, smooth_scale):
    y_q,y_scale = triton_slide_smooth_quant(y, smooth_scale)
    w_q,w_scale = triton_slide_tranpose_smooth_quant(w, 1/smooth_scale)
    return y_q,y_scale,w_q,w_scale 

def triton_slide_smooth_quant_tn(y, x, smooth_scale):
    y_q,y_scale = triton_slide_tranpose_smooth_quant(y, smooth_scale)
    x_q,x_scale = triton_slide_tranpose_smooth_quant(x, 1/smooth_scale)
    return y_q,y_scale,x_q,x_scale


def slide_smooth_quant_forward(x,w, smooth_scale):

    x_q,x_s,w_q,w_s = triton_slide_smooth_quant_nt(x, w, smooth_scale)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_s,
                                    scale_b=w_s.view(1,-1),
                                    out_dtype=x.dtype,
                                    use_fast_accum=True)
    return output

def slide_smooth_quant_backward(y,w, smooth_scale):

    y_q,y_s,w_q,w_s = triton_slide_smooth_quant_nn(y, w, smooth_scale)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_s,
                                    scale_b=w_s,
                                    out_dtype=y.dtype,
                                    use_fast_accum=True)
    return output


def slide_smooth_quant_update(y,x, smooth_scale):
    y_q,y_s,x_q,x_s = triton_slide_smooth_quant_tn(y, x, smooth_scale)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_s.view(-1,1),
                                    scale_b=x_s,
                                    out_dtype=y.dtype,
                                    use_fast_accum=True)
    return output


# # grid K//BLOCK_K
# @triton.jit
# def fused_smooth_kernel_nt(x_ptr, xs_ptr, xs_max_ptr, w_ptr, ws_ptr, ws_max_ptr, M, N, K, eps, BLOCK_SIZE: tl.constexpr, BLOCK_K: tl.constexpr):
#     pid = tl.program_id(axis=0)

#     offs = pid*BLOCK_K + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_K)[None,:]
#     m = tl.cdiv(M, BLOCK_SIZE)
#     x_max = tl.zeros((BLOCK_K,),dtype=tl.float32)
#     x_ptrs = x_ptr + offs
#     for i in range(m):
#         x = tl.load(x_ptrs)
#         x_max = tl.maximum(tl.max(tl.abs(x), axis=0),x_max)
#         x_ptrs += BLOCK_SIZE*K

#     # offs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_K)[None,:]
#     n = tl.cdiv(N, BLOCK_SIZE)
#     w_max = tl.zeros((BLOCK_K,),dtype=tl.float32)
#     w_ptrs = w_ptr + offs
#     for i in range(n):
#         w = tl.load(w_ptrs)
#         w_max = tl.maximum(tl.max(tl.abs(w), axis=0),x_max)
#         w_ptrs += BLOCK_SIZE*K

#     scale = tl.sqrt(x_max*w_max)
#     x_scale = x_max/scale
#     w_scale = w_max/scale

#     x_ptrs = x_ptr + offs
#     xs_ptrs = xs_ptr + offs
#     xs_offs = tl.arange(0, BLOCK_SIZE)
#     xs_max_ptrs =  xs_max_ptr + pid*M + xs_offs
#     # xs_max_ptrs =  xs_max_ptr + pid + tl.arange(0, BLOCK_SIZE)*m #2D 
#     for i in range(m):
#         x = tl.load(x_ptrs)
#         x = x / x_scale
#         xs_max = tl.maximum(tl.max(tl.abs(x), axis=1), eps)
#         tl.store(xs_ptrs, x)
#         tl.store(xs_max_ptrs, xs_max)
#         x_ptrs += BLOCK_SIZE*K
#         xs_ptrs += BLOCK_SIZE*K
#         xs_max_ptrs += BLOCK_SIZE

#     w_ptrs = w_ptr + offs
#     ws_ptrs = ws_ptr + offs
#     ws_offs = tl.arange(0, BLOCK_SIZE)
#     ws_max_ptrs = ws_max_ptr + pid*N + ws_offs
#     for i in range(n):
#         w = tl.load(w_ptrs)
#         ws = w / w_scale
#         ws_max = tl.maximum(tl.max(tl.abs(ws), axis=1),eps)
#         tl.store(ws_ptrs, ws)
#         tl.store(ws_max_ptrs, ws_max)
#         w_ptrs += BLOCK_SIZE*K
#         ws_ptrs += BLOCK_SIZE*K
#         ws_max_ptrs += BLOCK_SIZE






# @triton.jit
# def smooth_kernel_tn(y_ptr, yq_ptr,x_ptr, xq_ptr, x_quant_scale_ptr, M, N, K, eps, BLOCK_SIZE: tl.constexpr, BLOCK_K: tl.constexpr):
#     pid = tl.program_id(axis=0)
    
#     offs = pid*BLOCK_SIZE*N + tl.arange(0, BLOCK_SIZE)[:,None]*N + tl.arange(0, BLOCK_K)[None,:] 
#     toffs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_K)[:,None]*M + tl.arange(0, BLOCK_SIZE)[None,:]
#     scale_ptrs = x_quant_scale_ptr + pid * BLOCK_SIZE +tl.arange(0, BLOCK_SIZE)[None,:]
#     scale = tl.load(scale_ptrs)
#     n = tl.cdiv(N, BLOCK_K)
#     for i in range(n):
#         y = tl.trans(tl.load(y_ptr+offs))
#         o = (y*scale).to(yq_ptr.dtype.element_ty)
#         tl.store(yq_ptr+toffs, o)
#         offs += BLOCK_K
#         toffs += BLOCK_K*M
#         # scale_ptrs += BLOCK_K

#     offs = pid*BLOCK_SIZE*K + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_K)[None,:]
#     toffs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_K)[:,None]*M + tl.arange(0, BLOCK_SIZE)[None,:]
#     k = tl.cdiv(K, BLOCK_K)
#     for j in range(k):
#         x = tl.trans(tl.load(x_ptr+offs))
#         tl.store(xq_ptr+toffs, x)
#         offs += BLOCK_K
#         toffs += BLOCK_K*M

# @triton.jit
# def smooth_v3_kernel_nn(y_ptr, yq_ptr,w_ptr, wq_ptr, w_quant_scale_ptr, M, N, K, eps, BLOCK_SIZE: tl.constexpr, BLOCK_N: tl.constexpr):
#     pid = tl.program_id(axis=0)
    
#     # offs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:,None]*N + tl.arange(0, BLOCK_N)[None,:] 
#     offs = pid*BLOCK_SIZE*N + tl.arange(0, BLOCK_SIZE)[:,None]*N + tl.arange(0, BLOCK_N)[None,:] 
#     scale_ptrs = w_quant_scale_ptr + tl.arange(0, BLOCK_N)[None,:]
#     scale = tl.load(scale_ptrs)
#     # n = tl.cdiv(M, BLOCK_SIZE)
#     n = tl.cdiv(N, BLOCK_N)
#     for i in range(n):
#         y = tl.load(y_ptr+offs)
#         o = (y*scale).to(yq_ptr.dtype.element_ty)
#         tl.store(yq_ptr+offs, o)
#         # offs += BLOCK_SIZE * N
#         offs += BLOCK_N
#         scale_ptrs += BLOCK_N

# @triton.jit
# def smooth_kernel_wq(w_ptr, wq_ptr, M, N, K, eps, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
#     pid = tl.program_id(axis=0)
    
#     offs = pid*BLOCK_N*K + tl.arange(0, BLOCK_N)[:,None]*K + tl.arange(0, BLOCK_K)[None,:]
#     toffs = pid*BLOCK_N + tl.arange(0, BLOCK_K)[:,None]*N + tl.arange(0, BLOCK_N)[None,:]
#     k = tl.cdiv(K, BLOCK_K)
#     for j in range(k):
#         x = tl.trans(tl.load(w_ptr+offs))
#         tl.store(wq_ptr+toffs, x)
#         offs += BLOCK_K
#         toffs += BLOCK_K*N


# #grid M/BLOCK_M
# @triton.jit
# def row_quant_sm_kernel(x_ptr, q_ptr, s_ptr,  M, K,  BLOCK_SIZE: tl.constexpr, BLOCK_K: tl.constexpr, SCALE_K: tl.constexpr):
#     pid = tl.program_id(0)

#     offs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_K)[None,:]
#     n_block = SCALE_K
    
#     # offs_scale = pid * (M // BLOCK_SIZE) + tl.arange(0, 8)[None,:] #需要加边界 K // BLOCK_K
#     offs_scale = pid * BLOCK_SIZE * SCALE_K +  tl.arange(0, BLOCK_SIZE)[:,None]* SCALE_K + tl.arange(0, SCALE_K)[None,:]
#     scale_off = tl.load(s_ptr+offs_scale)
#     scale = tl.max(scale_off, axis=1).expand_dims(-1)
#     # scale = scale.expand_dims(-1)
#     # if pid == 0:
#     #     tl.device_print(scale)

#     x_ptrs = x_ptr + offs
#     q_ptrs = q_ptr + offs
#     for j in range(n_block):
#         x = tl.load(x_ptrs)
#         # if pid == 0:
#         #     tl.device_print(x)
#         y = x.to(tl.float32) / scale 
#         y = y.to(q_ptr.dtype.element_ty)
#         tl.store(q_ptrs, y)
#         x_ptrs += BLOCK_K
#         q_ptrs += BLOCK_K
        


# # v3: smooth + token/channel
# def triton_fused_sm_quant_nt(x, w):
#     eps = 1e-10
#     M, K = x.shape
#     N, K = w.shape
#     device = x.device 
#     x_s = torch.empty((M, K), device=x.device, dtype=x.dtype)
#     w_s = torch.empty((N, K), device=w.device, dtype=w.dtype)
#     x_q = torch.empty((M, K), device=x.device, dtype=torch.float8_e4m3fn)
#     w_q = torch.empty((N, K), device=w.device, dtype=torch.float8_e4m3fn)
#     x_scale = torch.empty((M,1), device=x.device, dtype=torch.float32)
#     w_scale = torch.empty((1,N), device=w.device, dtype=torch.float32)
#     # xs_max_tmp = torch.empty(M, device=x.device, dtype=torch.float32)
#     # ws_max_tmp = torch.empty(N, device=w.device, dtype=torch.float32)
    
#     BLOCK_SIZE = 512 
#     BLOCK_K = 64

#     # xs_max_tmp = torch.empty((M, K//BLOCK_K), device=x.device, dtype=torch.float32)
#     # ws_max_tmp = torch.empty((N, K//BLOCK_K), device=w.device, dtype=torch.float32)
    
#     xs_max_tmp = torch.empty(M*(K//BLOCK_K), device=x.device, dtype=torch.float32)
#     ws_max_tmp = torch.empty(N*(K//BLOCK_K), device=w.device, dtype=torch.float32)
#     # print(f"grid: {K//BLOCK_SIZE}")

#     grid = lambda META: (K//BLOCK_K, )
#     fused_smooth_kernel_nt[grid](
#         x, x_s, xs_max_tmp,
#         w, w_s, ws_max_tmp,
#         M,N,K,
#         eps, 
#         BLOCK_SIZE,
#         BLOCK_K,
#         num_stages=6,
#         num_warps=16
#     )

#     xs_max_tmp = xs_max_tmp.view(M,-1)
#     ws_max_tmp = ws_max_tmp.view(N,-1)
    
#     # print(x_s)
#     # print(torch.sum(x_s==0))
#     # print(torch.nonzero(x_s)[:514])
#     #70 us for bellow
#     # xs_max = xs_max_tmp.view(M, -1).max(1)[0]/448.0
#     # ws_max = ws_max_tmp.view(N, -1).max(1)[0]/448.0

#     BLOCK_SIZE = 32
#     BLOCK_K = 512
#     SCALE_K = K // BLOCK_K
#     grid = lambda META: (M//BLOCK_SIZE, )
#     row_quant_sm_kernel[grid](
#         x_s, x_q, xs_max_tmp,
#         M,K,
#         BLOCK_SIZE,
#         BLOCK_K,
#         SCALE_K,
#         num_stages=6,
#         num_warps=32
#     )

#     # BLOCK_SIZE = 4096
#     grid = lambda META: (N//BLOCK_SIZE, )
#     row_quant_sm_kernel[grid](
#         w_s, w_q, ws_max_tmp,
#         N,K,
#         BLOCK_SIZE,
#         BLOCK_K,
#         SCALE_K,
#         num_stages=6,
#         num_warps=32
#     )

#     return x_q,w_q

# def triton_sm_quant_tn(y, x):
#     eps = 1e-10
#     M, N = y.shape
#     M, K = x.shape
#     device = x.device 
#     # x_s = torch.empty((M, K), device=x.device, dtype=x.dtype)
#     # w_s = torch.empty((N, K), device=w.device, dtype=w.dtype)
#     y_q = torch.empty((N, M), device=device, dtype=torch.float8_e4m3fn)
#     x_q = torch.empty((K, M), device=x.device, dtype=torch.float8_e4m3fn)
#     x_quant_scale = torch.empty((M,1), device=x.device, dtype=torch.float32)#pass from outside 
    
#     BLOCK_SIZE = 64
#     BLOCK_K = 64 #128

#     grid = lambda META: (M//BLOCK_SIZE, )
#     smooth_kernel_tn[grid](
#         y, y_q,
#         x, x_q, x_quant_scale,
#         M,N,K,
#         eps, 
#         BLOCK_SIZE,
#         BLOCK_K,
#         num_stages=6,
#         num_warps=32
#     )
    
# def triton_sm_quant_nn(y, w):
#     eps = 1e-10
#     M, N = y.shape
#     N, K = w.shape
#     device = w.device 
#     # x_s = torch.empty((M, K), device=x.device, dtype=x.dtype)
#     # w_s = torch.empty((N, K), device=w.device, dtype=w.dtype)
#     y_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
#     w_q = torch.empty((K, N), device=device, dtype=torch.float8_e4m3fn)
#     w_quant_scale = torch.empty((1,N), device=device, dtype=torch.float32)#pass from outside 
    
#     BLOCK_SIZE = 64
#     BLOCK_N = 512

#     # grid = lambda META: (N//BLOCK_N, )
#     grid = lambda META: (M//BLOCK_SIZE, )
#     smooth_v3_kernel_nn[grid](
#         y, y_q,
#         w, w_q, w_quant_scale,
#         M,N,K,
#         eps, 
#         BLOCK_SIZE,
#         BLOCK_N,
#         num_stages=6,
#         num_warps=32
#     )

#     BLOCK_K = 64
#     BLOCK_N = 64
#     grid = lambda META: (N//BLOCK_N, )
#     smooth_kernel_wq[grid](
#         w, w_q,
#         M,N,K,
#         eps, 
#         BLOCK_K,
#         BLOCK_N,
#         num_stages=6,
#         num_warps=32
#     )


