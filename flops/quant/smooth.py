
import math
import torch
import triton
import triton.language as tl
from triton import Config
from flops.quant.channel import row_quant_kernel
from flops.utils.transpose import triton_transpose


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
def smooth_nt_kernel(x_ptr, xs_ptr, w_ptr, ws_ptr, smooth_scale_ptr, M, N, K, H: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)
    # col-wise read, col-wise write
    offs = pid*W + tl.arange(0, H)[:,None]*K + tl.arange(0, W)[None,:]
    x_max = tl.zeros((W,),dtype=tl.float32) + 1.17e-38
    m = tl.cdiv(M, H)
    for i in range(m):
        x = tl.load(x_ptr+offs)
        x_max = tl.maximum(tl.max(tl.abs(x), axis=0),x_max)
        offs += H*K

    n = tl.cdiv(N, H)
    offs = (pid*W + tl.arange(0, H)[:,None]*K + tl.arange(0, W)[None,:]).to(tl.int64)
    w_max = tl.zeros((W,),dtype=tl.float32) + 1.17e-38
    for i in range(n):
        w = tl.load(w_ptr+offs)
        w_max = tl.maximum(tl.max(tl.abs(w), axis=0),w_max)
        offs += H*K

    maxs = tl.sqrt(x_max*w_max)
    x_scale = x_max/maxs  # reciprocal of w_scale
    w_scale = w_max/maxs  # reciprocal of x_scale

    tl.store(smooth_scale_ptr + pid*W + tl.arange(0, W), x_scale)

    w_scale = w_scale.to(x_ptr.dtype.element_ty)  # reciprocal of x_scale
    offs = pid*W + tl.arange(0, H)[:,None]*K + tl.arange(0, W)[None,:]
    for i in range(m):
        x = tl.load(x_ptr+offs)
        x = x*w_scale   # x / x_scale
        tl.store(xs_ptr+offs, x)
        offs += H*K

    x_scale = x_scale.to(x_ptr.dtype.element_ty)  # reciprocal of w_scale
    offs = (pid*W + tl.arange(0, H)[:,None]*K + tl.arange(0, W)[None,:]).to(tl.int64)
    for i in range(n):
        w = tl.load(w_ptr+offs)
        w = w*x_scale  # w / w_scale
        tl.store(ws_ptr+offs, w)
        offs += H*K



def triton_smooth_quant_nt(x, w, smooth_scale=None):
    M, K = x.shape
    N, K = w.shape
    device = x.device 
    x_b = torch.empty((M, K), device=device, dtype=x.dtype)
    x_q = torch.empty((M, K), device=device, dtype=torch.float8_e4m3fn)
    w_b = torch.empty((N, K), device=device, dtype=x.dtype)
    w_q = torch.empty((N, K), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((M,1), device=device, dtype=torch.float32)
    w_scale = torch.empty((1,N), device=device, dtype=torch.float32)
    
    # smooth_scale may be initialized from SmoothQuantLinear, as forward can not output two params with a single grad
    if smooth_scale is None:
        smooth_scale = torch.empty((K,), device=device, dtype=torch.float32)

    H = max([x for x in [128,256,512,1024] if M%x == 0 and N%x == 0])
    W = 32
    grid = lambda META: (K//W, )
    smooth_nt_kernel[grid](
        x, x_b,
        w, w_b,
        smooth_scale,
        M,N,K,
        H, W, 
        num_stages=5,
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

    return x_q,w_q,x_scale,w_scale,smooth_scale



# dx = y @wT
@triton.jit
def smooth_nn_kernel(y_ptr, ys_ptr, w_ptr, ws_ptr, M, N, K, H: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)
    # y: col-wise read, col-wise write
    # w: row-wise read, col-wise write
    offs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:]
    y_max = tl.zeros((W,),dtype=tl.float32) + 1.17e-38
    m = tl.cdiv(M, H)
    for i in range(m):
        y = tl.load(y_ptr+offs)
        y_max = tl.maximum(tl.max(tl.abs(y), axis=0),y_max)
        offs += H*N

    k = tl.cdiv(K, H)
    w_max = tl.zeros((W,),dtype=tl.float32) + 1.17e-38
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
    y_max = tl.zeros((W,),dtype=tl.float32) + 1.17e-38
    offs = pid*W*N + tl.arange(0, W)[:,None]*N + tl.arange(0, H)[None,:]
    for i in range(n):
        y = tl.load(y_ptr+offs)
        y_max = tl.maximum(tl.max(tl.abs(y), axis=1),y_max)
        offs += H

    k = tl.cdiv(K, H)
    x_max = tl.zeros((W,),dtype=tl.float32) + 1.17e-38
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





def smooth_quant_forward(x,w,smooth_scale=None):

    x_q,w_q,x_s,w_s,smooth_scale = triton_smooth_quant_nt(x, w, smooth_scale=smooth_scale)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_s,
                                    scale_b=w_s,
                                    out_dtype=x.dtype,
                                    use_fast_accum=True)
    return output,x_q,w_q,x_s,w_s,smooth_scale

def smooth_quant_backward(y,w):

    y_q,w_q,y_s,w_s = triton_smooth_quant_nn(y, w)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_s,
                                    scale_b=w_s,
                                    out_dtype=y.dtype,
                                    use_fast_accum=True)
    return output



def smooth_quant_update(y,x):
    y_q,x_q,y_s,x_s = triton_smooth_quant_tn(y, x)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_s,
                                    scale_b=x_s,
                                    out_dtype=y.dtype,
                                    use_fast_accum=True)
    return output


def reused_smooth_quant_forward(x,w,smooth_scale):

    x_q,x_s = triton_slide_smooth_quant(x, 1/smooth_scale)
    w_q,w_s = triton_slide_smooth_quant(w, smooth_scale)

    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_s,
                                    scale_b=w_s.view(1,-1),
                                    out_dtype=x.dtype,
                                    use_fast_accum=True)
    return output,x_q,w_q,x_s,w_s,smooth_scale

def reused_smooth_quant_backward(y,w_q, smooth_scale, w_s):

    y_q,y_s = triton_slide_smooth_quant(y, w_s)
    w_q = triton_transpose(w_q)
    # print(f'{smooth_scale=} {y_s.view(-1)=} {w_s.view(-1)=} {y_q[:4,:4]=} {w_q[:4,:4]=}')

    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_s,
                                    scale_b=1.0/smooth_scale.view(1,-1),
                                    out_dtype=y.dtype,
                                    use_fast_accum=True)
    return output


def reused_smooth_quant_update(y,x_q, smooth_scale, x_s):

    y_q,y_s = triton_slide_transpose_smooth_quant(y, x_s)
    x_q = triton_transpose(x_q)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_s.view(-1,1),
                                    scale_b=smooth_scale.view(1,-1),
                                    out_dtype=y.dtype,
                                    use_fast_accum=True)
    return output


def reused_smooth_quant_f_and_b(x,w,y,smooth_scale=None):
    if smooth_scale is None:
        o,x_q,w_q,x_s,w_s,smooth_scale = smooth_quant_forward(x,w,smooth_scale=None)
    else:
        o,x_q,w_q,x_s,w_s,smooth_scale = reused_smooth_quant_forward(x,w,smooth_scale)
    dx = reused_smooth_quant_backward(y, w_q, smooth_scale, w_s)
    dw = reused_smooth_quant_update(y, x_q, smooth_scale, x_s)
    # print(f'{smooth_scale=} {x_s[:,0]=} {w_s=}')
    return o,dx,dw,smooth_scale


@triton.jit
def slide_smooth_quant_kernel(x_ptr, q_ptr, ss_ptr, qs_ptr, M, N, H: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    offs = pid*W*N + tl.arange(0, W)[:,None]*N + tl.arange(0, H)[None,:]
    soffs = tl.arange(0, H)
    x_max = tl.zeros((W,),dtype=tl.float32) + 1.17e-38
    n = tl.cdiv(N, H)
    for i in range(n):
        x = tl.load(x_ptr+offs)
        scale = tl.load(ss_ptr+soffs)
        x = x.to(tl.float32) * scale
        x_max = tl.maximum(tl.max(tl.abs(x), axis=1),x_max)
        offs += H 
        soffs += H

    scale = x_max/448.0
    tl.store(qs_ptr+pid*W+tl.arange(0, W), scale)

    s = (1.0/scale)[:,None]
    offs = pid*W*N + tl.arange(0, W)[:,None]*N + tl.arange(0, H)[None,:]
    soffs = tl.arange(0, H)
    for i in range(n):
        x = tl.load(x_ptr+offs)
        smooth_scale = tl.load(ss_ptr+soffs)
        xq = (x.to(tl.float32) * smooth_scale * s).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr+offs, xq)
        offs += H 
        soffs += H


# smooth_scale: w_max/tl.sqrt(x_max*w_max)
def triton_slide_smooth_quant(x, smooth_scale):
    M, N = x.shape
    device = x.device 
    x_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((M,1), device=device, dtype=torch.float32)
    H = max([x for x in [128,256,512,1024] if N%x == 0])
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
def slide_smooth_quant_tma_kernel(x_desc_ptr, q_desc_ptr, ss_ptr, qs_ptr, M, N, H: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    # offs = pid*W*N + tl.arange(0, W)[:,None]*N + tl.arange(0, H)[None,:]
    offs = pid*W*N
    soffs = tl.arange(0, H)
    x_max = tl.zeros((W,),dtype=tl.float32) + 1.17e-38
    n = tl.cdiv(N, H)
    for i in range(n):
        # x = tl.load(x_ptr+offs)
        offs_w = i*H
        x = tl._experimental_descriptor_load(x_desc_ptr, [offs, offs_w], [W, H], tl.float16)
        scale = tl.load(ss_ptr+soffs)
        x = x.to(tl.float32) * scale
        x_max = tl.maximum(tl.max(tl.abs(x), axis=1),x_max)
        # offs += H 
        soffs += H

    scale = x_max/448.0
    tl.store(qs_ptr+pid*W+tl.arange(0, W), scale)

    s = (1.0/scale)[:,None]
    # offs = pid*W*N + tl.arange(0, W)[:,None]*N + tl.arange(0, H)[None,:]
    offs = pid*W*N 
    soffs = tl.arange(0, H)
    for i in range(n):
        # x = tl.load(x_ptr+offs)
        offs_w = i*H
        x = tl._experimental_descriptor_load(x_desc_ptr, [offs, offs_w], [W, H], tl.float16)
        smooth_scale = tl.load(ss_ptr+soffs)
        xq = (x.to(tl.float32) * smooth_scale * s).to(tl.float8e4nv)
        # tl.store(q_ptr+offs, xq)
        tl._experimental_descriptor_store(q_desc_ptr, xq, [offs, offs_w])
        # offs += H 
        soffs += H

import numpy as np

def triton_slide_smooth_quant_tma(x, smooth_scale):
    M, N = x.shape
    device = x.device 
    x_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((M,1), device=device, dtype=torch.float32)
    # H = 1024 if N%1024 == N else 256
    H = 32
    W = 16


    TMA_SIZE = 128
    desc_x = np.empty(TMA_SIZE, dtype=np.int8)
    desc_xq = np.empty(TMA_SIZE, dtype=np.int8)

    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(x.data_ptr(), M, N, W, H, x.element_size(),
                                                              desc_x)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(x_q.data_ptr(), M, N, W, H, x_q.element_size(),
                                                              desc_xq)

    desc_x = torch.tensor(desc_x, device=device)
    desc_xq = torch.tensor(desc_xq, device=device)
    
    grid = lambda META: (M//W, )
    slide_smooth_quant_tma_kernel[grid](
        desc_x,
        desc_xq,
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
    x_max = tl.zeros((W,),dtype=tl.float32) + 1.17e-38
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
    soffs = tl.arange(0, H)
    toffs = pid*W*M + tl.arange(0, W)[:,None]*M + tl.arange(0, H)[None,:]
    for i in range(m):
        smooth_scale = tl.load(ss_ptr+soffs)
        x = tl.trans(tl.load(x_ptr+offs))
        x = (x*smooth_scale*s).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr+toffs, x)
        offs += H*N
        toffs += H


# smooth_scale: w_max/tl.sqrt(x_max*w_max)
def triton_slide_transpose_smooth_quant(x, smooth_scale):
    M, N = x.shape
    device = x.device 
    x_q = torch.empty((N, M), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((1,N), device=device, dtype=torch.float32)
    H = max([x for x in [128,256,512] if M%x == 0])
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
    x_q,x_scale = triton_slide_smooth_quant(x, 1/smooth_scale)
    w_q,w_scale = triton_slide_smooth_quant(w, smooth_scale)
    return x_q,x_scale,w_q,w_scale


def triton_slide_smooth_quant_nn(y, w, smooth_scale):
    y_q,y_scale = triton_slide_smooth_quant(y, smooth_scale)
    w_q,w_scale = triton_slide_transpose_smooth_quant(w, 1/smooth_scale)
    return y_q,y_scale,w_q,w_scale 

def triton_slide_smooth_quant_tn(y, x, smooth_scale):
    y_q,y_scale = triton_slide_transpose_smooth_quant(y, smooth_scale)
    x_q,x_scale = triton_slide_transpose_smooth_quant(x, 1/smooth_scale)
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






@triton.jit
def fused_smooth_kernel_nt(x_ptr, xb_ptr, xm_ptr, w_ptr, wb_ptr, wm_ptr, smooth_scale_ptr, M, N, K, H: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)

    offs = pid*W + tl.arange(0, H)[:,None]*K + tl.arange(0, W)[None,:]
    m = tl.cdiv(M, H)
    x_max = tl.zeros((W,),dtype=tl.float32)
    x_ptrs = x_ptr + offs
    indices = tl.arange(0, H)
    for i in range(m):
        x = tl.load(x_ptrs, mask=i*H+indices[:,None]<M)
        x_max = tl.maximum(tl.max(tl.abs(x), axis=0),x_max)
        x_ptrs += H*K

    n = tl.cdiv(N, H)
    w_max = tl.zeros((W,),dtype=tl.float32)
    w_ptrs = w_ptr + offs
    indices = tl.arange(0, H)
    for i in range(n):
        w = tl.load(w_ptrs, mask=i*H+indices[:,None]<N)
        w_max = tl.maximum(tl.max(tl.abs(w), axis=0),w_max)
        w_ptrs += H*K

    scale = tl.sqrt(x_max*w_max)
    x_scale = x_max/scale
    w_scale = w_max/scale

    smooth_scale_ptrs = smooth_scale_ptr + pid*W + tl.arange(0, W)
    tl.store(smooth_scale_ptrs, w_scale)

    x_ptrs = x_ptr + offs
    xs_ptrs = xb_ptr + offs
    xs_offs = tl.arange(0, H)
    xs_max_ptrs =  xm_ptr + pid*M + xs_offs
    for i in range(m):
        x = tl.load(x_ptrs)
        x = x * w_scale
        xs_max = tl.maximum(tl.max(tl.abs(x), axis=1), 1.17e-38)
        tl.store(xs_ptrs, x)
        tl.store(xs_max_ptrs, xs_max)
        x_ptrs += H*K
        xs_ptrs += H*K
        xs_max_ptrs += H

    w_ptrs = w_ptr + offs
    ws_ptrs = wb_ptr + offs
    ws_offs = tl.arange(0, H)
    ws_max_ptrs = wm_ptr + pid*N + ws_offs
    for i in range(n):
        w = tl.load(w_ptrs)
        ws = w * x_scale
        ws_max = tl.maximum(tl.max(tl.abs(ws), axis=1), 1.17e-38)
        tl.store(ws_ptrs, ws)
        tl.store(ws_max_ptrs, ws_max)
        w_ptrs += H*K
        ws_ptrs += H*K
        ws_max_ptrs += H

