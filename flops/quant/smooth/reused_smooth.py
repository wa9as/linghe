
import math
import numpy as np

import torch
import triton
import triton.language as tl
from triton import Config
from flops.quant.channel.channel import row_quant_kernel
from flops.utils.transpose import triton_transpose,triton_block_transpose,triton_block_pad_transpose
from flops.utils.util import round_up



@triton.jit
def reused_smooth_quant_kernel(x_ptr, q_ptr, ss_ptr, qs_ptr, M, N, H: tl.constexpr, W: tl.constexpr, EVEN: tl.constexpr, REVERSE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    offs = pid*W*N + tl.arange(0, W)[:,None]*N + tl.arange(0, H)[None,:]
    soffs = tl.arange(0, H)
    x_max = tl.zeros((W,),dtype=tl.float32) + 5.27e-36
    n = tl.cdiv(N, H)
    for i in range(n):
        if EVEN:
            x = tl.load(x_ptr+offs)
            smooth_scale = tl.load(ss_ptr+soffs)
        else:
            x = tl.load(x_ptr+offs, mask=(i*H+tl.arange(0, H)[None,:]<N)&(pid*W+tl.arange(0, W)[:,None]<M))
            other = 0.0 if REVERSE else 1e30
            smooth_scale = tl.load(ss_ptr+soffs, mask=soffs<N, other=other)
        if REVERSE:
            x = x.to(tl.float32) * smooth_scale
        else:
            x = x.to(tl.float32) / smooth_scale
        x_max = tl.maximum(tl.max(tl.abs(x), axis=1),x_max)
        offs += H 
        soffs += H

    scale = x_max/448.0
    tl.store(qs_ptr+pid*W+tl.arange(0, W), scale)

    s = (1.0/scale)[:,None]

    offs = pid*W*N + tl.arange(0, W)[:,None]*N + tl.arange(0, H)[None,:]
    soffs = tl.arange(0, H)
    for i in range(n):
        if EVEN:
            x = tl.load(x_ptr+offs)
            smooth_scale = tl.load(ss_ptr+soffs)
        else:
            x = tl.load(x_ptr+offs, mask=(i*H+tl.arange(0, H)[None,:]<N)&(pid*W+tl.arange(0, W)[:,None]<M))
            other = 0.0 if REVERSE else 1e30
            smooth_scale = tl.load(ss_ptr+soffs, mask=soffs<N, other=other)

        if REVERSE:
            xq = (x.to(tl.float32) * smooth_scale * s).to(q_ptr.dtype.element_ty)
        else:
            xq = (x.to(tl.float32) / smooth_scale * s).to(q_ptr.dtype.element_ty)

        if EVEN:
            tl.store(q_ptr+offs, xq)
        else:
            tl.store(q_ptr+offs, xq, mask=(i*H+tl.arange(0, H)[None,:]<N)&(pid*W+tl.arange(0, W)[:,None]<M))
        offs += H 
        soffs += H




# smooth_scale: w_max/tl.sqrt(x_max*w_max)
def triton_reused_smooth_quant(x, smooth_scale, x_q=None, x_scale=None, reverse=False, pad_scale=False):
    M, N = x.shape
    device = x.device 
    if x_q is None:
        x_q = torch.zeros((M, N), device=device, dtype=torch.float8_e4m3fn)
    if x_scale is None:
        scale_size = round_up(M, b=16) if pad_scale else M
        x_scale = torch.zeros((scale_size,), device=device, dtype=torch.float32)
    W = 8 if M < 132*10 else 16
    H = 512 if W == 16 else 1024
    # H = 512
    # W = 16
    if N%H == 0 and M%W == 0:
        EVEN = True
    else:
        EVEN = False
    grid = lambda META: (triton.cdiv(M, W), )
    reused_smooth_quant_kernel[grid](
        x,
        x_q,
        smooth_scale,
        x_scale,
        M, N,
        H, W,
        EVEN,
        reverse,
        num_stages=5,
        num_warps=4
    )

    return x_q,x_scale



@triton.jit
def reused_transpose_smooth_quant_kernel(x_ptr, q_ptr, ss_ptr, qs_ptr, M, N, H: tl.constexpr, W: tl.constexpr, EVEN: tl.constexpr, REVERSE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # col-wise read, row-wise write
    offs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:]
    soffs = tl.arange(0, H)
    x_max = tl.zeros((W,),dtype=tl.float32) + 5.27e-36
    m = tl.cdiv(M, H)
    for i in range(m):
        if EVEN:
            x = tl.load(x_ptr+offs)
            smooth_scale = tl.load(ss_ptr+soffs)[:,None]
        else:
            x = tl.load(x_ptr+offs, mask=(i*H+tl.arange(0,H)[:,None]<M) & (pid*W+tl.arange(0,W)[None,:]<N))
            other = 0.0 if REVERSE else 1e30
            smooth_scale = tl.load(ss_ptr+soffs, mask=soffs<M, other=other)[:,None]
        if REVERSE:
            x = x * smooth_scale
        else:
            x = x / smooth_scale
        x_max = tl.maximum(tl.max(tl.abs(x), axis=0),x_max)
        offs += H*N 
        soffs += H

    scale = (x_max/448.0)
    if EVEN:
        tl.store(qs_ptr+pid*W+tl.arange(0, W), scale)
    else:
        tl.store(qs_ptr+pid*W+tl.arange(0, W), scale, mask=pid*W+tl.arange(0,W)<N)


    s = (1.0/scale)[:,None]
    offs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:]
    soffs = tl.arange(0, H)
    toffs = pid*W*M + tl.arange(0, W)[:,None]*M + tl.arange(0, H)[None,:]
    for i in range(m):
        if EVEN:
            x = tl.trans(tl.load(x_ptr+offs))
            smooth_scale = tl.load(ss_ptr+soffs)
        else:
            x = tl.trans(tl.load(x_ptr+offs, mask=(i*H+tl.arange(0,H)[:,None]<M) & (pid*W+tl.arange(0,W)[None,:]<N)))
            other = 0.0 if REVERSE else 1e30
            smooth_scale = tl.load(ss_ptr+soffs, mask=soffs<M, other=other)

        if REVERSE:
            x = (x*smooth_scale*s).to(q_ptr.dtype.element_ty)
        else:
            x = (x/smooth_scale*s).to(q_ptr.dtype.element_ty)
        if EVEN:
            tl.store(q_ptr+toffs, x)
        else:
            tl.store(q_ptr+toffs, x, mask=(i*H+tl.arange(0,H)[None,:]<M) & (pid*W+tl.arange(0,W)[:,None]<N))
        offs += H*N
        toffs += H
        soffs += H


def triton_reused_transpose_smooth_quant(x, smooth_scale, reverse=False):
    M, N = x.shape
    device = x.device 
    x_q = torch.zeros((N, M), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.zeros((N,), device=device, dtype=torch.float32)
    H = max([x for x in [1,128,256,512] if M%x == 0])
    W = max([x for x in [1,16] if N%x == 0])
    if H > 1 and W > 1:
        EVEN = True 
    else:
        EVEN = False 
        H = 256
        W = 16

    grid = lambda META: (triton.cdiv(N, W), )
    reused_transpose_smooth_quant_kernel[grid](
        x,
        x_q,
        smooth_scale,
        x_scale,
        M, N,
        H, W, 
        EVEN, reverse,
        num_stages=5,
        num_warps=4
    )

    return x_q,x_scale



@triton.jit
def reused_transpose_pad_smooth_quant_kernel(x_ptr, q_ptr, ss_ptr, qs_ptr, M, N, P, H: tl.constexpr, W: tl.constexpr, EVEN: tl.constexpr, REVERSE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # col-wise read, row-wise write
    offs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:]
    soffs = tl.arange(0, H)
    x_max = tl.zeros((W,),dtype=tl.float32) + 5.27e-36
    m = tl.cdiv(M, H)
    for i in range(m):
        if EVEN:
            x = tl.load(x_ptr+offs)
            smooth_scale = tl.load(ss_ptr+soffs)[:,None]
        else:
            x = tl.load(x_ptr+offs, mask=(i*H+tl.arange(0,H)[:,None]<M) & (pid*W+tl.arange(0,W)[None,:]<N))
            other = 0.0 if REVERSE else 1e30
            smooth_scale = tl.load(ss_ptr+soffs, mask=soffs<M, other=other)[:,None]
        if REVERSE:
            x = x * smooth_scale
        else:
            x = x / smooth_scale
        x_max = tl.maximum(tl.max(tl.abs(x), axis=0),x_max)
        offs += H*N 
        soffs += H

    scale = (x_max/448.0)
    if EVEN:
        tl.store(qs_ptr+pid*W+tl.arange(0, W), scale)
    else:
        tl.store(qs_ptr+pid*W+tl.arange(0, W), scale, mask=pid*W+tl.arange(0,W)<N)


    s = (1.0/scale)[:,None]
    offs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:]
    soffs = tl.arange(0, H)
    toffs = pid*W*M + tl.arange(0, W)[:,None]*P + tl.arange(0, H)[None,:]
    for i in range(m):
        if EVEN:
            x = tl.trans(tl.load(x_ptr+offs))
            smooth_scale = tl.load(ss_ptr+soffs)
        else:
            x = tl.trans(tl.load(x_ptr+offs, mask=(i*H+tl.arange(0,H)[:,None]<M) & (pid*W+tl.arange(0,W)[None,:]<N)))
            other = 0.0 if REVERSE else 1e30
            smooth_scale = tl.load(ss_ptr+soffs, mask=soffs<M, other=other)

        if REVERSE:
            x = (x*smooth_scale*s).to(q_ptr.dtype.element_ty)
        else:
            x = (x/smooth_scale*s).to(q_ptr.dtype.element_ty)
        if EVEN:
            tl.store(q_ptr+toffs, x)
        else:
            tl.store(q_ptr+toffs, x, mask=(i*H+tl.arange(0,H)[None,:]<M) & (pid*W+tl.arange(0,W)[:,None]<N))
        offs += H*N
        toffs += H
        soffs += H



def triton_reused_transpose_pad_smooth_quant(x, smooth_scale, reverse=False, pad=False):
    M, N = x.shape
    device = x.device 
    P = round_up(M) if pad else M
    x_q = torch.zeros((N, P), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.zeros((N,), device=device, dtype=torch.float32)
    H = max([x for x in [1,64,128,256] if M%x == 0])
    W = max([x for x in [1,16,32] if M%x == 0])
    if H > 1 and W > 1: 
        EVEN = True 
    else:
        EVEN = False 
        H = 256 if H == 1 else H
        W = 32 if W == 1 else W

    grid = lambda META: (triton.cdiv(N, W), )
    reused_transpose_pad_smooth_quant_kernel[grid](
        x,
        x_q,
        smooth_scale,
        x_scale,
        M, N, P,
        H, W, 
        EVEN, reverse,
        num_stages=5,
        num_warps=4
    )

    return x_q,x_scale



@triton.jit
def reused_smooth_quant_tma_kernel(x_desc_ptr, q_desc_ptr, ss_ptr, qs_ptr, M, N, H: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    # offs = pid*W*N + tl.arange(0, W)[:,None]*N + tl.arange(0, H)[None,:]
    offs = pid*W*N
    soffs = tl.arange(0, H)
    x_max = tl.zeros((W,),dtype=tl.float32) + 5.27e-36  # torch.finfo(torch.float32).tiny*448
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


def triton_reused_smooth_quant_tma(x, smooth_scale):
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
    reused_smooth_quant_tma_kernel[grid](
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




def triton_reused_smooth_quant_nt(x, w, smooth_scale):
    x_q,x_scale = triton_reused_smooth_quant(x, 1/smooth_scale)
    w_q,w_scale = triton_reused_smooth_quant(w, smooth_scale)
    return x_q,x_scale,w_q,w_scale


def triton_reused_smooth_quant_nn(y, w, smooth_scale):
    y_q,y_scale = triton_reused_smooth_quant(y, smooth_scale)
    w_q,w_scale = triton_reused_transpose_smooth_quant(w, 1/smooth_scale)
    return y_q,y_scale,w_q,w_scale 

def triton_reused_smooth_quant_tn(y, x, smooth_scale):
    y_q,y_scale = triton_reused_transpose_smooth_quant(y, smooth_scale)
    x_q,x_scale = triton_reused_transpose_smooth_quant(x, 1/smooth_scale)
    return y_q,y_scale,x_q,x_scale


def reused_smooth_quant_forward(x,w, smooth_scale):

    x_q,x_s,w_q,w_s = triton_reused_smooth_quant_nt(x, w, smooth_scale)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_s,
                                    scale_b=w_s.view(1,-1),
                                    out_dtype=x.dtype,
                                    use_fast_accum=True)
    return output

def reused_smooth_quant_backward(y,w, smooth_scale):

    y_q,y_s,w_q,w_s = triton_reused_smooth_quant_nn(y, w, smooth_scale)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_s,
                                    scale_b=w_s,
                                    out_dtype=y.dtype,
                                    use_fast_accum=True)
    return output


def reused_smooth_quant_update(y,x, smooth_scale):
    y_q,y_s,x_q,x_s = triton_reused_smooth_quant_tn(y, x, smooth_scale)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_s.view(-1,1),
                                    scale_b=x_s,
                                    out_dtype=y.dtype,
                                    use_fast_accum=True)
    return output



