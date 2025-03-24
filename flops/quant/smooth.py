
from enum import IntEnum
from typing import Tuple
import math
import torch
import triton
import triton.language as tl
from triton import Config

from flops.gemm.fp8_gemm import persistent_fp8_gemm




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



# smooth quant
def fp8_smooth_v0_f_and_b(x,w,y):
    xq,wq,fwd_scale = triton_smooth_quant_v0_nt(x,w)
    o = persistent_fp8_gemm(xq, wq.t(), torch.bfloat16)
    
    yq,wq,bwd_scale = triton_smooth_quant_v0_nn(y,w)
    y_dummy_scale = torch.ones((x.size(0),1),dtype=torch.float32,device=x.device)
    dx = torch._scaled_mm(yq,
                            wq.t(),
                            scale_a=y_dummy_scale,
                            scale_b=bwd_scale[None],
                            out_dtype=torch.bfloat16,
                            use_fast_accum=True)

    dw = torch._scaled_mm(yq.t().contiguous(),
                                    xq.t().contiguous().t(),
                                    scale_a=bwd_scale[:,None],
                                    scale_b=fwd_scale[None],
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return o, dx, dw


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

    # offs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_K)[None,:]
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
    # xs_max_ptrs =  xs_max_ptr + pid + tl.arange(0, BLOCK_SIZE)*m #2D
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


#grid M/BLOCK_M
@triton.jit
def row_quant_sm_kernel(x_ptr, q_ptr, s_ptr,  M, K,  BLOCK_SIZE: tl.constexpr, BLOCK_K: tl.constexpr, SCALE_K: tl.constexpr):
    pid = tl.program_id(0)

    offs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_K)[None,:]
    n_block = SCALE_K
    
    # offs_scale = pid * (M // BLOCK_SIZE) + tl.arange(0, 8)[None,:] #需要加边界 K // BLOCK_K
    offs_scale = pid * BLOCK_SIZE * SCALE_K +  tl.arange(0, BLOCK_SIZE)[:,None]* SCALE_K + tl.arange(0, SCALE_K)[None,:]
    scale_off = tl.load(s_ptr+offs_scale)
    scale = tl.max(scale_off, axis=1).expand_dims(-1)
    # scale = scale.expand_dims(-1)
    # if pid == 0:
    #     tl.device_print(scale)

    x_ptrs = x_ptr + offs
    q_ptrs = q_ptr + offs
    for j in range(n_block):
        x = tl.load(x_ptrs)
        # if pid == 0:
        #     tl.device_print(x)
        y = x.to(tl.float32) / scale 
        y = y.to(q_ptr.dtype.element_ty)
        tl.store(q_ptrs, y)
        x_ptrs += BLOCK_K
        q_ptrs += BLOCK_K
        


# v3: smooth + token/channel
def triton_sm_v3_quant_nt(x, w):
    eps = 1e-10
    M, K = x.shape
    N, K = w.shape
    device = x.device 
    x_s = torch.empty((M, K), device=x.device, dtype=x.dtype)
    w_s = torch.empty((N, K), device=w.device, dtype=w.dtype)
    x_q = torch.empty((M, K), device=x.device, dtype=torch.float8_e4m3fn)
    w_q = torch.empty((N, K), device=w.device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((M,1), device=x.device, dtype=torch.float32)
    w_scale = torch.empty((1,N), device=w.device, dtype=torch.float32)
    # xs_max_tmp = torch.empty(M, device=x.device, dtype=torch.float32)
    # ws_max_tmp = torch.empty(N, device=w.device, dtype=torch.float32)
    
    BLOCK_SIZE = 512 
    BLOCK_K = 64

    # xs_max_tmp = torch.empty((M, K//BLOCK_K), device=x.device, dtype=torch.float32)
    # ws_max_tmp = torch.empty((N, K//BLOCK_K), device=w.device, dtype=torch.float32)
    
    xs_max_tmp = torch.empty(M*(K//BLOCK_K), device=x.device, dtype=torch.float32)
    ws_max_tmp = torch.empty(N*(K//BLOCK_K), device=w.device, dtype=torch.float32)
    # print(f"grid: {K//BLOCK_SIZE}")

    grid = lambda META: (K//BLOCK_K, )
    smooth_v3_kernel_nt[grid](
        x, x_s, xs_max_tmp,
        w, w_s, ws_max_tmp,
        M,N,K,
        eps, 
        BLOCK_SIZE,
        BLOCK_K,
        num_stages=6,
        num_warps=16
    )

    xs_max_tmp = xs_max_tmp.view(M,-1)
    ws_max_tmp = ws_max_tmp.view(N,-1)
    
    # print(x_s)
    # print(torch.sum(x_s==0))
    # print(torch.nonzero(x_s)[:514])
    #70 us for bellow
    # xs_max = xs_max_tmp.view(M, -1).max(1)[0]/448.0
    # ws_max = ws_max_tmp.view(N, -1).max(1)[0]/448.0

    BLOCK_SIZE = 32
    BLOCK_K = 512
    SCALE_K = K // BLOCK_K
    grid = lambda META: (M//BLOCK_SIZE, )
    row_quant_sm_kernel[grid](
        x_s, x_q, xs_max_tmp,
        M,K,
        BLOCK_SIZE,
        BLOCK_K,
        SCALE_K,
        num_stages=6,
        num_warps=32
    )

    # BLOCK_SIZE = 4096
    grid = lambda META: (N//BLOCK_SIZE, )
    row_quant_sm_kernel[grid](
        w_s, w_q, ws_max_tmp,
        N,K,
        BLOCK_SIZE,
        BLOCK_K,
        SCALE_K,
        num_stages=6,
        num_warps=32
    )

    return x_q,w_q
