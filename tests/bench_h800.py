import time 
from enum import IntEnum
from typing import Tuple
import math
import torch
import triton
import triton.language as tl
from triton import Config



def fp16_forward(x,w):
    return x @ w

def fp16_update(y,x):
    return y.t() @ x

def fp16_backward(y,w):
    return y @ w


def hadamard_matrix(n, device='cuda:0', dtype=torch.bfloat16):
    m2 = torch.tensor([[1,1],[1,-1]],device=device,dtype=dtype)
    m = m2
    for i in range(int(round(math.log2(n)-1))):
        m = torch.kron(m,m2)
    return m.to(dtype)

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


def triton_hadamard_nt(x, w, hm, R=2):
    assert w.size(1) == w.size(1)
    M, K = x.shape
    N, K = w.shape
    x_b = torch.empty_like(x)
    w_b = torch.empty_like(w)
    BLOCK_SIZE = hm.size(0)
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




def triton_hadamard_quant_nt(x, w, hm, R=2):
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

    BLOCK_SIZE = 4096
    grid = lambda META: (M, )
    row_quant_kernel[grid](
        x_b, x_q, x_scale,
        M,K,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=8
    )

    BLOCK_SIZE = 4096
    grid = lambda META: (N, )
    row_quant_kernel[grid](
        w_b, w_q, w_scale,
        N,K,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=8
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
def triton_hadamard_quant_tn(y, x, hm, R=2):
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
    grid = lambda META: (M//BLOCK_SIZE, )
    hadamard_tn_kernel[grid](
        y, y_b, 
        x, x_b,
        hm,
        M,N,K,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=8
    )

    BLOCK_SIZE = 4096
    grid = lambda META: (K, )
    row_quant_kernel[grid](
        x_b, x_q, x_scale,
        K,M,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=8
    )

    BLOCK_SIZE = 4096
    grid = lambda META: (N, )
    row_quant_kernel[grid](
        y_b, y_q, y_scale,
        N,M,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=8
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


def triton_hadamard_quant_nn(y, w, hm, R=2):
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
    hadamard_nn_kernel[grid](
        y, y_b,
        w, w_b, 
        hm,
        M,N,K,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=8
    )

    BLOCK_SIZE = 4096
    grid = lambda META: (M, )
    row_quant_kernel[grid](
        y_b, y_q, y_scale,
        M,N,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=8
    )

    BLOCK_SIZE = 4096
    grid = lambda META: (K, )
    row_quant_kernel[grid](
        w_b, w_q, w_scale,
        K,N,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=8
    )

    return y_q,w_q,y_scale,w_scale



@triton.jit
def fused_hadamard_kernel(x_ptr, b_ptr, s_ptr, q_ptr, hm_ptr, M, N, BLOCK_SIZE: tl.constexpr, R: tl.constexpr, SIDE: tl.constexpr):
    # apply hadamard transform and row-wise quant
    # SIDE=0: hadamard@block  
    # SIDE=1: block@hadamard
    # row-wise read, row-wise write
    pid = tl.program_id(axis=0)
    hm = tl.load(hm_ptr + tl.arange(0, BLOCK_SIZE)[:,None]*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None,:])
    offs = pid*BLOCK_SIZE*N + tl.arange(0, BLOCK_SIZE)[:,None]*N + tl.arange(0, BLOCK_SIZE)[None,:]
    n = tl.cdiv(N, BLOCK_SIZE)
    maxs = tl.zeros((BLOCK_SIZE,),dtype=tl.float32)
    for i in range(n):
        x = tl.load(x_ptr+offs)
        if SIDE == 0:
            x = tl.dot(hm, x)
        else:
            x = tl.dot(x, hm)
        tl.store(b_ptr+offs, x)
        maxs = tl.maximum(maxs, tl.max(tl.abs(x),1))
        offs += BLOCK_SIZE

    scales = maxs/448.0

    tl.store(s_ptr + pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), scales)
    rs = (448.0/maxs)[:,None]

    offs = pid*BLOCK_SIZE*N + tl.arange(0, BLOCK_SIZE)[:,None]*N + tl.arange(0, R*BLOCK_SIZE)[None,:]
    n = tl.cdiv(N, R*BLOCK_SIZE)
    for i in range(n):
        x = tl.load(b_ptr+offs)
        y = (x.to(tl.float32)*rs).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr+offs, y)
        offs += R*BLOCK_SIZE



def triton_fused_hadamard(x, hm, hm_side=1, op_side=0, R=2):
    # for x in y = x @ w, op_side = 0, hm_side=1
    # for wT in dx = y @ wT, hm_side=1: hm_side = 0 in logical format, but hm_side will be 1 with transpose format
    # for yT in dwT = yT @ x, hm_side=1, op_side = 0
    # for x in dwT = yT @ x, hm_side=1, op_side = 1
    M, N = x.shape
    x_b = torch.empty((M,N),dtype=x.dtype,device=x.device)
    if op_side == 0:
        x_s = torch.empty((M,1),dtype=torch.float32,device=x.device)
    else:
        x_s = torch.empty((1,M),dtype=torch.float32,device=x.device)
    x_q = torch.empty((M,N),dtype=torch.float8_e4m3fn,device=x.device)
    BLOCK_SIZE = hm.size(0)
    SIDE = hm_side
    grid = lambda META: (M//BLOCK_SIZE, )
    fused_hadamard_kernel[grid](
        x, 
        x_b,
        x_s,
        x_q, 
        hm,
        M,N,
        BLOCK_SIZE,
        R,
        SIDE,
        num_stages=6,
        num_warps=8
    )
    return x_q,x_s



@triton.jit
def fused_transpose_hadamard_kernel(x_ptr, b_ptr, s_ptr, q_ptr, hm_ptr, M, N, BLOCK_SIZE: tl.constexpr, R: tl.constexpr, SIDE: tl.constexpr):
    # transpose x: [M, N] -> [N, M] 
    # and then apply hadamard transform
    # SIDE=0: hadamard@block  
    # SIDE=1: block@hadamard
    pid = tl.program_id(axis=0)
    hm = tl.load(hm_ptr + tl.arange(0, BLOCK_SIZE)[:,None]*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None,:])
    # col-wise read, row-wise write
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
        maxs = tl.maximum(maxs, tl.max(tl.abs(x),1))
        tl.store(b_ptr+toffs, x)
        offs += BLOCK_SIZE*N
        toffs += BLOCK_SIZE

    scales = maxs/448.0
    rs = (448.0/maxs)[:,None]

    tl.store(s_ptr + pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), scales)

    toffs = pid*BLOCK_SIZE*M + tl.arange(0, BLOCK_SIZE)[:,None]*M + tl.arange(0, R*BLOCK_SIZE)[None,:]
    m = tl.cdiv(M, R*BLOCK_SIZE)
    for i in range(m):
        x = tl.load(b_ptr+toffs).to(tl.float32)
        y = (x*rs).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr+toffs, y)
        toffs += R*BLOCK_SIZE


def triton_fused_transpose_hadamard(x, hm, hm_side=1, op_side=0, R=2):
    # for wT in dx = y @ wT, op_side = 0, hm_side = 1: hm_side = 0 in logical format, but hm_side will be 1 with transpose format
    # for yT in dwT = yT @ x, op_side = 0, hm_side = 0
    # for x in dwT = yT @ x, op_side = 1, hm_side = 1
    M, N = x.shape
    x_b = torch.empty((N,M),dtype=x.dtype,device=x.device)
    if op_side == 0:
        x_s = torch.empty((N,1),dtype=torch.float32,device=x.device)
    else:
        x_s = torch.empty((1,N),dtype=torch.float32,device=x.device)
    x_q = torch.empty((N,M),dtype=torch.float8_e4m3fn,device=x.device)
    BLOCK_SIZE = hm.size(0)
    SIDE = hm_side
    grid = lambda META: (N//BLOCK_SIZE, )
    fused_transpose_hadamard_kernel[grid](
        x, 
        x_b,
        x_s,
        x_q, 
        hm,
        M,N,
        BLOCK_SIZE,
        R,
        SIDE,
        num_stages=6,
        num_warps=8
    )
    return x_q,x_s



# write h@x and h@w as well, bit for bilateral transformer
# y = x @ w
# dx = y @ wT
# dwT = yT @ x
@triton.jit
def bit_hadamard_nt_kernel(x_ptr, xb_ptr, xbt_ptr, w_ptr, wb_ptr, wbt_ptr, hm_ptr, M, N, K, BLOCK_SIZE: tl.constexpr, R: tl.constexpr):
    pid = tl.program_id(axis=0)

    hm = tl.load(hm_ptr + tl.arange(0, BLOCK_SIZE)[:,None]*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None,:])

    offs = pid*BLOCK_SIZE + tl.arange(0, R*BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_SIZE)[None,:]
    toffs = pid*BLOCK_SIZE*M + tl.arange(0, BLOCK_SIZE)[:,None]*M + tl.arange(0, R*BLOCK_SIZE)[None,:]
    m = tl.cdiv(M, R*BLOCK_SIZE)
    for i in range(m):
        x = tl.load(x_ptr+offs)
        tl.store(xb_ptr+offs, tl.dot(x, hm) )
        tl.store(xbt_ptr+toffs, tl.dot(hm, x) )
        offs += R*BLOCK_SIZE*K
        toffs += R*BLOCK_SIZE

    hm = (hm/BLOCK_SIZE).to(x_ptr.dtype.element_ty)
    offs = pid*BLOCK_SIZE + tl.arange(0, R*BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_SIZE)[None,:]
    toffs = pid*BLOCK_SIZE*N + tl.arange(0, BLOCK_SIZE)[:,None]*N + tl.arange(0, R*BLOCK_SIZE)[None,:]
    n = tl.cdiv(N, R*BLOCK_SIZE)
    for i in range(n):
        w = tl.load(w_ptr+offs)
        tl.store(wb_ptr+offs, tl.dot(w, hm))
        tl.store(wbt_ptr+toffs, tl.dot(hm, w))
        offs += R*BLOCK_SIZE*K
        toffs += R*BLOCK_SIZE


# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_bit_hadamard_nt(x, w, hm, R=1):
    assert R==1
    assert w.size(1) == w.size(1)
    M, K = x.shape
    N, K = w.shape
    x_b = torch.empty_like(x)
    w_b = torch.empty_like(w)
    x_bt = torch.empty((K,N),dtype=x.dtype,device=x.device)
    w_bt = torch.empty((K,N),dtype=x.dtype,device=x.device)
    BLOCK_SIZE = hm.size(0)
    grid = lambda META: (K//BLOCK_SIZE, )
    bit_hadamard_nt_kernel[grid](
        x, x_b, x_bt,
        w, w_b, w_bt,
        hm,
        M,N,K,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=8
    )
    return x_b,x_bt,w_b,w_bt

# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_bit_hadamard_quant_nt(x,w,hm, R=1):

    assert R==1
    assert w.size(1) == w.size(1)
    M, K = x.shape
    N, K = w.shape
    device = x.device
    x_b = torch.empty_like(x)
    w_b = torch.empty_like(w)
    x_bt = torch.empty((K,M),dtype=x.dtype,device=device)
    w_bt = torch.empty((K,N),dtype=x.dtype,device=device)
    BLOCK_SIZE = hm.size(0)
    grid = lambda META: (K//BLOCK_SIZE, )
    bit_hadamard_nt_kernel[grid](
        x, x_b, x_bt,
        w, w_b, w_bt,
        hm,
        M,N,K,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=8
    )

    x_q = torch.empty((M, K), device=device, dtype=torch.float8_e4m3fn)
    w_q = torch.empty((N, K), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((M,1), device=device, dtype=torch.float32)
    w_scale = torch.empty((1,N), device=device, dtype=torch.float32)


    BLOCK_SIZE = 4096
    grid = lambda META: (M, )
    row_quant_kernel[grid](
        x_b, x_q, x_scale,
        M,K,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=8
    )

    BLOCK_SIZE = 4096
    grid = lambda META: (N, )
    row_quant_kernel[grid](
        w_b, w_q, w_scale,
        N,K,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=8
    )

    return x_bt,w_bt,x_q,w_q,x_scale,w_scale


# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_bit_hadamard_quant_nn(y,w,hm,R=1):
    # w is transposed and hadamard tranformed
    assert R==1
    assert y.size(1) == w.size(1)
    M, N = y.shape
    K, N = w.shape

    device = y.device
    y_b = torch.empty_like(y)
    y_bt = torch.empty((N,M),dtype=y.dtype,device=device)
    BLOCK_SIZE = hm.size(0)
    grid = lambda META: (N//BLOCK_SIZE, )
    bit_hadamard_dy_kernel[grid](
        y, y_b, y_bt,
        hm,
        M,N,K,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=8
    )

    y_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    w_q = torch.empty((K, N), device=device, dtype=torch.float8_e4m3fn)
    y_scale = torch.empty((M,1), device=device, dtype=torch.float32)
    w_scale = torch.empty((1,K), device=device, dtype=torch.float32)


    BLOCK_SIZE = 4096
    grid = lambda META: (M, )
    row_quant_kernel[grid](
        y_b, y_q, y_scale,
        M,N,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=8
    )

    BLOCK_SIZE = 4096
    grid = lambda META: (K, )
    row_quant_kernel[grid](
        w, w_q, w_scale,
        K,N,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=8
    )

    return y_bt,y_q,w_q,y_scale,w_scale



# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_bit_hadamard_quant_tn(y,x,hm,R=1):
    # y and x is transposed and hadamard tranformed
    assert R==1
    # print(y.shape,x.shape )
    assert y.size(1) == x.size(1)
    N, M = y.shape
    K, M = x.shape

    device = y.device
    y_q = torch.empty((N, M), device=device, dtype=torch.float8_e4m3fn)
    x_q = torch.empty((K, M), device=device, dtype=torch.float8_e4m3fn)
    y_scale = torch.empty((N,1), device=device, dtype=torch.float32)
    x_scale = torch.empty((1,K), device=device, dtype=torch.float32)

    BLOCK_SIZE = 4096
    grid = lambda META: (N, )
    row_quant_kernel[grid](
        y, y_q, y_scale,
        N,M,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=8
    )

    BLOCK_SIZE = 4096
    grid = lambda META: (K, )
    row_quant_kernel[grid](
        x, x_q, x_scale,
        K,M,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=8
    )

    return y_q,x_q,y_scale,x_scale


# write h@x and h@w as well, bit for bilateral transformer
# y = x @ w
# dx = y @ wT
# dwT = yT @ x
@triton.jit
def bit_hadamard_dy_kernel(y_ptr, yb_ptr, ybt_ptr, hm_ptr, M, N, K, BLOCK_SIZE: tl.constexpr, R: tl.constexpr):
    pid = tl.program_id(axis=0)

    hm = tl.load(hm_ptr + tl.arange(0, BLOCK_SIZE)[:,None]*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None,:])

    offs = pid*BLOCK_SIZE + tl.arange(0, R*BLOCK_SIZE)[:,None]*N + tl.arange(0, BLOCK_SIZE)[None,:]
    toffs = pid*BLOCK_SIZE*M + tl.arange(0, BLOCK_SIZE)[:,None]*M + tl.arange(0, R*BLOCK_SIZE)[None,:]
    m = tl.cdiv(M, R*BLOCK_SIZE)
    for i in range(m):
        x = tl.load(y_ptr+offs)
        tl.store(yb_ptr+offs, tl.dot(x, hm) )
        tl.store(ybt_ptr+toffs, tl.dot(hm, x) )
        offs += R*BLOCK_SIZE*N
        toffs += R*BLOCK_SIZE



def triton_bit_hadamard_dy(y, hm, R=1):
    assert R==1
    M, N = y.shape
    K = 0
    y_b = torch.empty_like(x)
    y_bt = torch.empty((N,M),dtype=x.dtype,device=x.device)
    BLOCK_SIZE = hm.size(0)
    grid = lambda META: (N//BLOCK_SIZE, )
    bit_hadamard_nt_kernel[grid](
        y, y_b, y_bt,
        hm,
        M,N,K,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=8
    )
    return y_b,y_bt



def hadamard_quant_forward(x,w,hm):
    x_q,w_q,x_scale,w_scale = triton_hadamard_quant_nt(x, w, hm)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_scale,
                                    scale_b=w_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,x_q,w_q,x_scale,w_scale

def hadamard_quant_backward(y,w,hm):

    y_q,w_q,y_scale,w_scale = triton_hadamard_quant_nn(y, w, hm)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_scale,
                                    scale_b=w_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_q,w_q,y_scale,w_scale


def hadamard_quant_update(y,x,hm):
    y_q,x_q,y_scale,x_scale = triton_hadamard_quant_tn(y, x, hm)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_scale,
                                    scale_b=x_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_q,x_q,y_scale,x_scale


def triton_hadamard_quant_nt_nn_tn(x,w,y,hm):
    triton_hadamard_quant_nt(x, w, hm)
    triton_hadamard_quant_nn(y, w, hm)
    triton_hadamard_quant_tn(y, x, hm)


def fp8_hadamard_f_and_b(x,w,y,hm):
    hadamard_quant_forward(x, w, hm)
    hadamard_quant_backward(y, w, hm)
    hadamard_quant_update(y,x, hm)


# y = x @ w
def triton_fused_hadamard_quant_nt(x, w, hm):
    stream = torch.cuda.Stream(device=0)
    x_q,x_s = triton_fused_hadamard(x, hm, hm_side=1, op_side=0)
    with torch.cuda.stream(stream):
        w_q,w_s = triton_fused_hadamard(w, hm, hm_side=1, op_side=1)
    torch.cuda.current_stream().wait_stream(stream)
    return x_q,x_s,w_q,w_s


# dx = y @ wT
def triton_fused_hadamard_quant_nn(y, w, hm):
    stream = torch.cuda.Stream(device=0)
    y_q,y_s = triton_fused_hadamard(y, hm, hm_side=1, op_side=0)
    with torch.cuda.stream(stream):
        w_q,w_s = triton_fused_transpose_hadamard(w, hm, hm_side=1, op_side=1)
    torch.cuda.current_stream().wait_stream(stream)
    return y_q,y_s,w_q,w_s


# dwT = yT @ x
def triton_fused_hadamard_quant_tn(y, x, hm):
    stream = torch.cuda.Stream(device=0)
    y_q,y_s = triton_fused_transpose_hadamard(y, hm, hm_side=1, op_side=0)
    with torch.cuda.stream(stream):
        x_q,x_s = triton_fused_transpose_hadamard(x, hm, hm_side=1, op_side=1)
    torch.cuda.current_stream().wait_stream(stream)
    return y_q,y_s,x_q,x_s

def triton_fuse_hadamard_quant_nt_nn_tn(x,w,y,hm):
    triton_fused_hadamard_quant_nt(x, w, hm)
    triton_fused_hadamard_quant_nn(y, w, hm)
    triton_fused_hadamard_quant_tn(y, x, hm)

def fp8_fuse_hadamard_f_and_b(x,w,y,hm):
    fuse_hadamard_quant_forward(x, w, hm)
    fuse_hadamard_quant_backward(y, w, hm)
    fuse_hadamard_quant_update(y,x, hm)





def fuse_hadamard_quant_forward(x,w,hm):

    x_q,x_s,w_q,w_s = triton_fused_hadamard_quant_nt(x, w, hm)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_s,
                                    scale_b=w_s,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,x_q,x_s,w_q,w_s

def fuse_hadamard_quant_backward(y,w,hm):

    y_q,y_s,w_q,w_s = triton_fused_hadamard_quant_nn(y, w, hm)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_s,
                                    scale_b=w_s,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_q,y_s,w_q,w_s



def fuse_hadamard_quant_update(y,x,hm):
    y_q,y_s,x_q,x_s = triton_fused_hadamard_quant_tn(y, x, hm)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_s,
                                    scale_b=x_s,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_q,y_s,x_q,x_s


def bit_hadamard_quant_forward(x,w,hm):

    x_bt,w_bt, x_q,w_q,x_scale,w_scale = triton_bit_hadamard_quant_nt(x, w, hm)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_scale,
                                    scale_b=w_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,x_bt,w_bt,x_q,w_q,x_scale,w_scale

def bit_hadamard_quant_backward(y,w,hm):

    y_bt,y_q,w_q,y_scale,w_scale = triton_bit_hadamard_quant_nn(y, w, hm)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_scale,
                                    scale_b=w_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_bt,y_q,w_q,y_scale,w_scale



def bit_hadamard_quant_update(y,x,hm):
    y_q,x_q,y_scale,x_scale = triton_bit_hadamard_quant_tn(y, x, hm)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_scale,
                                    scale_b=x_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_q,x_q,y_scale,x_scale



def fp8_bit_hadamard_f_and_b(x,w,y,hm):
    output,x_bt,w_bt,x_q,w_q,x_scale,w_scale = bit_hadamard_quant_forward(x, w, hm)
    output,y_bt,y_q,w_q,y_scale,w_scale=bit_hadamard_quant_backward(y, w_bt, hm)
    output,y_q,x_q,y_scale,x_scale=bit_hadamard_quant_update(y_bt,x_bt, hm)



@triton.jit
def row_quant_kernel(x_ptr, q_ptr, s_ptr,  M, N,  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    n_block = tl.cdiv(N, BLOCK_SIZE)
    indices = tl.arange(0, BLOCK_SIZE)
    max_val = 1e-6
    for j in range(n_block):
        offs = pid*N + j*BLOCK_SIZE + indices
        x = tl.load(x_ptr + offs, mask=j*BLOCK_SIZE + indices<N, other=0)
        max_val = tl.maximum(tl.max(tl.abs(x.to(tl.float32))), max_val)
    scale = max_val/448.0
    tl.store(s_ptr + pid, scale)
    for j in range(n_block):
        offs = pid*N + j*BLOCK_SIZE + indices
        x = tl.load(x_ptr + offs, mask=j*BLOCK_SIZE + indices<N, other=0)
        y = x.to(tl.float32) / scale
        y = y.to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + offs, y, mask=j*BLOCK_SIZE + indices<N)


def triton_row_quant(x):
    M, N = x.shape 
    BLOCK_SIZE = 4096
    x_q = torch.empty((M,N),dtype=torch.float8_e4m3fn,device=x.device)
    x_scale = torch.empty((M,1),dtype=torch.float32,device=x.device)
    grid = lambda META: (M, )
    row_quant_kernel[grid](
        x, x_q, x_scale,
        M,N,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=8
    )
    return x_q, x_scale



def fp16_f_and_b(x,w,y):
    y = x@w.t()
    dw = y.t()@x
    dx = y@w
    return y, dw, dx



def benchmark_func(fn, *args, n_repeat=1000, ref_flops=None, ref_time=None, name='', **kwargs):
    func_name = fn.__name__

    for i in range(100):
        fn(*args,**kwargs)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    
    ts = time.time()
    for i in range(n_repeat):
        start_events[i].record()
        fn(*args,**kwargs)
        end_events[i].record()
    
    torch.cuda.synchronize() 
    te = time.time()
    
    # times = sum([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
    # average_event_time = times * 1000 / n_repeat

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times = sorted(times)
    clip = max(1,n_repeat//100)
    times = sum(times[clip:-clip])
    
    average_event_time = times * 1000 / (n_repeat - 2*clip)
    
    fs = ''
    if ref_flops is not None:
        flops = ref_flops/1e12/(average_event_time/1e6)
        fs = f'FLOPS:{flops:.2f}T'
    ss = ''
    if ref_time is not None:
        ss = f'speedup:{ref_time/average_event_time:.3f}'
    print(f'{func_name} {name} time:{average_event_time:.1f} us {fs} {ss}')
    return average_event_time


# 5b: hidden_size:4k  seq_length:8K shape:(M,N,K)
# qkv: 8192, 6144, 4096
# out: 8192, 4096, 4096
# up/gate: 8192, 13312, 4096
# down: 8192, 4096, 13312   # benchmark setting


# 80b: hidden_size:8k  seq_length:8K shape:(M,N,K)
# qkv: 8192, 10240, 8192
# out: 8192, 8192, 8192
# up/gate: 8192, 34048, 8192
# down: 8192, 4096, 13312



def benchmark_with_shape(shape):
    batch_size, out_dim, in_dim = shape
    device = 'cuda:0'
    dtype = torch.bfloat16
    qtype = torch.float8_e4m3fn  # torch.float8_e5m2
    n_repeat = 1000
    gpu = torch.cuda.get_device_properties(0).name


    x = torch.randn(batch_size, in_dim, dtype=dtype, device=device)
    w = torch.randn(out_dim, in_dim, dtype=dtype, device=device)
    y = torch.randn(batch_size, out_dim, dtype=dtype, device=device)
    x_f8 = x.to(qtype)
    w_f8 = w.to(qtype)
    y_f8 = y.to(qtype)
    B = 64
    hm = hadamard_matrix(B, dtype=dtype, device=device)

    org_out = fp16_forward(x, w.t())
    print(f'\ndevice:{gpu} M:{batch_size} N:{out_dim} K:{in_dim}')

    # y = x @ w
    # dx = y @ wT
    # dwT = yT @ x

    # benchmark_func(triton_hadamard_nt, x, w, hm, n_repeat=n_repeat)
    # benchmark_func(triton_row_quant, x, n_repeat=n_repeat)
    # benchmark_func(triton_row_quant, w, n_repeat=n_repeat)

    benchmark_func(triton_hadamard_quant_nt, x, w, hm, n_repeat=n_repeat)
    benchmark_func(triton_hadamard_quant_tn, y, x, hm, n_repeat=n_repeat)
    benchmark_func(triton_hadamard_quant_nn, y, w, hm, n_repeat=n_repeat)

    benchmark_func(triton_fused_hadamard, x, hm, hm_side=1, op_side=0)
    benchmark_func(triton_fused_transpose_hadamard, x, hm, hm_side=1, op_side=0)
    benchmark_func(triton_fused_hadamard_quant_nt, x,w,hm, n_repeat=n_repeat)
    benchmark_func(triton_fused_hadamard_quant_nn, y,x,hm, n_repeat=n_repeat)
    benchmark_func(triton_fused_hadamard_quant_tn, y,w,hm, n_repeat=n_repeat)
    
    benchmark_func(triton_hadamard_quant_nt_nn_tn, x,w,y,hm, n_repeat=n_repeat)
    benchmark_func(triton_fuse_hadamard_quant_nt_nn_tn, x,w,y,hm, n_repeat=n_repeat)


    # benchmark_func(triton_bit_hadamard_nt, x, w, hm, n_repeat=n_repeat)
    # benchmark_func(triton_bit_hadamard_quant_nt, x, w, hm, n_repeat=n_repeat)
    # benchmark_func(triton_bit_hadamard_quant_nn, y, w.t().contiguous(), hm, n_repeat=n_repeat)
    # benchmark_func(triton_bit_hadamard_quant_tn, y.t().contiguous(), x.t().contiguous(), hm, n_repeat=n_repeat)


    # ref_time = benchmark_func(fp16_forward, x, w.t(), n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
    # benchmark_func(hadamard_quant_forward, x, w, hm, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)
    # ref_time = benchmark_func(fp16_backward, y, w, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
    # benchmark_func(hadamard_quant_backward, y, w, hm, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)
    # ref_time = benchmark_func(fp16_update, y, x, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
    # benchmark_func(hadamard_quant_update, y, x, hm, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)

    ref_time = benchmark_func(fp16_f_and_b, x, w, y, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*6)
    benchmark_func(fp8_hadamard_f_and_b, x, w, y, hm, n_repeat=n_repeat, ref_time=ref_time,ref_flops=batch_size*in_dim*out_dim*6)
    benchmark_func(fp8_fuse_hadamard_f_and_b, x, w, y, hm, n_repeat=n_repeat, ref_time=ref_time,ref_flops=batch_size*in_dim*out_dim*6)
    benchmark_func(fp8_bit_hadamard_f_and_b, x, w, y, hm, n_repeat=n_repeat, ref_time=ref_time,ref_flops=batch_size*in_dim*out_dim*6)



# 5b: hidden_size:4k  seq_length:8K shape:(M,N,K)
# qkv: 8192, 6144, 4096
# out: 8192, 4096, 4096
# up/gate: 8192, 13312, 4096
# down: 8192, 4096, 13312   # benchmark setting


# 80b: hidden_size:8k  seq_length:8K shape:(M,N,K)
# qkv: 8192, 10240, 8192
# out: 8192, 8192, 8192
# up/gate: 8192, 34048, 8192
# down: 8192, 4096, 34048


# benchmark_with_shape([8192, 4096, 13312])

for shape in [[8192, 6144, 4096], [8192, 4096, 4096], [8192, 13312, 4096], [8192, 4096, 13312],
            [8192, 10240, 8192],[8192, 8192, 8192],[8192, 34048, 8192],[8192, 4096, 34048]]:
    benchmark_with_shape(shape)