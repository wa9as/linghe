import math
import torch
import triton
import triton.language as tl
from triton import Config
from flops.quant.channel import row_quant_kernel



def hadamard_matrix(n, device='cuda:0', dtype=torch.bfloat16, norm=False):
    assert 2**int(math.log2(n)) == n
    m2 = torch.tensor([[1,1],[1,-1]],device=device,dtype=torch.float32)
    m = m2
    for i in range(int(math.log2(n))-1):
        m = torch.kron(m,m2)
    if norm:
        m = m / n**0.5
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
    # hm = (hm/BLOCK_SIZE).to(x_ptr.dtype.element_ty)
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
        num_warps=4,
        num_ctas=1
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
        num_warps=4,
        num_ctas=1
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
    # hm = (hm/BLOCK_SIZE).to(x_ptr.dtype.element_ty)
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
    # hm = (hm/BLOCK_SIZE).to(w_ptr.dtype.element_ty)
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
def fused_hadamard_kernel(x_ptr, b_ptr, s_ptr, q_ptr, hm_ptr, M, N, BLOCK_SIZE: tl.constexpr, R: tl.constexpr, SIDE: tl.constexpr):
    # apply hadamard transform and row-wise quant
    # SIDE=0: hadamard@block  
    # SIDE=1: block@hadamard
    # row-wise read, row-wise write
    pid = tl.program_id(axis=0)
    hm = tl.load(hm_ptr + tl.arange(0, BLOCK_SIZE)[:,None]*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None,:])
    offs = pid*BLOCK_SIZE*N + tl.arange(0, BLOCK_SIZE)[:,None]*N + tl.arange(0, BLOCK_SIZE)[None,:]
    n = tl.cdiv(N, BLOCK_SIZE)
    maxs = tl.zeros((BLOCK_SIZE,),dtype=tl.float32)+1.17e-38
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



def triton_fused_hadamard(x, hm, op_side=0, hm_side=1, R=2):
    # y = x @ w
    #   x: op_side = 0, hm_side=1
    #   w: op_side = 1, hm_side=1
    # dx = y @ wT, 
    #   y: op_side = 0, hm_side=1

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
        num_warps=4
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
    maxs = tl.zeros((BLOCK_SIZE,),dtype=tl.float32)+1.17e-38
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


def triton_fused_transpose_hadamard(x, hm, op_side=0, hm_side=1, R=2):
    # dx = y @ wT
    #   wT: op_side = 1, hm_side = 1
    # dwT = yT @ x:
    #   yT: op_side = 0, hm_side = 0
    #   x: op_side = 1, hm_side = 1
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
        num_warps=4
    )
    return x_q,x_s


@triton.jit
def hadamard_quant_kernel(
    x_ptr,
    hm_ptr,
    xq_ptr,
    xtq_ptr,
    x_scale_ptr,
    xt_scale_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    R: tl.constexpr,
):
    pid = tl.program_id(0)
    range_m = tl.arange(0, BLOCK_SIZE)
    range_n = tl.arange(0, BLOCK_SIZE)
    hm = tl.load(hm_ptr + range_m[:, None] * BLOCK_SIZE + range_n[None, :])
    
    offs_m = pid * BLOCK_SIZE + range_m
    offs_n = pid * BLOCK_SIZE + range_n
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    max_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) + 1e-9
    max_col = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) + 1e-9
    
    for i in range(0, tl.cdiv(N, BLOCK_SIZE)):
        offs_n = i * BLOCK_SIZE + range_n
        mask_n = offs_n < N
        mask = mask_m[:, None] & mask_n[None, :]
        x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0)
        x_hm = tl.dot(x, hm)
        abs_x = tl.abs(x_hm.to(tl.float32))
        max_row = tl.maximum(max_row, tl.max(abs_x, axis=1))
        col_max = tl.max(abs_x, axis=0)
        max_col = tl.maximum(max_col, col_max)
    
    scale_row = 448.0 / max_row
    scale_col = 448.0 / max_col
    tl.store(x_scale_ptr + offs_m, max_row / 448.0, mask=mask_m)
    tl.store(xt_scale_ptr + offs_n, max_col / 448.0, mask=mask_n)
    
    for i in range(0, tl.cdiv(N, BLOCK_SIZE)):
        offs_n = i * BLOCK_SIZE + range_n
        mask_n = offs_n < N
        mask = mask_m[:, None] & mask_n[None, :]
        x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0)
        x_hm = tl.dot(x, hm)
        xq = (x_hm.to(tl.float32) * scale_row[:, None]).to(xq_ptr.dtype.element_ty)
        tl.store(xq_ptr + offs_m[:, None] * N + offs_n[None, :], xq, mask=mask)
        xtq = (x_hm.to(tl.float32) * scale_col[None, :]).to(xtq_ptr.dtype.element_ty)
        tl.store(xtq_ptr + offs_n[:, None] * M + offs_m[None, :], xtq, mask=mask_n[:, None] & mask_m[None, :])


# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_hadamard_quant_x(x, hm):
    # apply hadamard transformation and quantization for x
    # y = x @ w: x->x@h and rowwise quant
    # dwT = yT @ x: x->xT@h and rowwise quant
    M, N = x.shape
    B = hm.size(0)
    device = x.device 
    x_q = torch.empty((M,N),dtype=torch.float8_e4m3fn,device=device)
    xt_q = torch.empty((N,M),dtype=torch.float8_e4m3fn,device=device)
    x_scale = torch.empty((1,M),dtype=torch.float32,device=device)
    xt_scale = torch.empty((N,1),dtype=torch.float32,device=device)

    BLOCK_SIZE = hm.size(0)
    R = 2
    grid = lambda META: (M//BLOCK_SIZE, )
    hadamard_quant_kernel[grid](
        x, 
        hm,
        x_q,
        xt_q,
        x_scale, 
        xt_scale,
        M,
        N,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=4
    )
    return x_q,xt_q,x_scale,xt_scale



# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_hadamard_quant_w(w, hm):
    # apply hadamard transformation and quantization for w
    # y = x @ w: w->w@h and rowwise quant
    # dx = y @ wT: w->h@wT and rowwise quant
    M, N = w.shape
    B = hm.size(0)
    device = w.device
    w_q = torch.empty((M,N),dtype=torch.float8_e4m3fn,device=device)
    wt_q = torch.empty((N,M),dtype=torch.float8_e4m3fn,device=device)
    w_scale = torch.empty((1,M),dtype=torch.float32,device=device)
    wt_scale = torch.empty((N,1),dtype=torch.float32,device=device)

    BLOCK_SIZE = hm.size(0)
    R = 2
    grid = lambda META: (M//BLOCK_SIZE, )
    hadamard_quant_kernel[grid](
        w, 
        hm,
        w_q,
        wt_q,
        w_scale, 
        wt_scale,
        M,
        N,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=4
    )
    return w_q,wt_q,w_scale,wt_scale


# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_hadamard_quant_y(y, hm):
    # apply hadamard transformation and quantization for dy
    # dx = y @ wT: y->y@h and rowwise quant
    # dwT = yT @ x: y->h@yT and rowwise quant
    M, N = y.shape
    B = hm.size(0)
    device = y.device
    y_q = torch.empty((M,N),dtype=torch.float8_e4m3fn,device=device)
    yt_q = torch.empty((N,M),dtype=torch.float8_e4m3fn,device=device)
    y_scale = torch.empty((M,1),dtype=torch.float32,device=device)
    yt_scale = torch.empty((1,N),dtype=torch.float32,device=device)
    
    BLOCK_SIZE = hm.size(0)
    R = 2
    grid = lambda META: (M//BLOCK_SIZE, )
    hadamard_quant_kernel[grid](
        y, 
        hm,
        y_q,
        yt_q,
        y_scale, 
        yt_scale,
        M,
        N,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=4
    )
    return y_q,yt_q,y_scale,yt_scale



"""
write h@x and h@w, bit for BIlateral Transform
y = x @ w
dx = y @ wT
dwT = yT @ x
x in yT @ x should be h@x and transposed
w in y @ wT should be h@w and transposed
"""

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
        tl.store(xbt_ptr+toffs, tl.trans(tl.dot(hm, x)) )
        offs += R*BLOCK_SIZE*K
        toffs += R*BLOCK_SIZE

    # hm = (hm/BLOCK_SIZE).to(x_ptr.dtype.element_ty)
    offs = pid*BLOCK_SIZE + tl.arange(0, R*BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_SIZE)[None,:]
    toffs = pid*BLOCK_SIZE*N + tl.arange(0, BLOCK_SIZE)[:,None]*N + tl.arange(0, R*BLOCK_SIZE)[None,:]
    n = tl.cdiv(N, R*BLOCK_SIZE)
    for i in range(n):
        w = tl.load(w_ptr+offs)
        tl.store(wb_ptr+offs, tl.dot(w, hm))
        tl.store(wbt_ptr+toffs, tl.trans(tl.dot(hm, w)))
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
        num_warps=4
    )
    return x_b,x_bt,w_b,w_bt



"""
y = x @ w
dx = y @ wT
dwT = yT @ x
yT in yT @ x should be h@y and transposed
"""
@triton.jit
def bit_hadamard_dy_kernel(y_ptr, yb_ptr, ybt_ptr, hm_ptr, M, N, K, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    hm = tl.load(hm_ptr + tl.arange(0, BLOCK_SIZE)[:,None]*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None,:])

    offs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:,None]*N + tl.arange(0, BLOCK_SIZE)[None,:]
    toffs = pid*BLOCK_SIZE*M + tl.arange(0, BLOCK_SIZE)[:,None]*M + tl.arange(0, BLOCK_SIZE)[None,:]
    m = tl.cdiv(M, BLOCK_SIZE)
    for i in range(m):
        x = tl.load(y_ptr+offs)
        tl.store(yb_ptr+offs, tl.dot(x, hm) )
        tl.store(ybt_ptr+toffs, tl.trans(tl.dot(hm, x)) )
        offs += BLOCK_SIZE*N
        toffs += BLOCK_SIZE


def triton_bit_hadamard_dy(y, hm, R=1):
    assert R==1
    M, N = y.shape
    K = 0
    y_b = torch.empty_like(y)
    y_bt = torch.empty((N,M),dtype=y.dtype,device=y.device)
    BLOCK_SIZE = hm.size(0)
    grid = lambda META: (N//BLOCK_SIZE, )
    bit_hadamard_nt_kernel[grid](
        y, y_b, y_bt,
        hm,
        M,N,K,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=4
    )
    return y_b,y_bt


# y = x @ w
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
        num_warps=4
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

    return x_bt,w_bt,x_q,w_q,x_scale,w_scale

"""
y = x @ w
dx = y @ wT
dwT = yT @ x

"""
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
        num_stages=6,
        num_warps=4
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
        num_warps=4
    )

    BLOCK_SIZE = 4096
    grid = lambda META: (K, )
    row_quant_kernel[grid](
        w, w_q, w_scale,
        K,N,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=4
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
        num_warps=4
    )

    BLOCK_SIZE = 4096
    grid = lambda META: (K, )
    row_quant_kernel[grid](
        x, x_q, x_scale,
        K,M,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=4
    )

    return y_q,x_q,y_scale,x_scale



def hadamard_quant_forward(x,w,hm):
    x_q,w_q,x_scale,w_scale = triton_hadamard_quant_nt(x, w, hm)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_scale,
                                    scale_b=w_scale,
                                    out_dtype=x.dtype,
                                    use_fast_accum=True)
    return output

def hadamard_quant_backward(y,w,hm):

    y_q,w_q,y_scale,w_scale = triton_hadamard_quant_nn(y, w, hm)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_scale,
                                    scale_b=w_scale,
                                    out_dtype=y.dtype,
                                    use_fast_accum=True)
    return output


def hadamard_quant_update(y,x,hm):
    y_q,x_q,y_scale,x_scale = triton_hadamard_quant_tn(y, x, hm)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_scale,
                                    scale_b=x_scale,
                                    out_dtype=x.dtype,
                                    use_fast_accum=True)
    return output


def hadamard_quant_forward_megatron(x,w,hm):
    x_q, x_scale, w_q, w_scale = triton_hadamard_quant_nt_megatron(x, w, hm)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_scale,
                                    scale_b=w_scale,
                                    out_dtype=x.dtype,
                                    use_fast_accum=True
    )
    return output


def hadamard_quant_backward_megatron(y,w,hm):
    y_q, y_scale, w_q, w_scale = triton_hadamard_quant_nn_megatron(y, w, hm)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_scale,
                                    scale_b=w_scale,
                                    out_dtype=y.dtype,
                                    use_fast_accum=True
    )
    return output


def hadamard_quant_update_megatron(y,x,hm):
    y_q, y_scale, x_q, x_scale = triton_hadamard_quant_tn_megatron(y, x, hm)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_scale,
                                    scale_b=x_scale,
                                    out_dtype=x.dtype,
                                    use_fast_accum=True
    )
    return output


def hadamard_quant_forward_debug(x,w,hm):
    x_q,w_q,x_scale,w_scale = triton_hadamard_quant_nt(x, w, hm)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_scale,
                                    scale_b=w_scale,
                                    out_dtype=x.dtype,
                                    use_fast_accum=True)
    return output,x_q,w_q,x_scale,w_scale

def hadamard_quant_backward_debug(y,w,hm):

    y_q,w_q,y_scale,w_scale = triton_hadamard_quant_nn(y, w, hm)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_scale,
                                    scale_b=w_scale,
                                    out_dtype=y.dtype,
                                    use_fast_accum=True)
    return output,y_q,w_q,y_scale,w_scale


def hadamard_quant_update_debug(y,x,hm):
    y_q,x_q,y_scale,x_scale = triton_hadamard_quant_tn(y, x, hm)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_scale,
                                    scale_b=x_scale,
                                    out_dtype=y.dtype,
                                    use_fast_accum=True)
    return output,y_q,x_q,y_scale,x_scale


def hadamard_quant_forward_debug_megatron(x, w, hm):
    x_q, x_scale, w_q, w_scale = triton_hadamard_quant_nt_megatron(x, w, hm)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_scale,
                                    scale_b=w_scale,
                                    out_dtype=x.dtype,
                                    use_fast_accum=True
    )
    return output, x_q, w_q, x_scale, w_scale


def hadamard_quant_backward_debug_megatron(y,w,hm):
    y_q, y_scale, w_q, w_scale = triton_hadamard_quant_nn_megatron(y, w, hm)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_scale,
                                    scale_b=w_scale,
                                    out_dtype=y.dtype,
                                    use_fast_accum=True
    )
    return output, y_q, w_q, y_scale, w_scale


def hadamard_quant_update_debug_megatron(y, x, hm):
    y_q, y_scale, x_q, x_scale = triton_hadamard_quant_tn_megatron(y, x, hm)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_scale,
                                    scale_b=x_scale,
                                    out_dtype=x.dtype,
                                    use_fast_accum=True
    )
    return output, y_q, x_q, y_scale, x_scale


def triton_hadamard_quant_nt_nn_tn(x,w,y,hm):
    triton_hadamard_quant_nt(x, w, hm)
    triton_hadamard_quant_nn(y, w, hm)
    triton_hadamard_quant_tn(y, x, hm)


def triton_hadamard_quant_nt_nn_tn_megatron(x,w,y,hm):
    triton_hadamard_quant_nt_megatron(x, w, hm)
    triton_hadamard_quant_nn_megatron(y, w, hm)
    triton_hadamard_quant_tn_megatron(y, x, hm)


def fp8_hadamard_f_and_b(x,w,y,hm):
    hadamard_quant_forward(x, w, hm)
    hadamard_quant_backward(y, w, hm)
    hadamard_quant_update(y,x, hm)


# y = x @ w
def triton_fused_hadamard_quant_nt(x, w, hm):
    # stream = torch.cuda.Stream(device=0)
    # x_q,x_s = triton_fused_hadamard(x, hm, hm_side=1, op_side=0)
    # with torch.cuda.stream(stream):
    #     w_q,w_s = triton_fused_hadamard(w, hm, hm_side=1, op_side=1)
    # torch.cuda.current_stream().wait_stream(stream)
    x_q,x_s = triton_fused_hadamard(x, hm, hm_side=1, op_side=0)
    w_q,w_s = triton_fused_hadamard(w, hm, hm_side=1, op_side=1)
    return x_q,x_s,w_q,w_s




# dx = y @ wT
def triton_fused_hadamard_quant_nn(y, w, hm):
    # stream = torch.cuda.Stream(device=0)
    # y_q,y_s = triton_fused_hadamard(y, hm, hm_side=1, op_side=0)
    # with torch.cuda.stream(stream):
    #     w_q,w_s = triton_fused_transpose_hadamard(w, hm, hm_side=1, op_side=1)
    # torch.cuda.current_stream().wait_stream(stream)
    y_q,y_s = triton_fused_hadamard(y, hm, hm_side=1, op_side=0)
    w_q,w_s = triton_fused_transpose_hadamard(w, hm, hm_side=1, op_side=1)
    return y_q,y_s,w_q,w_s


# dwT = yT @ x
def triton_fused_hadamard_quant_tn(y, x, hm):
    # stream = torch.cuda.Stream(device=0)
    # y_q,y_s = triton_fused_transpose_hadamard(y, hm, hm_side=1, op_side=0)
    # with torch.cuda.stream(stream):
    #     x_q,x_s = triton_fused_transpose_hadamard(x, hm, hm_side=1, op_side=1)
    # torch.cuda.current_stream().wait_stream(stream)
    y_q,y_s = triton_fused_transpose_hadamard(y, hm, hm_side=1, op_side=0)
    x_q,x_s = triton_fused_transpose_hadamard(x, hm, hm_side=1, op_side=1)
    return y_q,y_s,x_q,x_s


def triton_hadamard_quant_nt_megatron(x, w, hm):
    x_q, _, x_scale, _ = triton_hadamard_quant_x(x, hm)
    w_q, _, w_scale, _ = triton_hadamard_quant_w(w, hm)
    return x_q, x_scale.t(), w_q, w_scale


def triton_hadamard_quant_nn_megatron(y, w, hm):
    y_q, _, y_scale, _ = triton_hadamard_quant_y(y, hm)
    w_q, _, w_scale, _ = triton_hadamard_quant_w(w.t(), hm)
    return y_q, y_scale, w_q, w_scale


def triton_hadamard_quant_tn_megatron(y, x, hm):
    y_q, _, y_scale, _ = triton_hadamard_quant_y(y.t(), hm)
    x_q, _, x_scale, _ = triton_hadamard_quant_x(x.t(), hm)
    return y_q, y_scale, x_q, x_scale


def triton_fused_hadamard_quant_nt_nn_tn(x,w,y,hm):
    triton_fused_hadamard_quant_nt(x, w, hm)
    triton_fused_hadamard_quant_nn(y, w, hm)
    triton_fused_hadamard_quant_tn(y, x, hm)

def fp8_fused_hadamard_f_and_b(x,w,y,hm):
    fused_hadamard_quant_forward(x, w, hm)
    fused_hadamard_quant_backward(y, w, hm)
    fused_hadamard_quant_update(y,x, hm)




def fused_hadamard_quant_forward(x,w,hm):

    x_q,x_s,w_q,w_s = triton_fused_hadamard_quant_nt(x, w, hm)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_s,
                                    scale_b=w_s,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output

def fused_hadamard_quant_backward(y,w,hm):

    y_q,y_s,w_q,w_s = triton_fused_hadamard_quant_nn(y, w, hm)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_s,
                                    scale_b=w_s,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output



def fused_hadamard_quant_update(y,x,hm):
    y_q,y_s,x_q,x_s = triton_fused_hadamard_quant_tn(y, x, hm)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_s,
                                    scale_b=x_s,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output



def fused_hadamard_quant_forward_debug(x,w,hm):

    x_q,x_s,w_q,w_s = triton_fused_hadamard_quant_nt(x, w, hm)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_s,
                                    scale_b=w_s,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,x_q,x_s,w_q,w_s

def fused_hadamard_quant_backward_debug(y,w,hm):

    y_q,y_s,w_q,w_s = triton_fused_hadamard_quant_nn(y, w, hm)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_s,
                                    scale_b=w_s,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_q,y_s,w_q,w_s



def fused_hadamard_quant_update_debug(y,x,hm):
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
    return output,x_bt,w_bt

def bit_hadamard_quant_backward(y,w,hm):

    y_bt,y_q,w_q,y_scale,w_scale = triton_bit_hadamard_quant_nn(y, w, hm)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_scale,
                                    scale_b=w_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_bt



def bit_hadamard_quant_update(y,x,hm):
    y_q,x_q,y_scale,x_scale = triton_bit_hadamard_quant_tn(y, x, hm)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_scale,
                                    scale_b=x_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output



def bit_hadamard_quant_forward_debug(x,w,hm):

    x_bt,w_bt, x_q,w_q,x_scale,w_scale = triton_bit_hadamard_quant_nt(x, w, hm)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_scale,
                                    scale_b=w_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,x_bt,w_bt,x_q,w_q,x_scale,w_scale

def bit_hadamard_quant_backward_debug(y,w,hm):

    y_bt,y_q,w_q,y_scale,w_scale = triton_bit_hadamard_quant_nn(y, w, hm)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_scale,
                                    scale_b=w_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_bt,y_q,w_q,y_scale,w_scale



def bit_hadamard_quant_update_debug(y,x,hm):
    y_q,x_q,y_scale,x_scale = triton_bit_hadamard_quant_tn(y, x, hm)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_scale,
                                    scale_b=x_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_q,x_q,y_scale,x_scale


def fp8_bit_hadamard_f_and_b(x,w,y,hm):
    output,x_bt,w_bt = bit_hadamard_quant_forward(x, w, hm)
    output,y_bt=bit_hadamard_quant_backward(y, w_bt, hm)
    output=bit_hadamard_quant_update(y_bt,x_bt, hm)