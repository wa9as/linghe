import math
import torch
import triton
import triton.language as tl
from triton import Config
from flops.quant.channel.channel import row_quant_kernel
from flops.utils.util import round_up





"""
write h@x and h@w, duplex for BIlateral Transform
y = x @ w
dx = y @ wT
dwT = yT @ x
x in yT @ x should be h@x and transposed
w in y @ wT should be h@w and transposed
"""

@triton.jit
def duplex_hadamard_nt_kernel(x_ptr, xb_ptr, xbt_ptr, w_ptr, wb_ptr, wbt_ptr, hm_ptr, M, N, K, BLOCK_SIZE: tl.constexpr, R: tl.constexpr):
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
def triton_duplex_hadamard_nt(x, w, hm, R=1):
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
    duplex_hadamard_nt_kernel[grid](
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
def duplex_hadamard_dy_kernel(y_ptr, yb_ptr, ybt_ptr, hm_ptr, M, N, K, BLOCK_SIZE: tl.constexpr):
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


def triton_duplex_hadamard_dy(y, hm, R=1):
    assert R==1
    M, N = y.shape
    K = 0
    y_b = torch.empty_like(y)
    y_bt = torch.empty((N,M),dtype=y.dtype,device=y.device)
    BLOCK_SIZE = hm.size(0)
    grid = lambda META: (N//BLOCK_SIZE, )
    duplex_hadamard_nt_kernel[grid](
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
def triton_duplex_hadamard_quant_nt(x,w,hm, R=1):

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
    duplex_hadamard_nt_kernel[grid](
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
def triton_duplex_hadamard_quant_nn(y,w,hm,R=1):
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
    duplex_hadamard_dy_kernel[grid](
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
def triton_duplex_hadamard_quant_tn(y,x,hm,R=1):
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



def duplex_hadamard_quant_forward(x,w,hm):

    x_bt,w_bt, x_q,w_q,x_scale,w_scale = triton_duplex_hadamard_quant_nt(x, w, hm)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_scale,
                                    scale_b=w_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,x_bt,w_bt

def duplex_hadamard_quant_backward(y,w,hm):

    y_bt,y_q,w_q,y_scale,w_scale = triton_duplex_hadamard_quant_nn(y, w, hm)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_scale,
                                    scale_b=w_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_bt



def duplex_hadamard_quant_update(y,x,hm):
    y_q,x_q,y_scale,x_scale = triton_duplex_hadamard_quant_tn(y, x, hm)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_scale,
                                    scale_b=x_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output



def duplex_hadamard_quant_forward_debug(x,w,hm):

    x_bt,w_bt, x_q,w_q,x_scale,w_scale = triton_duplex_hadamard_quant_nt(x, w, hm)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_scale,
                                    scale_b=w_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,x_bt,w_bt,x_q,w_q,x_scale,w_scale

def duplex_hadamard_quant_backward_debug(y,w,hm):

    y_bt,y_q,w_q,y_scale,w_scale = triton_duplex_hadamard_quant_nn(y, w, hm)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_scale,
                                    scale_b=w_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_bt,y_q,w_q,y_scale,w_scale



def duplex_hadamard_quant_update_debug(y,x,hm):
    y_q,x_q,y_scale,x_scale = triton_duplex_hadamard_quant_tn(y, x, hm)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_scale,
                                    scale_b=x_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_q,x_q,y_scale,x_scale


def fp8_duplex_hadamard_f_and_b(x,w,y,hm):
    output,x_bt,w_bt = duplex_hadamard_quant_forward(x, w, hm)
    output,y_bt=duplex_hadamard_quant_backward(y, w_bt, hm)
    output=duplex_hadamard_quant_update(y_bt,x_bt, hm)