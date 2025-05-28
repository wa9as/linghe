
import math
import torch
import triton
import triton.language as tl
from triton import Config
from flops.quant.channel.channel import row_quant_kernel
from flops.utils.transpose import triton_transpose,triton_block_transpose,triton_block_pad_transpose
from flops.utils.util import round_up





@triton.jit
def smooth_nt_kernel(x_ptr, xs_ptr, w_ptr, ws_ptr, smooth_scale_ptr, M, N, K, H: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)
    # col-wise read, col-wise write
    offs = pid*W + tl.arange(0, H)[:,None]*K + tl.arange(0, W)[None,:]
    x_max = tl.zeros((W,),dtype=tl.float32) + 5.27e-36
    m = tl.cdiv(M, H)
    for i in range(m):
        x = tl.load(x_ptr+offs)
        x_max = tl.maximum(tl.max(tl.abs(x), axis=0),x_max)
        offs += H*K

    n = tl.cdiv(N, H)
    offs = (pid*W + tl.arange(0, H)[:,None]*K + tl.arange(0, W)[None,:]).to(tl.int64)
    w_max = tl.zeros((W,),dtype=tl.float32) + 5.27e-36
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
    y_max = tl.zeros((W,),dtype=tl.float32) + 5.27e-36
    m = tl.cdiv(M, H)
    for i in range(m):
        y = tl.load(y_ptr+offs)
        y_max = tl.maximum(tl.max(tl.abs(y), axis=0),y_max)
        offs += H*N

    k = tl.cdiv(K, H)
    w_max = tl.zeros((W,),dtype=tl.float32) + 5.27e-36
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
    y_max = tl.zeros((W,),dtype=tl.float32) + 5.27e-36
    offs = pid*W*N + tl.arange(0, W)[:,None]*N + tl.arange(0, H)[None,:]
    for i in range(n):
        y = tl.load(y_ptr+offs)
        y_max = tl.maximum(tl.max(tl.abs(y), axis=1),y_max)
        offs += H

    k = tl.cdiv(K, H)
    x_max = tl.zeros((W,),dtype=tl.float32) + 5.27e-36
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

