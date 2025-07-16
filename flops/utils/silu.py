import math
import os
import torch
import triton
import triton.language as tl
from triton import Config





@triton.jit
def weighted_silu_forward_kernel(x_ptr, weight_ptr, out_ptr, M, T, N: tl.constexpr, n:tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)

    row_offs = pid*W*T*n+tl.arange(0, W)[:,None]*n
    col_offs = tl.arange(0, n)[None,:]
    
    for i in range(T):
        indices = pid*W*T+i*W+tl.arange(0, W)
        mask = indices[:,None]<M
        x1 = tl.load(x_ptr+row_offs*2+col_offs, mask=mask).to(tl.float32)
        x2 = tl.load(x_ptr+n+row_offs*2+col_offs, mask=mask).to(tl.float32)
        w = tl.load(weight_ptr+indices, mask=indices<M).to(tl.float32)[:,None]
        x = x1/(1+tl.exp(-x1))*x2*w
        tl.store(out_ptr+row_offs+col_offs, x, mask=mask)
        row_offs += n*W

def triton_weighted_silu_forward(x, weight, out=None):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device 
    if out is None:
        out = torch.empty((M, N//2), device=device, dtype=x.dtype)
    W = 8192//N 
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    T = triton.cdiv(M, sm*W)
    grid = (sm, )
    weighted_silu_forward_kernel[grid](
        x,
        weight,
        out,
        M, T,
        N, 
        N//2,
        W,
        num_stages=3,
        num_warps=16
    )
    return out




@triton.jit
def weighted_silu_backward_kernel(g_ptr, x_ptr, weight_ptr, dx_ptr, dw_ptr, M, T, N: tl.constexpr, n:tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)

    offs = pid*W*T*N+tl.arange(0, W)[:,None]*N+tl.arange(0, n)[None,:]
    hoffs = pid*W*T*n+tl.arange(0, W)[:,None]*n+tl.arange(0, n)[None,:]
    for i in range(T):
        mask = pid*W*T+i*W+tl.arange(0, W)
        x1 = tl.load(x_ptr+offs, mask=mask[:,None]<M).to(tl.float32)
        x2 = tl.load(x_ptr+offs+n, mask=mask[:,None]<M).to(tl.float32)
        g = tl.load(g_ptr+hoffs, mask=mask[:,None]<M).to(tl.float32)
        w = tl.load(weight_ptr+mask, mask=mask<M).to(tl.float32)[:,None]
        sigmoid = 1/(1+tl.exp(-x1))
        dw = tl.sum(x1*sigmoid*x2*g,1)
        tl.store(dw_ptr+mask, dw, mask=mask<M)
        dx1 = g*x2*w*sigmoid*(1+x1*tl.exp(-x1)* sigmoid)
        tl.store(dx_ptr+offs, dx1, mask=mask[:,None]<M)

        dx2 = g*x1*sigmoid*w
        tl.store(dx_ptr+offs+n, dx2, mask=mask[:,None]<M)

        offs += N*W
        hoffs += n*W


def triton_weighted_silu_backward(g, x, weight):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device 
    dw = torch.empty((M, 1), device=device, dtype=x.dtype)
    dx = torch.empty((M, N), device=device, dtype=x.dtype)
    W = 8192//N 
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    T = triton.cdiv(M, sm*W)
    grid = (sm, )
    weighted_silu_backward_kernel[grid](
        g,
        x,
        weight,
        dx,
        dw,
        M, T,
        N, 
        N//2,
        W,
        num_stages=3,
        num_warps=16
    )
    return dx,dw



@triton.jit
def weighted_silu_and_quant_forward_kernel(x_ptr, weight_ptr, smooth_scale_ptr, out_ptr, scale_ptr, M, T, N: tl.constexpr, n:tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)

    row_offs = pid*T*W*n+tl.arange(0, W)[:,None]*n
    col_offs = tl.arange(0, n)[None,:]
    smooth_scale = tl.load(smooth_scale_ptr+tl.arange(0, n))
    smooth_scale = 1.0/tl.maximum(smooth_scale, 1e-30)
    
    for i in range(T):
        indices = pid*T*W+i*W+tl.arange(0, W)
        mask = indices[:,None]<M
        x1 = tl.load(x_ptr+row_offs*2+col_offs, mask=mask).to(tl.float32)
        x2 = tl.load(x_ptr+n+row_offs*2+col_offs, mask=mask).to(tl.float32)
        w = tl.load(weight_ptr+indices, mask=indices<M).to(tl.float32)[:,None]
        x = x1/(1+tl.exp(-x1))*x2*w*smooth_scale
        scale = tl.maximum(tl.max(x.abs(), 1)/448, 1e-30)
        tl.store(scale_ptr+indices, scale, mask=indices<M)
        x = (x/scale[:,None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr+row_offs+col_offs, x, mask=mask)
        row_offs += n*W

# not used, shared expert used the triton_silu_and_quant_forward kernel
def triton_weighted_silu_and_quant_forward(x, weight, smooth_scale, out=None, scale=None):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device 
    if out is None:
        out = torch.empty((M, N//2), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.empty((M,), device=device, dtype=torch.float32)
    W = 8192//N 
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    T = triton.cdiv(M, sm*W)
    grid = (sm, )
    weighted_silu_and_quant_forward_kernel[grid](
        x,
        weight,
        smooth_scale,
        out,
        scale,
        M, T,
        N, 
        N//2,
        W,
        num_stages=2,
        num_warps=16
    )
    return out, scale



@triton.jit
def silu_and_quant_forward_kernel(x_ptr, smooth_scale_ptr, out_ptr, scale_ptr, M, T, N: tl.constexpr, n:tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)

    row_offs = pid*T*W*n+tl.arange(0, W)[:,None]*n
    col_offs = tl.arange(0, n)[None,:]
    smooth_scale = tl.load(smooth_scale_ptr+tl.arange(0, n))
    smooth_scale = 1.0/tl.maximum(smooth_scale, 1e-30)
    
    for i in range(T):
        indices = pid*T*W+i*W+tl.arange(0, W)
        mask = indices[:,None]<M
        x1 = tl.load(x_ptr+row_offs*2+col_offs, mask=mask).to(tl.float32)
        x2 = tl.load(x_ptr+n+row_offs*2+col_offs, mask=mask).to(tl.float32)
        x = x1/(1+tl.exp(-x1))*x2*smooth_scale
        scale = tl.maximum(tl.max(x.abs(), 1)/448, 1e-30)
        tl.store(scale_ptr+indices, scale, mask=indices<M)
        x = (x/scale[:,None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr+row_offs+col_offs, x, mask=mask)
        row_offs += n*W

# used in shared expert
def triton_silu_and_quant_forward(x, smooth_scale, out=None, scale=None):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device 
    if out is None:
        out = torch.empty((M, N//2), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.empty((M,), device=device, dtype=torch.float32)
    W = 8192//N 
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    T = triton.cdiv(M, sm*W)
    grid = (sm, )
    silu_and_quant_forward_kernel[grid](
        x,
        smooth_scale,
        out,
        scale,
        M, T,
        N, 
        N//2,
        W,
        num_stages=2,
        num_warps=16
    )
    return out, scale




@triton.jit
def weighted_silu_and_quant_and_calibrate_forward_kernel(x_ptr, weight_ptr, smooth_scale_ptr, out_ptr, scale_ptr, max_ptr, M, T, N: tl.constexpr, n:tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)

    row_offs = pid*T*W*n+tl.arange(0, W)[:,None]*n
    col_offs = tl.arange(0, n)[None,:]
    smooth_scale = tl.load(smooth_scale_ptr+tl.arange(0, n))
    smooth_scale = 1.0/tl.maximum(smooth_scale, 1e-30)
    maxs = tl.zeros((W,n), dtype=tl.float32) + 1e-30

    for i in range(T):
        indices = pid*T*W+i*W+tl.arange(0, W)
        mask = indices[:,None]<M
        x1 = tl.load(x_ptr+row_offs*2+col_offs, mask=mask).to(tl.float32)
        x2 = tl.load(x_ptr+n+row_offs*2+col_offs, mask=mask).to(tl.float32)
        w = tl.load(weight_ptr+indices, mask=indices<M).to(tl.float32)[:,None]
        x = x1/(1+tl.exp(-x1))*x2 
        maxs = tl.maximum(x, maxs)
        x = x*w*smooth_scale
        scale = tl.maximum(tl.max(x.abs(), 1)/448, 1e-30)
        tl.store(scale_ptr+indices, scale, mask=indices<M)
        x = (x/scale[:,None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr+row_offs+col_offs, x, mask=mask)
        row_offs += n*W

    maxs = tl.sqrt(tl.max(maxs,0))
    maxs = tl.where(maxs<4, 1.0, maxs)
    # tl.atomic_max(max_ptr+tl.arange(0, n), max_1)  # very slow
    tl.store(max_ptr+pid*n+tl.arange(0, n), maxs)



# not used, shared expert uses the triton_silu_and_quant_and_calibrate_forward
def triton_weighted_silu_and_quant_and_calibrate_forward(x, weight, smooth_scale, out=None, scale=None, maxs=None):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device 
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    if out is None:
        out = torch.empty((M, N//2), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.empty((M,), device=device, dtype=torch.float32)
    if maxs is None:
        maxs = torch.empty((sm,N//2), device=device, dtype=torch.float32)
    W = 8192//N 
    T = triton.cdiv(M, sm*W)
    grid = (sm, )
    weighted_silu_and_quant_and_calibrate_forward_kernel[grid](
        x,
        weight,
        smooth_scale,
        out,
        scale,
        maxs,
        M, T,
        N, 
        N//2,
        W,
        num_stages=2,
        num_warps=16
    )
    maxs = maxs.amax(0)
    return out, scale, maxs





@triton.jit
def silu_and_quant_and_calibrate_forward_kernel(x_ptr, smooth_scale_ptr, out_ptr, scale_ptr, max_ptr, M, T, N: tl.constexpr, n:tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)

    row_offs = pid*T*W*n+tl.arange(0, W)[:,None]*n
    col_offs = tl.arange(0, n)[None,:]
    smooth_scale = tl.load(smooth_scale_ptr+tl.arange(0, n))
    smooth_scale = 1.0/tl.maximum(smooth_scale, 1e-30)
    maxs = tl.zeros((W,n), dtype=tl.float32) + 1e-30

    for i in range(T):
        indices = pid*T*W+i*W+tl.arange(0, W)
        mask = indices[:,None]<M
        x1 = tl.load(x_ptr+row_offs*2+col_offs, mask=mask).to(tl.float32)
        x2 = tl.load(x_ptr+n+row_offs*2+col_offs, mask=mask).to(tl.float32)
        x = x1/(1+tl.exp(-x1))*x2 
        maxs = tl.maximum(x, maxs)
        x = x*smooth_scale
        scale = tl.maximum(tl.max(x.abs(), 1)/448, 1e-30)
        tl.store(scale_ptr+indices, scale, mask=indices<M)
        x = (x/scale[:,None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr+row_offs+col_offs, x, mask=mask)
        row_offs += n*W

    maxs = tl.sqrt(tl.max(maxs,0))
    maxs = tl.where(maxs<4, 1.0, maxs)
    # tl.atomic_max(max_ptr+tl.arange(0, n), max_1)  # very slow
    tl.store(max_ptr+pid*n+tl.arange(0, n), maxs)



# used in shared expert
def triton_silu_and_quant_and_calibrate_forward(x, smooth_scale, out=None, scale=None, maxs=None):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device 
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    if out is None:
        out = torch.empty((M, N//2), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.empty((M,), device=device, dtype=torch.float32)
    if maxs is None:
        maxs = torch.empty((sm,N//2), device=device, dtype=torch.float32)
    W = 8192//N 
    T = triton.cdiv(M, sm*W)
    grid = (sm, )
    silu_and_quant_and_calibrate_forward_kernel[grid](
        x,
        smooth_scale,
        out,
        scale,
        maxs,
        M, T,
        N, 
        N//2,
        W,
        num_stages=2,
        num_warps=16
    )
    maxs = maxs.amax(0)
    return out, scale, maxs



@triton.jit
def weighted_silu_and_quant_backward_kernel(g_ptr, x_ptr, weight_ptr, smooth_scale_ptr, dx_ptr, dx_scale_ptr, dw_ptr, M, T, N: tl.constexpr, n:tl.constexpr, W: tl.constexpr, REVERSE: tl.constexpr):
    pid = tl.program_id(axis=0)

    smooth_scale_1 = tl.load(smooth_scale_ptr+tl.arange(0, n))
    smooth_scale_2 = tl.load(smooth_scale_ptr+n+tl.arange(0, n))
    if not REVERSE:
        smooth_scale_1 = 1/tl.maximum(smooth_scale_1,1e-30)
        smooth_scale_2 = 1/tl.maximum(smooth_scale_2,1e-30)


    offs = pid*W*T*N+tl.arange(0, W)[:,None]*N+tl.arange(0, n)[None,:]
    hoffs = pid*W*T*n+tl.arange(0, W)[:,None]*n+tl.arange(0, n)[None,:]
    for i in range(T):
        mask = pid*W*T+i*W+tl.arange(0, W)
        x1 = tl.load(x_ptr+offs, mask=mask[:,None]<M).to(tl.float32)
        x2 = tl.load(x_ptr+offs+n, mask=mask[:,None]<M).to(tl.float32)
        g = tl.load(g_ptr+hoffs, mask=mask[:,None]<M).to(tl.float32)
        w = tl.load(weight_ptr+mask, mask=mask<M).to(tl.float32)[:,None]
        sigmoid = 1/(1+tl.exp(-x1))
        dw = tl.sum(x1*sigmoid*x2*g,1)
        tl.store(dw_ptr+mask, dw, mask=mask<M)
        dx1 = g*x2*w*sigmoid*(1+x1*tl.exp(-x1)* sigmoid)*smooth_scale_1
        dx2 = g*x1*sigmoid*w*smooth_scale_2

        scale = tl.maximum(tl.maximum(tl.max(dx1, 1), tl.max(dx2, 1)), 1e-30)/448
        dx1 = (dx1/scale[:,None]).to(dx_ptr.dtype.element_ty)
        dx2 = (dx2/scale[:,None]).to(dx_ptr.dtype.element_ty)

        tl.store(dx_scale_ptr+mask, scale, mask=mask<M)
        tl.store(dx_ptr+offs, dx1, mask=mask[:,None]<M)
        tl.store(dx_ptr+offs+n, dx2, mask=mask[:,None]<M)

        offs += N*W
        hoffs += n*W

# not used, shared expert use the triton_silu_and_quant_backward kernel
def triton_weighted_silu_and_quant_backward(g, x, weight, smooth_scale, reverse=True):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device 
    dx = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    dx_scale = torch.empty((M, ), device=device, dtype=torch.float32)
    dw = torch.empty((M, 1), device=device, dtype=x.dtype)
    W = 8192//N 
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    T = triton.cdiv(M, sm*W)
    grid = (sm, )
    weighted_silu_and_quant_backward_kernel[grid](
        g,
        x,
        weight,
        smooth_scale,
        dx,
        dx_scale,
        dw,
        M, T,
        N, 
        N//2,
        W,
        reverse,
        num_stages=3,
        num_warps=16
    )
    return dx, dx_scale, dw




@triton.jit
def silu_and_quant_backward_kernel(g_ptr, x_ptr, smooth_scale_ptr, dx_ptr, dx_scale_ptr, M, T, N: tl.constexpr, n:tl.constexpr, W: tl.constexpr, REVERSE: tl.constexpr):
    pid = tl.program_id(axis=0)

    smooth_scale_1 = tl.load(smooth_scale_ptr+tl.arange(0, n))
    smooth_scale_2 = tl.load(smooth_scale_ptr+n+tl.arange(0, n))
    if not REVERSE:
        smooth_scale_1 = 1/tl.maximum(smooth_scale_1,1e-30)
        smooth_scale_2 = 1/tl.maximum(smooth_scale_2,1e-30)


    offs = pid*W*T*N+tl.arange(0, W)[:,None]*N+tl.arange(0, n)[None,:]
    hoffs = pid*W*T*n+tl.arange(0, W)[:,None]*n+tl.arange(0, n)[None,:]
    for i in range(T):
        mask = pid*W*T+i*W+tl.arange(0, W)
        x1 = tl.load(x_ptr+offs, mask=mask[:,None]<M).to(tl.float32)
        x2 = tl.load(x_ptr+offs+n, mask=mask[:,None]<M).to(tl.float32)
        g = tl.load(g_ptr+hoffs, mask=mask[:,None]<M).to(tl.float32)
        sigmoid = 1/(1+tl.exp(-x1))
        dx1 = g*x2*sigmoid*(1+x1*tl.exp(-x1)* sigmoid)*smooth_scale_1
        dx2 = g*x1*sigmoid*smooth_scale_2

        scale = tl.maximum(tl.maximum(tl.max(dx1, 1), tl.max(dx2, 1)), 1e-30)/448
        dx1 = (dx1/scale[:,None]).to(dx_ptr.dtype.element_ty)
        dx2 = (dx2/scale[:,None]).to(dx_ptr.dtype.element_ty)

        tl.store(dx_scale_ptr+mask, scale, mask=mask<M)
        tl.store(dx_ptr+offs, dx1, mask=mask[:,None]<M)
        tl.store(dx_ptr+offs+n, dx2, mask=mask[:,None]<M)

        offs += N*W
        hoffs += n*W

# used in shared expert
def triton_silu_and_quant_backward(g, x, smooth_scale, reverse=True):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device 
    dx = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    dx_scale = torch.empty((M, ), device=device, dtype=torch.float32)
    W = 8192//N 
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    T = triton.cdiv(M, sm*W)
    grid = (sm, )
    silu_and_quant_backward_kernel[grid](
        g,
        x,
        smooth_scale,
        dx,
        dx_scale,
        M, T,
        N, 
        N//2,
        W,
        reverse,
        num_stages=3,
        num_warps=16
    )
    return dx, dx_scale



@triton.jit
def batch_weighted_silu_and_quant_forward_kernel(x_ptr, weight_ptr, smooth_scale_ptr, out_ptr, scale_ptr, count_ptr, accum_ptr, M, N: tl.constexpr, n:tl.constexpr, W: tl.constexpr):
    eid = tl.program_id(axis=0)
    tid = tl.program_id(axis=1)
    sm = tl.num_programs(axis=1)

    count = tl.load(count_ptr+eid)
    ei = tl.load(accum_ptr+eid)
    si = ei - count
    c = tl.cdiv(count, sm*W)

    row_offs = si*n + tid*c*W*n+tl.arange(0, W)[:,None]*n
    col_offs = tl.arange(0, n)[None,:]
    smooth_scale = tl.load(smooth_scale_ptr+n*eid+tl.arange(0, n))
    smooth_scale = 1.0/tl.maximum(smooth_scale, 1e-30)

    for i in range(c):
        indices = si + tid*c*W+i*W+tl.arange(0, W)
        mask = indices[:,None]<M
        x1 = tl.load(x_ptr+row_offs*2+col_offs, mask=mask).to(tl.float32)
        x2 = tl.load(x_ptr+n+row_offs*2+col_offs, mask=mask).to(tl.float32)

        w = tl.load(weight_ptr+indices, mask=indices<M).to(tl.float32)[:,None]
        x = x1/(1+tl.exp(-x1))*x2*w*smooth_scale
        scale = tl.maximum(tl.max(x, 1)/448, 1e-30)
        tl.store(scale_ptr+indices, scale, mask=indices<M)
        x = (x/scale[:,None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr+row_offs+col_offs, x, mask=mask)
        row_offs += n*W


# used in shared expert
def triton_batch_weighted_silu_and_quant_forward(x, weight, smooth_scale, counts, out=None, scale=None):
    # row-wise read, row-wise write
    M, N = x.shape
    n_experts = counts.shape[0]
    assert N <= 8192
    device = x.device 
    if out is None:
        out = torch.empty((M, N//2), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.empty((M,), device=device, dtype=torch.float32)
    accums = torch.cumsum(counts,0)
    W = 8192//N 
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    grid = (n_experts, sm)
    batch_weighted_silu_and_quant_forward_kernel[grid](
        x,
        weight,
        smooth_scale,
        out,
        scale,
        counts,
        accums,
        M,
        N, 
        N//2,
        W,
        num_stages=3,
        num_warps=16
    )
    return out, scale




@triton.jit
def batch_weighted_silu_and_quant_and_calibrate_forward_kernel(x_ptr, weight_ptr, smooth_scale_ptr, out_ptr, scale_ptr, max_ptr, count_ptr, accum_ptr, M, N: tl.constexpr, n:tl.constexpr, W: tl.constexpr):
    eid = tl.program_id(axis=0)
    tid = tl.program_id(axis=1)
    sm = tl.num_programs(axis=1)

    count = tl.load(count_ptr+eid)
    ei = tl.load(accum_ptr+eid)
    si = ei - count
    c = tl.cdiv(count, sm*W)

    row_offs = si*n + tid*c*W*n+tl.arange(0, W)[:,None]*n
    col_offs = tl.arange(0, n)[None,:]
    smooth_scale = tl.load(smooth_scale_ptr+n*eid+tl.arange(0, n))
    smooth_scale = 1.0/tl.maximum(smooth_scale, 1e-30)

    maxs = tl.zeros((W,n), dtype=tl.float32) + 1e-30

    for i in range(c):
        indices = si + tid*c*W+i*W+tl.arange(0, W)
        mask = indices[:,None]<M
        x1 = tl.load(x_ptr+row_offs*2+col_offs, mask=mask).to(tl.float32)
        x2 = tl.load(x_ptr+n+row_offs*2+col_offs, mask=mask).to(tl.float32)

        w = tl.load(weight_ptr+indices, mask=indices<M).to(tl.float32)[:,None]
        x = x1/(1+tl.exp(-x1))*x2  

        maxs = tl.maximum(x, maxs)

        x *= w*smooth_scale
        scale = tl.maximum(tl.max(x, 1)/448, 1e-30)
        tl.store(scale_ptr+indices, scale, mask=indices<M)
        x = (x/scale[:,None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr+row_offs+col_offs, x, mask=mask)
        row_offs += n*W

    maxs = tl.sqrt(tl.max(maxs,0))
    maxs = tl.where(maxs<4, 1.0, maxs)
    tl.store(max_ptr+eid*sm*n + tid*n+tl.arange(0, n), maxs)




@triton.jit
def batch_sum_kernel(x_ptr, out_ptr, M, N, W:tl.constexpr, H: tl.constexpr):
    eid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    maxs = tl.zeros((W,), dtype=tl.float32) + 1e-30
    c = tl.cdiv(M, H)
    offs = eid * M * N + bid * W + tl.arange(0, H)[:,None] * N + tl.arange(0,W)[None,:]
    for i in range(c):
        x = tl.load(x_ptr+offs, mask=i*H+tl.arange(0,H)[:,None]<M).to(tl.float32)
        maxs = tl.maximum(tl.max(x,0), maxs)
        offs += H*N
        
    tl.store(out_ptr+eid*N + bid*W+tl.arange(0, W), maxs)



# used in shared expert
def triton_batch_weighted_silu_and_quant_and_calibrate_forward(x, weight, smooth_scale, counts, out=None, scale=None, maxs=None):
    # row-wise read, row-wise write
    M, N = x.shape
    n_experts = counts.shape[0]
    assert N <= 8192
    device = x.device 
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    if out is None:
        out = torch.empty((M, N//2), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.empty((M,), device=device, dtype=torch.float32)
    if maxs is None:
        maxs = torch.empty((n_experts, N//2), device=device, dtype=torch.float32)
    tmp_maxs = torch.empty((n_experts, sm, N//2), device=device, dtype=torch.bfloat16)
    
    accums = torch.cumsum(counts, 0)
    W = 8192//N 
    grid = (n_experts, sm)
    batch_weighted_silu_and_quant_and_calibrate_forward_kernel[grid](
        x,
        weight,
        smooth_scale,
        out,
        scale,
        tmp_maxs,
        counts,
        accums,
        M,
        N, 
        N//2,
        W,
        num_stages=3,
        num_warps=16
    )
    # maxs = tmp_maxs.amax(1)

    T = 128//n_experts
    W = N//2//T
    H = 16
    grid = (n_experts, T)
    batch_sum_kernel[grid](tmp_maxs, 
                           maxs, 
                           sm, 
                           N//2, 
                           W, 
                           H)
    return out, scale, maxs




@triton.jit
def batch_weighted_silu_and_quant_backward_kernel(g_ptr, x_ptr, weight_ptr, smooth_scale_ptr, count_ptr, accum_ptr, dx_ptr, dx_scale_ptr, dw_ptr, M, N: tl.constexpr, n:tl.constexpr, W: tl.constexpr, REVERSE: tl.constexpr):
    eid = tl.program_id(axis=0)
    tid = tl.program_id(axis=1)
    sm = tl.num_programs(axis=1)

    count = tl.load(count_ptr+eid)
    ei = tl.load(accum_ptr+eid)
    si = ei - count
    c = tl.cdiv(count, sm*W)

    smooth_scale_1 = tl.load(smooth_scale_ptr+n*eid*2+tl.arange(0, n))
    smooth_scale_2 = tl.load(smooth_scale_ptr+n*eid*2+n+tl.arange(0, n))
    if not REVERSE:
        smooth_scale_1 = 1/tl.maximum(smooth_scale_1,1e-30)
        smooth_scale_2 = 1/tl.maximum(smooth_scale_2,1e-30)

    offs = si*N + tid*c*W*N +tl.arange(0, W)[:,None]*N+tl.arange(0, n)[None,:]
    hoffs = si*n + tid*c*W*n+tl.arange(0, W)[:,None]*n+tl.arange(0, n)[None,:]
    for i in range(c):
        mask = si + tid*c*W +i*W+tl.arange(0, W)
        x1 = tl.load(x_ptr+offs, mask=mask[:,None]<M).to(tl.float32)
        x2 = tl.load(x_ptr+offs+n, mask=mask[:,None]<M).to(tl.float32)
        g = tl.load(g_ptr+hoffs, mask=mask[:,None]<M).to(tl.float32)
        w = tl.load(weight_ptr+mask, mask=mask<M).to(tl.float32)[:,None]
        sigmoid = 1/(1+tl.exp(-x1))
        dw = tl.sum(x1*sigmoid*x2*g,1)
        tl.store(dw_ptr+mask, dw, mask=mask<M)
        dx1 = g*x2*w*sigmoid*(1+x1*tl.exp(-x1)* sigmoid)*smooth_scale_1
        dx2 = g*x1*sigmoid*w*smooth_scale_2

        scale = tl.maximum(tl.maximum(tl.max(dx1, 1), tl.max(dx2, 1)), 1e-30)/448
        dx1 = (dx1/scale[:,None]).to(dx_ptr.dtype.element_ty)
        dx2 = (dx2/scale[:,None]).to(dx_ptr.dtype.element_ty)

        tl.store(dx_scale_ptr+mask, scale, mask=mask<M)
        tl.store(dx_ptr+offs, dx1, mask=mask[:,None]<M)
        tl.store(dx_ptr+offs+n, dx2, mask=mask[:,None]<M)

        offs += N*W
        hoffs += n*W

# used in routed experts
def triton_batch_weighted_silu_and_quant_backward(g, x, weight, smooth_scales, counts, reverse=True):
    # row-wise read, row-wise write
    M, N = x.shape
    n_expert = counts.shape[0]
    assert 128%n_expert == 0
    assert N <= 8192
    device = x.device 
    dx = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    dx_scale = torch.empty((M, ), device=device, dtype=torch.float32)
    dw = torch.empty((M, 1), device=device, dtype=x.dtype)
    accums = torch.cumsum(counts,0)
    
    W = 8192//N 
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    grid = (n_expert, sm)
    batch_weighted_silu_and_quant_backward_kernel[grid](
        g,
        x,
        weight,
        smooth_scales,
        counts,
        accums,
        dx,
        dx_scale,
        dw,
        M,
        N, 
        N//2,
        W,
        reverse,
        num_stages=3,
        num_warps=16
    )
    return dx, dx_scale, dw



