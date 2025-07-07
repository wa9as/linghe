import math
import os
import torch
import triton
import triton.language as tl
from triton import Config


# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

@triton.jit
def rms_norm_forward_kernel(x_ptr, weight_ptr, out_ptr, norm_ptr, eps, M, T, N: tl.constexpr, W: tl.constexpr, OUTPUT: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    weight = tl.load(weight_ptr+tl.arange(0, N))[None,:]

    offs = pid*W*T*N+tl.arange(0, W)[:,None]*N+tl.arange(0, N)[None,:]
    for i in range(T):
        x = tl.load(x_ptr+offs, mask=pid*W*T+i*W+tl.arange(0, W)[:,None]<M).to(tl.float32)
        rms = tl.sqrt(tl.sum(x*x, axis=1)/N+eps)
        rms = 1/rms
        if OUTPUT:
            tl.store(norm_ptr+pid*W*T+i*W+tl.arange(0, W), rms, mask=pid*W*T+i*W+tl.arange(0, W)<M)

        x = (x*rms[:,None])*weight

        tl.store(out_ptr+offs, x, mask=pid*W*T+i*W+tl.arange(0, W)[:,None]<M)
        offs += N*W

# rms is stored as 1/rms
def triton_rms_norm_forward(x, weight, eps=1e-6, out=None, rms=None, output_rms=False):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device 
    if out is None:
        out = torch.empty((M, N), device=device, dtype=x.dtype)
    if rms is None and output_rms:
        rms = torch.empty((M,), device=device, dtype=torch.float32)
    W = 8192//N 
    T = triton.cdiv(M, 132*W)
    grid = lambda META: (132, )
    rms_norm_forward_kernel[grid](
        x,
        weight,
        out,
        rms,
        eps,
        M, T,
        N, 
        W,
        output_rms,
        num_stages=3,
        num_warps=16
    )
    return out, rms



@triton.jit
def depracated_rms_norm_backward_kernel(
    dx_ptr, dy_ptr, x_ptr, weight_ptr, norm_ptr, M, N: tl.constexpr, BLOCK_N: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return
        
    row_start = row_idx * N
    norm = tl.load(norm_ptr + row_idx).to(tl.float32)
    
    dot = 0.0
    for offset in range(0, N, BLOCK_N):
        col_offsets = offset + tl.arange(0, BLOCK_N)
        mask = col_offsets < N
        
        x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        weighted_dy = dy * weight
        dot += tl.sum(weighted_dy * x)
    
    factor = norm * norm * norm * (dot / N)
    
    for offset in range(0, N, BLOCK_N):
        col_offsets = offset + tl.arange(0, BLOCK_N)
        mask = col_offsets < N
        
        x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        weighted_dy = dy * weight
        dx = norm * weighted_dy - x * factor
        tl.store(dx_ptr + row_start + col_offsets, dx, mask=mask)



def triton_depracated_rms_norm_backward(dy, x, weight, norm):
    M, N = x.shape
    dx = torch.empty(M, N, dtype=torch.float32, device=x.device)
    BLOCK_N = min(triton.next_power_of_2(N), 1024)

    grid = lambda META: (M,)
    depracated_rms_norm_backward_kernel[grid](
        dx,
        dy,
        x,
        weight,
        norm,
        M, N,
        BLOCK_N,
        num_stages=3,
        num_warps=8
    )
    
    return dx, torch.sum(dy * x * norm.unsqueeze(-1), dim=0)





@triton.jit
def rms_norm_backward_kernel(
    grad_output_ptr,
    x_ptr,
    w_ptr,
    dx_ptr,
    dw_ptr,
    eps,
    M,
    T,
    N: tl.constexpr
):
    pid = tl.program_id(0)

    w = tl.load(w_ptr+tl.arange(0, N))

    offsets = pid*T*N+tl.arange(0, N)
    w_grads = tl.zeros((N,), dtype=tl.float32)
    for i in range(T):
        mask = pid*T+i<M
        x = tl.load(x_ptr+offsets, mask=mask).to(tl.float32)
        g = tl.load(grad_output_ptr+offsets, mask=mask)
        rms = tl.sqrt(tl.sum(x*x)/N+eps)
        r = 1.0/rms
        w_grad = x * g * r
        w_grads += w_grad

        dx = r*g*w - r*r*r/N*x*tl.sum(x*g*w)
        
        tl.store(dx_ptr+offsets, dx, mask=mask)

        offsets += N
    
    tl.store(dw_ptr+pid*N+tl.arange(0, N), w_grads)


def triton_rms_norm_backward(grad_output, x, w, eps=1e-6):
    M, N = x.shape
    dx = torch.empty(M, N, dtype=x.dtype, device=x.device)

    T = triton.cdiv(M, 132)
    tmp_dw = torch.empty(132, N, dtype=torch.float32, device=x.device)

    grid = lambda META: (132, )
    rms_norm_backward_kernel[grid](
        grad_output,
        x,
        w,
        dx,
        tmp_dw,
        eps,
        M,
        T,
        N,
        num_stages=3,
        num_warps=16
    )
    return dx, tmp_dw.sum(dim=0).to(x.dtype)
    



@triton.jit
def rms_norm_and_quant_forward_kernel(x_ptr, weight_ptr, smooth_scale_ptr, out_ptr, scale_ptr, max_ptr, rms_ptr, eps, M, T, N: tl.constexpr, W: tl.constexpr, CALIBRATE: tl.constexpr, OUTPUT: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    weight = tl.load(weight_ptr+tl.arange(0, N))[None,:]
    smooth_scale = tl.load(smooth_scale_ptr+tl.arange(0, N))[None,:]
    smooth_scale = 1.0/tl.maximum(smooth_scale, 1e-37)
    if CALIBRATE:
        maxs = tl.zeros((W,N), dtype=tl.float32) + 1e-37
    offs = pid*W*T*N+tl.arange(0, W)[:,None]*N+tl.arange(0, N)[None,:]
    for i in range(T):
        indices = pid*W*T+i*W+tl.arange(0, W)
        x = tl.load(x_ptr+offs, mask=indices[:,None]<M).to(tl.float32)
        if CALIBRATE:
            maxs = tl.maximum(maxs, x)
        rms = tl.sqrt(tl.sum(x*x, axis=1)/N+eps)
        rms = 1/rms
        if OUTPUT:
            tl.store(rms_ptr+indices, rms, mask=indices<M)
        x = (x*rms[:,None])*(weight*smooth_scale)
        scale = tl.maximum(tl.max(tl.abs(x),1)/448.0, 1e-37)
        x = (x/scale).to(out_ptr.dtype.element_ty)
        tl.store(scale_ptr+indices, scale, mask=indices<M)
        tl.store(out_ptr+offs, x, mask=indices[:,None]<M)
        offs += N*W
    
    if CALIBRATE:
        maxs = tl.max(maxs, 0)
        maxs = tl.sqrt(tl.max(maxs,0))
        maxs = tl.where(maxs<4, 1.0, maxs)
        tl.store(max_ptr+tl.arange(0, N), maxs)


# rms is used for moe routing, it is stored as 1/rms
def triton_rms_norm_and_quant_forward(x, weight, smooth_scale, eps=1e-6, out=None, scale=None, calibrate=False, output_rms=False):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device 
    if out is None:
        out = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.empty((M, ), device=device, dtype=torch.float32)
    if calibrate:
        maxs = torch.empty((N,), dtype=torch.float32, device=device)
    else:
        maxs = None
    if output_rms:
        rms = torch.empty((M,), dtype=torch.float32, device=device)
    else:
        rms = None
    W = 8192//N 
    T = triton.cdiv(M, 132*W)
    grid = lambda META: (132, )
    rms_norm_and_quant_forward_kernel[grid](
        x,
        weight,
        smooth_scale,
        out,
        scale,
        maxs,
        rms,
        eps,
        M, 
        T,
        N, 
        W,
        calibrate,
        output_rms,
        num_stages=3,
        num_warps=16
    )
    return out,scale,maxs,rms



