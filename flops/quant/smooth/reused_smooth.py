
import math

import torch
import triton
import triton.language as tl
from triton import Config
from flops.quant.channel.channel import row_quant_kernel
from flops.utils.transpose import triton_transpose,triton_block_transpose,triton_block_pad_transpose
from flops.utils.util import round_up



@triton.jit
def reused_smooth_quant_kernel(x_ptr, q_ptr, ss_ptr, qs_ptr, M, N, H: tl.constexpr, W: tl.constexpr, EVEN: tl.constexpr, REVERSE: tl.constexpr, ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    offs = pid*W*N + tl.arange(0, W)[:,None]*N + tl.arange(0, H)[None,:]
    soffs = tl.arange(0, H)
    x_max = tl.zeros((W,),dtype=tl.float32) + 1e-30
    n = tl.cdiv(N, H)
    for i in range(n):
        if EVEN:
            x = tl.load(x_ptr+offs)
            smooth_scale = tl.load(ss_ptr+soffs)
        else:
            # x = tl.load(x_ptr+offs, mask=(i*H+tl.arange(0, H)[None,:]<N)&(pid*W+tl.arange(0, W)[:,None]<M))
            x = tl.load(x_ptr+offs, mask=pid*W+tl.arange(0, W)[:,None]<M)
            other = 0.0 if REVERSE else 1e30
            smooth_scale = tl.load(ss_ptr+soffs, mask=soffs<N, other=other)
        if REVERSE:
            x = x.to(tl.float32) * smooth_scale
        else:
            x = x.to(tl.float32) / smooth_scale
        x_max = tl.maximum(tl.max(tl.abs(x), axis=1),x_max)
        offs += H 
        soffs += H

    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(x_max/448.0)))
    else:
        scale = x_max/448.0

    # paddings should be filled with 0
    tl.store(qs_ptr+pid*W+tl.arange(0, W), scale)
        
    s = (1.0/scale)[:,None]

    offs = pid*W*N + tl.arange(0, W)[:,None]*N + tl.arange(0, H)[None,:]
    soffs = tl.arange(0, H)
    for i in range(n):
        if EVEN:
            x = tl.load(x_ptr+offs)
            smooth_scale = tl.load(ss_ptr+soffs)
        else:
            # x = tl.load(x_ptr+offs, mask=(i*H+tl.arange(0, H)[None,:]<N)&(pid*W+tl.arange(0, W)[:,None]<M))
            x = tl.load(x_ptr+offs, mask=pid*W+tl.arange(0, W)[:,None]<M)
            other = 0.0 if REVERSE else 1e30
            smooth_scale = tl.load(ss_ptr+soffs, mask=soffs<N, other=other)

        if REVERSE:
            xq = (x.to(tl.float32) * smooth_scale * s).to(q_ptr.dtype.element_ty)
        else:
            xq = (x.to(tl.float32) / smooth_scale * s).to(q_ptr.dtype.element_ty)

        if EVEN:
            tl.store(q_ptr+offs, xq)
        else:
            # tl.store(q_ptr+offs, xq, mask=(i*H+tl.arange(0, H)[None,:]<N)&(pid*W+tl.arange(0, W)[:,None]<M))
            tl.store(q_ptr+offs, xq, mask=pid*W+tl.arange(0, W)[:,None]<M)
        offs += H 
        soffs += H


def triton_reused_smooth_quant(x, smooth_scale, x_q=None, x_scale=None, reverse=False, pad_scale=False, round_scale=False):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N%1024 == 0
    assert pad_scale or M%32==0, "scale should be padding to be multiple of 32"
    device = x.device 
    if x_q is None:
        x_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    P = round_up(M, b=32) if pad_scale else M
    if x_scale is None:
        x_scale = torch.empty((P,), device=device, dtype=torch.float32)
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    W = 8 if M <= sm*8 else 16
    H = 1024 if W == 8 else 512
    EVEN = M%W == 0 and P==M
    grid = (triton.cdiv(P, W), )
    reused_smooth_quant_kernel[grid](
        x,
        x_q,
        smooth_scale,
        x_scale,
        M, N,
        H, W,
        EVEN,
        reverse,
        round_scale,
        num_stages=5,
        num_warps=4
    )

    return x_q,x_scale



@triton.jit
def depracated_tokenwise_reused_smooth_quant_kernel(x_ptr, q_ptr, ss_ptr, qs_ptr, M, P, W, N: tl.constexpr, REVERSE: tl.constexpr, ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    smooth_scale = tl.load(ss_ptr+tl.arange(0, N))
    if not REVERSE:
        smooth_scale = 1.0/smooth_scale
    
    for i in range(W):
        x = tl.load(x_ptr+pid*W*N+i*N+tl.arange(0, N), mask=pid*W+i<M).to(tl.float32)
        x *= smooth_scale
        x_max = tl.maximum(tl.max(tl.abs(x)), 1e-30)

        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(x_max/448.0)))
        else:
            scale = x_max/448.0

        s = 1.0/scale
        x *= s
        xq = x.to(q_ptr.dtype.element_ty)
        tl.store(qs_ptr+pid*W+i, scale, mask=pid*W+i<P)
        tl.store(q_ptr+pid*W*N+i*N+tl.arange(0, N), xq, mask=pid*W+i<M)


def triton_depracated_tokenwise_reused_smooth_quant(x, smooth_scale, x_q=None, x_scale=None, reverse=False, pad_scale=False, round_scale=False):
    # row-wise read, row-wise write
    M, N = x.shape
    device = x.device 
    if x_q is None:
        x_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    P = round_up(M, b=32) if pad_scale else M
    assert P%32==0, "scale should be multiple of 32"
    if x_scale is None:
        x_scale = torch.empty((P,), device=device, dtype=torch.float32)
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    W = triton.cdiv(P, sm)
    grid = (sm, )
    depracated_tokenwise_reused_smooth_quant_kernel[grid](
        x,
        x_q,
        smooth_scale,
        x_scale,
        M, P, W,
        N, 
        reverse,
        round_scale,
        num_stages=3,
        num_warps=8
    )
    return x_q,x_scale


@triton.jit
def tokenwise_reused_smooth_quant_kernel(x_ptr, q_ptr, ss_ptr, qs_ptr, M, P, T, N: tl.constexpr, W: tl.constexpr, REVERSE: tl.constexpr, ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    smooth_scale = tl.load(ss_ptr+tl.arange(0, N))[None,:]
    if not REVERSE:
        smooth_scale = 1.0/smooth_scale
    
    for i in range(T):
        x = tl.load(x_ptr+pid*W*T*N+i*N*W+tl.arange(0, W)[:,None]*N+tl.arange(0, N)[None,:], mask=pid*W*T+i*W+tl.arange(0, W)[:,None]<M).to(tl.float32)
        x *= smooth_scale
        x_max = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)

        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(x_max/448.0)))
        else:
            scale = x_max/448.0

        s = 1.0/scale
        x *= s[:,None]
        xq = x.to(q_ptr.dtype.element_ty)
        tl.store(qs_ptr+pid*W*T+i*W+tl.arange(0, W), scale, mask=pid*W*T+i*W+tl.arange(0, W)<P)
        tl.store(q_ptr+pid*W*T*N+i*N*W+tl.arange(0, W)[:,None]*N+tl.arange(0, N)[None,:], xq, mask=pid*W*T+i*W+tl.arange(0, W)[:,None]<M)


def triton_tokenwise_reused_smooth_quant(x, smooth_scale, x_q=None, x_scale=None, reverse=False, pad_scale=False, round_scale=False):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device 
    if x_q is None:
        x_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    P = round_up(M, b=32) if pad_scale else M
    assert P%32==0, "scale should be multiple of 32"
    if x_scale is None:
        x_scale = torch.empty((P,), device=device, dtype=torch.float32)
    W = 8192//N 
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    T = triton.cdiv(P, sm*W)
    grid = (sm, )
    tokenwise_reused_smooth_quant_kernel[grid](
        x,
        x_q,
        smooth_scale,
        x_scale,
        M, P, T,
        N, 
        W,
        reverse,
        round_scale,
        num_stages=3,
        num_warps=32
    )
    return x_q,x_scale





@triton.jit
def batch_smooth_quant_kernel(x_ptr, q_ptr, ss_ptr, qs_ptr, xm_ptr, count_ptr, accum_ptr, T, N: tl.constexpr, REVERSE: tl.constexpr, ROUND: tl.constexpr, CALIBRATE: tl.constexpr):
    pid = tl.program_id(axis=0)

    i_expert = pid//T 
    i_batch = pid%T

    # row-wise read, row-wise write
    smooth_scale = tl.load(ss_ptr+i_expert*N+tl.arange(0, N))
    if not REVERSE:
        smooth_scale = 1.0/smooth_scale

    if CALIBRATE:
        x_maxs = tl.zeros((N,),dtype=tl.float32) + 1e-30

    count = tl.load(count_ptr+i_expert)
    ei = tl.load(accum_ptr+i_expert)
    si = ei - count

    n = (count-1)//T+1  # samples for each task
    for i in range(i_batch*n, min((i_batch+1)*n, count)):
        x = tl.load(x_ptr + si*N + i*N + tl.arange(0, N)).to(tl.float32)
        if CALIBRATE:
            x_maxs = tl.maximum(x_maxs, x)
        x *= smooth_scale
        x_max = tl.maximum(tl.max(tl.abs(x)), 1e-30)

        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(x_max/448.0)))
        else:
            scale = x_max/448.0

        tl.store(qs_ptr+si+i, scale)

        s = 1.0/scale
        x *= s
        xq = x.to(q_ptr.dtype.element_ty)
        tl.store(q_ptr+si*N+i*N+tl.arange(0, N), xq)

    if CALIBRATE:
        x_maxs = tl.sqrt(x_maxs)
        x_maxs = tl.where(x_maxs<4, 1.0, x_maxs)
        tl.store(xm_ptr+pid*N+tl.arange(0, N), x_maxs)


"""
select and smooth and quant
x: [bs, dim]
smooth_scales: [n_experts, dim]
token_count_per_expert: [n_experts]
x_q: [bs, dim]
x_scale: [bs]
"""
def triton_batch_smooth_quant(x, smooth_scales, token_count_per_expert, x_q=None, x_scale=None, x_maxs=None, reverse=False, round_scale=False, calibrate=False):
    # row-wise read, row-wise write
    M, N = x.shape
    device = x.device 
    n_expert = token_count_per_expert.shape[0]
    assert 128 % n_expert == 0
    if x_q is None:
        x_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    if x_scale is None:
        x_scale = torch.empty((M,), device=device, dtype=torch.float32)
    accum_token_count = torch.cumsum(token_count_per_expert, 0)
    T = 128//n_expert
    if calibrate and x_maxs is None:
        x_maxs = torch.empty((128,N), device=device, dtype=torch.float32)

    grid = (128, )
    batch_smooth_quant_kernel[grid](
        x,
        x_q,
        smooth_scales,
        x_scale,
        x_maxs,
        token_count_per_expert,
        accum_token_count,
        T, N,
        reverse,
        round_scale,
        calibrate,
        num_stages=3,
        num_warps=8
    )
    if calibrate:
        x_maxs = x_maxs.view(n_expert, T, N).amax(1)
    return x_q,x_scale,x_maxs




# @triton.jit
# def batch_calibrate_kernel(x_ptr, q_ptr, ss_ptr, qs_ptr, count_ptr, accum_ptr, T, N: tl.constexpr, REVERSE: tl.constexpr, ROUND: tl.constexpr):
#     pid = tl.program_id(axis=0)

#     i_expert = pid//T 
#     i_batch = pid%T

#     # row-wise read, row-wise write
#     smooth_scale = tl.load(ss_ptr+i_expert*N+tl.arange(0, N))
#     if not REVERSE:
#         smooth_scale = 1.0/smooth_scale

#     count = tl.load(count_ptr+i_expert)
#     ei = tl.load(accum_ptr+i_expert)
#     si = ei - count

#     n = (count-1)//T+1  # samples for each task
#     for i in range(i_batch*n, min((i_batch+1)*n, count)):
#         x = tl.load(x_ptr + si*N + i*N + tl.arange(0, N)).to(tl.float32)
#         x *= smooth_scale
#         x_max = tl.maximum(tl.max(tl.abs(x)), 1e-30)

#         if ROUND:
#             scale = tl.exp2(tl.ceil(tl.log2(x_max/448.0)))
#         else:
#             scale = x_max/448.0

#         tl.store(qs_ptr+si+i, scale)

#         s = 1.0/scale
#         x *= s
#         xq = x.to(q_ptr.dtype.element_ty)
#         tl.store(q_ptr+si*N+i*N+tl.arange(0, N), xq)

# """
# select and smooth and quant
# x: [bs, dim]
# smooth_scales: [n_experts, dim]
# token_count_per_expert: [n_experts]
# x_q: [bs, dim]
# x_scale: [bs]
# """
# def triton_batch_calibrate(x, token_count_per_expert, x_m=None):
#     # row-wise read, row-wise write
#     M, N = x.shape
#     device = x.device 
#     n_expert = token_count_per_expert.shape[0]
#     assert 128 % n_expert == 0
#     if x_m is None:
#         x_m = torch.empty((n_expert, N), device=device, dtype=torch.float32)
#     accum_token_count = torch.cumsum(token_count_per_expert, 0)
#     T = 128//n_expert
#     assert N % T == 0
#     W = N//T

#     grid = (128, )
#     batch_calibrate_kernel[grid](
#         x,
#         x_m,
#         token_count_per_expert,
#         accum_token_count,
#         T, W,
#         num_stages=3,
#         num_warps=8
#     )
#     return x_m


@triton.jit
def reused_transpose_pad_smooth_quant_kernel(x_ptr, q_ptr, ss_ptr, qs_ptr, M, N, P, H: tl.constexpr, W: tl.constexpr, EVEN: tl.constexpr, REVERSE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # col-wise read, row-wise write
    offs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:]
    soffs = tl.arange(0, H)
    x_max = tl.zeros((W,),dtype=tl.float32) + 1e-30
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

    scale = x_max/448.0
    if EVEN:
        tl.store(qs_ptr+pid*W+tl.arange(0, W), scale)
    else:
        tl.store(qs_ptr+pid*W+tl.arange(0, W), scale, mask=pid*W+tl.arange(0,W)<N)


    s = (1.0/scale)[:,None]
    offs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:]
    soffs = tl.arange(0, H)
    toffs = pid*W*P + tl.arange(0, W)[:,None]*P + tl.arange(0, H)[None,:]
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
            # mask with P instead of M
            tl.store(q_ptr+toffs, x, mask=(i*H+tl.arange(0,H)[None,:]<P) & (pid*W+tl.arange(0,W)[:,None]<N))
        offs += H*N
        toffs += H
        soffs += H



def triton_reused_transpose_pad_smooth_quant(x, smooth_scale, reverse=False, pad=False):
    # col-wise read, row-wise write
    # M should be padded
    M, N = x.shape
    device = x.device 
    P = round_up(M, b=32) if pad else M
    x_q = torch.empty((N, P), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((N,), device=device, dtype=torch.float32)
    H = 256
    W = 32
    EVEN = M%H == 0 and N%W == 0 

    grid = (triton.cdiv(N, W), )
    reused_transpose_pad_smooth_quant_kernel[grid](
        x,
        x_q,
        smooth_scale,
        x_scale,
        M, N, P,
        H, W, 
        EVEN, reverse,
        num_stages=3,
        num_warps=4
    )

    return x_q,x_scale



@triton.jit
def reused_transpose_pad_rescale_smooth_quant_kernel(x_ptr, q_ptr, org_smooth_scale_ptr, org_quant_scale_ptr, transpose_smooth_scale_ptr, transpose_quant_scale_ptr, M, N, P, H: tl.constexpr, W: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(axis=0)
    # col-wise read, row-wise write
    offs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:]
    soffs = tl.arange(0, H)
    x_max = tl.zeros((W,),dtype=tl.float32) + 1e-30
    if EVEN:
        org_smooth_scale = tl.load(org_smooth_scale_ptr + pid*W + tl.arange(0, W))[None,:]
    else:
        org_smooth_scale = tl.load(org_smooth_scale_ptr + pid*W + tl.arange(0, W), mask=pid*W + tl.arange(0, W)<N, other=1e30)[None,:]

    m = tl.cdiv(M, H)
    for i in range(m):
        if EVEN:
            x = tl.load(x_ptr+offs)
            org_quant_scale = tl.load(org_quant_scale_ptr+soffs)[:,None]
            transpose_smooth_scale = tl.load(transpose_smooth_scale_ptr+soffs)[:,None]
        else:
            x = tl.load(x_ptr+offs, mask=(i*H+tl.arange(0,H)[:,None]<M) & (pid*W+tl.arange(0,W)[None,:]<N))
            org_quant_scale = tl.load(org_quant_scale_ptr+soffs, mask=soffs<M, other=0.0)[:,None]
            transpose_smooth_scale = tl.load(transpose_smooth_scale_ptr+soffs, mask=soffs<M, other=0.0)[:,None]

        x = x.to(tl.float32) / org_smooth_scale * (org_quant_scale*transpose_smooth_scale)
        x_max = tl.maximum(tl.max(tl.abs(x), axis=0),x_max)
        offs += H*N 
        soffs += H

    scale = tl.exp2(tl.ceil(tl.log2(x_max/448.0)))
    if EVEN:
        tl.store(transpose_quant_scale_ptr+pid*W+tl.arange(0, W), scale)
    else:
        tl.store(transpose_quant_scale_ptr+pid*W+tl.arange(0, W), scale, mask=pid*W+tl.arange(0,W)<N)

    s = (1.0/scale)[None,:]

    offs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:]
    soffs = tl.arange(0, H)
    toffs = pid*W*P + tl.arange(0, W)[:,None]*P + tl.arange(0, H)[None,:]
    for i in range(m):

        if EVEN:
            x = tl.load(x_ptr+offs)
            org_quant_scale = tl.load(org_quant_scale_ptr+soffs)[:,None]
            transpose_smooth_scale = tl.load(transpose_smooth_scale_ptr+soffs)[:,None]
        else:
            x = tl.load(x_ptr+offs, mask=(i*H+tl.arange(0,H)[:,None]<M) & (pid*W+tl.arange(0,W)[None,:]<N))
            org_quant_scale = tl.load(org_quant_scale_ptr+soffs, mask=soffs<M, other=0.0)[:,None]
            transpose_smooth_scale = tl.load(transpose_smooth_scale_ptr+soffs, mask=soffs<M, other=0.0)[:,None]

        x = x.to(tl.float32) * (s / org_smooth_scale) * (org_quant_scale*transpose_smooth_scale)
        x = tl.maximum(tl.minimum(x, 448.0), -448.0)
        x = tl.trans(x.to(q_ptr.dtype.element_ty))
        if EVEN:
            tl.store(q_ptr+toffs, x)
        else:
            tl.store(q_ptr+toffs, x, mask=(i*H+tl.arange(0,H)[None,:]<P) & (pid*W+tl.arange(0,W)[:,None]<N))
        offs += H*N
        toffs += H
        soffs += H


"""
x_q is colwise smooth and rowwise quant
org_smooth_scale and transpose_smooth_scale is reversed
smooth scale and quant scale should be power of 2
step: dequant x_q -> apply smooth scale -> quant -> transpose -> pad
implement: x_q/org_smooth_scale*(org_quant_scale*smooth_scale) -> colwise quant and transpose
"""
def triton_reused_transpose_pad_rescale_smooth_quant(x_q, org_smooth_scale, org_quant_scale, transpose_smooth_scale, reverse=True, pad=False):
    # col-wise read, row-wise write
    assert reverse
    M, N = x_q.shape
    device = x_q.device 
    P = round_up(M, b=32) if pad else M
    xt_q = torch.empty((N, P), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((N,), device=device, dtype=torch.float32)
    H = 256
    W = 16
    EVEN = P == M and M%H == 0 and N%W == 0 

    grid = (triton.cdiv(N, W), )
    reused_transpose_pad_rescale_smooth_quant_kernel[grid](
        x_q,
        xt_q,
        org_smooth_scale,
        org_quant_scale,
        transpose_smooth_scale,
        x_scale,
        M, N, P,
        H, W, 
        EVEN,
        num_stages=4,
        num_warps=8
    )

    return xt_q,x_scale





@triton.jit
def index_select_smooth_quant_kernel(x_ptr, q_ptr, ss_ptr, qs_ptr, count_ptr, accum_ptr, index_ptr, M, N: tl.constexpr, REVERSE: tl.constexpr, ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    smooth_scale = tl.load(ss_ptr+pid*N+tl.arange(0, N))
    if not REVERSE:
        smooth_scale = 1.0/smooth_scale
    count = tl.load(count_ptr+pid)
    ei = tl.load(accum_ptr+pid)
    si = ei - count
    for i in range(count):
        index = tl.load(index_ptr+si+i)
        x = tl.load(x_ptr+ index*N+tl.arange(0, N)).to(tl.float32)
        x *= smooth_scale
        x_max = tl.maximum(tl.max(tl.abs(x)), 1e-30)

        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(x_max/448.0)))
        else:
            scale = x_max/448.0

        tl.store(qs_ptr+si+i, scale)

        s = 1.0/scale
        x *= s
        xq = x.to(q_ptr.dtype.element_ty)
        tl.store(q_ptr+si*N+i*N+tl.arange(0, N), xq)

"""
select and smooth and quant
x: [bs, dim]
smooth_scales: [n_experts, dim]
indices: [n_experts*topk]
x_q: [bs*topk, dim]
x_scale: [bs*topk]
"""
def triton_index_select_smooth_quant(x, smooth_scales, token_count_per_expert, indices, x_q=None, x_scale=None, reverse=False, round_scale=False):
    # row-wise read, row-wise write
    M, N = x.shape
    n_expert = smooth_scales.shape[0]
    E = indices.shape[0]
    device = x.device 
    if x_q is None:
        x_q = torch.empty((E, N), device=device, dtype=torch.float8_e4m3fn)
    if x_scale is None:
        x_scale = torch.empty((E,), device=device, dtype=torch.float32)
    accum_token_count = torch.cumsum(token_count_per_expert, 0)
    # TODO: adapt for n_expert <= 64
    grid = (n_expert, )
    index_select_smooth_quant_kernel[grid](
        x,
        x_q,
        smooth_scales,
        x_scale,
        token_count_per_expert,
        accum_token_count,
        indices,
        M, N, 
        reverse,
        round_scale,
        num_stages=3,
        num_warps=8
    )
    return x_q,x_scale




@triton.jit
def index_select_smooth_quant_and_sum_kernel(grads_ptr, tokens_ptr, q_ptr, ss_ptr, qs_ptr, count_ptr, accum_ptr, index_ptr, sum_ptr, M, N: tl.constexpr, REVERSE: tl.constexpr, ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    smooth_scale = tl.load(ss_ptr+pid*N+tl.arange(0, N))
    if not REVERSE:
        smooth_scale = 1.0/smooth_scale
    count = tl.load(count_ptr+pid)
    ei = tl.load(accum_ptr+pid)
    si = ei - count
    for i in range(count):
        index = tl.load(index_ptr+si+i)
        x = tl.load(grads_ptr + index*N+tl.arange(0, N)).to(tl.float32)
        t = tl.load(tokens_ptr+si*N+i*N+tl.arange(0, N)).to(tl.float32)
        sums = tl.sum(x*t)
        tl.store(sum_ptr+si+i, sums)

        x *= smooth_scale
        x_max = tl.maximum(tl.max(tl.abs(x)), 1e-30)

        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(x_max/448.0)))
        else:
            scale = x_max/448.0

        tl.store(qs_ptr+si+i, scale)

        s = 1.0/scale
        x *= s
        xq = x.to(q_ptr.dtype.element_ty)
        tl.store(q_ptr+si*N+i*N+tl.arange(0, N), xq)

"""
select and smooth and quant
x: [bs, dim]
smooth_scales: [n_experts, dim]
indices: [n_experts*topk]
x_q: [bs*topk, dim]
x_scale: [bs*topk]
"""
def triton_index_select_smooth_quant_and_sum(grads, tokens, smooth_scales, token_count_per_expert, indices, x_q=None, x_scale=None, x_sum=None, reverse=False, round_scale=False):
    # row-wise read, row-wise write
    M, N = grads.shape
    n_expert, n = smooth_scales.shape
    assert N == n, f'{N=} {n=}'
    E = indices.shape[0]
    device = grads.device 
    if x_q is None:
        x_q = torch.empty((E, N), device=device, dtype=torch.float8_e4m3fn)
    if x_scale is None:
        x_scale = torch.empty((E,), device=device, dtype=torch.float32)
    if x_sum is None:
        x_sum = torch.empty((E,), device=device, dtype=grads.dtype)
    accum_token_count = torch.cumsum(token_count_per_expert, 0)
    grid = (n_expert, )
    index_select_smooth_quant_and_sum_kernel[grid](
        grads,
        tokens,
        x_q,
        smooth_scales,
        x_scale,
        token_count_per_expert,
        accum_token_count,
        indices,
        x_sum,
        M, N, 
        reverse,
        round_scale,
        num_stages=3,
        num_warps=8
    )
    return x_q,x_scale,x_sum



@triton.jit
def smooth_unpermute_backward_kernel(grads_data_ptr, grads_scale_ptr, q_ptr, ss_ptr, qs_ptr, count_ptr, accum_ptr, index_ptr, N: tl.constexpr, n: tl.constexpr, hs: tl.constexpr, REVERSE: tl.constexpr, ROUND: tl.constexpr, GROUP: tl.constexpr):
    eid = tl.program_id(axis=0)
    wid = tl.program_id(axis=1)
    T = tl.num_programs(axis=1)

    # row-wise read, row-wise write
    smooth_scale = tl.load(ss_ptr+eid*N+tl.arange(0, N))
    if not REVERSE:
        smooth_scale = 1.0/smooth_scale
    count = tl.load(count_ptr+eid)
    ei = tl.load(accum_ptr+eid)
    si = ei - count
    c = tl.cdiv(count, T)
    for i in range(si+wid*c, tl.minimum(si+wid*c+c,ei)):
        index = tl.load(index_ptr+i)
        x = tl.load(grads_data_ptr + index*N+tl.arange(0, N)).to(tl.float32)
        if GROUP:
            gs = tl.load(grads_scale_ptr + index*n + tl.arange(0,hs))
            x = tl.reshape(tl.reshape(x, (hs, N//hs))*gs[:,None], (N,))
        else:
            gs = tl.load(grads_scale_ptr + index)
            x *= gs

        x *= smooth_scale
        x_max = tl.maximum(tl.max(tl.abs(x)), 1e-30)

        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(x_max/448.0)))
        else:
            scale = x_max/448.0

        tl.store(qs_ptr+i, scale)

        s = 1.0/scale
        x *= s
        xq = x.to(q_ptr.dtype.element_ty)
        tl.store(q_ptr+i*N+tl.arange(0, N), xq)

"""
select and smooth and quant
grad_data: [bs, dim]
grad_scale: [bs, dim/128]
smooth_scales: [n_experts, dim]
indices: [n_experts*topk]
x_q: [bs*topk, dim]
x_scale: [bs*topk]
"""
def triton_smooth_unpermute_backward(grad_data, grad_scale, smooth_scales, token_count_per_expert, indices, x_q=None, x_scale=None, reverse=False, round_scale=False):
    # row-wise read, row-wise write
    M, N = grad_data.shape
    n_expert, n = smooth_scales.shape
    assert 128%n_expert == 0
    assert N == n, f'{N=} {n=}'
    
    group = grad_scale.ndim > 1
    hs = grad_scale.shape[1] if group else 1

    E = indices.size(0)
    device = grad_data.device 
    if x_q is None:
        x_q = torch.empty((E, N), device=device, dtype=torch.float8_e4m3fn)
    if x_scale is None:
        x_scale = torch.empty((E,), device=device, dtype=torch.float32)
    accum_token_count = torch.cumsum(token_count_per_expert, 0)
    W = 128//n_expert
    n = N//128
    grid = (n_expert, W)
    smooth_unpermute_backward_kernel[grid](
        grad_data,
        grad_scale,
        x_q,
        smooth_scales,
        x_scale,
        token_count_per_expert,
        accum_token_count,
        indices,
        N, 
        n,
        hs,
        reverse,
        round_scale,
        group,
        num_stages=3,
        num_warps=16
    )
    return x_q,x_scale



@triton.jit
def smooth_permute_with_mask_map_kernel(grads_data_ptr, quant_data_ptr, mask_map_ptr, grads_scale_ptr, smooth_scale_ptr, quant_scale_ptr, M, T, N: tl.constexpr, REVERSE: tl.constexpr, ROUND: tl.constexpr):
    eid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    # smooth_scale_ptr = tl.load(smooth_scale_ptrs + eid).to(tl.pointer_type(tl.float32))
    smooth_scale = tl.load(smooth_scale_ptr+eid*N+tl.arange(0, N))
    if not REVERSE:
        smooth_scale = 1.0/smooth_scale
    for i in range(bid*T, tl.minimum(bid*T+T,M)):
        index = tl.load(mask_map_ptr+eid*M+i)
        mask = index >= 0
        x = tl.load(grads_data_ptr + i*N+tl.arange(0, N), mask=mask).to(tl.float32)
        gs = tl.load(grads_scale_ptr + i, mask=mask)
        x *= gs

        x *= smooth_scale
        x_max = tl.maximum(tl.max(tl.abs(x)), 1e-30)

        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(x_max/448.0)))
        else:
            scale = x_max/448.0

        tl.store(quant_scale_ptr+index, scale, mask=mask)

        s = 1.0/scale
        x *= s
        xq = x.to(quant_data_ptr.dtype.element_ty)
        tl.store(quant_data_ptr+index*N+tl.arange(0, N), xq, mask=mask)

# """
# gather and smooth quant
# inp: [num_tokens, hidden_size], rowwise_data
# row_id_map: [n_experts, num_tokens], indices
# scale: [num_tokens], rowwise_scale_inv
# smooth_scale_ptrs: [n_experts], data_ptr
# """

def triton_smooth_permute_with_mask_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    scale: torch.Tensor,
    num_experts: int,
    num_tokens: int,
    num_out_tokens: int,
    hidden_size: int,
    smooth_scales: torch.Tensor,
    reverse=True,
    round_scale=True
    ):
    output = torch.empty((num_out_tokens, hidden_size), dtype=inp.dtype, device="cuda")

    permuted_scale = torch.empty(
        (num_out_tokens, ), dtype=scale.dtype, device="cuda"
    )
    sm = torch.cuda.get_device_properties(inp.device).multi_processor_count
    T = triton.cdiv(num_tokens, sm)
    grid = (num_experts, sm)
    smooth_permute_with_mask_map_kernel[grid](
        inp,
        output,
        row_id_map,
        scale,
        smooth_scales,
        permuted_scale,
        num_tokens,
        T,
        hidden_size,
        reverse,
        round_scale
    )
    return output, permuted_scale


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
                                    scale_a=x_s.view(-1,1),
                                    scale_b=w_s.view(1,-1),
                                    out_dtype=x.dtype,
                                    use_fast_accum=True)
    return output

def reused_smooth_quant_backward(y,w, smooth_scale):

    y_q,y_s,w_q,w_s = triton_reused_smooth_quant_nn(y, w, smooth_scale)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_s.view(-1,1),
                                    scale_b=w_s.view(1,-1),
                                    out_dtype=y.dtype,
                                    use_fast_accum=True)
    return output


def reused_smooth_quant_update(y,x, smooth_scale):
    y_q,y_s,x_q,x_s = triton_reused_smooth_quant_tn(y, x, smooth_scale)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_s.view(-1,1),
                                    scale_b=x_s.view(1,-1),
                                    out_dtype=y.dtype,
                                    use_fast_accum=True)
    return output

def reused_smooth_quant_f_and_b(x, w, y, smooth_scale):
    reused_smooth_quant_forward(x,w, smooth_scale)
    reused_smooth_quant_backward(y,w, smooth_scale)
    reused_smooth_quant_update(y,x, smooth_scale)