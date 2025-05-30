
import math
import torch
import triton
import triton.language as tl
from triton import Config
from flops.quant.channel.channel import row_quant_kernel
from flops.quant.smooth.reused_smooth import triton_reused_smooth_quant, triton_reused_transpose_smooth_quant, triton_reused_transpose_pad_smooth_quant
from flops.utils.transpose import triton_transpose,triton_block_transpose,triton_block_pad_transpose
from flops.utils.util import round_up



@triton.jit
def calc_smooth_scale_kernel(x_ptr, smooth_scale_ptr, inv_smooth_scale_ptr, M, N, H: tl.constexpr, W: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(axis=0)
    # col-wise read, col-wise write
    offs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:]
    x_max = tl.zeros((W,),dtype=tl.float32) + 5.27e-36
    m = tl.cdiv(M, H)
    for i in range(m):
        if EVEN:
            x = tl.load(x_ptr+offs)
        else:
            x = tl.load(x_ptr+offs, mask=(i*H+tl.arange(0, H)[:,None]<M) & (pid*W+tl.arange(0, W)[None,:]<N) )
        x_max = tl.maximum(tl.max(tl.abs(x), axis=0),x_max)
        offs += H*N

    maxs = tl.sqrt(x_max)
    scale = tl.where(maxs<4, 1.0, maxs)

    if EVEN:
        tl.store(smooth_scale_ptr + pid*W + tl.arange(0, W), scale)
        tl.store(inv_smooth_scale_ptr + pid*W + tl.arange(0, W), 1.0/scale)
    else:
        mask = pid*W+tl.arange(0, W)<N
        tl.store(smooth_scale_ptr + pid*W + tl.arange(0, W), scale, mask=mask)
        tl.store(inv_smooth_scale_ptr + pid*W + tl.arange(0, W), 1.0/scale, mask=mask)

# adapt for megatron fp8 training
def triton_calc_smooth_scale(x):
    M, N = x.shape
    device = x.device 
    x_smooth_scale = torch.empty((N,), device=device, dtype=torch.float32)  # 
    x_inv_smooth_scale = torch.empty((N,), device=device, dtype=torch.float32)
    # H = max([x for x in [128,256,512] if M%x == 0])
    H = 256
    W = 16
    if M%H == 0 and N%W == 0:
        EVEN = True 
    else:
        EVEN = False
    grid = lambda META: ((N-1)//W+1, )
    calc_smooth_scale_kernel[grid](
        x,
        x_smooth_scale,
        x_inv_smooth_scale,
        M, N,
        H, W, EVEN,
        num_stages=5,
        num_warps=4
    )
    return x_smooth_scale, x_inv_smooth_scale



"""
divide x by smooth_scale and row-wise quantization
smooth scale is updated by square root of x's column-wise maxs, and set in weight's x_maxs attr

transpose: transpose quantized x for wgrad
pad: # pad M to be multiplier of 16, including quant scales and transposed x

"""

# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_smooth_quant_x(x, smooth_scale, transpose=True, pad=False):
    assert x.size(1) == smooth_scale.size(0)

    x_q,x_scale = triton_reused_smooth_quant(x, smooth_scale, pad_scale=pad)

    if transpose:
        xt_q = triton_block_pad_transpose(x_q, pad=pad)  
    else:
        xt_q = None 
    xt_scale = smooth_scale

    # if torch.isnan(x_q).count_nonzero()>0:
    #     print(f'{x_q.float().max()=}')
    #     raise ValueError('triton_smooth_quant_x nan')

    return x_q,xt_q,x_scale,xt_scale


def triton_smooth_quant_w(w, smooth_scale, transpose=True):
    assert w.size(1) == smooth_scale.size(0)

    w_q,w_scale = triton_reused_smooth_quant(w, smooth_scale, pad_scale=False)

    if transpose:
        wt_q = triton_block_pad_transpose(w_q, pad=False)  # x_q has be padded
    else:
        wt_q = None 
    wt_scale = smooth_scale

    # if torch.isnan(w_q).count_nonzero()>0:
    #     print(f'{w_q.float().max()=}')
    #     raise ValueError('triton_smooth_quant_w nan')

    return w_q,wt_q,w_scale,wt_scale


# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_smooth_quant_y(y, smooth_scale, transpose_smooth_scale, reverse=True, pad=False):
    assert reverse, "args `smooth_scale` and `transpose_smooth_scale` must be reciprocal in triton_smooth_quant_y"
    assert y.size(1) == smooth_scale.size(0)
    assert pad or y.size(0) == transpose_smooth_scale.size(0)
    y_q,y_scale = triton_reused_smooth_quant(y, smooth_scale, reverse=True, pad_scale=pad)
    yt_q, yt_scale = triton_reused_transpose_pad_smooth_quant(y, transpose_smooth_scale, reverse=True, pad=pad)

    # if torch.isnan(yt_q).count_nonzero()>0:
    #     print(f'{yt_q.float().max()=}')
    #     raise ValueError('triton_smooth_quant_yT nan')

    return y_q,yt_q,y_scale,yt_scale