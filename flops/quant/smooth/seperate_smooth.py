
import torch
import triton
import triton.language as tl
from triton import Config
from flops.quant.smooth.reused_smooth import triton_reused_smooth_quant, triton_reused_transpose_pad_smooth_quant
from flops.utils.transpose import triton_transpose_and_pad


"""
megatron fp8 training steps:
step 0: init w smooth scale w_smooth
step 1: smooth and quant w when w is updated
step 2: in forward step, columnwise smooth x and rowwise quant x, calc y=x@w; meanwhile, record the columnwise max of x, it is used to update w_smooth
step 3: in dgrad step, columnwise smooth y and rowwise quant y, transpose x, calc dx=y@wT 
step 4: in wgrad step, dequant then smooth an then quant y_q to get yt_q, calc dw=yT@x

alternative (it's not suitable for fp8 combine):
step 4: in wgrad step, rowwise smooth y and columnwise quant y and transpose to get yt_q, calc dw=yT@x

"""

"""
divide x by smooth_scale and row-wise quantization
smooth scale is updated by square root of x's column-wise maxs, and set in weight's x_maxs attr

transpose: transpose quantized x for wgrad
pad: # pad M to be multiplier of 32, including quant scales and transposed x

"""
# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_smooth_quant_x(x, smooth_scale, x_q=None, x_scale=None, xt_q=None, transpose=True, pad=False, round_scale=False):
    
    #assert x.size(1) == smooth_scale.size(0)
    x_q,x_scale = triton_reused_smooth_quant(x, smooth_scale, x_q=x_q, x_scale=x_scale, reverse=False, round_scale=round_scale)

    if transpose:
        xt_q = triton_transpose_and_pad(x_q, out=xt_q, pad=pad)  
    else:
        xt_q = None 
    xt_scale = smooth_scale

    return x_q,xt_q,x_scale,xt_scale


# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_smooth_quant_y(y, smooth_scale, transpose_smooth_scale, reverse=True, transpose=True,  pad=False, round_scale=False):
    
    assert reverse, "args `smooth_scale` and/or `transpose_smooth_scale` must be in reciprocal format in triton_smooth_quant_y"
    N = y.size(1)
    # assert N == smooth_scale.size(0)
    y_q,y_scale = triton_reused_smooth_quant(y, smooth_scale, reverse=True, round_scale=round_scale)
    if transpose:
        # assert pad or y.size(0) == transpose_smooth_scale.size(0)
        yt_q, yt_scale = triton_reused_transpose_pad_smooth_quant(y, transpose_smooth_scale, reverse=reverse, pad=pad, round_scale=round_scale)
    else:
        yt_q, yt_scale = None, None

    return y_q,yt_q,y_scale,yt_scale


def triton_smooth_quant_w(w, smooth_scale, w_q, quant_scale, offset=0, round_scale=False):
     
    M,N = w_q.shape 
    assert w.size(1) == smooth_scale.size(0)
    size = w.numel()
    m = size//N 
    ms = offset//N
    w_q_slice = w_q[ms:ms+m].view(torch.float8_e4m3fn)
    quant_scale_slice = quant_scale[ms:ms+m]
    w_q,w_scale = triton_reused_smooth_quant(w, smooth_scale, x_q=w_q_slice, x_scale=quant_scale_slice, round_scale=round_scale)

    return w_q,w_scale

