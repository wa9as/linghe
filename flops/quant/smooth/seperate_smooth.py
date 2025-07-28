import torch

from flops.quant.smooth.reused_smooth import triton_reused_smooth_quant, \
    triton_reused_transpose_smooth_quant, round_up, triton_subrow_reused_smooth_quant
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
def triton_smooth_quant_x(x, smooth_scale, x_q=None, x_scale=None, xt_q=None,
                          transpose=True, pad=True, round_scale=False):
    # assert x.size(1) == smooth_scale.size(0)
    x_q, x_scale = triton_reused_smooth_quant(x, smooth_scale, x_q=x_q,
                                              x_scale=x_scale, reverse=False,
                                              round_scale=round_scale)

    if transpose:
        xt_q = triton_transpose_and_pad(x_q, out=xt_q, pad=pad)
    else:
        xt_q = None
    xt_scale = smooth_scale

    return x_q, xt_q, x_scale, xt_scale


# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_smooth_quant_y(y, smooth_scale, transpose_smooth_scale, reverse=True,
                          transpose=True, pad=True, round_scale=False):
    assert reverse, "args `smooth_scale` and/or `transpose_smooth_scale` must be in reciprocal format in triton_smooth_quant_y"
    N = y.size(1)
    # assert N == smooth_scale.size(0)
    y_q, y_scale = triton_reused_smooth_quant(y, smooth_scale, reverse=True,
                                              round_scale=round_scale)
    if transpose:
        # assert pad or y.size(0) == transpose_smooth_scale.size(0)
        yt_q, yt_scale = triton_reused_transpose_smooth_quant(y,
                                                              transpose_smooth_scale,
                                                              reverse=reverse,
                                                              pad=pad,
                                                              round_scale=round_scale)
    else:
        yt_q, yt_scale = None, None

    return y_q, yt_q, y_scale, yt_scale


def triton_smooth_quant_w(w, smooth_scale, w_q, quant_scale, offset=0,
                          round_scale=False, default_scale=1e-5):
    assert w.ndim == 1
    assert w_q.size(1) == smooth_scale.size(0)

    size = w.numel()
    M, N = w_q.shape

    if size == M * N:
        triton_reused_smooth_quant(w.view(M, N), smooth_scale, x_q=w_q,
                                                x_scale=quant_scale,
                                                round_scale=round_scale)
    elif offset % N == 0 and size % N == 0:
        n_row = size // N
        row_id = offset // N
        w_q_slice = w_q[row_id:row_id + n_row]
        quant_scale_slice = quant_scale[row_id:row_id + n_row]
        triton_reused_smooth_quant(w.view(n_row,N), smooth_scale, x_q=w_q_slice,
                                                x_scale=quant_scale_slice,
                                                round_scale=round_scale)
    else:
        # import pdb; pdb.set_trace()
        row_si = (offset - 1)//N + 1
        row_ei = (offset + size) // N
        col_si = offset % N 
        col_ei = (offset + size ) % N
        n_row = row_ei - row_si
        mw_offset = 0 if col_si == 0 else N - col_si 
        w_q_slice = w_q[row_si:row_ei]
        quant_scale_slice = quant_scale[row_si:row_ei]
        triton_reused_smooth_quant(w[mw_offset:mw_offset+n_row*N].view(n_row,N), smooth_scale, x_q=w_q_slice,
                                                x_scale=quant_scale_slice,
                                                round_scale=round_scale)

        # subrow scale is writed by the row with leading master weight
        if col_ei != 0:
            weight_offset = row_ei*N
            master_weight_offset = mw_offset + n_row * N 
            triton_subrow_reused_smooth_quant(w, smooth_scale, w_q, quant_scale, weight_offset, master_weight_offset, col_ei,
                                      reverse=False, round_scale=round_scale, scale=default_scale)

