
import torch
import triton
import triton.language as tl
from triton import Config
from flops.quant.channel.channel import row_quant_kernel
from flops.quant.smooth.reused_smooth import triton_reused_smooth_quant, triton_reused_transpose_pad_smooth_quant
from flops.utils.transpose import triton_block_pad_transpose

"""
megatron fp8 training steps:
step 0: init w smooth scale w_smooth
v
step 2: in forward step, columnwise smooth x and rowwise quant x, calc y=x@w; meanwhile, record the columnwise max of x, it is used to update w_smooth
step 3: in dgrad step, columnwise smooth y and rowwise quant y, transpose x, calc dx=y@wT 
step 4: in wgrad step, dequant then smooth an then quant y_q to get yt_q, calc dw=yT@x

alternative (it's not suitable for fp8 combine):
step 4: in wgrad step, rowwise smooth y and columnwise quant y and transpose to get yt_q, calc dw=yT@x

"""




@triton.jit
def calc_smooth_scale_kernel(x_ptr, smooth_scale_ptr, inv_smooth_scale_ptr, M, N, H: tl.constexpr, W: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(axis=0)
    # col-wise read, col-wise write
    col_offs = tl.arange(0, W)
    row_offs = tl.arange(0, H)

    x_max = tl.zeros((W,),dtype=tl.float32) + 5.27e-36
    m = tl.cdiv(M, H)
    
    for i in range(m):
        current_offs = pid * W + i * H * N
        mask = None
        
        if EVEN:
            x = tl.load(x_ptr + current_offs + row_offs[:, None] * N)
        else:
            col_mask = (pid * W + col_offs) < N
            row_mask = (i * H + row_offs) < M
            mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(x_ptr + current_offs + row_offs[:, None] * N, mask=mask, other=0.0)
        
        abs_x = tl.abs(x)
        col_max = tl.max(abs_x, axis=0)
        x_max = tl.maximum(x_max, col_max)
    
    maxs = tl.sqrt(tl.maximum(x_max, 5.27e-36))
    scale = tl.where(maxs < 4.0, 1.0, maxs)
    inv_scale = 1.0 / tl.maximum(scale, 5.27e-36)
    
    tl.store(smooth_scale_ptr + pid * W + col_offs, scale)
    tl.store(inv_smooth_scale_ptr + pid * W + col_offs, inv_scale)


# adapt for megatron fp8 training
def triton_calc_smooth_scale(x):
    M, N = x.shape
    device = x.device 
    x_smooth_scale = torch.empty((N,), device=device, dtype=torch.float32)
    x_inv_smooth_scale = torch.empty((N,), device=device, dtype=torch.float32)
    H = 512 if M >= 512 else 256
    W = 32 if N >= 2048 else 16
    if M%H == 0 and N%W == 0:
        EVEN = True 
    else:
        EVEN = False
    grid = lambda META: (triton.cdiv(N, W), )
    calc_smooth_scale_kernel[grid](
        x,
        x_smooth_scale,
        x_inv_smooth_scale,
        M, N,
        H, W, EVEN,
        num_stages=4,
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

    return x_q,xt_q,x_scale,xt_scale


# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_smooth_quant_y(y, smooth_scale, transpose_smooth_scale, reverse=True, transpose=True,  pad=False):
    assert reverse, "args `smooth_scale` and/or `transpose_smooth_scale` must be in reciprocal format in triton_smooth_quant_y"
    #assert y.size(1) == smooth_scale.size(0)
    y_q,y_scale = triton_reused_smooth_quant(y, smooth_scale, reverse=reverse, pad_scale=pad)
    if transpose:
        #assert pad or y.size(0) == transpose_smooth_scale.size(0)
        yt_q, yt_scale = triton_reused_transpose_pad_smooth_quant(y, transpose_smooth_scale, reverse=reverse, pad=pad)
    else:
        yt_q, yt_scale = None, None

    return y_q, yt_q, y_scale, yt_scale


def triton_smooth_quant_partial_w(w, smooth_scale, w_q, w_scale, offset=0):
    M,N = w_q.shape 
    assert w.size(1) == smooth_scale.size(0)
    size = w.numel()
    m = size//N
    w_q_slice = w_q.view(-1)[offset:offset+size].view(m,N).view(torch.float8_e4m3fn)
    w_scale_slice = w_scale[offset//N:(offset+size)//N]
    w_q,w_scale = triton_reused_smooth_quant(w, smooth_scale, x_q=w_q_slice, x_scale=w_scale_slice, pad_scale=False)

    return w_q,w_scale


def seperate_smooth_quant_forward(x, w):
    x_smooth_scale, x_inv_smooth_scale = triton_calc_smooth_scale(x)
    x_q, _, x_scale, _ = triton_smooth_quant_x(x, x_inv_smooth_scale, transpose=False)
    w_q, _, w_scale, _ = triton_smooth_quant_x(w, x_smooth_scale, transpose=False)

    output = torch._scaled_mm(
        x_q,
        w_q.t(),
        scale_a=x_scale.view(-1, 1),
        scale_b=w_scale.view(1, -1),
        out_dtype=x.dtype,
        use_fast_accum=True
    )
    
    return output, x_q, w_q, x_scale, w_scale


def seperate_smooth_quant_backward(y, w):
    y_smooth_scale, y_inv_smooth_scale = triton_calc_smooth_scale(y)
    y_q, _, y_scale, _ = triton_smooth_quant_x(y, y_smooth_scale, transpose=False)
    _, wt_q, _, wt_scale = triton_smooth_quant_y(w, y_smooth_scale, y_inv_smooth_scale)
    
    output = torch._scaled_mm(
        y_q,
        wt_q.t(),
        scale_a=y_scale.view(-1, 1),
        scale_b=wt_scale.view(1, -1),
        out_dtype=y.dtype,
        use_fast_accum=True
    )
    
    return output, y_q, wt_q, y_scale, wt_scale


def seperate_smooth_quant_update(y, x):
    y_smooth_scale, y_inv_smooth_scale = triton_calc_smooth_scale(y)
    _, yt_q, _, yt_scale = triton_smooth_quant_y(y, y_inv_smooth_scale, y_smooth_scale)
    _, xt_q, _, xt_scale = triton_smooth_quant_y(x, y_smooth_scale, y_inv_smooth_scale)

    output = torch._scaled_mm(
        yt_q,
        xt_q.t(),
        scale_a=yt_scale.view(-1, 1),
        scale_b=xt_scale.view(1, -1),
        out_dtype=y.dtype,
        use_fast_accum=True
    )
    
    return output, yt_q, xt_q, yt_scale, xt_scale


def seperate_smooth_quant_f_and_b(x, w, y, w_smooth_scale):
    # Initialize w_smooth_scale if not provided
    if w_smooth_scale is None:
        w_smooth_scale, _ = triton_calc_smooth_scale(w)
    
    # Smooth and quant weights using w_smooth
    w_q, _, w_scale, _ = triton_smooth_quant_x(
        w, 
        1/w_smooth_scale,
        transpose=False, 
        pad=False
    )
    
    # ===== forward =====
    # Calc smooth scale from x and update w_smooth
    x_smooth_scale, _ = triton_calc_smooth_scale(x)

    # Quant x using inverse smooth scale
    x_q, _, x_scale, _ = triton_smooth_quant_x(
        x, 
        w_smooth_scale, 
        transpose=False, 
        pad=False
    )
    
    # y=x_q@w_q
    o = torch._scaled_mm(
        x_q,
        w_q.t(),
        scale_a=x_scale.view(-1, 1),
        scale_b=w_scale.view(1, -1),
        out_dtype=x.dtype,
        use_fast_accum=True
    )

    w_smooth_scale = x_smooth_scale.clone().detach()
    
    # ===== dgrad =====
    # Smooth and quant y for gradient calculation
    y_smooth_scale, y_inv_smooth_scale = triton_calc_smooth_scale(y)
    y_q, _, y_scale, _ = triton_smooth_quant_x(
        y,
        y_inv_smooth_scale,
        transpose=False,
        pad=False
    )
    
    # Transpose and quant weights
    _, wt_q, _, wt_scale = triton_smooth_quant_y(
        w,
        y_smooth_scale,
        y_inv_smooth_scale,
        reverse=True,
        transpose=True,
        pad=False
    )
    
    # dx=y@wT
    dx = torch._scaled_mm(
        y_q,
        wt_q.t(),
        scale_a=y_scale.view(-1, 1),
        scale_b=wt_scale.view(1, -1),
        out_dtype=y.dtype,
        use_fast_accum=True
    )

    # ===== wgrad =====
    # Prepare transposed y and x for weight gradient
    _, yt_q, _, yt_scale = triton_smooth_quant_y(
        y,
        y_inv_smooth_scale,
        y_smooth_scale,
        reverse=True,
        transpose=True,
        pad=False
    )

    _, xt_q, _, xt_scale = triton_smooth_quant_y(
        x,
        y_smooth_scale,
        y_inv_smooth_scale,
        reverse=True,
        transpose=True,
        pad=False
    )
    
    # dw=yT@x
    dw = torch._scaled_mm(
        yt_q,
        xt_q.t(),
        scale_a=yt_scale.view(-1, 1),
        scale_b=xt_scale.view(1, -1),
        out_dtype=y.dtype,
        use_fast_accum=True
    )
    
    return o, dx, dw, w_smooth_scale