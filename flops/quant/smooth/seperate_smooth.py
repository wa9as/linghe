
import torch
import triton
import triton.language as tl
from triton import Config
from flops.quant.channel.channel import row_quant_kernel
from flops.quant.smooth.reused_smooth import triton_reused_smooth_quant, triton_tokenwise_reused_smooth_quant, triton_reused_transpose_pad_smooth_quant
from flops.utils.transpose import triton_transpose,triton_block_transpose,triton_block_pad_transpose
from flops.utils.util import round_up


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




@triton.jit
def update_weight_smooth_scale_kernel(x_ptr, smooth_scale_ptr, M, N, H: tl.constexpr, W: tl.constexpr, EVEN: tl.constexpr, ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    # col-wise read, col-wise write
    col_offs = tl.arange(0, W)
    row_offs = tl.arange(0, H)

    x_max = tl.zeros((W,),dtype=tl.float32) + 1e-30
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
            x = tl.load(x_ptr + current_offs + row_offs[:, None] * N, mask=mask)
        
        abs_x = tl.abs(x)
        col_max = tl.max(abs_x, axis=0)
        x_max = tl.maximum(x_max, col_max)
    
    scale = tl.sqrt(x_max)
    scale = tl.where(scale<4, 1.0, scale)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    inv_scale = 1.0 / scale
    
    tl.store(smooth_scale_ptr + pid * W + col_offs, inv_scale)


# update weight smooth scale for next step with x input 
def triton_update_weight_smooth_scale(x, round_scale=False):
    assert round_scale
    N = x.size(-1)
    M = x.nelement()//N
    device = x.device 
    weight_smooth_scale = torch.empty((N,), device=device, dtype=torch.float32)
    H = 256
    W = 16
    EVEN = M%H == 0 and N%W == 0
    grid = (triton.cdiv(N, W), )
    update_weight_smooth_scale_kernel[grid](
        x,
        weight_smooth_scale,
        M, N,
        H, W, 
        EVEN,
        round_scale,
        num_stages=4,
        num_warps=4
    )
    return weight_smooth_scale


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
    assert round_scale
    #assert x.size(1) == smooth_scale.size(0)
    N = x.size(1)
    assert triton.next_power_of_2(N) == N
    x_q,x_scale = triton_tokenwise_reused_smooth_quant(x, smooth_scale, x_q=x_q, x_scale=x_scale, reverse=False, round_scale=round_scale)

    if transpose:
        xt_q = triton_block_pad_transpose(x_q, x_t=xt_q, pad=pad)  
    else:
        xt_q = None 
    xt_scale = smooth_scale

    return x_q,xt_q,x_scale,xt_scale


# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_smooth_quant_y(y, smooth_scale, transpose_smooth_scale, reverse=True, transpose=True,  pad=False, round_scale=False):
    assert round_scale
    assert reverse, "args `smooth_scale` and/or `transpose_smooth_scale` must be in reciprocal format in triton_smooth_quant_y"
    N = y.size(1)
    # assert N == smooth_scale.size(0)
    if triton.next_power_of_2(N) == N:
        y_q,y_scale = triton_tokenwise_reused_smooth_quant(y, smooth_scale, reverse=True, round_scale=round_scale)
    else:
        y_q,y_scale = triton_reused_smooth_quant(y, smooth_scale, reverse=True, round_scale=round_scale)
    if transpose:
        # assert pad or y.size(0) == transpose_smooth_scale.size(0)
        yt_q, yt_scale = triton_reused_transpose_pad_smooth_quant(y, transpose_smooth_scale, reverse=reverse, pad=pad, round_scale=round_scale)
    else:
        yt_q, yt_scale = None, None

    return y_q,yt_q,y_scale,yt_scale


def triton_smooth_quant_partial_w(w, smooth_scale, w_q, w_scale, offset=0, round_scale=False):
    assert round_scale 
    M,N = w_q.shape 
    assert w.size(1) == smooth_scale.size(0)
    size = w.numel()
    m = size//N 
    ms = offset//N
    w_q_slice = w_q[ms:ms+m].view(torch.float8_e4m3fn)
    w_scale_slice = w_scale[offset//N:(offset+size)//N]
    w_q,w_scale = triton_tokenwise_reused_smooth_quant(w, smooth_scale, x_q=w_q_slice, x_scale=w_scale_slice, round_scale=round_scale)

    return w_q,w_scale

