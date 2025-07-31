import torch
import triton
import triton.language as tl


@triton.jit
def calculate_smooth_scale_kernel(x_ptr, y_ptr, min_value, N,
                                  B: tl.constexpr,
                                  EVEN: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * B + tl.arange(0, B)
    if EVEN:
        x = tl.load(x_ptr+offs).to(tl.float32)
    else:
        x = tl.load(x_ptr+offs, mask=offs<N).to(tl.float32)
    x = 1.0/tl.sqrt(tl.maximum(x, min_value))
    x = tl.exp2(tl.ceil(tl.log2(x)))
    if EVEN:
        tl.store(y_ptr + offs, x)
    else:
        tl.store(y_ptr + offs, x, mask=offs<N)

"""
input_smooth_scales = torch.sqrt(torch.maximum(input_smooth_scales, torch.ones([1], dtype=torch.float32, device=input_smooth_scales.device)))
weight_smooth_scales = 1/input_smooth_scales
weight_smooth_scales = torch.exp2(torch.ceil(torch.log2(weight_smooth_scales)))
first `offset` values are ignored 
"""
def triton_calculate_smooth_scale(x, min_value=1.0, inplace=False):
    N = x.shape[0]
    B = 4096
    if inplace:
        output = x 
    else:
        output = torch.empty((N,), dtype=x.dtype, device=x.device)

    EVEN = N % B == 0
    num_stages = 3
    num_warps = 4
    grid = (triton.cdiv(N, B), )
    calculate_smooth_scale_kernel[grid](
        x, output, min_value,
        N,
        B,
        EVEN,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return output