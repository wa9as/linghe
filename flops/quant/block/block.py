
from enum import IntEnum
from typing import Tuple

import torch
import triton
import triton.language as tl
from triton import Config


# Some triton kernels for tilewise and blockwise quantization are from the link below with modification:
# https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py


@triton.jit
def block_quant_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr, ROUND: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x)) / 448.0, 1e-30)
    if ROUND:
        s = tl.exp2(tl.ceil(tl.log2(s)))
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n + pid_n, s)


def block_quant(x, dtype=torch.float8_e4m3fn, block_size = 128, round_scale=False):
    M, N = x.size()
    y = torch.empty_like(x, dtype=dtype)
    s = x.new_empty(x.size(-2) // block_size, x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE"]), triton.cdiv(N, META["BLOCK_SIZE"]))  # noqa: E731
    block_quant_kernel[grid](x, 
                             y, 
                             s, 
                             M, 
                             N, 
                             BLOCK_SIZE=block_size, 
                             ROUND=round_scale,
                             num_stages=6, 
                             num_warps=8)
    return y, s



