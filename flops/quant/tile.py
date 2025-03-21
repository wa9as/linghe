
from enum import IntEnum
from typing import Tuple

import torch
import triton
import triton.language as tl
from triton import Config



# Some triton kernels for tilewise and blockwise quantization are from the link below with modification:
# https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py


@triton.jit
def block_quant_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    eps = 1e-10
    s = tl.maximum(tl.max(tl.abs(x)), eps) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n + pid_n, s)


def block_quant(x: torch.Tensor, dtype=torch.float8_e4m3fn, block_size: int = 128) -> torch.Tensor:
    M, N = x.size()
    y = torch.empty_like(x, dtype=dtype)
    s = x.new_empty(x.size(-2) // block_size, x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))  # noqa: E731
    block_quant_kernel[grid](x, y, s, M, N, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def tile_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    eps = 1e-10
    s = tl.maximum(tl.max(tl.abs(x)), eps) / 448.0
    # # ceil rounding to power of 2
    # s = tl.ceil(tl.log2(s))
    # s = tl.exp2(s)
    # # floor rounding to power of 2
    # s = tl.floor(tl.log2(s))
    # s = tl.exp2(s)
    # nearest rounding to power of 2
    s = tl.floor(tl.log2(s) + 0.5)
    s = tl.exp2(s)
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)



def tile_quant(
    x: torch.Tensor, dtype=torch.float8_e4m3fn, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0
    y = torch.empty_like(x, dtype=dtype)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)  # noqa: E731
    tile_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s

