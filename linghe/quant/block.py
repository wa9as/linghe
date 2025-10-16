# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def block_quant_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr,
                       ROUND: tl.constexpr):
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


def triton_block_quant(x,
                block_size=128,
                round_scale=False):
    """
    blockwise quantize x
    Args:
        x: input tensor
        block_size: block wise
        round_scale: whether round scale to power of 2

    Returns:
        y: quantized tensor, float8_e4m3fn
        s: quantization scale, float32
    """
    M, N = x.size()
    y = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    s = x.new_empty(x.size(-2) // block_size, x.size(-1) // block_size,
                    dtype=torch.float32)
    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
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
