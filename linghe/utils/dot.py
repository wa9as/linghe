# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def dot_kernel(x_ptr, y_ptr, sum_ptr, M, N, H: tl.constexpr, W: tl.constexpr):
    # rowwise read, rowwise write
    pid = tl.program_id(axis=0)
    offs = pid * W * N + tl.arange(0, W)[:, None] * N + tl.arange(0, H)[None, :]

    n = tl.cdiv(N, H)
    sums = tl.zeros((W,), dtype=tl.float32)
    for i in range(n):
        x = tl.load(x_ptr + offs).to(tl.float32)
        y = tl.load(y_ptr + offs).to(tl.float32)
        sums += tl.sum(x * y, axis=1)
        offs += H

    tl.store(sum_ptr + pid * W + tl.arange(0, W), sums)


def triton_dot(x, y):
    """
    vector dot multiply, output = sum(x*y, 1),
    it is used to calculate gradient of router weight
    Args:
        x:
        y:

    Returns:
        output of sum(x*y, 1)
    """
    M, N = x.shape
    H = 128
    W = 16
    assert M % W == 0

    num_stages = 5
    num_warps = 8
    device = x.device
    s = torch.empty((M,), device=device, dtype=x.dtype)
    grid = (triton.cdiv(M, W),)
    dot_kernel[grid](x, y, s, M, N, H, W, num_stages=num_stages, num_warps=num_warps)
    return s
