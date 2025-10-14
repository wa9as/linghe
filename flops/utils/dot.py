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
        q = tl.load(y_ptr + offs).to(tl.float32)
        sums += tl.sum(x * q, axis=1)
        offs += H

    tl.store(sum_ptr + pid * W + tl.arange(0, W), sums)


def triton_dot(x, y):
    M, N = x.shape
    H = 128
    W = 16
    assert M % W == 0

    num_stages = 5
    num_warps = 8
    device = x.device
    s = torch.empty((M,), device=device, dtype=x.dtype)
    grid = (triton.cdiv(M, W),)
    dot_kernel[grid](
        x, y, s,
        M, N,
        H, W,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return s


@triton.jit
def mix_precise_dot_kernel(x_ptr, q_ptr, sum_ptr, smooth_scale_ptr,
                           quant_scale_ptr, M, N, H: tl.constexpr,
                           W: tl.constexpr):
    # rowwise read, rowwise write
    pid = tl.program_id(axis=0)
    offs = pid * W * N + tl.arange(0, W)[:, None] * N + tl.arange(0, H)[None, :]
    soffs = tl.arange(0, H)
    quant_scale = tl.load(quant_scale_ptr + pid * W + tl.arange(0, W))

    n = tl.cdiv(N, H)
    sums = tl.zeros((W,), dtype=tl.float32)
    for i in range(n):
        x = tl.load(x_ptr + offs)
        q = tl.load(q_ptr + offs)
        smooth_scale = tl.load(smooth_scale_ptr + soffs)[None, :]
        q = q.to(tl.float32) * smooth_scale
        x = x.to(tl.float32)
        sums += tl.sum(x * q, axis=1) * quant_scale
        offs += H
        soffs += H

    tl.store(sum_ptr + pid * W + tl.arange(0, W), sums)


# q should be dequant
def triton_mix_precise_dot(x, q, smooth_scale, quant_scale, reverse=False):
    assert reverse
    M, N = x.shape
    device = x.device
    s = torch.empty((M,), device=device, dtype=x.dtype)

    H = 128
    W = 16
    num_stages = 5
    num_warps = 8

    grid = (triton.cdiv(M, W),)
    mix_precise_dot_kernel[grid](
        x, q, s,
        smooth_scale,
        quant_scale,
        M, N,
        H, W,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return s
