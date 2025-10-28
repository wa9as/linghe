# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.tools.benchmark import benchmark_func
from linghe.tools.util import output_check
from linghe.utils.dot import triton_dot


def torch_fp16_dot(x, y):
    return (x * y).sum(1)


def test_dot(M=4096, N=4096, bench=False):
    dtype = torch.bfloat16
    device = "cuda:0"

    n_repeat = 100

    x = torch.randn(M, N, dtype=dtype, device=device)
    y = torch.randn(M, N, dtype=dtype, device=device)
    q = torch.randn(M, N, dtype=dtype, device=device).to(torch.float8_e4m3fn)
    quant_scale = torch.randn(M, dtype=torch.float32, device=device).abs()
    smooth_scale = torch.randn(N, dtype=torch.float32, device=device).abs()

    sums = triton_dot(x, q)
    sums_ref = torch_fp16_dot(x, q.float().to(dtype))
    output_check(sums_ref, sums, "sum")

    sums_ref = (
        x.float() * (q.to(torch.float32) * quant_scale[:, None] * smooth_scale[None, :])
    ).sum(dim=1)

    if bench:
        ref_time = benchmark_func(torch_fp16_dot, x, y, n_repeat=n_repeat)


if __name__ == "__main__":
    test_dot(M=4096, N=4096)
