# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.quant.block.group import (triton_group_quant,
                                     triton_persist_group_quant)
from linghe.tools.benchmark import benchmark_func
from linghe.tools.util import (output_check,
                              torch_group_quant)


def test_group_quant(M=4096, N=4096, B=128, round_scale=False, bench=False):
    x = torch.randn((M, N), dtype=torch.bfloat16, device='cuda:0') ** 3
    xq_ref, x_scale_ref = torch_group_quant(x, B, round_scale=round_scale)
    xq, x_scale = triton_group_quant(x, group_size=B, round_scale=round_scale)
    output_check(xq_ref.float(), xq.float(), mode='data')
    output_check(x_scale_ref.float(), x_scale.float(), mode='scale')

    xq, x_scale = triton_persist_group_quant(x, group_size=B,
                                             round_scale=round_scale)
    output_check(xq_ref.float(), xq.float(), mode='data')
    output_check(x_scale_ref.float(), x_scale.float(), mode='scale')

    # torch.testing.assert_close(xq_ref.float(), xq.float(), rtol=0.02, atol=0.02)

    if bench:
        n_repeat = 100
        ref_time = benchmark_func(triton_group_quant, x, group_size=B,
                                  n_repeat=n_repeat, ref_bytes=M * N * 3)
        benchmark_func(triton_persist_group_quant, x, group_size=B,
                       n_repeat=n_repeat, ref_time=ref_time,
                       ref_bytes=M * N * 3)


if __name__ == '__main__':
    test_group_quant(M=4096, N=4096, B=128)
    test_group_quant(M=4096, N=8192, B=128)
    test_group_quant(M=2049, N=8192, B=128)
