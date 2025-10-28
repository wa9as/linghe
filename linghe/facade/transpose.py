# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.utils.transpose import triton_transpose


class TransposeDim01Function(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx, x):
        return triton_transpose(x, dim0=0, dim1=1)

    @staticmethod
    def backward(ctx, grad_output):
        return triton_transpose(grad_output, dim0=0, dim1=1)


def transpose_dim01(x):
    """
    transpose a tensor with the first two dims, x.ndims should not greater than 4
    Args:
        x: input tensor

    Returns:
        a transposed tensor
    """
    return TransposeDim01Function.apply(x)
