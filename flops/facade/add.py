# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from flops.utils.add import triton_inplace_add


class InplaceAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        return triton_inplace_add(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output
