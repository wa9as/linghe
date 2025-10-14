# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from flops.utils.loss import triton_softmax_cross_entropy_forward, \
    triton_softmax_cross_entropy_backward


class SoftmaxCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, inplace=False):
        shape = logits.shape
        if len(shape) == 3:
            logits = logits.view(-1, shape[-1])
        loss, sum_exp, max_logit = triton_softmax_cross_entropy_forward(logits,
                                                                        labels)
        ctx.save_for_backward(logits, labels, sum_exp, max_logit)
        ctx.inplace = inplace
        ctx.shape = shape
        if len(shape) == 3:
            loss = loss.view(shape[0], shape[1])
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, sum_exp, max_logit = ctx.saved_tensors
        shape = ctx.shape
        grad = logits if ctx.inplace else None
        grad = triton_softmax_cross_entropy_backward(logits, labels, sum_exp,
                                                     max_logit,
                                                     grad_output,
                                                     output_grad=grad)
        if len(shape) == 3:
            grad = grad.view(shape)
        return grad, None, None, None


class GradScalingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coef=0.2):
        ctx.coef = coef
        return x

    @staticmethod
    def backward(ctx, grad_output):
        shape = grad_output.shape
        assert len(shape) == 2
        bs, length = grad_output.shape
        array = length - torch.arange(0, length, device=grad_output.device)
        scale = 1 / torch.pow(array.float(), ctx.coef)
        grad = grad_output * scale
        return grad, None
