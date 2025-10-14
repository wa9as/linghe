# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.utils.rope import triton_qk_norm_and_half_rope_forward, \
    triton_qk_norm_and_half_rope_backward


class QkNormHalfRopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, q_norm_weight, k_norm_weight, freqs, H=32, h=4,
                eps=1e-6):
        shape = qkv.shape
        qo, ko, vo = triton_qk_norm_and_half_rope_forward(qkv,
                                                          q_norm_weight.data,
                                                          k_norm_weight.data,
                                                          freqs,
                                                          H=H,
                                                          h=h,
                                                          eps=eps,
                                                          interleave=True,
                                                          transpose=True)

        ctx.save_for_backward(qkv, q_norm_weight.data, k_norm_weight.data,
                              freqs)
        ctx.H = H
        ctx.h = h
        ctx.eps = eps
        ctx.shape = shape
        return qo, ko, vo

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        qkv, q_norm_weight, k_norm_weight, freqs = ctx.saved_tensors

        dqkv, dqw, dkw = triton_qk_norm_and_half_rope_backward(grad_q,
                                                               grad_k,
                                                               grad_v,
                                                               qkv,
                                                               q_norm_weight,
                                                               k_norm_weight,
                                                               freqs,
                                                               eps=ctx.eps,
                                                               transpose=True,
                                                               interleave=True)
        return dqkv, dqw, dkw, None, None, None, None
