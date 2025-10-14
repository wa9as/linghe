# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core.fusions.fused_cross_entropy import \
    fused_vocab_parallel_cross_entropy
from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

from flops.tools.benchmark import benchmark_func
from flops.tools.util import output_check
from flops.utils.loss import (triton_softmax_cross_entropy_backward,
                              triton_softmax_cross_entropy_forward)


def torch_cross_entropy(logits, targets):
    float_logits = logits.float()
    losses = torch.nn.functional.cross_entropy(
        float_logits.view(-1, float_logits.size()[-1]),
        targets.view(-1),
        reduction='none')
    loss = losses.mean()
    loss.backward()
    return losses, logits.grad


def te_cross_entropy_forward_backward(logits, targets):
    losses = parallel_cross_entropy(logits[None],
                                    targets[None])
    loss = losses.mean()
    loss.backward()
    return losses, logits.grad


def fused_cross_entropy_forward_backward(logits, targets, pg):
    losses = fused_vocab_parallel_cross_entropy(logits[None],
                                                targets[None],
                                                pg)[0]
    loss = losses.mean()
    loss.backward()
    return losses, logits.grad


def triton_cross_entropy_forward_backward(logits, targets, input_grad):
    losses, sum_exp, max_logits = triton_softmax_cross_entropy_forward(logits,
                                                                       targets)
    output_grad = triton_softmax_cross_entropy_backward(logits, targets,
                                                        sum_exp, max_logits,
                                                        input_grad)
    return losses, output_grad


def bench_triton_softmax_cross_entropy(M=4096, N=157184):
    device = 'cuda:0'
    logits = torch.randn((M, N), dtype=torch.bfloat16, device=device,
                         requires_grad=True) ** 1
    logits = logits.detach().clone().requires_grad_()
    # targets = (torch.rand((M,), dtype=torch.float32, device=device) * N).to(
    #     torch.int64)
    targets = torch.topk(logits, 4)[1][:, 3].contiguous()
    input_grad = 1 / M * torch.ones((M,), dtype=torch.bfloat16, device=device)

    sum_exp = torch.rand((M,), dtype=torch.float32, device=device)
    max_logits = torch.rand((M,), dtype=torch.float32, device=device)

    pg = dist.new_group(ranks=[0], backend='nccl')
    torch_losses, torch_grad = torch_cross_entropy(
        logits.detach().clone().requires_grad_(), targets)
    fused_losses, fused_grad = fused_cross_entropy_forward_backward(
        logits.detach().clone().requires_grad_(), targets, pg)
    triton_losses, sum_exp_, max_logit_ = triton_softmax_cross_entropy_forward(
        logits, targets)
    output_check(torch_losses, triton_losses)
    output_check(fused_losses, triton_losses)
    output_check(torch_losses, fused_losses)

    ref_time = benchmark_func(torch_cross_entropy,
                              logits.detach().clone().requires_grad_(), targets,
                              ref_bytes=M * N * 4)
    benchmark_func(triton_softmax_cross_entropy_forward, logits, targets,
                   ref_bytes=M * N * 2, ref_time=ref_time)
    benchmark_func(triton_softmax_cross_entropy_backward, logits, targets,
                   sum_exp, max_logits, input_grad, ref_bytes=M * N * 4,
                   ref_time=ref_time)

    benchmark_func(te_cross_entropy_forward_backward,
                   logits.detach().clone().requires_grad_(), targets,
                   ref_bytes=M * N * 4,
                   ref_time=ref_time)
    benchmark_func(te_cross_entropy_forward_backward,
                   logits.detach().clone().requires_grad_(), targets,
                   ref_bytes=M * N * 4,
                   ref_time=ref_time)
    benchmark_func(triton_cross_entropy_forward_backward, logits, targets,
                   input_grad, ref_bytes=M * N * 4,
                   ref_time=ref_time)


if __name__ == '__main__':
    init_method = "env://"
    dist.init_process_group(backend='nccl', init_method=init_method,
                            world_size=1, rank=0,
                            timeout=timedelta(seconds=30))
    bench_triton_softmax_cross_entropy(M=8192, N=157184)
