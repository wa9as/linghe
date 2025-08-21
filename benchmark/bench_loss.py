import torch
from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

from flops.utils.benchmark import benchmark_func
from flops.utils.loss import (triton_softmax_cross_entropy_backward,
                              triton_softmax_cross_entropy_forward)


def torch_cross_entropy(logits, targets):
    float_logits = logits.float()
    loss = torch.nn.functional.cross_entropy(
        float_logits.view(-1, logits.size()[-1]),
        targets.view(-1),
        reduction='none').mean()
    loss.backward()
    return loss, logits.grad

def te_cross_entropy_forward(logits, targets):
    float_logits = logits[None]
    loss = parallel_cross_entropy(float_logits,
                                  targets[None]).mean()
    return loss

def te_cross_entropy_forward_backward(logits, targets):
    float_logits = logits[None]
    loss = parallel_cross_entropy(float_logits,
                                  targets[None]).mean()
    loss.backward()
    return loss, logits.grad

def triton_cross_entropy_forward_backward(logits, targets, input_grad):
    loss, sum_exp, max_logits = triton_softmax_cross_entropy_forward(logits, targets)
    output_grad = triton_softmax_cross_entropy_backward(logits, targets,
                   sum_exp, max_logits, input_grad)
    return loss, output_grad

def bench_triton_softmax_cross_entropy(M=4096, N=157184):
    device = 'cuda:0'
    logits = torch.randn((M, N), dtype=torch.bfloat16, device=device,
                         requires_grad=True)
    targets = (torch.rand((M,), dtype=torch.float32, device=device) * N).to(
        torch.int64)
    input_grad = 1 / M * torch.ones((M,), dtype=torch.bfloat16, device=device)

    sum_exp = torch.rand((M,), dtype=torch.float32, device=device)
    max_logits = torch.rand((M,), dtype=torch.float32, device=device)

    ref_time = benchmark_func(torch_cross_entropy, logits, targets,
                              ref_bytes=M * N * 4)
    benchmark_func(triton_softmax_cross_entropy_forward, logits, targets,
                   ref_bytes=M * N * 2, ref_time=ref_time)
    benchmark_func(te_cross_entropy_forward, logits, targets,
                   ref_bytes=M * N * 4, ref_time=ref_time)

    benchmark_func(triton_softmax_cross_entropy_backward, logits, targets,
                   sum_exp, max_logits, input_grad, ref_bytes=M * N * 4, ref_time=ref_time)

    benchmark_func(te_cross_entropy_forward_backward, logits, targets, ref_bytes=M * N * 4,
                   ref_time=ref_time)
    benchmark_func(triton_cross_entropy_forward_backward, logits, targets, input_grad, ref_bytes=M * N * 4,
                   ref_time=ref_time)

if __name__ == '__main__':
    bench_triton_softmax_cross_entropy(M=8192, N=157184)
