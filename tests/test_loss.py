import torch

from flops.utils.benchmark import benchmark_func
from flops.utils.loss import triton_softmax_cross_entropy_forward, \
    triton_softmax_cross_entropy_backward
from flops.utils.util import output_check


def torch_cross_entropy(logits, targets):
    float_logits = logits.float()
    loss = torch.nn.functional.cross_entropy(
        float_logits.view(-1, logits.size()[-1]),
        targets.view(-1),
        reduction='none').mean()
    loss.backward()
    return loss, logits.grad

def test_triton_softmax_cross_entropy(M=4096, N=157184, coef=1.0, bench=False):
    device = 'cuda:0'
    logits = torch.randn((M, N), dtype=torch.bfloat16, device=device,
                         requires_grad=True)
    logits = (logits*coef).detach().clone().requires_grad_()
    targets = (torch.rand((M,), dtype=torch.float32, device=device) * N).to(
        torch.int64)
    input_grad = 1 / M * torch.ones((M,), dtype=torch.bfloat16, device=device)
    loss_ref, grad_ref = torch_cross_entropy(logits, targets)
    loss, sum_exp, max_logit = triton_softmax_cross_entropy_forward(logits, targets)
    grad = triton_softmax_cross_entropy_backward(logits, targets, sum_exp, max_logit,
                                                 input_grad)
    output_check(loss_ref, loss.mean(), mode='loss')
    output_check(grad_ref.float(), grad.float(), mode='grad')
    if bench:
        benchmark_func(torch_cross_entropy, logits, targets,
                                  ref_bytes=M * N * 6)
        benchmark_func(triton_softmax_cross_entropy_forward, logits, targets,
                       ref_bytes=M * N * 2)
        benchmark_func(triton_softmax_cross_entropy_backward, logits, targets,
                       sum_exp, max_logit, input_grad, ref_bytes=M * N * 4)
if __name__ == '__main__':
    test_triton_softmax_cross_entropy(M=8192, N=157184, coef=1.0, bench=True)
    test_triton_softmax_cross_entropy(M=8192, N=157184, coef=100.0, bench=True)