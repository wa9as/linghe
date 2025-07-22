
import torch

from flops.facade.loss import SoftmaxCrossEntropyFunction
from flops.utils.loss import *
from flops.utils.util import output_check
from flops.utils.benchmark import benchmark_func
from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy


def torch_cross_entropy(logits, targets):
    float_logits = logits.float()
    loss = torch.nn.functional.cross_entropy(float_logits.view(-1, logits.size()[-1]),
                           targets.view(-1),
                           reduction='none').mean()
    loss.backward()
    return loss, logits.grad


def te_cross_entropy(logits, targets):
    float_logits = logits[None].float()
    loss = parallel_cross_entropy(float_logits,
                           targets[None]).mean()
    loss.backward()
    return loss, logits.grad


M, N = 4096*4, 157184
device = 'cuda:0'
logits = torch.randn((M,N), dtype=torch.bfloat16, device=device, requires_grad=True)
targets = (torch.rand((M,), dtype=torch.float32, device=device)*N).to(torch.int64)
input_grad = 1/M*torch.ones((M,),dtype=torch.bfloat16,device=device)

loss_ref, grad_ref = torch_cross_entropy(logits, targets)
loss, sum_exp = triton_softmax_cross_entropy_forward(logits, targets)
grad = triton_softmax_cross_entropy_backward(logits, targets, sum_exp, input_grad)
output_check(loss_ref,loss.mean(),mode='loss')
output_check(grad_ref.float(),grad.float(),mode='grad')

ref_time = benchmark_func(torch_cross_entropy, logits, targets, ref_bytes=M*N*4)
benchmark_func(triton_softmax_cross_entropy_forward, logits, targets, ref_bytes=M*N*2, ref_time=ref_time)
benchmark_func(triton_softmax_cross_entropy_backward, logits, targets, sum_exp, input_grad, ref_bytes=M*N*4, ref_time=ref_time)
benchmark_func(parallel_cross_entropy, logits[None], targets[None], ref_bytes=M*N*4, ref_time=ref_time)
benchmark_func(te_cross_entropy, logits, targets, ref_bytes=M*N*4, ref_time=ref_time)
