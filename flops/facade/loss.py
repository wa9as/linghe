
import torch
from flops.utils.loss import triton_softmax_cross_entropy_forward, triton_softmax_cross_entropy_backward

class SoftmaxCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, est_max_logit=0.0, inplace=False):
        shape = logits.shape
        if len(shape) == 3:
            logits = logits.view(-1, shape[-1])
        loss, sum_exp = triton_softmax_cross_entropy_forward(logits, labels, est_max_logit=est_max_logit)
        ctx.save_for_backward(logits, labels, sum_exp)
        ctx.est_max_logit = est_max_logit
        ctx.inplace = inplace
        ctx.shape = shape 
        if len(shape) == 3:
            loss = loss.view(shape[0],shape[1])
        return loss
        
    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, sum_exp = ctx.saved_tensors
        shape = ctx.shape 
        grad = logits if ctx.inplace else None
        grad = triton_softmax_cross_entropy_backward(logits, labels, sum_exp, grad_output, output_grad=grad, est_max_logit=ctx.est_max_logit)
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
        scale = 1/torch.pow(array.float(), ctx.coef)
        grad = grad_output * scale
        return grad, None