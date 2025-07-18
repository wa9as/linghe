import math
import os
import torch
import triton
import triton.language as tl
from triton import Config


# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

@triton.jit
def softmax_cross_entropy_forward_kernel(logit_ptr, label_ptr, loss_ptr, M, N, H: tl.constexpr, W: tl.constexpr):
    
    pid = tl.program_id(axis=0)
    weight = tl.load(weight_ptr+tl.arange(0, N))[None,:]

    offs = pid*W*T*N+tl.arange(0, W)[:,None]*N+tl.arange(0, N)[None,:]
    for i in range(T):
        x = tl.load(x_ptr+offs, mask=pid*W*T+i*W+tl.arange(0, W)[:,None]<M).to(tl.float32)
        rms = tl.sqrt(tl.sum(x*x, axis=1)/N+eps)

        x = (x/rms[:,None])*weight

        tl.store(out_ptr+offs, x, mask=pid*W*T+i*W+tl.arange(0, W)[:,None]<M)
        offs += N*W


def triton_softmax_cross_entropy_forward(logits, labels):
    M, N = logits.shape
    device = logits.device 
    loss = torch.empty((M, ), device=device, dtype=torch.float32)

    # sm = torch.cuda.get_device_properties(device).multi_processor_count
    H = 8
    W = 128
    assert N%W == 0
    grid = (M//H, )
    softmax_cross_entropy_forward_kernel[grid](
        logits,
        labels,
        loss,
        M, 
        N, 
        H,
        W,
        num_stages=3,
        num_warps=16
    )
    return loss.sum()



# @triton.jit
# def softmax_cross_entropy_backward_kernel(
#     grad_output_ptr,
#     x_ptr,
#     w_ptr,
#     dx_ptr,
#     dw_ptr,
#     eps,
#     M,
#     T,
#     N: tl.constexpr
# ):
#     pid = tl.program_id(0)

#     w = tl.load(w_ptr+tl.arange(0, N))

#     offsets = pid*T*N+tl.arange(0, N)
#     w_grads = tl.zeros((N,), dtype=tl.float32)
#     for i in range(T):
#         mask = pid*T+i<M
#         x = tl.load(x_ptr+offsets, mask=mask).to(tl.float32)
#         g = tl.load(grad_output_ptr+offsets, mask=mask)
#         rms = tl.sqrt(tl.sum(x*x)/N+eps)
#         r = 1.0/rms
#         w_grad = x * g * r
#         w_grads += w_grad

#         dx = r*g*w - r*r*r/N*x*tl.sum(x*g*w)
        
#         tl.store(dx_ptr+offsets, dx, mask=mask)

#         offsets += N
    
#     tl.store(dw_ptr+pid*N+tl.arange(0, N), w_grads)


# def triton_softmax_cross_entropy_backward(grad_output, x, w, eps=1e-6):
#     M, N = x.shape
#     dx = torch.empty(M, N, dtype=x.dtype, device=x.device)

#     sm = torch.cuda.get_device_properties(x.device).multi_processor_count

#     T = triton.cdiv(M, sm)
#     tmp_dw = torch.empty(sm, N, dtype=torch.float32, device=x.device)
#     grid = (sm, )
#     softmax_cross_entropy_backward_kernel[grid](
#         grad_output,
#         x,
#         w,
#         dx,
#         tmp_dw,
#         eps,
#         M,
#         T,
#         N,
#         num_stages=3,
#         num_warps=16
#     )
#     return dx, tmp_dw.sum(dim=0).to(x.dtype)
    