from flops.utils.norm import *

class RMSNormtriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        M, N = x.shape
        assert N <= 8192
        device = x.device 
        # if out is None:
        out = torch.empty((M, N), device=device, dtype=x.dtype)
        # if norm is None:
        norm = torch.empty((M,), device=device, dtype=torch.float32)
        W = 8192//N 
        T = triton.cdiv(M, 132*W)
        grid = lambda META: (132, )

        rms_norm_forward_kernel[grid](
            x,
            weight,
            out,
            norm,
            eps,
            M, T,
            N, 
            W,
            num_stages=3,
            num_warps=16
        )

        ctx.save_for_backward(x, weight, norm)
        ctx.eps = eps
        ctx.N = N

        return out
    
    @staticmethod
    def backward(ctx, dy):
        x, weight, norm = ctx.saved_tensors

        M, N = x.shape
        dx = torch.empty(M, N, dtype=torch.float32, device=x.device)
        BLOCK_N = min(triton.next_power_of_2(N), 1024)
        
        grid = lambda META: (M,)

        rms_norm_backward_kernel[grid](
            dx,
            dy,
            x,
            weight,
            norm,
            M, N,
            BLOCK_N,
            num_stages=3,
            num_warps=8
        )

        return dx, torch.sum(dy * x * norm.unsqueeze(-1), dim=0)