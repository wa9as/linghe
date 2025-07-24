from flops.utils.norm import *


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        M, N = x.shape
        assert N <= 8192
        device = x.device 
        out = torch.empty((M, N), device=device, dtype=x.dtype)

        sm = torch.cuda.get_device_properties(device).multi_processor_count
        
        W = 8192//N 
        T = triton.cdiv(M, sm*W)
        grid = (sm, )

        rms_norm_forward_kernel[grid](
            x,
            weight,
            out,
            # norm,
            eps,
            M, T,
            N, 
            W,
            num_stages=3,
            num_warps=16
        )

        # ctx.save_for_backward(x, weight, norm)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        ctx.N = N

        return out
    
    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors

        sm = torch.cuda.get_device_properties(x.device).multi_processor_count

        M, N = x.shape
        T = triton.cdiv(M, sm)
        dx = torch.empty(M, N, dtype=x.dtype, device=x.device)
        tmp_dw = torch.empty(sm, N, dtype=torch.float32, device=x.device)
        
        eps = ctx.eps
        grid = (sm, )

        rms_norm_backward_kernel[grid](
            dy,
            x,
            weight,
            dx,
            tmp_dw,
            eps,
            M,
            T,
            N,
            num_stages=3,
            num_warps=16
        )

        return dx, tmp_dw.sum(dim=0).to(x.dtype), None