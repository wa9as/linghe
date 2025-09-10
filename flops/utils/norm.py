from re import S
import torch
import triton
import triton.language as tl



@triton.jit
def rms_norm_forward_kernel(x_ptr, weight_ptr, out_ptr, eps, M, T,
                            N: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)
    weight = tl.load(weight_ptr + tl.arange(0, N))[None, :]

    offs = pid * W * T * N + tl.arange(0, W)[:, None] * N + tl.arange(0, N)[
                                                            None, :]
    for i in range(T):
        x = tl.load(x_ptr + offs,
                    mask=pid * W * T + i * W + tl.arange(0, W)[:, None] < M).to(
            tl.float32)
        rms = tl.sqrt(tl.sum(x * x, axis=1) / N + eps)

        x = (x / rms[:, None]) * weight

        tl.store(out_ptr + offs, x,
                 mask=pid * W * T + i * W + tl.arange(0, W)[:, None] < M)
        offs += N * W


def triton_rms_norm_forward(x, weight, eps=1e-6, out=None):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device
    if out is None:
        out = torch.empty((M, N), device=device, dtype=x.dtype)

    W = 8192 // N
    T = 4
    grid = (M//(T*W),)
    rms_norm_forward_kernel[grid](
        x,
        weight,
        out,
        eps,
        M,
        T,
        N,
        W,
        num_stages=3,
        num_warps=4
    )
    return out


@triton.jit
def rms_norm_backward_kernel(
        grad_output_ptr,
        x_ptr,
        w_ptr,
        dx_ptr,
        dw_ptr,
        eps,
        M,
        T,
        N: tl.constexpr,
        W: tl.constexpr
):
    pid = tl.program_id(0)

    w = tl.load(w_ptr + tl.arange(0, N)).to(tl.float32)

    offs = pid * W * T * N + tl.arange(0, W)[:, None] * N + tl.arange(0, N)[
                                                            None, :]
    w_grads = tl.zeros((N,), dtype=tl.float32)
    for i in range(T):
        mask = pid * T + i < M
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        g = tl.load(grad_output_ptr + offs, mask=mask).to(tl.float32)
        rms = tl.sqrt(tl.sum(x * x, 1) / N + eps)
        r = 1.0 / rms[:, None]
        w_grad = x * g * r
        w_grads += tl.sum(w_grad, 0)

        dx = r * g * w - r * r * r * x * tl.sum(x * g * w, 1, keep_dims=True) / N

        tl.store(dx_ptr + offs, dx, mask=mask)

        offs += N * W

    tl.store(dw_ptr + pid * N + tl.arange(0, N), w_grads)


def triton_rms_norm_backward(grad_output, x, w, eps=1e-6):
    M, N = x.shape
    dx = torch.empty(M, N, dtype=x.dtype, device=x.device)

    W = 8192 // N
    T = 16
    assert 8192 % N ==0 and M % (T*W) == 0
    g = M//(T*W)
    tmp_dw = torch.empty(g, N, dtype=torch.float32, device=w.device)
    grid = (g,)
    rms_norm_backward_kernel[grid](
        grad_output,
        x,
        w,
        dx,
        tmp_dw,
        eps,
        M,
        T,
        N,
        W,
        num_stages=3,
        num_warps=4
    )
    return dx, tmp_dw.sum(dim=0).to(w.dtype)


@triton.jit
def rms_norm_and_smooth_quant_forward_kernel(x_ptr, weight_ptr, smooth_scale_ptr,
                                      out_ptr, scale_ptr, max_ptr, rms_ptr, 
                                      eps,
                                      M, 
                                      T, 
                                      N: tl.constexpr, 
                                      W: tl.constexpr,
                                      CALIBRATE: tl.constexpr,
                                      OUTPUT: tl.constexpr,
                                      ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    weight = tl.load(weight_ptr + tl.arange(0, N)).to(tl.float32)[None, :]
    smooth_scale = tl.load(smooth_scale_ptr + tl.arange(0, N))[None, :]
    smooth_scale = 1.0 / tl.maximum(smooth_scale, 1e-30)
    if CALIBRATE:
        # triton 3.3.1 has bug with N = 2048 and calibrate=True 
        maxs = tl.zeros((N, ), dtype=tl.float32)
    offs = pid * W * T * N + tl.arange(0, W)[:, None] * N + tl.arange(0, N)[
                                                            None, :]
    for i in range(T):
        indices = pid * W * T + i * W + tl.arange(0, W)
        x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32)
        rms = 1/tl.sqrt(tl.sum(x * x, axis=1) / N + eps)
        if OUTPUT:
            tl.store(rms_ptr + indices, rms, mask=indices < M)
        x = x * rms[:, None] * weight

        if CALIBRATE:
            maxs = tl.maximum(maxs, tl.max(tl.abs(x),0))

        x = x * smooth_scale
        scale = tl.maximum(tl.max(tl.abs(x), 1) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        q = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(scale_ptr + indices, scale, mask=indices < M)
        tl.store(out_ptr + offs, q, mask=indices[:, None] < M)
        offs += N * W

    if CALIBRATE:
        tl.store(max_ptr + pid * N + tl.arange(0, N), maxs)

# output non-transposed and transposed together
# should used with batchsize >= 16384
@triton.jit
def rms_norm_and_block_quant_forward_kernel(x_ptr, 
                                      weight_ptr,
                                      out_ptr, 
                                      scale_ptr, 
                                      transpose_output_ptr, 
                                      transpose_scale_ptr,
                                      rms_ptr, 
                                      eps,
                                      M, 
                                      T: tl.constexpr,
                                      N: tl.constexpr,
                                      nb: tl.constexpr,
                                      W: tl.constexpr, 
                                      H : tl.constexpr,
                                      ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)

    # row-wise read, row-wise write
    weight = tl.load(weight_ptr + tl.arange(0, N)).to(tl.float32)[None, :]
    offs = pid * W * T * N + tl.arange(0, W)[:, None] * N + tl.arange(0, N)[
                                                            None, :]
    for i in range(T):
        indices = pid * W * T + i * W + tl.arange(0, W)
        x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32)
        rms = tl.rsqrt(tl.sum(x * x, axis=1) / N + eps)
        tl.store(rms_ptr + indices, rms, mask=indices < M)
        x = (x * rms[:, None]) * weight
        x = tl.reshape(x, [W, nb, 128])
        scale = tl.maximum(tl.max(tl.abs(x), 2) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        x = (x / scale[:,:, None]).to(out_ptr.dtype.element_ty)
        x = tl.reshape(x, [W, N])
        tl.store(scale_ptr + indices[:, None] * nb + tl.arange(0, nb)[None, :], scale, mask=indices[:, None] < M)
        tl.store(out_ptr + offs, x, mask=indices[:, None] < M)
        offs += N * W

    tl.debug_barrier()

    offs = pid * W * T * N + tl.arange(0, 128)[:, None] * N + tl.arange(0, H)[
                                                            None, :]
    toffs = pid * 128 + tl.arange(0, H)[:, None] * M + tl.arange(0, 128)[
                                                            None, :]
    indices = pid * W * T + tl.arange(0, 128)
    rms = tl.load(rms_ptr + indices, mask=indices < M)[:, None]
    for i in range(N//H):
        x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32) 
        wgt = tl.load(weight_ptr + i * H +  tl.arange(0, H)).to(tl.float32)
        x = x * rms * wgt
        scale = tl.maximum(tl.max(x.abs(), 0) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(transpose_scale_ptr + pid * N + i * H + tl.arange(0, H), scale)
        x = (x/scale).to(transpose_output_ptr.dtype.element_ty)
        tl.store(transpose_output_ptr + toffs, tl.trans(x), mask=indices[None, :] < M)
        offs += H
        toffs += M * H



# output non-transposed tensor only
@triton.jit
def rms_norm_and_block_quant_forward_n_kernel(x_ptr, 
                                      weight_ptr,
                                      out_ptr, 
                                      scale_ptr, 
                                      rms_ptr, 
                                      eps,
                                      M: tl.constexpr, 
                                      T: tl.constexpr,
                                      N: tl.constexpr,
                                      nb: tl.constexpr,
                                      W: tl.constexpr, 
                                      ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)

    # row-wise read, row-wise write
    weight = tl.load(weight_ptr + tl.arange(0, N)).to(tl.float32)[None, :]
    offs = pid * W * T * N + tl.arange(0, W)[:, None] * N + tl.arange(0, N)[
                                                            None, :]
    for i in range(T):
        indices = pid * W * T + i * W + tl.arange(0, W)
        x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32)
        rms = tl.rsqrt(tl.sum(x * x, axis=1) / N + eps)
        tl.store(rms_ptr + indices, rms, mask=indices < M)

        x = x * rms[:, None] * weight
        x = tl.reshape(x, [W, nb, 128])
        scale = tl.maximum(tl.max(tl.abs(x), 2) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        # x = (x / scale[:,:, None]).to(out_ptr.dtype.element_ty)
        # x = tl.reshape(x, [W, N])

        x = x / scale[:,:, None]
        x = tl.reshape(x, [W, N])

        tl.store(scale_ptr + indices[:, None] * nb + tl.arange(0, nb)[None, :], scale, mask=indices[:, None] < M)
        tl.store(out_ptr + offs, x, mask=indices[:, None] < M)
        offs += N * W


# output transposed tensor only
@triton.jit
def rms_norm_and_block_quant_forward_t_kernel(x_ptr, 
                                      weight_ptr,
                                      transpose_output_ptr, 
                                      transpose_scale_ptr,
                                      rms_ptr, 
                                      M, 
                                      N,
                                      W: tl.constexpr, 
                                      ROUND: tl.constexpr):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)

    offs = rid * 128 * N + cid * W + tl.arange(0, 128)[:, None] * N + tl.arange(0, W)[
                                                            None, :]
    toffs = rid * 128 + cid * M * W + tl.arange(0, W)[:, None] * M + tl.arange(0, 128)[
                                                            None, :]

    weight = tl.load(weight_ptr + cid * W + tl.arange(0, W)).to(tl.float32)
    indices = rid * 128 + tl.arange(0, 128)
    rms = tl.load(rms_ptr + indices, mask=indices < M)[:, None]
    x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32) 
    x = x * rms * weight
    scale = tl.maximum(tl.max(x.abs(), 0) / 448.0, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    tl.store(transpose_scale_ptr + rid * N + cid * W + tl.arange(0, W), scale)

    x = (tl.trans(x/scale)).to(transpose_output_ptr.dtype.element_ty)
    tl.store(transpose_output_ptr + toffs, x, mask=indices[None, :] < M)


# rms is used for moe routing, it is stored as 1/rms
def triton_rms_norm_and_quant_forward(x, weight, smooth_scale=None, eps=1e-6,
                                      out=None, scale=None, rms=None, calibrate=False,
                                      output_rms=False, round_scale=False,
                                      output_mode=2):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192 and 8192 % N == 0
    device = x.device
    smooth = smooth_scale is not None

    if out is None and (smooth or output_mode in (0, 2)):
        out = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)

    if smooth:
        if scale is None:
            scale = torch.empty((M,), device=device, dtype=torch.float32)
        transpose_output = None 
        transpose_scale = None
        W = 8192 // N
        T = 8 if M // W >= 4096 else 4
        assert M % (T * W) == 0
        g = M // (T*W)
        if calibrate:
            maxs = torch.empty((g, N), dtype=torch.float32, device=device)
        else:
            maxs = None
        if output_rms and rms is None:
            rms = torch.empty((M,), dtype=torch.float32, device=device)
        grid = (g,)
        rms_norm_and_smooth_quant_forward_kernel[grid](
            x,
            weight,
            smooth_scale,
            out,
            scale,
            maxs,
            rms,
            eps,
            M,
            T,
            N,
            W,
            calibrate,
            output_rms,
            round_scale,
            num_stages=3,
            num_warps=2 if N == 2048 else 4
        )
        if calibrate:
            maxs = maxs.amax(0)
    else:
        if scale is None and output_mode in (0, 2):
            scale = torch.empty((M, N//128), device=device, dtype=torch.float32)
        if rms is None:
            rms = torch.empty((M,), dtype=torch.float32, device=device)
        maxs = None 
        # transpose_output should be initialized, or else can not make splitted tensors
        transpose_output = torch.empty((N, M), device=device, dtype=torch.float8_e4m3fn)
        transpose_scale = torch.empty(((M+127)//128, N), device=device, dtype=torch.float32)
        if output_mode == 0: # only output non-transpose tensor
            W = 8192 // N
            T = 16 // W  
            grid = (triton.cdiv(M, 16),)
            rms_norm_and_block_quant_forward_n_kernel[grid](
                x,
                weight,
                out,
                scale,
                rms,
                eps,
                M,
                T,
                N,
                N//128,
                W,
                round_scale,
                num_stages=3,
                num_warps=4
            )
            scale = scale.t().contiguous()

        elif output_mode == 1:  # only output transposed tensor
            # W = N//512
            # grid = (512,)
            W = 32 
            grid = (triton.cdiv(M, 128), N//W)
            rms_norm_and_block_quant_forward_t_kernel[grid](x, 
                                      weight,
                                      transpose_output, 
                                      transpose_scale,
                                      rms, 
                                      M, 
                                      N,
                                      W, 
                                      round_scale,
                                      num_stages=3,
                                      num_warps=4)
        
        elif output_mode == 2:  # output non-transposed and transposed tensor together
            W = 8192 // N
            T = 128 // W  # BLOCK SIZE
            H = 64
            grid = (triton.cdiv(M, 128),)
            rms_norm_and_block_quant_forward_kernel[grid](
                x,
                weight,
                out,
                scale,
                transpose_output,
                transpose_scale,
                rms,
                eps,
                M,
                T,
                N,
                N//128,
                W,
                H,
                round_scale,
                num_stages=3,
                num_warps=16
            )
            scale = scale.t().contiguous()

    return out, scale, maxs, rms, transpose_output, transpose_scale
