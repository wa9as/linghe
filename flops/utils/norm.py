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

    sm = torch.cuda.get_device_properties(device).multi_processor_count

    W = 8192 // N
    T = triton.cdiv(M, sm * W)
    grid = (sm,)
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
        num_warps=16
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
        N: tl.constexpr
):
    pid = tl.program_id(0)

    w = tl.load(w_ptr + tl.arange(0, N))

    offsets = pid * T * N + tl.arange(0, N)
    w_grads = tl.zeros((N,), dtype=tl.float32)
    for i in range(T):
        mask = pid * T + i < M
        x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
        g = tl.load(grad_output_ptr + offsets, mask=mask)
        rms = tl.sqrt(tl.sum(x * x) / N + eps)
        r = 1.0 / rms
        w_grad = x * g * r
        w_grads += w_grad

        dx = r * g * w - r * r * r / N * x * tl.sum(x * g * w)

        tl.store(dx_ptr + offsets, dx, mask=mask)

        offsets += N

    tl.store(dw_ptr + pid * N + tl.arange(0, N), w_grads)


def triton_rms_norm_backward(grad_output, x, w, eps=1e-6):
    M, N = x.shape
    dx = torch.empty(M, N, dtype=x.dtype, device=x.device)

    sm = torch.cuda.get_device_properties(x.device).multi_processor_count

    T = triton.cdiv(M, sm)
    tmp_dw = torch.empty(sm, N, dtype=torch.float32, device=x.device)
    grid = (sm,)
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
        num_stages=3,
        num_warps=16
    )
    return dx, tmp_dw.sum(dim=0).to(x.dtype)


@triton.jit
def rms_norm_and_smooth_quant_forward_kernel(x_ptr, weight_ptr, smooth_scale_ptr,
                                      out_ptr, scale_ptr, max_ptr, rms_ptr, eps,
                                      M, T, N: tl.constexpr, W: tl.constexpr,
                                      CALIBRATE: tl.constexpr,
                                      OUTPUT: tl.constexpr,
                                      ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    weight = tl.load(weight_ptr + tl.arange(0, N))[None, :]
    smooth_scale = tl.load(smooth_scale_ptr + tl.arange(0, N))[None, :]
    smooth_scale = 1.0 / smooth_scale
    if CALIBRATE:
        maxs = tl.zeros((N, ), dtype=tl.float32)
    offs = pid * W * T * N + tl.arange(0, W)[:, None] * N + tl.arange(0, N)[
                                                            None, :]
    for i in range(T):
        indices = pid * W * T + i * W + tl.arange(0, W)
        x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32)
        rms = tl.rsqrt(tl.sum(x * x, axis=1) / N + eps)
        if OUTPUT:
            tl.store(rms_ptr + indices, rms, mask=indices < M)
        x = x * rms[:, None] * weight
        if CALIBRATE:
            maxs = tl.maximum(maxs, tl.max(tl.abs(x), 0))
        x = x * smooth_scale
        scale = tl.maximum(tl.max(tl.abs(x), 1) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        x = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(scale_ptr + indices, scale, mask=indices < M)
        tl.store(out_ptr + offs, x, mask=indices[:, None] < M)
        offs += N * W

    if CALIBRATE:
        # maxs = tl.max(maxs, 0)
        tl.store(max_ptr + pid * N + tl.arange(0, N), maxs)




@triton.jit
def rms_norm_and_block_quant_forward_kernel(x_ptr, weight_ptr,
                                      out_ptr, scale_ptr, 
                                      transpose_output_ptr, transpose_scale_ptr,
                                      rms_ptr, 
                                      eps,
                                      M, 
                                      T: tl.constexpr,
                                      N: tl.constexpr,
                                      nb: tl.constexpr,
                                      W: tl.constexpr, 
                                      H : tl.constexpr,
                                      OUTPUT_RMS: tl.constexpr,
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
        if OUTPUT_RMS:
            tl.store(rms_ptr + indices, rms, mask=indices < M)
        x = (x * rms[:, None]) * weight
        # maxs = tl.maximum(maxs, tl.abs(x))
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
        indices = pid * W * T + tl.arange(0, 128)
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


# rms is used for moe routing, it is stored as 1/rms
def triton_rms_norm_and_quant_forward(x, weight, smooth_scale=None, eps=1e-6,
                                      out=None, scale=None, calibrate=False,
                                      output_rms=False, round_scale=False,
                                      transpose=False):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device
    smooth = smooth_scale is not None
    if out is None:
        out = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)

    # blockwise must write rms
    if output_rms:
        rms = torch.empty((M,), dtype=torch.float32, device=device)
    else:
        rms = None

    sm = torch.cuda.get_device_properties(
        x.device).multi_processor_count

    if smooth:
        if scale is None:
            scale = torch.empty((M,), device=device, dtype=torch.float32)
        transpose_output = None 
        transpose_scale = None
        W = 8192 // N
        sm = torch.cuda.get_device_properties(device).multi_processor_count
        T = triton.cdiv(M, sm * W)
        if calibrate:
            maxs = torch.empty((sm, N), dtype=torch.float32, device=device)
        else:
            maxs = None
        grid = (sm,)
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
            num_warps=16
        )
        maxs = maxs.amax(0)
    else:
        if scale is None:
            scale = torch.empty((M, N//128), device=device, dtype=torch.float32)
        # recomputation must be used with this kernel
        # transpose = False: forward, output rms, only output non-transposed tensor
        # transpose = True: backward, input rms, only output transposed tensor
        transpose_output = torch.empty((N, M), device=device, dtype=torch.float8_e4m3fn)
        transpose_scale = torch.empty(((M+127)//128, N), device=device, dtype=torch.float32)
        maxs = None 
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
            output_rms,
            round_scale,
            num_stages=3,
            num_warps=16
        )
        scale = scale.t().contiguous()
    return out, scale, maxs, rms, transpose_output, transpose_scale
