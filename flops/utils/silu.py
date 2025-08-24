import torch
import triton
import triton.language as tl




# n is power of 2
@triton.jit
def silu_and_smooth_quant_forward_kernel(x_ptr, smooth_scale_ptr, out_ptr, scale_ptr,
                                  max_ptr, M, T, n: tl.constexpr,
                                  W: tl.constexpr, ROUND: tl.constexpr,
                                  CALIBRATE: tl.constexpr):
    pid = tl.program_id(axis=0)

    row_offs = pid * T * W * n + tl.arange(0, W)[:, None] * n
    col_offs = tl.arange(0, n)[None, :]
    smooth_scale = tl.load(smooth_scale_ptr + tl.arange(0, n))
    smooth_scale = 1.0 / smooth_scale
    if CALIBRATE:
        maxs = tl.zeros((W, n), dtype=tl.float32)

    for i in range(T):
        indices = pid * T * W + i * W + tl.arange(0, W)
        mask = indices[:, None] < M
        x1 = tl.load(x_ptr + row_offs * 2 + col_offs, mask=mask).to(tl.float32)
        x2 = tl.load(x_ptr + n + row_offs * 2 + col_offs, mask=mask).to(
            tl.float32)
        x = x1 / (1 + tl.exp(-x1)) * x2
        if CALIBRATE:
            maxs = tl.maximum(x.abs(), maxs)
        x = x * smooth_scale
        scale = tl.maximum(tl.max(x.abs(), 1) / 448, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(scale_ptr + indices, scale, mask=indices < M)
        x = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row_offs + col_offs, x, mask=mask)
        row_offs += n * W

    if CALIBRATE:
        maxs = tl.max(maxs, 0)
        tl.store(max_ptr + pid * n + tl.arange(0, n), maxs)


# n is NOT power of 2
@triton.jit
def compatible_silu_and_smooth_quant_forward_kernel(x_ptr, smooth_scale_ptr, out_ptr,
                                             scale_ptr, max_ptr, M,
                                             T: tl.constexpr, n: tl.constexpr,
                                             B: tl.constexpr,
                                             ROUND: tl.constexpr,
                                             CALIBRATE: tl.constexpr):
    pid = tl.program_id(axis=0)

    # rowwise read with block size [T, B]
    row_offs = pid * T * n + tl.arange(0, T)[:, None] * n
    col_offs = tl.arange(0, B)[None, :]

    nb = n // B
    maxs = tl.zeros((T,), dtype=tl.float32)
    for i in range(nb):

        smooth_scale = tl.load(smooth_scale_ptr + i * B + tl.arange(0, B))
        x1 = tl.load(x_ptr + row_offs * 2 + col_offs).to(tl.float32)
        x2 = tl.load(x_ptr + n + row_offs * 2 + col_offs).to(tl.float32)
        x = x1 / (1 + tl.exp(-x1)) * x2
        if CALIBRATE:
            x_maxs = tl.max(x.abs(), 0)
            tl.store(max_ptr + pid * n + i * B + tl.arange(0, B), x_maxs)
        x = x / smooth_scale
        maxs = tl.maximum(tl.max(x.abs(), 1), maxs)
        col_offs += B

    scale = tl.maximum(maxs / 448, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    tl.store(scale_ptr + pid * T + tl.arange(0, T), scale)

    col_offs = tl.arange(0, B)[None, :]
    for i in range(nb):
        smooth_scale = tl.load(smooth_scale_ptr + i * B + tl.arange(0, B))

        x1 = tl.load(x_ptr + row_offs * 2 + col_offs).to(tl.float32)
        x2 = tl.load(x_ptr + n + row_offs * 2 + col_offs).to(tl.float32)
        x = x1 / (1 + tl.exp(-x1)) * x2
        x = x / smooth_scale

        x = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row_offs + col_offs, x)
        col_offs += B




@triton.jit
def silu_and_block_quant_forward_kernel(x_ptr, 
                                        out_ptr, scale_ptr,
                                        transpose_output_ptr, transpose_scale_ptr,
                                        M, 
                                        n: tl.constexpr, 
                                        ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)

    offs = pid * 128 * n * 2 + tl.arange(0, 128)[:, None] * n * 2 + tl.arange(0, 128)[None, :]
    hoffs = pid * 128 * n + tl.arange(0, 128)[:, None] * n + tl.arange(0, 128)[None, :]
    toffs = pid * 128 + tl.arange(0, 128)[:, None] * M + tl.arange(0, 128)[
                                                            None, :]
    indices = pid * 128 + tl.arange(0, 128)
    mask = indices[:, None] < M
    for i in range(n//128):
        x1 = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        x2 = tl.load(x_ptr + n + offs, mask=mask).to(
            tl.float32)
        x = x1 / (1 + tl.exp(-x1)) * x2
        scale = tl.maximum(tl.max(x.abs(), 1) / 448, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(scale_ptr + pid * 128 + i * M + tl.arange(0, 128), scale, mask=indices < M)
        xq = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + hoffs, xq, mask=mask)
        offs += 128
        hoffs += 128

        scale = tl.maximum(tl.max(x.abs(), 0) / 448, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(transpose_scale_ptr + pid * n + i * 128 + tl.arange(0, 128), scale)
        xq = (x / scale).to(out_ptr.dtype.element_ty)
        tl.store(transpose_output_ptr + toffs, tl.trans(xq), mask=indices[None, :] < M)
        toffs += M * 128


# used in shared expert
def triton_silu_and_quant_forward(x, smooth_scale=None, out=None, scale=None,
                                  maxs=None, round_scale=False,
                                  calibrate=False):
    # row-wise read, row-wise write
    M, N = x.shape
    device = x.device
    smooth = smooth_scale is not None 
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    if out is None:
        out = torch.empty((M, N // 2), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        if smooth:
            scale = torch.empty((M,), device=device, dtype=torch.float32)
        else:
            scale = torch.empty((N // 2 // 128, M), device=device, dtype=torch.float32)

    if smooth:
        transpose_output = None 
        transpose_scale = None
    else:
        transpose_output = torch.empty((N // 2, M), device=device, dtype=torch.float8_e4m3fn) 
        transpose_scale = torch.empty((triton.cdiv(M, 128), N // 2), device=device, dtype=torch.float32)




    if smooth:
        if triton.next_power_of_2(N) == N and N <= 8192:
            W = 8192 // N
            T = triton.cdiv(M, sm * W)
            if maxs is None and calibrate:
                maxs = torch.empty((sm, N // 2), device=device, dtype=torch.float32)
            grid = (sm,)
            silu_and_smooth_quant_forward_kernel[grid](
                x,
                smooth_scale,
                out,
                scale,
                maxs,
                M,
                T,
                N // 2,
                W,
                round_scale,
                calibrate,
                num_stages=2,
                num_warps=16
            )
        else:
            B = 512
            T = 16
            assert N // 2 % B == 0 and M % T == 0
            grid = (M // T,)
            if maxs is None and calibrate:
                maxs = torch.empty((M // T, N // 2), device=device, dtype=torch.float32)
            compatible_silu_and_smooth_quant_forward_kernel[grid](
                x,
                smooth_scale,
                out,
                scale,
                maxs,
                M,
                T,
                N // 2,
                B,
                round_scale,
                calibrate,
                num_stages=2,
                num_warps=16
            )

        if calibrate:
            maxs = maxs.amax(0)

    else:
        grid = (triton.cdiv(M, 128), )
        silu_and_block_quant_forward_kernel[grid](
            x,
            out,
            scale,
            transpose_output, 
            transpose_scale,
            M,
            N // 2,
            round_scale,
            num_stages=2,
            num_warps=16
        )

    return out, scale, maxs, transpose_output, transpose_scale



@triton.jit
def silu_and_smooth_quant_backward_kernel(g_ptr, x_ptr, smooth_scale_ptr, dx_ptr,
                                   dx_scale_ptr, M, T, n: tl.constexpr,
                                   W: tl.constexpr, REVERSE: tl.constexpr,
                                   ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)

    smooth_scale_1 = tl.load(smooth_scale_ptr + tl.arange(0, n))
    smooth_scale_2 = tl.load(smooth_scale_ptr + n + tl.arange(0, n))
    if not REVERSE:
        smooth_scale_1 = 1 / smooth_scale_1
        smooth_scale_2 = 1 / smooth_scale_2

    offs = pid * W * T * n * 2 + tl.arange(0, W)[:, None] * n * 2 + tl.arange(0,
                                                                              n)[
                                                                    None, :]
    hoffs = pid * W * T * n + tl.arange(0, W)[:, None] * n + tl.arange(0, n)[
                                                             None, :]
    for i in range(T):
        mask = pid * W * T + i * W + tl.arange(0, W)
        x1 = tl.load(x_ptr + offs, mask=mask[:, None] < M).to(tl.float32)
        x2 = tl.load(x_ptr + offs + n, mask=mask[:, None] < M).to(tl.float32)
        g = tl.load(g_ptr + hoffs, mask=mask[:, None] < M).to(tl.float32)
        sigmoid = 1 / (1 + tl.exp(-x1))
        dx1 = g * x2 * sigmoid * (
                1 + x1 * (1 - sigmoid)) * smooth_scale_1
        dx2 = g * x1 * sigmoid * smooth_scale_2

        scale = tl.maximum(
            tl.maximum(tl.max(dx1.abs(), 1), tl.max(dx2.abs(), 1)) / 448, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        dx1 = (dx1 / scale[:, None]).to(dx_ptr.dtype.element_ty)
        dx2 = (dx2 / scale[:, None]).to(dx_ptr.dtype.element_ty)

        tl.store(dx_scale_ptr + mask, scale, mask=mask < M)
        tl.store(dx_ptr + offs, dx1, mask=mask[:, None] < M)
        tl.store(dx_ptr + offs + n, dx2, mask=mask[:, None] < M)

        offs += n * W * 2
        hoffs += n * W



@triton.jit
def compatible_silu_and_quant_backward_kernel(g_ptr, x_ptr, smooth_scale_ptr,
                                              dx_ptr, dx_scale_ptr, M,
                                              T: tl.constexpr, n: tl.constexpr,
                                              B: tl.constexpr,
                                              REVERSE: tl.constexpr,
                                              ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)

    offs = pid * T * n * 2 + tl.arange(0, T)[:, None] * n * 2 + tl.arange(0, B)[
                                                                None, :]
    hoffs = pid * T * n + tl.arange(0, T)[:, None] * n + tl.arange(0, B)[None,
                                                         :]
    nb = n // B
    maxs = tl.zeros((T,), dtype=tl.float32)
    for i in range(nb):
        smooth_scale_1 = tl.load(smooth_scale_ptr + i * B + tl.arange(0, B))
        smooth_scale_2 = tl.load(smooth_scale_ptr + n + i * B + tl.arange(0, B))
        if not REVERSE:
            smooth_scale_1 = 1 / smooth_scale_1
            smooth_scale_2 = 1 / smooth_scale_2

        x1 = tl.load(x_ptr + offs).to(tl.float32)
        x2 = tl.load(x_ptr + offs + n).to(tl.float32)
        g = tl.load(g_ptr + hoffs).to(tl.float32)
        sigmoid = 1 / (1 + tl.exp(-x1))
        dx1 = g * x2 * sigmoid * (
                1 + x1 * (1 - sigmoid)) * smooth_scale_1
        dx2 = g * x1 * sigmoid * smooth_scale_2

        maxs = tl.maximum(
            tl.maximum(tl.max(dx1.abs(), 1), tl.max(dx2.abs(), 1)), maxs)

        offs += B
        hoffs += B

    scale = tl.maximum(maxs / 448, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    tl.store(dx_scale_ptr + pid * T + tl.arange(0, T), scale)

    s = 1 / scale[:, None]
    offs = pid * T * n * 2 + tl.arange(0, T)[:, None] * n * 2 + tl.arange(0, B)[
                                                                None, :]
    hoffs = pid * T * n + tl.arange(0, T)[:, None] * n + tl.arange(0, B)[None,
                                                         :]
    for i in range(nb):
        smooth_scale_1 = tl.load(smooth_scale_ptr + i * B + tl.arange(0, B))
        smooth_scale_2 = tl.load(smooth_scale_ptr + n + i * B + tl.arange(0, B))
        if not REVERSE:
            smooth_scale_1 = 1 / smooth_scale_1
            smooth_scale_2 = 1 / smooth_scale_2

        x1 = tl.load(x_ptr + offs).to(tl.float32)
        x2 = tl.load(x_ptr + offs + n).to(tl.float32)
        g = tl.load(g_ptr + hoffs).to(tl.float32)
        sigmoid = 1 / (1 + tl.exp(-x1))
        dx1 = g * x2 * sigmoid * (
                1 + x1 * (1 - sigmoid)) * smooth_scale_1
        dx2 = g * x1 * sigmoid * smooth_scale_2

        dx1 = (dx1 * s).to(dx_ptr.dtype.element_ty)
        dx2 = (dx2 * s).to(dx_ptr.dtype.element_ty)

        tl.store(dx_ptr + offs, dx1)
        tl.store(dx_ptr + n + offs, dx2)
        offs += B
        hoffs += B




@triton.jit
def silu_and_block_quant_backward_kernel(g_ptr, x_ptr,  
                                        dx_ptr,
                                        dx_scale_ptr, 
                                        transpose_dx_ptr,
                                        transpose_dx_scale_ptr,
                                        M, 
                                        n: tl.constexpr,
                                        ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    nb = n // 128
    offs = pid * 128 * n * 2 + tl.arange(0, 128)[:, None] * n * 2 + tl.arange(0, 128)[None, :]
    hoffs = pid * 128 * n + tl.arange(0, 128)[:, None] * n + tl.arange(0, 128)[None, :]
    toffs = pid * 128 + tl.arange(0, 128)[:, None] * M + tl.arange(0, 128)[
                                                            None, :]
    idx = pid * 128 + tl.arange(0, 128)
    mask = idx[:, None] < M
    for i in range(n//128):
        x1 = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        x2 = tl.load(x_ptr + n + offs, mask=mask).to(
            tl.float32)
        g = tl.load(g_ptr + hoffs, mask=mask).to(tl.float32)
        sigmoid = 1 / (1 + tl.exp(-x1))
        dx1 = g * x2 * sigmoid * (
                1 + x1 * (1 - sigmoid))
        scale1 = tl.maximum(
            tl.max(dx1.abs(), 1) / 448, 1e-30)
        if ROUND:
            scale1 = tl.exp2(tl.ceil(tl.log2(scale1)))
        tl.store(dx_scale_ptr + i * M + pid * 128 + tl.arange(0, 128), scale1, mask=idx < M)
    
        qdx1 = (dx1 / scale1[:, None]).to(dx_ptr.dtype.element_ty)
        tl.store(dx_ptr + offs, qdx1, mask=mask)

        scale1 = tl.maximum(
            tl.max(dx1.abs(), 0) / 448, 1e-30)
        if ROUND:
            scale1 = tl.exp2(tl.ceil(tl.log2(scale1)))
        tl.store(transpose_dx_scale_ptr + pid * n * 2 + i * 128 + tl.arange(0, 128), scale1)

        qdx1 = (dx1 / scale1[None, :]).to(dx_ptr.dtype.element_ty)
        tl.store(transpose_dx_ptr + toffs, tl.trans(qdx1), mask=idx[None, :] < M)

        # may save memory?
        # sigmoid = 1 / (1 + tl.exp(-x1))
        dx2 = g * x1 * sigmoid
        scale2 = tl.maximum(
            tl.max(dx2.abs(), 1) / 448, 1e-30)
        if ROUND:
            scale2 = tl.exp2(tl.ceil(tl.log2(scale2)))
        tl.store(dx_scale_ptr + i * M + pid * 128 + M * nb + tl.arange(0, 128), scale2, mask=idx < M)

        qdx2 = (dx2 / scale2[:, None]).to(dx_ptr.dtype.element_ty)
        tl.store(dx_ptr + offs + n, qdx2, mask=idx[:, None] < M)

        scale2 = tl.maximum(
            tl.max(dx2.abs(), 0) / 448, 1e-30)
        if ROUND:
            scale2 = tl.exp2(tl.ceil(tl.log2(scale2)))
        tl.store(transpose_dx_scale_ptr + pid * n * 2 + n + i * 128 + tl.arange(0, 128), scale2)

        qdx2 = (dx2 / scale2[None, :]).to(dx_ptr.dtype.element_ty)
        tl.store(transpose_dx_ptr + M * n + toffs, tl.trans(qdx2), mask=idx[None,:] < M)

        offs += 128
        hoffs += 128
        toffs += 128 * M


# used in shared expert
def triton_silu_and_quant_backward(g, x, smooth_scale=None, reverse=True,
                                   round_scale=False):
    # row-wise read, row-wise write
    M, N = x.shape
    device = x.device
    smooth = smooth_scale is not None
    dx = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    if smooth:
        dx_scale = torch.empty((M,), device=device, dtype=torch.float32)
    else:
        dx_scale = torch.empty((N//128, M), device=device, dtype=torch.float32)
    if smooth:
        transpose_dx = None 
        transpose_dx_scale = None 
    else:
        transpose_dx = torch.empty((N, M), device=device, dtype=torch.float8_e4m3fn)
        transpose_dx_scale = torch.empty((triton.cdiv(M, 128), N), device=device, dtype=torch.float32)

    if smooth:
        if triton.next_power_of_2(N) == N and N <= 8192:
            W = 8192 // N
            sm = torch.cuda.get_device_properties(device).multi_processor_count
            T = triton.cdiv(M, sm * W)
            grid = (sm,)
            silu_and_smooth_quant_backward_kernel[grid](
                g,
                x,
                smooth_scale,
                dx,
                dx_scale,
                M,
                T,
                N // 2,
                W,
                reverse,
                round_scale,
                num_stages=3,
                num_warps=16
            )
        else:
            B = 512
            T = 16
            assert M % T == 0 and N // 2 % B == 0
            grid = (M // T,)
            if smooth:
                compatible_silu_and_quant_backward_kernel[grid](
                    g,
                    x,
                    smooth_scale,
                    dx,
                    dx_scale,
                    M,
                    T,
                    N // 2,
                    B,
                    reverse,
                    round_scale,
                    num_stages=3,
                    num_warps=16
                )
    else:
        grid = (triton.cdiv(M ,128),)
        silu_and_block_quant_backward_kernel[grid](
            g,
            x,
            dx,
            dx_scale,
            transpose_dx,
            transpose_dx_scale,
            M,
            N // 2,
            round_scale,
            num_stages=2,
            num_warps=16
        )
    return dx, dx_scale, transpose_dx, transpose_dx_scale


@triton.jit
def weighted_silu_forward_kernel(x_ptr, weight_ptr, out_ptr, M, T,
                                 N: tl.constexpr, n: tl.constexpr,
                                 W: tl.constexpr):
    pid = tl.program_id(axis=0)

    row_offs = pid * W * T * n + tl.arange(0, W)[:, None] * n
    col_offs = tl.arange(0, n)[None, :]

    for i in range(T):
        indices = pid * W * T + i * W + tl.arange(0, W)
        mask = indices[:, None] < M
        x1 = tl.load(x_ptr + row_offs * 2 + col_offs, mask=mask).to(tl.float32)
        x2 = tl.load(x_ptr + n + row_offs * 2 + col_offs, mask=mask).to(
            tl.float32)
        w = tl.load(weight_ptr + indices, mask=indices < M).to(tl.float32)[:,
            None]
        x = x1 / (1 + tl.exp(-x1)) * x2 * w
        tl.store(out_ptr + row_offs + col_offs, x, mask=mask)
        row_offs += n * W


def triton_weighted_silu_forward(x, weight, out=None):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device
    if out is None:
        out = torch.empty((M, N // 2), device=device, dtype=x.dtype)
    W = 8192 // N
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    T = triton.cdiv(M, sm * W)
    grid = (sm,)
    weighted_silu_forward_kernel[grid](
        x,
        weight,
        out,
        M, T,
        N,
        N // 2,
        W,
        num_stages=3,
        num_warps=16
    )
    return out


@triton.jit
def weighted_silu_backward_kernel(g_ptr, x_ptr, weight_ptr, dx_ptr, dw_ptr, M,
                                  T, N: tl.constexpr, n: tl.constexpr,
                                  W: tl.constexpr):
    pid = tl.program_id(axis=0)

    offs = pid * W * T * N + tl.arange(0, W)[:, None] * N + tl.arange(0, n)[
                                                            None, :]
    hoffs = pid * W * T * n + tl.arange(0, W)[:, None] * n + tl.arange(0, n)[
                                                             None, :]
    for i in range(T):
        mask = pid * W * T + i * W + tl.arange(0, W)
        x1 = tl.load(x_ptr + offs, mask=mask[:, None] < M).to(tl.float32)
        x2 = tl.load(x_ptr + offs + n, mask=mask[:, None] < M).to(tl.float32)
        g = tl.load(g_ptr + hoffs, mask=mask[:, None] < M).to(tl.float32)
        w = tl.load(weight_ptr + mask, mask=mask < M).to(tl.float32)[:, None]
        sigmoid = 1 / (1 + tl.exp(-x1))
        dw = tl.sum(x1 * sigmoid * x2 * g, 1)
        tl.store(dw_ptr + mask, dw, mask=mask < M)
        dx1 = g * x2 * w * sigmoid * (1 + x1 * (1 - sigmoid))
        tl.store(dx_ptr + offs, dx1, mask=mask[:, None] < M)

        dx2 = g * x1 * sigmoid * w
        tl.store(dx_ptr + offs + n, dx2, mask=mask[:, None] < M)

        offs += N * W
        hoffs += n * W


def triton_weighted_silu_backward(g, x, weight):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device
    dw = torch.empty((M, 1), device=device, dtype=x.dtype)
    dx = torch.empty((M, N), device=device, dtype=x.dtype)
    W = 8192 // N
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    T = triton.cdiv(M, sm * W)
    grid = (sm,)
    weighted_silu_backward_kernel[grid](
        g,
        x,
        weight,
        dx,
        dw,
        M, T,
        N,
        N // 2,
        W,
        num_stages=3,
        num_warps=16
    )
    return dx, dw


@triton.jit
def weighted_silu_and_quant_forward_kernel(x_ptr, weight_ptr, smooth_scale_ptr,
                                           out_ptr, scale_ptr, max_ptr, M, T,
                                           N: tl.constexpr, n: tl.constexpr,
                                           W: tl.constexpr, ROUND: tl.constexpr,
                                           CALIBRATE: tl.constexpr):
    pid = tl.program_id(axis=0)

    row_offs = pid * T * W * n + tl.arange(0, W)[:, None] * n
    col_offs = tl.arange(0, n)[None, :]
    smooth_scale = tl.load(smooth_scale_ptr + tl.arange(0, n))
    smooth_scale = 1.0 / smooth_scale
    if CALIBRATE:
        maxs = tl.zeros((W, n), dtype=tl.float32)

    for i in range(T):
        indices = pid * T * W + i * W + tl.arange(0, W)
        mask = indices[:, None] < M
        x1 = tl.load(x_ptr + row_offs * 2 + col_offs, mask=mask).to(tl.float32)
        x2 = tl.load(x_ptr + n + row_offs * 2 + col_offs, mask=mask).to(
            tl.float32)
        w = tl.load(weight_ptr + indices, mask=indices < M).to(tl.float32)[:,
            None]
        x = x1 / (1 + tl.exp(-x1)) * x2
        if CALIBRATE:
            maxs = tl.maximum(x, maxs)
        x = x * w * smooth_scale
        scale = tl.maximum(tl.max(x.abs(), 1) / 448, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(scale_ptr + indices, scale, mask=indices < M)
        x = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row_offs + col_offs, x, mask=mask)
        row_offs += n * W

    if CALIBRATE:
        maxs = tl.max(maxs, 0)
        tl.store(max_ptr + pid * n + tl.arange(0, n), maxs)


# not used, shared expert uses the triton_silu_and_quant_and_calibrate_forward
def triton_weighted_silu_and_quant_forward(x, weight, smooth_scale, out=None,
                                           scale=None, maxs=None,
                                           round_scale=False, calibrate=False):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    if out is None:
        out = torch.empty((M, N // 2), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.empty((M,), device=device, dtype=torch.float32)
    if maxs is None and calibrate:
        maxs = torch.empty((sm, N // 2), device=device, dtype=torch.float32)
    W = 8192 // N
    T = triton.cdiv(M, sm * W)
    grid = (sm,)
    weighted_silu_and_quant_forward_kernel[grid](
        x,
        weight,
        smooth_scale,
        out,
        scale,
        maxs,
        M,
        T,
        N,
        N // 2,
        W,
        round_scale,
        calibrate,
        num_stages=2,
        num_warps=16
    )
    if calibrate:
        maxs = maxs.amax(0)
    return out, scale, maxs


@triton.jit
def weighted_silu_and_quant_backward_kernel(g_ptr, x_ptr, weight_ptr,
                                            smooth_scale_ptr, dx_ptr,
                                            dx_scale_ptr, dw_ptr, M, T,
                                            N: tl.constexpr, n: tl.constexpr,
                                            W: tl.constexpr,
                                            REVERSE: tl.constexpr,
                                            ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)

    smooth_scale_1 = tl.load(smooth_scale_ptr + tl.arange(0, n))
    smooth_scale_2 = tl.load(smooth_scale_ptr + n + tl.arange(0, n))
    if not REVERSE:
        smooth_scale_1 = 1 / smooth_scale_1
        smooth_scale_2 = 1 / smooth_scale_2

    offs = pid * W * T * N + tl.arange(0, W)[:, None] * N + tl.arange(0, n)[
                                                            None, :]
    hoffs = pid * W * T * n + tl.arange(0, W)[:, None] * n + tl.arange(0, n)[
                                                             None, :]
    for i in range(T):
        mask = pid * W * T + i * W + tl.arange(0, W)
        x1 = tl.load(x_ptr + offs, mask=mask[:, None] < M).to(tl.float32)
        x2 = tl.load(x_ptr + offs + n, mask=mask[:, None] < M).to(tl.float32)
        g = tl.load(g_ptr + hoffs, mask=mask[:, None] < M).to(tl.float32)
        w = tl.load(weight_ptr + mask, mask=mask < M).to(tl.float32)[:, None]
        sigmoid = 1 / (1 + tl.exp(-x1))
        dw = tl.sum(x1 * sigmoid * x2 * g, 1)
        tl.store(dw_ptr + mask, dw, mask=mask < M)
        dx1 = g * x2 * w * sigmoid * (
                1 + x1 * (1 - sigmoid)) * smooth_scale_1
        dx2 = g * x1 * sigmoid * w * smooth_scale_2

        scale = tl.maximum(
            tl.maximum(tl.max(dx1.abs(), 1), tl.max(dx2.abs(), 1)) / 448, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        dx1 = (dx1 / scale[:, None]).to(dx_ptr.dtype.element_ty)
        dx2 = (dx2 / scale[:, None]).to(dx_ptr.dtype.element_ty)

        tl.store(dx_scale_ptr + mask, scale, mask=mask < M)
        tl.store(dx_ptr + offs, dx1, mask=mask[:, None] < M)
        tl.store(dx_ptr + offs + n, dx2, mask=mask[:, None] < M)

        offs += N * W
        hoffs += n * W


# not used, shared expert use the triton_silu_and_quant_backward kernel
def triton_weighted_silu_and_quant_backward(g, x, weight, smooth_scale,
                                            reverse=True, round_scale=False):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device
    dx = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    dx_scale = torch.empty((M,), device=device, dtype=torch.float32)
    dw = torch.empty((M, 1), device=device, dtype=x.dtype)
    W = 8192 // N
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    T = triton.cdiv(M, sm * W)
    grid = (sm,)
    weighted_silu_and_quant_backward_kernel[grid](
        g,
        x,
        weight,
        smooth_scale,
        dx,
        dx_scale,
        dw,
        M, T,
        N,
        N // 2,
        W,
        reverse,
        round_scale,
        num_stages=3,
        num_warps=16
    )
    return dx, dx_scale, dw




@triton.jit
def batch_weighted_silu_and_smooth_quant_forward_kernel(x_ptr, weight_ptr,
                                                 smooth_scale_ptr, out_ptr,
                                                 scale_ptr, max_ptr, count_ptr,
                                                 accum_ptr, M,
                                                 n: tl.constexpr,
                                                 W: tl.constexpr,
                                                 ROUND: tl.constexpr,
                                                 REVERSE: tl.constexpr,
                                                 CALIBRATE: tl.constexpr):
    eid = tl.program_id(axis=0)
    tid = tl.program_id(axis=1)
    sm = tl.num_programs(axis=1)

    count = tl.load(count_ptr + eid)
    ei = tl.load(accum_ptr + eid)
    si = ei - count
    c = tl.cdiv(count, sm * W)

    row_offs = si * n + tid * c * W * n + tl.arange(0, W)[:, None] * n
    col_offs = tl.arange(0, n)[None, :]
    smooth_scale = tl.load(smooth_scale_ptr + n * eid + tl.arange(0, n))
    if not REVERSE:
        smooth_scale = 1.0 / smooth_scale

    if CALIBRATE:
        maxs = tl.zeros((W, n), dtype=tl.float32)

    for i in range(c):
        indices = tid * c * W + i * W + tl.arange(0, W)
        mask = indices[:, None] < count
        x1 = tl.load(x_ptr + row_offs * 2 + col_offs, mask=mask).to(tl.float32)
        x2 = tl.load(x_ptr + n + row_offs * 2 + col_offs, mask=mask).to(
            tl.float32)

        w = tl.load(weight_ptr + si + indices, mask=indices < count).to(
            tl.float32)[:,
            None]
        x = x1 / (1 + tl.exp(-x1)) * x2

        if CALIBRATE:
            maxs = tl.maximum(x.abs(), maxs)

        x *= w * smooth_scale
        scale = tl.maximum(tl.max(x.abs(), 1) / 448, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(scale_ptr + si + indices, scale, mask=indices < count)
        x = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row_offs + col_offs, x, mask=mask)
        row_offs += n * W

    if CALIBRATE:
        maxs = tl.max(maxs, 0)
        tl.store(max_ptr + eid * sm * n + tid * n + tl.arange(0, n), maxs)


@triton.jit
def batch_weighted_silu_and_block_quant_forward_kernel(x_ptr, weight_ptr,
                                                 out_ptr,
                                                 scale_ptr, 
                                                 transpose_output_ptr, 
                                                 transpose_scale_ptr,
                                                 count_ptr,
                                                 accum_ptr, 
                                                 M,
                                                 n: tl.constexpr,
                                                 E: tl.constexpr,
                                                 ROUND: tl.constexpr):
    eid = tl.program_id(axis=0)
    tid = tl.program_id(axis=1)

    count = tl.load(count_ptr + eid)
    ei = tl.load(accum_ptr + eid)
    si = ei - count
    c = tl.cdiv(count, 128)
    nb = n // 128

    counts = tl.load(count_ptr + tl.arange(0, E))
    n_blocks = tl.cdiv(counts, 128)
    transpose_scale_off = tl.sum(tl.where(tl.arange(0, E)< eid, n_blocks, 0))

    if tid < c:
        offs = si * n * 2 + tid * 128 * n * 2 + tl.arange(0, 128)[:, None] * n * 2 + tl.arange(0, 128)[None, :]
        hoffs = si * n + tid * 128 * n + tl.arange(0, 128)[:, None] * n + tl.arange(0, 128)[None, :]
        toffs = si * n + tid * 128 + tl.arange(0, 128)[:, None] * count + tl.arange(0, 128)[
                                                                None, :]
        indices = tid * 128 + tl.arange(0, 128)
        mask = indices[:, None] < count
        w = tl.load(weight_ptr + si + indices, mask=indices < count).to(
                tl.float32)[:,
                None]
        for i in range(n//128):
            x1 = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
            x2 = tl.load(x_ptr + n + offs, mask=mask).to(
                tl.float32)
            x = x1 * tl.sigmoid(x1) * x2 * w

            scale = tl.maximum(tl.max(x.abs(), 1) / 448, 1e-30)
            if ROUND:
                scale = tl.exp2(tl.ceil(tl.log2(scale)))
            tl.store(scale_ptr + si * nb + i * count + tid * 128 + tl.arange(0, 128), scale, mask=indices < count)

            xq = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
            tl.store(out_ptr + hoffs, xq, mask=mask)

            scale = tl.maximum(tl.max(x.abs(), 0) / 448, 1e-30)
            if ROUND:
                scale = tl.exp2(tl.ceil(tl.log2(scale)))
            tl.store(transpose_scale_ptr + transpose_scale_off * n + tid * n + i * 128 + tl.arange(0, 128), scale)

            xq = (x / scale).to(out_ptr.dtype.element_ty)
            tl.store(transpose_output_ptr + toffs, tl.trans(xq), mask=indices[None, :] < count)
            offs += 128
            hoffs += 128
            toffs += count * 128

# used in routed experts
def triton_batch_weighted_silu_and_quant_forward(x, 
                                                 weight, 
                                                 counts, 
                                                 smooth_scale=None,
                                                 splits=None,
                                                 out=None, 
                                                 scale=None,
                                                 maxs=None, 
                                                 round_scale=False,
                                                 reverse=False,
                                                 calibrate=False):
    # row-wise read, row-wise write
    M, N = x.shape
    n = N // 2
    n_experts = counts.shape[0]
    smooth = smooth_scale is not None
    assert N <= 8192
    device = x.device
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    if out is None:
        out = torch.empty((M, n), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        if smooth:
            scale = torch.empty((M,), device=device, dtype=torch.float32)
        else:
            # intra layout and inner layput are not consist,
            # tensors will be viewed after splitting
            scale = torch.empty((M * n // 128), device=device, dtype=torch.float32)
    if smooth:
        transpose_output = None
        transpose_scale = None
    else:
        assert splits is not None, 'batch mode need splits to launch kernels'
        calibrate = False
        blocks = sum([(x+127)//128 for x in splits])
        transpose_output = torch.empty((M * n), device=device, dtype=torch.float8_e4m3fn)
        transpose_scale = torch.empty((blocks * n), device=device, dtype=torch.float32)

    if calibrate:
        if maxs is None:
            maxs = torch.empty((n_experts, n), device=device,
                               dtype=torch.float32)
        tmp_maxs = torch.empty((n_experts, sm, n), device=device,
                               dtype=torch.float32)
    else:
        tmp_maxs = None
        maxs = None

    accums = torch.cumsum(counts, 0)
    if smooth:
        W = 8192 // N
        grid = (n_experts, sm)
        batch_weighted_silu_and_smooth_quant_forward_kernel[grid](
            x,
            weight,
            smooth_scale,
            out,
            scale,
            tmp_maxs,
            counts,
            accums,
            M,
            n,
            W,
            round_scale,
            reverse,
            calibrate,
            num_stages=3,
            num_warps=16
        )
        if calibrate:
            maxs = tmp_maxs.amax(1)
    else:
        grid = (n_experts, triton.cdiv(max(splits), 128))
        batch_weighted_silu_and_block_quant_forward_kernel[grid](
            x,
            weight,
            out,
            scale,
            transpose_output,
            transpose_scale,
            counts,
            accums,
            M,
            n,
            len(splits),
            round_scale,
            num_stages=2,
            num_warps=16
        )


    return out, scale, maxs, transpose_output, transpose_scale


@triton.jit
def batch_weighted_silu_and_smooth_quant_backward_kernel(g_ptr, x_ptr, weight_ptr,
                                                  smooth_scale_ptr, count_ptr,
                                                  accum_ptr, dx_ptr,
                                                  dx_scale_ptr, dw_ptr, M,
                                                  n: tl.constexpr,
                                                  W: tl.constexpr,
                                                  REVERSE: tl.constexpr,
                                                  ROUND: tl.constexpr):
    eid = tl.program_id(axis=0)
    tid = tl.program_id(axis=1)
    sm = tl.num_programs(axis=1)

    count = tl.load(count_ptr + eid)
    ei = tl.load(accum_ptr + eid)
    si = ei - count
    c = tl.cdiv(count, sm * W)

    smooth_scale_1 = tl.load(smooth_scale_ptr + n * eid * 2 + tl.arange(0, n))
    smooth_scale_2 = tl.load(
        smooth_scale_ptr + n * eid * 2 + n + tl.arange(0, n))
    if not REVERSE:
        smooth_scale_1 = 1 / smooth_scale_1
        smooth_scale_2 = 1 / smooth_scale_2

    offs = si * n * 2 + tid * c * W * n * 2 + tl.arange(0, W)[:,
                                              None] * n * 2 + tl.arange(
        0, n)[None, :]
    hoffs = si * n + tid * c * W * n + tl.arange(0, W)[:, None] * n + tl.arange(
        0, n)[None, :]
    for i in range(c):
        indices = tid * c * W + i * W + tl.arange(0, W)
        x1 = tl.load(x_ptr + offs, mask=indices[:, None] < count).to(tl.float32)
        # x1 = tl.maximum(x1, -80.7)
        x2 = tl.load(x_ptr + offs + n, mask=indices[:, None] < count).to(
            tl.float32)
        g = tl.load(g_ptr + hoffs, mask=indices[:, None] < count).to(tl.float32)
        w = tl.load(weight_ptr + si + indices, mask=indices < count).to(
            tl.float32)[:, None]
        sigmoid = 1 / (1 + tl.exp(-x1))
        dw = tl.sum(x1 * sigmoid * x2 * g, 1)
        tl.store(dw_ptr + si + indices, dw, mask=indices < count)

        dx1 = g * x2 * w * sigmoid * (
                1 + x1 * (1 - sigmoid)) * smooth_scale_1
        dx2 = g * x1 * sigmoid * w * smooth_scale_2

        dx1_max = tl.max(dx1.abs(), 1)
        dx2_max = tl.max(dx2.abs(), 1)
        scale = tl.maximum(tl.maximum(dx1_max, dx2_max) / 448, 1e-30)
        # scale = tl.maximum(dx2_max / 448, 1e-30)
        # tl.device_print('scale', scale)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(dx_scale_ptr + si + indices, scale, mask=indices < count)

        dx1 = (dx1 / scale[:, None]).to(dx_ptr.dtype.element_ty)
        dx2 = (dx2 / scale[:, None]).to(dx_ptr.dtype.element_ty)

        tl.store(dx_ptr + offs, dx1, mask=indices[:, None] < count)
        tl.store(dx_ptr + offs + n, dx2, mask=indices[:, None] < count)

        offs += n * W * 2
        hoffs += n * W




@triton.jit
def batch_weighted_silu_and_block_quant_backward_kernel(g_ptr, x_ptr, weight_ptr,
                                                  count_ptr,
                                                  accum_ptr, 
                                                  dx_ptr,
                                                  dx_scale_ptr, 
                                                  transpose_dx_ptr, 
                                                  transpose_dx_scale_ptr,
                                                  dw_ptr, 
                                                  M,
                                                  n: tl.constexpr,
                                                  E: tl.constexpr,
                                                  ROUND: tl.constexpr):
    eid = tl.program_id(axis=0)
    tid = tl.program_id(axis=1)

    count = tl.load(count_ptr + eid)
    ei = tl.load(accum_ptr + eid)
    si = ei - count
    c = tl.cdiv(count, 128)
    nb = n // 128

    counts = tl.load(count_ptr + tl.arange(0, E))
    block_counts = tl.cdiv(counts, 128)
    transpose_off = tl.sum(tl.where(tl.arange(0, E) < eid, block_counts, 0))

    if tid < c:
        offs = si * n * 2 + tid * 128 * n * 2 + tl.arange(0, 128)[:, None] * n * 2 + tl.arange(0, 128)[None, :]
        hoffs = si * n + tid * 128 * n + tl.arange(0, 128)[:, None] * n + tl.arange(0, 128)[None, :]
        toffs = si * n * 2 + tid * 128 + tl.arange(0, 128)[:, None] * count + tl.arange(0, 128)[
                                                                None, :]
        idx = tid * 128 + tl.arange(0, 128)
        mask = idx[:, None] < count
        w = tl.load(weight_ptr + si + idx, mask=idx < count).to(
            tl.float32)[:, None]
        dw = tl.zeros((128,), dtype=tl.float32)
        for i in range(n//128):
            x1 = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
            x2 = tl.load(x_ptr + n + offs, mask=mask).to(tl.float32)
            g = tl.load(g_ptr + hoffs, mask=mask).to(tl.float32)
            sigmoid = tl.sigmoid(x1)

            dw += tl.sum(x1 * sigmoid * x2 * g, 1)

            dx = g * x2 * w * sigmoid * (
                    1 + x1 * (1 - sigmoid))
            scale = tl.maximum(
                tl.max(dx.abs(), 1) / 448, 1e-30)
            if ROUND:
                scale = tl.exp2(tl.ceil(tl.log2(scale)))
            tl.store(dx_scale_ptr + si * nb * 2 + i * count + tid * 128 + tl.arange(0, 128), scale, mask=idx < count)

            qdx = (dx / scale[:, None]).to(dx_ptr.dtype.element_ty)
            tl.store(dx_ptr + offs, qdx, mask=mask)

            scale = tl.maximum(
                tl.max(dx.abs(), 0) / 448, 1e-30)
            if ROUND:
                scale = tl.exp2(tl.ceil(tl.log2(scale)))
            tl.store(transpose_dx_scale_ptr + transpose_off * n * 2 + tid * n * 2 + i * 128 + tl.arange(0, 128), scale)

            qdx = (dx / scale[None, :]).to(dx_ptr.dtype.element_ty)
            tl.store(transpose_dx_ptr + toffs, tl.trans(qdx), mask=idx[None, :] < count)

            # may save memory?
            # sigmoid = 1 / (1 + tl.exp(-x1))
            dx = g * x1 * sigmoid * w
            scale = tl.maximum(
                tl.max(dx.abs(), 1) / 448, 1e-30)
            if ROUND:
                scale = tl.exp2(tl.ceil(tl.log2(scale)))
            qdx = (dx / scale[:, None]).to(dx_ptr.dtype.element_ty)
            tl.store(dx_scale_ptr + si * nb * 2 + i * count + tid * 128 + count * nb + tl.arange(0, 128), scale, mask=idx < count)
            tl.store(dx_ptr + n + offs, qdx, mask=mask)

            scale = tl.maximum(
                tl.max(dx.abs(), 0) / 448, 1e-30)
            if ROUND:
                scale = tl.exp2(tl.ceil(tl.log2(scale)))
            qdx = (dx / scale[None, :]).to(dx_ptr.dtype.element_ty)
            tl.store(transpose_dx_scale_ptr + transpose_off * n * 2 + tid * n * 2 + n + i * 128 + tl.arange(0, 128), scale)
            tl.store(transpose_dx_ptr + count * n + toffs, tl.trans(qdx), mask=idx[None, :] < count)

            offs += 128
            hoffs += 128
            toffs += 128 * count

        tl.store(dw_ptr + si + idx, dw, mask=idx < count)


# used in routed experts
def triton_batch_weighted_silu_and_quant_backward(g, x, weight, 
                                                  counts,
                                                  smooth_scale=None,
                                                  splits=None,
                                                  reverse=True,
                                                  round_scale=False):
    # row-wise read, row-wise write
    M, N = x.shape
    n_expert = counts.shape[0]
    smooth = smooth_scale is not None
    assert N <= 8192 and 8192 % N == 0
    device = x.device
    dx = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn) 
    if smooth:
        dx_scale = torch.empty((M,), device=device, dtype=torch.float32)
        transpose_dx = None 
        transpose_dx_scale = None 
    else:
        assert splits is not None, 'batch mode need splits to launch kernels'
        # intra layout and inner layput are not consist,
        # tensors will be viewed after splitting
        dx_scale = torch.empty((N // 128 * M), device=device, dtype=torch.float32)
        transpose_dx = torch.empty((N * M), device=device, dtype=torch.float8_e4m3fn)  
        s = sum([((x-1)//128+1) for x in splits])
        transpose_dx_scale = torch.empty((s * N), device=device, dtype=torch.float32) 

    dw = torch.empty((M, 1), device=device, dtype=weight.dtype)
    accums = torch.cumsum(counts, 0)

    if smooth:
        W = 8192 // N
        sm = torch.cuda.get_device_properties(device).multi_processor_count
        grid = (n_expert, sm)
        batch_weighted_silu_and_smooth_quant_backward_kernel[grid](
            g,
            x,
            weight,
            smooth_scale,
            counts,
            accums,
            dx,
            dx_scale,
            dw,
            M,
            N // 2,
            W,
            reverse,
            round_scale,
            num_stages=3,
            num_warps=16
        )
    else:
        grid = (n_expert, triton.cdiv(max(splits), 128))
        batch_weighted_silu_and_block_quant_backward_kernel[grid](
            g,
            x,
            weight,
            counts,
            accums,
            dx,
            dx_scale,
            transpose_dx,
            transpose_dx_scale,
            dw,
            M,
            N // 2,
            n_expert,
            round_scale,
            num_stages=2,
            num_warps=16
        )
    return dx, dx_scale, dw, transpose_dx, transpose_dx_scale
