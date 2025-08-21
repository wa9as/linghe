import torch
import triton
import triton.language as tl




# n is power of 2
@triton.jit
def silu_and_quant_forward_kernel(x_ptr, smooth_scale_ptr, out_ptr, scale_ptr,
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
def compatible_silu_and_quant_forward_kernel(x_ptr, smooth_scale_ptr, out_ptr,
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


# used in shared expert
def triton_silu_and_quant_forward(x, smooth_scale, out=None, scale=None,
                                  maxs=None, round_scale=False,
                                  calibrate=False):
    # row-wise read, row-wise write
    M, N = x.shape
    device = x.device
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    if out is None:
        out = torch.empty((M, N // 2), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.empty((M,), device=device, dtype=torch.float32)
    if triton.next_power_of_2(N) == N and N <= 8192:
        if maxs is None and calibrate:
            maxs = torch.empty((sm, N // 2), device=device, dtype=torch.float32)
        W = 8192 // N
        T = triton.cdiv(M, sm * W)
        grid = (sm,)
        silu_and_quant_forward_kernel[grid](
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
        if maxs is None and calibrate:
            maxs = torch.empty((M // T, N // 2), device=device, dtype=torch.float32)
        assert N // 2 % B == 0 and M % T == 0
        grid = (M // T,)
        compatible_silu_and_quant_forward_kernel[grid](
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
    return out, scale, maxs



@triton.jit
def silu_and_quant_backward_kernel(g_ptr, x_ptr, smooth_scale_ptr, dx_ptr,
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


# used in shared expert
def triton_silu_and_quant_backward(g, x, smooth_scale, reverse=True,
                                   round_scale=False):
    # row-wise read, row-wise write
    M, N = x.shape
    device = x.device
    dx = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    dx_scale = torch.empty((M,), device=device, dtype=torch.float32)

    if triton.next_power_of_2(N) == N and N <= 8192:
        W = 8192 // N
        sm = torch.cuda.get_device_properties(device).multi_processor_count
        T = triton.cdiv(M, sm * W)
        grid = (sm,)
        silu_and_quant_backward_kernel[grid](
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
    return dx, dx_scale


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
def batch_weighted_silu_and_quant_forward_kernel(x_ptr, weight_ptr,
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
def batch_max_kernel(x_ptr, out_ptr, M, N, W: tl.constexpr, H: tl.constexpr):
    eid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    maxs = tl.zeros((W,), dtype=tl.float32)
    c = tl.cdiv(M, H)
    offs = eid * M * N + bid * W + tl.arange(0, H)[:, None] * N + tl.arange(0,
                                                                            W)[
                                                                  None, :]
    for i in range(c):
        x = tl.load(x_ptr + offs, mask=i * H + tl.arange(0, H)[:, None] < M).to(
            tl.float32)
        maxs = tl.maximum(tl.max(x, 0), maxs)
        offs += H * N

    tl.store(out_ptr + eid * N + bid * W + tl.arange(0, W), maxs)


# used in routed experts
def triton_batch_weighted_silu_and_quant_forward(x, weight, smooth_scale,
                                                 counts, out=None, scale=None,
                                                 maxs=None, round_scale=False,
                                                 reverse=False,
                                                 calibrate=False):
    # row-wise read, row-wise write
    M, N = x.shape
    n_experts = counts.shape[0]
    assert N <= 8192
    device = x.device
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    if out is None:
        out = torch.empty((M, N // 2), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.empty((M,), device=device, dtype=torch.float32)

    if calibrate:
        if maxs is None:
            maxs = torch.empty((n_experts, N // 2), device=device,
                               dtype=torch.float32)
        tmp_maxs = torch.empty((n_experts, sm, N // 2), device=device,
                               dtype=torch.float32)
    else:
        tmp_maxs = None
        maxs = None

    accums = torch.cumsum(counts, 0)
    W = 8192 // N
    grid = (n_experts, sm)
    batch_weighted_silu_and_quant_forward_kernel[grid](
        x,
        weight,
        smooth_scale,
        out,
        scale,
        tmp_maxs,
        counts,
        accums,
        M,
        N // 2,
        W,
        round_scale,
        reverse,
        calibrate,
        num_stages=3,
        num_warps=16
    )

    if calibrate:
        if 128 % n_experts == 0:
            M = tmp_maxs.shape[1]
            T = 128 // n_experts
            W = N // 2 // T
            H = 16
            grid = (n_experts, T)
            batch_max_kernel[grid](tmp_maxs,
                                   maxs,
                                   M,
                                   N // 2,
                                   W,
                                   H)
        else:
            maxs = tmp_maxs.amax(1)
    return out, scale, maxs


@triton.jit
def batch_weighted_silu_and_quant_backward_kernel(g_ptr, x_ptr, weight_ptr,
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


# used in routed experts
def triton_batch_weighted_silu_and_quant_backward(g, x, weight, smooth_scales,
                                                  counts,
                                                  reverse=True,
                                                  round_scale=False):
    # row-wise read, row-wise write
    M, N = x.shape
    n_expert = counts.shape[0]
    assert N <= 8192 and 8192 % N == 0
    device = x.device
    dx = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    dx_scale = torch.empty((M,), device=device, dtype=torch.float32)
    dw = torch.empty((M, 1), device=device, dtype=weight.dtype)
    accums = torch.cumsum(counts, 0)

    W = 8192 // N
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    grid = (n_expert, sm)
    batch_weighted_silu_and_quant_backward_kernel[grid](
        g,
        x,
        weight,
        smooth_scales,
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
    return dx, dx_scale, dw
