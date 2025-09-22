import torch
import triton
import triton.language as tl


@triton.jit
def hadamard_quant_row_kernel(
        x_ptr,
        hm_ptr,
        x_q_ptr,
        x_scale_ptr,
        M,
        N,
        BLOCK_SIZE: tl.constexpr,
        R: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * R * BLOCK_SIZE
    rows = row_start + tl.arange(0, R * BLOCK_SIZE)
    mask_rows = rows < M

    hm = tl.load(
        hm_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE + tl.arange(0,
                                                                            BLOCK_SIZE)[
                                                                  None, :])

    max_val = tl.zeros((R * BLOCK_SIZE,), dtype=tl.float32) + 1.17e-38

    num_col_blocks = tl.cdiv(N, BLOCK_SIZE)
    for col_block in range(num_col_blocks):
        col_start = col_block * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask_cols = cols < N

        offs = rows[:, None] * N + cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask_rows[:, None] & mask_cols[None, :],
                    other=0.0)
        x_transformed = tl.dot(x, hm)
        current_max = tl.max(tl.abs(x_transformed), axis=1)
        max_val = tl.maximum(max_val, current_max)

    scale = max_val / 448.0
    tl.store(x_scale_ptr + rows, scale, mask=mask_rows)
    s = 448.0 / tl.where(max_val > 0, max_val, 1.0)

    for col_block in range(num_col_blocks):
        col_start = col_block * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask_cols = cols < N

        offs = rows[:, None] * N + cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask_rows[:, None] & mask_cols[None, :],
                    other=0.0)
        x_transformed = tl.dot(x, hm)
        quantized = (x_transformed * s[:, None]).to(x_q_ptr.dtype.element_ty)
        tl.store(x_q_ptr + offs, quantized,
                 mask=mask_rows[:, None] & mask_cols[None, :])


@triton.jit
def hadamard_quant_col_kernel(
        x_ptr,
        hm_ptr,
        xt_q_ptr,
        xt_scale_ptr,
        M,
        N,
        BLOCK_SIZE: tl.constexpr,
        R: tl.constexpr,
):
    pid = tl.program_id(0)
    col_start = pid * R * BLOCK_SIZE
    cols = col_start + tl.arange(0, R * BLOCK_SIZE)
    mask_cols = cols < N

    hm = tl.load(
        hm_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE + tl.arange(0,
                                                                            BLOCK_SIZE)[
                                                                  None, :])

    max_val = tl.zeros((R * BLOCK_SIZE,), dtype=tl.float32) + 1.17e-38

    num_row_blocks = tl.cdiv(M, BLOCK_SIZE)
    for row_block in range(num_row_blocks):
        row_start = row_block * BLOCK_SIZE
        rows = row_start + tl.arange(0, BLOCK_SIZE)
        mask_rows = rows < M

        offs = rows[:, None] * N + cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask_rows[:, None] & mask_cols[None, :],
                    other=0.0)
        x_transformed = tl.dot(hm, x)
        current_max = tl.max(tl.abs(x_transformed), axis=0)
        max_val = tl.maximum(max_val, current_max)

    scale = max_val / 448.0
    tl.store(xt_scale_ptr + cols, scale, mask=mask_cols)
    s = 448.0 / tl.where(max_val > 0, max_val, 1.0)

    for row_block in range(num_row_blocks):
        row_start = row_block * BLOCK_SIZE
        rows = row_start + tl.arange(0, BLOCK_SIZE)
        mask_rows = rows < M

        offs = rows[:, None] * N + cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask_rows[:, None] & mask_cols[None, :],
                    other=0.0)
        x_transformed = tl.dot(hm, x)
        quantized = (x_transformed * s[None, :]).to(xt_q_ptr.dtype.element_ty)
        quantized_t = tl.trans(quantized)
        store_offs = cols[:, None] * M + rows[None, :]
        tl.store(xt_q_ptr + store_offs, quantized_t,
                 mask=mask_cols[:, None] & mask_rows[None, :])


# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_hadamard_quant_x(x, hm):
    # apply hadamard transformation and quantization for x
    # y = x @ w: x->x@h and rowwise quant
    # dwT = yT @ x: x->xT@h and rowwise quant
    M, N = x.shape
    device = x.device
    BLOCK_SIZE = hm.size(0)
    R = 1
    x_q = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device)
    xt_q = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=device)
    x_scale = torch.empty((M, ), dtype=torch.float32, device=device)
    xt_scale = torch.empty((N, ), dtype=torch.float32, device=device)

    grid_row = (triton.cdiv(M, R * BLOCK_SIZE),)
    hadamard_quant_row_kernel[grid_row](
        x,
        hm,
        x_q,
        x_scale,
        M,
        N,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=4
    )

    grid_col = (triton.cdiv(N, R * BLOCK_SIZE),)
    hadamard_quant_col_kernel[grid_col](
        x,
        hm,
        xt_q,
        xt_scale,
        M,
        N,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=4
    )

    return x_q, x_scale,xt_q, xt_scale


# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_hadamard_quant_w(w, hm):
    # apply hadamard transformation and quantization for w
    # y = x @ w: w->w@h and rowwise quant
    # dx = y @ wT: w->h@wT and rowwise quant
    M, N = w.shape
    device = w.device
    w_q = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device)
    wt_q = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=device)
    w_scale = torch.empty((M, ), dtype=torch.float32, device=device)
    wt_scale = torch.empty((N, ), dtype=torch.float32, device=device)

    BLOCK_SIZE = hm.size(0)
    R = 1

    grid_row = (triton.cdiv(M, R * BLOCK_SIZE),)
    hadamard_quant_row_kernel[grid_row](
        w,
        hm,
        w_q,
        w_scale,
        M,
        N,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=4
    )

    grid_col = (triton.cdiv(N, R * BLOCK_SIZE),)
    hadamard_quant_col_kernel[grid_col](
        w,
        hm,
        wt_q,
        wt_scale,
        M,
        N,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=4
    )

    return w_q, w_scale, wt_q, wt_scale


# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_hadamard_quant_y(y, hm):
    # apply hadamard transformation and quantization for dy
    # dx = y @ wT: y->y@h and rowwise quant
    # dwT = yT @ x: y->h@yT and rowwise quant
    M, N = y.shape
    device = y.device
    BLOCK_SIZE = hm.size(0)
    R = 1
    y_q = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device)
    yt_q = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=device)
    y_scale = torch.empty((M, ), dtype=torch.float32, device=device)
    yt_scale = torch.empty((N, ), dtype=torch.float32, device=device)

    grid_row = (triton.cdiv(M, R * BLOCK_SIZE),)
    hadamard_quant_row_kernel[grid_row](
        y,
        hm,
        y_q,
        y_scale,
        M,
        N,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=4
    )

    grid_col = (triton.cdiv(N, R * BLOCK_SIZE),)
    hadamard_quant_col_kernel[grid_col](
        y,
        hm,
        yt_q,
        yt_scale,
        M,
        N,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=4
    )

    return y_q, y_scale, yt_q, yt_scale


def triton_hadamard_quant_nt_megatron(x, w, hm):
    x_q, _, x_scale, _ = triton_hadamard_quant_x(x, hm)
    w_q, _, w_scale, _ = triton_hadamard_quant_w(w, hm)
    return x_q, x_scale, w_q, w_scale


def triton_hadamard_quant_nn_megatron(y, w, hm):
    y_q, _, y_scale, _ = triton_hadamard_quant_y(y, hm)
    _, wt_q, _, wt_scale = triton_hadamard_quant_w(w, hm)
    return y_q, y_scale, wt_q, wt_scale


def triton_hadamard_quant_tn_megatron(y, x, hm):
    _, yt_q, _, yt_scale = triton_hadamard_quant_y(y, hm)
    _, xt_q, _, xt_scale = triton_hadamard_quant_x(x, hm)
    return yt_q, yt_scale, xt_q, xt_scale


def hadamard_quant_forward_megatron(x, w, hm):
    x_q, x_scale, w_q, w_scale = triton_hadamard_quant_nt_megatron(x, w, hm)
    output = torch._scaled_mm(x_q,
                              w_q.t(),
                              scale_a=x_scale,
                              scale_b=w_scale,
                              out_dtype=x.dtype,
                              use_fast_accum=True
                              )
    return output, x_q, w_q, x_scale, w_scale


def hadamard_quant_backward_megatron(y, w, hm):
    y_q, y_scale, wt_q, wt_scale = triton_hadamard_quant_nn_megatron(y, w, hm)
    output = torch._scaled_mm(
        y_q,
        wt_q.t(),
        scale_a=y_scale,
        scale_b=wt_scale,
        out_dtype=y.dtype,
        use_fast_accum=True
    )
    return output, y_q, wt_q.t(), y_scale, wt_scale


def hadamard_quant_update_megatron(y, x, hm):
    yt_q, yt_scale, xt_q, xt_scale = triton_hadamard_quant_tn_megatron(y, x, hm)
    output = torch._scaled_mm(yt_q,
                              xt_q.t(),
                              scale_a=yt_scale.t(),
                              scale_b=xt_scale,
                              out_dtype=x.dtype,
                              use_fast_accum=True
                              )
    return output, yt_q, xt_q, yt_scale, xt_scale

