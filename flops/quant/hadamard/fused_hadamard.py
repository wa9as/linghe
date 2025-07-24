import torch
import triton
import triton.language as tl


@triton.jit
def fused_hadamard_kernel(x_ptr, b_ptr, s_ptr, q_ptr, hm_ptr, M, N,
                          BLOCK_SIZE: tl.constexpr, R: tl.constexpr,
                          SIDE: tl.constexpr):
    # apply hadamard transform and row-wise quant
    # SIDE=0: hadamard@block  
    # SIDE=1: block@hadamard
    # row-wise read, row-wise write
    pid = tl.program_id(axis=0)
    hm = tl.load(
        hm_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE + tl.arange(0,
                                                                            BLOCK_SIZE)[
                                                                  None, :])
    offs = pid * BLOCK_SIZE * N + tl.arange(0, BLOCK_SIZE)[:,
                                  None] * N + tl.arange(0, BLOCK_SIZE)[None, :]
    n = tl.cdiv(N, BLOCK_SIZE)
    maxs = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) + 5.27e-36
    for i in range(n):
        x = tl.load(x_ptr + offs)
        if SIDE == 0:
            x = tl.dot(hm, x)
        else:
            x = tl.dot(x, hm)
        tl.store(b_ptr + offs, x)
        maxs = tl.maximum(maxs, tl.max(tl.abs(x), 1))
        offs += BLOCK_SIZE

    scales = maxs / 448.0

    tl.store(s_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), scales)
    rs = (448.0 / maxs)[:, None]

    offs = pid * BLOCK_SIZE * N + tl.arange(0, BLOCK_SIZE)[:,
                                  None] * N + tl.arange(0, R * BLOCK_SIZE)[None,
                                              :]
    n = tl.cdiv(N, R * BLOCK_SIZE)
    for i in range(n):
        x = tl.load(b_ptr + offs)
        y = (x.to(tl.float32) * rs).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + offs, y)
        offs += R * BLOCK_SIZE


# add out for `_cast_master_weights_to_fp8_hadamard_scaling`
def triton_fused_hadamard(x, hm, out=None, op_side=0, hm_side=1, R=2):
    # y = x @ w
    #   x: op_side = 0, hm_side=1
    #   w: op_side = 1, hm_side=1
    # dx = y @ wT, 
    #   y: op_side = 0, hm_side=1

    M, N = x.shape
    x_b = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if op_side == 0:
        x_s = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    else:
        x_s = torch.empty((1, M), dtype=torch.float32, device=x.device)
    if out is None:
        x_q = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    else:
        out_dtype = out.dtype
        x_q = out.view((M, N)).view(torch.float8_e4m3fn)
        if x_q.device != x.device:
            x_q = x_q.to(x.device)
        if hm.device != x.device:
            hm = hm.to(x.device)
    BLOCK_SIZE = hm.size(0)
    SIDE = hm_side
    grid = (M // BLOCK_SIZE,)
    fused_hadamard_kernel[grid](
        x,
        x_b,
        x_s,
        x_q,
        hm,
        M, N,
        BLOCK_SIZE,
        R,
        SIDE,
        num_stages=6,
        num_warps=4
    )
    if out is not None:
        x_q = x_q.view(out_dtype)
    return x_q, x_s


@triton.jit
def fused_transpose_hadamard_kernel(x_ptr, b_ptr, s_ptr, q_ptr, hm_ptr, M, N,
                                    BLOCK_SIZE: tl.constexpr, R: tl.constexpr,
                                    SIDE: tl.constexpr):
    # transpose x: [M, N] -> [N, M] 
    # and then apply hadamard transform
    # SIDE=0: hadamard@block  
    # SIDE=1: block@hadamard
    pid = tl.program_id(axis=0)
    hm = tl.load(
        hm_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE + tl.arange(0,
                                                                            BLOCK_SIZE)[
                                                                  None, :])
    # col-wise read, row-wise write
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None] * N + tl.arange(
        0, BLOCK_SIZE)[None, :]
    toffs = pid * BLOCK_SIZE * M + tl.arange(0, BLOCK_SIZE)[:,
                                   None] * M + tl.arange(0, BLOCK_SIZE)[None, :]
    m = tl.cdiv(M, BLOCK_SIZE)
    maxs = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) + 5.27e-36
    for i in range(m):
        x = tl.trans(tl.load(x_ptr + offs))
        if SIDE == 0:
            x = tl.dot(hm, x)
        else:
            x = tl.dot(x, hm)
        maxs = tl.maximum(maxs, tl.max(tl.abs(x), 1))
        tl.store(b_ptr + toffs, x)
        offs += BLOCK_SIZE * N
        toffs += BLOCK_SIZE

    scales = maxs / 448.0
    rs = (448.0 / maxs)[:, None]

    tl.store(s_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), scales)

    toffs = pid * BLOCK_SIZE * M + tl.arange(0, BLOCK_SIZE)[:,
                                   None] * M + tl.arange(0, R * BLOCK_SIZE)[
                                               None, :]
    m = tl.cdiv(M, R * BLOCK_SIZE)
    for i in range(m):
        x = tl.load(b_ptr + toffs).to(tl.float32)
        y = (x * rs).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + toffs, y)
        toffs += R * BLOCK_SIZE


def triton_fused_transpose_hadamard(x, hm, op_side=0, hm_side=1, R=2):
    # dx = y @ wT
    #   wT: op_side = 1, hm_side = 1
    # dwT = yT @ x:
    #   yT: op_side = 0, hm_side = 0
    #   x: op_side = 1, hm_side = 1
    M, N = x.shape
    x_b = torch.empty((N, M), dtype=x.dtype, device=x.device)
    if op_side == 0:
        x_s = torch.empty((N, 1), dtype=torch.float32, device=x.device)
    else:
        x_s = torch.empty((1, N), dtype=torch.float32, device=x.device)
    x_q = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=x.device)
    BLOCK_SIZE = hm.size(0)
    SIDE = hm_side
    grid = (N // BLOCK_SIZE,)
    fused_transpose_hadamard_kernel[grid](
        x,
        x_b,
        x_s,
        x_q,
        hm,
        M, N,
        BLOCK_SIZE,
        R,
        SIDE,
        num_stages=6,
        num_warps=4
    )
    return x_q, x_s


def triton_fused_hadamard_quant_nt_nn_tn(x, w, y, hm):
    triton_fused_hadamard_quant_nt(x, w, hm)
    triton_fused_hadamard_quant_nn(y, w, hm)
    triton_fused_hadamard_quant_tn(y, x, hm)


def fused_hadamard_quant_forward(x, w, hm):
    x_q, x_s, w_q, w_s = triton_fused_hadamard_quant_nt(x, w, hm)
    output = torch._scaled_mm(x_q,
                              w_q.t(),
                              scale_a=x_s,
                              scale_b=w_s,
                              out_dtype=torch.bfloat16,
                              use_fast_accum=True)
    return output


def fused_hadamard_quant_backward(y, w, hm):
    y_q, y_s, w_q, w_s = triton_fused_hadamard_quant_nn(y, w, hm)
    output = torch._scaled_mm(y_q,
                              w_q.t(),
                              scale_a=y_s,
                              scale_b=w_s,
                              out_dtype=torch.bfloat16,
                              use_fast_accum=True)
    return output


def fused_hadamard_quant_update(y, x, hm):
    y_q, y_s, x_q, x_s = triton_fused_hadamard_quant_tn(y, x, hm)
    output = torch._scaled_mm(y_q,
                              x_q.t(),
                              scale_a=y_s,
                              scale_b=x_s,
                              out_dtype=torch.bfloat16,
                              use_fast_accum=True)
    return output


def fused_hadamard_quant_forward_debug(x, w, hm):
    x_q, x_s, w_q, w_s = triton_fused_hadamard_quant_nt(x, w, hm)
    output = torch._scaled_mm(x_q,
                              w_q.t(),
                              scale_a=x_s,
                              scale_b=w_s,
                              out_dtype=torch.bfloat16,
                              use_fast_accum=True)
    return output, x_q, x_s, w_q, w_s


def fused_hadamard_quant_backward_debug(y, w, hm):
    y_q, y_s, w_q, w_s = triton_fused_hadamard_quant_nn(y, w, hm)
    output = torch._scaled_mm(y_q,
                              w_q.t(),
                              scale_a=y_s,
                              scale_b=w_s,
                              out_dtype=torch.bfloat16,
                              use_fast_accum=True)
    return output, y_q, y_s, w_q, w_s


def fused_hadamard_quant_update_debug(y, x, hm):
    y_q, y_s, x_q, x_s = triton_fused_hadamard_quant_tn(y, x, hm)
    output = torch._scaled_mm(y_q,
                              x_q.t(),
                              scale_a=y_s,
                              scale_b=x_s,
                              out_dtype=torch.bfloat16,
                              use_fast_accum=True)
    return output, y_q, y_s, x_q, x_s


# y = x @ w
def triton_fused_hadamard_quant_nt(x, w, hm):
    # stream = torch.cuda.Stream(device=0)
    # x_q,x_s = triton_fused_hadamard(x, hm, hm_side=1, op_side=0)
    # with torch.cuda.stream(stream):
    #     w_q,w_s = triton_fused_hadamard(w, hm, hm_side=1, op_side=1)
    # torch.cuda.current_stream().wait_stream(stream)
    x_q, x_s = triton_fused_hadamard(x, hm, hm_side=1, op_side=0)
    w_q, w_s = triton_fused_hadamard(w, hm, hm_side=1, op_side=1)
    return x_q, x_s, w_q, w_s


# dx = y @ wT
def triton_fused_hadamard_quant_nn(y, w, hm):
    # stream = torch.cuda.Stream(device=0)
    # y_q,y_s = triton_fused_hadamard(y, hm, hm_side=1, op_side=0)
    # with torch.cuda.stream(stream):
    #     w_q,w_s = triton_fused_transpose_hadamard(w, hm, hm_side=1, op_side=1)
    # torch.cuda.current_stream().wait_stream(stream)
    y_q, y_s = triton_fused_hadamard(y, hm, hm_side=1, op_side=0)
    w_q, w_s = triton_fused_transpose_hadamard(w, hm, hm_side=1, op_side=1)
    return y_q, y_s, w_q, w_s


# dwT = yT @ x
def triton_fused_hadamard_quant_tn(y, x, hm):
    # stream = torch.cuda.Stream(device=0)
    # y_q,y_s = triton_fused_transpose_hadamard(y, hm, hm_side=1, op_side=0)
    # with torch.cuda.stream(stream):
    #     x_q,x_s = triton_fused_transpose_hadamard(x, hm, hm_side=1, op_side=1)
    # torch.cuda.current_stream().wait_stream(stream)
    y_q, y_s = triton_fused_transpose_hadamard(y, hm, hm_side=1, op_side=0)
    x_q, x_s = triton_fused_transpose_hadamard(x, hm, hm_side=1, op_side=1)
    return y_q, y_s, x_q, x_s


def fp8_fused_hadamard_f_and_b(x, w, y, hm):
    fused_hadamard_quant_forward(x, w, hm)
    fused_hadamard_quant_backward(y, w, hm)
    fused_hadamard_quant_update(y, x, hm)
