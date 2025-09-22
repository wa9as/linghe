import math

import torch
import triton
import triton.language as tl

from flops.quant.channel.channel import row_quant_kernel




@triton.jit
def hadamard_nt_kernel(x_ptr, xb_ptr, w_ptr, wb_ptr, hm_ptr, M, N, K,
                       BLOCK_SIZE: tl.constexpr, R: tl.constexpr):
    pid = tl.program_id(axis=0)

    hm = tl.load(
        hm_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE + tl.arange(0,
                                                                            BLOCK_SIZE)[
                                                                  None, :])
    offs = pid * BLOCK_SIZE + tl.arange(0, R * BLOCK_SIZE)[:,
                              None] * K + tl.arange(0, BLOCK_SIZE)[None, :]
    n = tl.cdiv(N, R * BLOCK_SIZE)
    for i in range(n):
        w = tl.load(w_ptr + offs)
        tl.store(wb_ptr + offs, tl.dot(w, hm))
        offs += R * BLOCK_SIZE * K

    # norm hm in x
    # hm = (hm/BLOCK_SIZE).to(x_ptr.dtype.element_ty)
    offs = pid * BLOCK_SIZE + tl.arange(0, R * BLOCK_SIZE)[:,
                              None] * K + tl.arange(0, BLOCK_SIZE)[None, :]
    m = tl.cdiv(M, R * BLOCK_SIZE)
    for i in range(m):
        x = tl.load(x_ptr + offs)
        tl.store(xb_ptr + offs, tl.dot(x, hm))
        offs += R * BLOCK_SIZE * K


def triton_hadamard_nt(x, w, hm, R=2):
    assert w.size(1) == w.size(1)
    M, K = x.shape
    N, K = w.shape
    x_b = torch.empty_like(x)
    w_b = torch.empty_like(w)
    BLOCK_SIZE = hm.size(0)
    grid = (K // BLOCK_SIZE,)
    hadamard_nt_kernel[grid](
        x, x_b,
        w, w_b,
        hm,
        M, N, K,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=4,
        num_ctas=1
    )
    return x_b, w_b


def triton_hadamard_quant_nt(x, w, hm, R=2):
    M, K = x.shape
    N, K = w.shape
    device = x.device
    x_b = torch.empty_like(x)
    w_b = torch.empty_like(w)
    x_q = torch.empty((M, K), device=device, dtype=torch.float8_e4m3fn)
    w_q = torch.empty((N, K), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((M, 1), device=device, dtype=torch.float32)
    w_scale = torch.empty((1, N), device=device, dtype=torch.float32)

    BLOCK_SIZE = hm.size(0)
    grid = (K // BLOCK_SIZE,)
    hadamard_nt_kernel[grid](
        x, x_b,
        w, w_b,
        hm,
        M, N, K,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=4,
        num_ctas=1
    )

    BLOCK_SIZE = 4096
    grid = (M,)
    row_quant_kernel[grid](
        x_b, x_q, x_scale,
        M, K,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=4
    )

    BLOCK_SIZE = 4096
    grid = (N,)
    row_quant_kernel[grid](
        w_b, w_q, w_scale,
        N, K,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=4
    )

    return x_q, w_q, x_scale, w_scale


@triton.jit
def hadamard_tn_kernel(y_ptr, yb_ptr, x_ptr, xb_ptr, hm_ptr, M, N, K,
                       BLOCK_SIZE: tl.constexpr, R: tl.constexpr):
    # dwT = yT @ x
    # both need transpose
    # y: [M,N] -> [N,M]
    # x: [M,K] -> [K,M]
    pid = tl.program_id(axis=0)

    hm = tl.load(
        hm_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE + tl.arange(0,
                                                                            BLOCK_SIZE)[
                                                                  None, :])

    offs = pid * BLOCK_SIZE * N + tl.arange(0, BLOCK_SIZE)[:,
                                  None] * N + tl.arange(0, R * BLOCK_SIZE)[None,
                                              :]
    toffs = pid * BLOCK_SIZE + tl.arange(0, R * BLOCK_SIZE)[:,
                               None] * M + tl.arange(0, BLOCK_SIZE)[None, :]
    n = tl.cdiv(N, R * BLOCK_SIZE)
    for i in range(n):
        y = tl.trans(tl.load(y_ptr + offs))
        o = tl.dot(y, hm)
        tl.store(yb_ptr + toffs, o)
        offs += R * BLOCK_SIZE
        toffs += R * M * BLOCK_SIZE

    # # norm hm in x
    # hm = (hm/BLOCK_SIZE).to(x_ptr.dtype.element_ty)
    offs = pid * BLOCK_SIZE * K + tl.arange(0, BLOCK_SIZE)[:,
                                  None] * K + tl.arange(0, R * BLOCK_SIZE)[None,
                                              :]
    toffs = pid * BLOCK_SIZE + tl.arange(0, R * BLOCK_SIZE)[:,
                               None] * M + tl.arange(0, BLOCK_SIZE)[None, :]
    k = tl.cdiv(K, R * BLOCK_SIZE)
    for i in range(k):
        x = tl.trans(tl.load(x_ptr + offs))
        x = tl.dot(x, hm)
        tl.store(xb_ptr + toffs, x)
        offs += R * BLOCK_SIZE
        toffs += R * M * BLOCK_SIZE


# v1: hadamard+token/channelx quant
def triton_hadamard_quant_tn(y, x, hm, R=2):
    # dwT = yT @ x
    # both need transpose
    # y: [M,N] -> [N,M]
    # x: [M,K] -> [K,M]
    assert y.size(0) == x.size(0)
    M, N = y.shape
    M, K = x.shape
    device = x.device

    y_b = torch.empty((N, M), device=device, dtype=y.dtype)
    y_q = torch.empty((N, M), device=device, dtype=torch.float8_e4m3fn)
    y_scale = torch.empty((N, 1), device=device, dtype=torch.float32)

    x_b = torch.empty((K, M), device=device, dtype=x.dtype)
    x_q = torch.empty((K, M), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((1, K), device=device, dtype=torch.float32)

    BLOCK_SIZE = hm.size(0)
    grid = (M // BLOCK_SIZE,)
    hadamard_tn_kernel[grid](
        y, y_b,
        x, x_b,
        hm,
        M, N, K,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=4
    )

    BLOCK_SIZE = 4096
    grid = (K,)
    row_quant_kernel[grid](
        x_b, x_q, x_scale,
        K, M,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=4
    )

    BLOCK_SIZE = 4096
    grid = (N,)
    row_quant_kernel[grid](
        y_b, y_q, y_scale,
        N, M,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=4
    )

    return y_q, x_q, y_scale, x_scale


@triton.jit
def hadamard_nn_kernel(y_ptr, yb_ptr, w_ptr, wb_ptr, hm_ptr, M, N, K,
                       BLOCK_SIZE: tl.constexpr, R: tl.constexpr):
    # w need transpose
    # dx = y @ w
    # y: [M,N]
    # w: [N,K] -> [K,N]
    pid = tl.program_id(axis=0)

    hm = tl.load(
        hm_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE + tl.arange(0,
                                                                            BLOCK_SIZE)[
                                                                  None, :])

    offs = pid * BLOCK_SIZE + tl.arange(0, R * BLOCK_SIZE)[:,
                              None] * N + tl.arange(0, BLOCK_SIZE)[None, :]
    m = tl.cdiv(M, R * BLOCK_SIZE)
    for i in range(m):
        y = tl.load(y_ptr + offs)
        o = tl.dot(y, hm)
        tl.store(yb_ptr + offs, o)
        offs += R * BLOCK_SIZE * N

    # norm hm in y 
    # hm = (hm/BLOCK_SIZE).to(w_ptr.dtype.element_ty)
    offs = pid * BLOCK_SIZE * K + tl.arange(0, BLOCK_SIZE)[:,
                                  None] * K + tl.arange(0, R * BLOCK_SIZE)[None,
                                              :]
    toffs = pid * BLOCK_SIZE + tl.arange(0, R * BLOCK_SIZE)[:,
                               None] * N + tl.arange(0, BLOCK_SIZE)[None, :]
    k = tl.cdiv(K, R * BLOCK_SIZE)
    for i in range(k):
        w = tl.trans(tl.load(w_ptr + offs))
        o = tl.dot(w, hm)
        tl.store(wb_ptr + toffs, o)
        offs += R * BLOCK_SIZE
        toffs += R * BLOCK_SIZE * N


def triton_hadamard_quant_nn(y, w, hm, R=2):
    # w need transpose
    # dx = y @ w
    # y: [M,N]
    # w: [N,K] -> [K,N]
    M, N = y.shape
    N, K = w.shape
    device = y.device
    y_b = torch.empty((M, N), device=device, dtype=y.dtype)
    y_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    y_scale = torch.empty((M, 1), device=device, dtype=torch.float32)

    w_b = torch.empty((K, N), device=device, dtype=y.dtype)
    w_q = torch.empty((K, N), device=device, dtype=torch.float8_e4m3fn)
    w_scale = torch.empty((1, K), device=device, dtype=torch.float32)

    BLOCK_SIZE = hm.size(0)
    grid = (N // BLOCK_SIZE,)
    hadamard_nn_kernel[grid](
        y, y_b,
        w, w_b,
        hm,
        M, N, K,
        BLOCK_SIZE,
        R,
        num_stages=6,
        num_warps=4
    )

    BLOCK_SIZE = 4096
    grid = (M,)
    row_quant_kernel[grid](
        y_b, y_q, y_scale,
        M, N,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=4
    )

    BLOCK_SIZE = 4096
    grid = (K,)
    row_quant_kernel[grid](
        w_b, w_q, w_scale,
        K, N,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=4
    )

    return y_q, w_q, y_scale, w_scale


def hadamard_quant_forward(x, w, hm):
    x_q, w_q, x_scale, w_scale = triton_hadamard_quant_nt(x, w, hm)
    output = torch._scaled_mm(x_q,
                              w_q.t(),
                              scale_a=x_scale,
                              scale_b=w_scale,
                              out_dtype=x.dtype,
                              use_fast_accum=True)
    return output


def hadamard_quant_backward(y, w, hm):
    y_q, w_q, y_scale, w_scale = triton_hadamard_quant_nn(y, w, hm)
    output = torch._scaled_mm(y_q,
                              w_q.t(),
                              scale_a=y_scale,
                              scale_b=w_scale,
                              out_dtype=y.dtype,
                              use_fast_accum=True)
    return output


def hadamard_quant_update(y, x, hm):
    y_q, x_q, y_scale, x_scale = triton_hadamard_quant_tn(y, x, hm)
    output = torch._scaled_mm(y_q,
                              x_q.t(),
                              scale_a=y_scale,
                              scale_b=x_scale,
                              out_dtype=x.dtype,
                              use_fast_accum=True)
    return output


def hadamard_quant_forward_debug(x, w, hm):
    x_q, x_scale, w_q, w_scale = triton_hadamard_quant_nt(x, w, hm)
    output = torch._scaled_mm(x_q,
                              w_q.t(),
                              scale_a=x_scale,
                              scale_b=w_scale,
                              out_dtype=x.dtype,
                              use_fast_accum=True)
    return output, x_q, w_q, x_scale, w_scale


def hadamard_quant_backward_debug(y, w, hm):
    y_q, w_q, y_scale, w_scale = triton_hadamard_quant_nn(y, w, hm)
    output = torch._scaled_mm(y_q,
                              w_q.t(),
                              scale_a=y_scale,
                              scale_b=w_scale,
                              out_dtype=y.dtype,
                              use_fast_accum=True)
    return output, y_q, w_q, y_scale, w_scale


def hadamard_quant_update_debug(y, x, hm):
    y_q, x_q, y_scale, x_scale = triton_hadamard_quant_tn(y, x, hm)
    output = torch._scaled_mm(y_q,
                              x_q.t(),
                              scale_a=y_scale,
                              scale_b=x_scale,
                              out_dtype=y.dtype,
                              use_fast_accum=True)
    return output, y_q, x_q, y_scale, x_scale


def triton_hadamard_quant_nt_nn_tn(x, w, y, hm):
    triton_hadamard_quant_nt(x, w, hm)
    triton_hadamard_quant_nn(y, w, hm)
    triton_hadamard_quant_tn(y, x, hm)


def fp8_hadamard_f_and_b(x, w, y, hm):
    hadamard_quant_forward(x, w, hm)
    hadamard_quant_backward(y, w, hm)
    hadamard_quant_update(y, x, hm)
