import os
import torch
import triton
import triton.language as tl
from triton import Config


# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


# fp8_gemm_configs = [
#     Config({"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n}, num_stages=num_stages, num_warps=8)
#     for block_m in [64, 128, 256]
#     for block_n in [64, 128, 256]
#     for num_stages in [3, 4, 5, 6]
# ]

# @triton.autotune(configs=fp8_gemm_configs, key=["N", "K"])
@triton.jit
def scaled_mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    ACCUM: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if EVEN:
        for i in range(k):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K
    else:
        for i in range(k-1):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None,:]<K-(k-1)*BLOCK_SIZE_K)
        b = tl.load(b_ptrs, mask=offs_k[:, None]<K-(k-1)*BLOCK_SIZE_K)
        accumulator = tl.dot(a, b, accumulator)

    a_scale = tl.load(a_scale_ptr+offs_m)
    b_scale = tl.load(b_scale_ptr+offs_n)

    accumulator = accumulator*a_scale[:,None]*b_scale[None,:]
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    if ACCUM:
        accumulator += tl.load(c_ptrs).to(tl.float32)
    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c)


def triton_scaled_mm(a: torch.Tensor,  b: torch.Tensor, a_scale: torch.Tensor, b_scale:torch.Tensor, out_dtype = torch.float32, c = None, accum=True):
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.size()
    N, K = b.size()
    if c is None:
        c = torch.empty(M, N, dtype=out_dtype, device=a.device)
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256  # 256
    EVEN = K%BLOCK_SIZE_K == 0
    grid = lambda META: (M//BLOCK_SIZE_M, N//BLOCK_SIZE_N)  # noqa: E731
    scaled_mm_kernel[grid](a, b, c, 
                                  a_scale,
                                  b_scale,
                                  N, K, 
                                  accum,
                                  EVEN,
                                  BLOCK_SIZE_K,
                                  BLOCK_SIZE_M,
                                  BLOCK_SIZE_N,
                                  num_stages=3,
                                  num_warps=8
                                 )
    return c
