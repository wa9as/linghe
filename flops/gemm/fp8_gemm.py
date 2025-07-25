import torch
import triton
import triton.language as tl


# fp8_gemm_configs = [
#     Config({"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n}, num_stages=num_stages, num_warps=8)
#     for block_m in [32, 64, 128]
#     for block_n in [32, 64, 128]
#     for num_stages in [3, 4, 5, 6]
# ]

# fp8_gemm_configs = [
#     Config({"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n}, num_stages=num_stages, num_warps=8)
#     for block_m in [128]
#     for block_n in [128]
#     for num_stages in [3, 4, 5, 6]
# ]

# @triton.autotune(configs=fp8_gemm_configs, key=["N", "K"])
@triton.jit
def trival_fp8_gemm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
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
    for i in range(k):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # accumulator += tl.dot(a, b)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)


def trival_fp8_gemm(a: torch.Tensor, b: torch.Tensor, dtype: torch.types):
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.size()
    N, K = b.size()
    c = torch.empty(M, N, dtype=dtype, device=a.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),  # noqa: E731
                         triton.cdiv(N, META["BLOCK_SIZE_N"]))  # noqa: E731
    BLOCK_SIZE_K = 128
    trival_fp8_gemm_kernel[grid](a, b, c, M, N, K,
                                 BLOCK_SIZE_K,
                                 128,
                                 128,
                                 num_stages=3,
                                 num_warps=8
                                 )
    return c


# @triton.autotune(configs=fp8_gemm_configs, key=["N", "K"])
@triton.jit
def fp8_gemm_nn_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
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
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # accumulator += tl.dot(a, b)
        accumulator = tl.dot(a, b, accumulator)

        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)


def fp8_gemm_nn(a: torch.Tensor, b: torch.Tensor, dtype: torch.types):
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.size()
    K, N = b.size()
    c = torch.empty(M, N, dtype=dtype, device=a.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),  # noqa: E731
                         triton.cdiv(N, META["BLOCK_SIZE_N"]))  # noqa: E731
    BLOCK_SIZE_K = 128
    fp8_gemm_nn_kernel[grid](a, b, c, M, N, K,
                             BLOCK_SIZE_K,
                             128,
                             128,
                             num_stages=3,
                             num_warps=8
                             )
    return c


def get_cuda_autotune_config():
    return [
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,
             'GROUP_SIZE_M': 8}, num_stages=3,
            num_warps=8),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32,
             'GROUP_SIZE_M': 8}, num_stages=4,
            num_warps=4),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
             'GROUP_SIZE_M': 8}, num_stages=4,
            num_warps=4),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32,
             'GROUP_SIZE_M': 8}, num_stages=4,
            num_warps=4),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
             'GROUP_SIZE_M': 8}, num_stages=4,
            num_warps=4),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32,
             'GROUP_SIZE_M': 8}, num_stages=4,
            num_warps=4),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32,
             'GROUP_SIZE_M': 8}, num_stages=5,
            num_warps=2),
        triton.Config(
            {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32,
             'GROUP_SIZE_M': 8}, num_stages=5,
            num_warps=2),
        # Good config for fp8 inputs.
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128,
             'GROUP_SIZE_M': 8}, num_stages=3,
            num_warps=8),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128,
             'GROUP_SIZE_M': 8}, num_stages=3,
            num_warps=8),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128,
             'GROUP_SIZE_M': 8}, num_stages=4,
            num_warps=4),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128,
             'GROUP_SIZE_M': 8}, num_stages=4,
            num_warps=4),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128,
             'GROUP_SIZE_M': 8}, num_stages=4,
            num_warps=4),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64,
             'GROUP_SIZE_M': 8}, num_stages=4,
            num_warps=4),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,
             'GROUP_SIZE_M': 8}, num_stages=4,
            num_warps=4),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64,
             'GROUP_SIZE_M': 8}, num_stages=4,
            num_warps=4)
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def persistent_fp8_gemm_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (
                offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        # b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # accumulator += tl.dot(a, b)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(tl.bfloat16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    # c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    # tl.store(c_ptrs, c, mask=c_mask)
    tl.store(c_ptrs, c)


def persistent_fp8_gemm(a, b, dtype):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N,META['BLOCK_SIZE_N']),) # noqa: E731
    persistent_fp8_gemm_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1)
    )
    return c
