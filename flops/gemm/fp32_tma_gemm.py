import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor
from typing import Optional

@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

def matmul_tma_persistent_get_configs():
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K" : BK, "GROUP_SIZE_M" : group_size_m, "EPILOGUE_SUBTILE" : SUBTILE}, num_stages=s, num_warps=w) \
        for BM in [128] \
        for BN in [128, 256] \
        for BK in [64, 128] \
        for s in ([2, 3, 4]) \
        for w in [4, 8, 16] \
        for SUBTILE in [True, False] \
        for group_size_m in [6, 8 ,16, 32]
    ]

@triton.autotune(
    configs=matmul_tma_persistent_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit()
def matmul_kernel_descriptor_persistent(a_ptr, b_ptr, c_ptr,  
                                        M, N, K,  
                                        BLOCK_SIZE_M: tl.constexpr,  
                                        BLOCK_SIZE_N: tl.constexpr,  
                                        BLOCK_SIZE_K: tl.constexpr,  
                                        GROUP_SIZE_M: tl.constexpr,  
                                        EPILOGUE_SUBTILE: tl.constexpr,  
                                        NUM_SMS: tl.constexpr):  

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl._experimental_make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl._experimental_make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl._experimental_make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2],
    )

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        # tile_id_c = tile_id
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_cm, offs_cn], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], c1)
        else:
            c_desc.store([offs_cm, offs_cn], accumulator)



def matmul_descriptor_persistent(a, b):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = torch.float32

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_descriptor_persistent[grid](
        a, b, c, 
        M, N, K, 
        NUM_SMS=NUM_SMS, 
    )
    return c

@triton.autotune(
    configs=matmul_tma_persistent_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit()
def matmul_kernel_descriptor_persistent_nn(a_ptr, b_ptr, c_ptr,  
                                            M, N, K,  
                                            BLOCK_SIZE_M: tl.constexpr,  
                                            BLOCK_SIZE_N: tl.constexpr,  
                                            BLOCK_SIZE_K: tl.constexpr,  
                                            GROUP_SIZE_M: tl.constexpr,  
                                            EPILOGUE_SUBTILE: tl.constexpr,  
                                            NUM_SMS: tl.constexpr):  

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl._experimental_make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl._experimental_make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    c_desc = tl._experimental_make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_k, offs_bn])
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        # tile_id_c = tile_id
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        c_desc.store([offs_cm, offs_cn], accumulator)

def matmul_descriptor_persistent_nn(a, b):

    M, K = a.shape
    K, N = b.shape
    dtype = torch.float32

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_descriptor_persistent_nn[grid](
        a, b, c, 
        M, N, K, 
        NUM_SMS=NUM_SMS, 
    )
    return c

@triton.autotune(
    configs=matmul_tma_persistent_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit()
def matmul_kernel_descriptor_persistent_tn(a_ptr, b_ptr, c_ptr,  
                                            M, N, K,  
                                            BLOCK_SIZE_M: tl.constexpr,  
                                            BLOCK_SIZE_N: tl.constexpr,  
                                            BLOCK_SIZE_K: tl.constexpr,  
                                            GROUP_SIZE_M: tl.constexpr,  
                                            EPILOGUE_SUBTILE: tl.constexpr,  
                                            NUM_SMS: tl.constexpr):  

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl._experimental_make_tensor_descriptor(
        a_ptr,
        shape=[K, M],
        strides=[M, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_M],
    )
    b_desc = tl._experimental_make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    c_desc = tl._experimental_make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_k, offs_am])
            b = b_desc.load([offs_k, offs_bn])
            accumulator = tl.dot(a.T, b, accumulator)

        tile_id_c += NUM_SMS
        # tile_id_c = tile_id
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        c_desc.store([offs_cm, offs_cn], accumulator)

def matmul_descriptor_persistent_tn(a, b):

    K, M = a.shape
    K, N = b.shape
    dtype = torch.float32
    
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    
    triton.set_allocator(alloc_fn)
    
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_descriptor_persistent_tn[grid](
        a, b, c, 
        M, N, K, 
        NUM_SMS=NUM_SMS, 
    )
    return c

def torch_matmul(x,w):
    return torch.nn.functional.linear(x.float(),w.float())
    # return torch.nn.functional.linear(x,w)


# M = 4096
# N = 4096
# K = 8192
# dtype = torch.bfloat16

# a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16).to(dtype)
# b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).to(dtype)
# b = b.T.contiguous()
# y_ref = a.float() @ b.float()
# # y_ref = torch_matmul(a,b)
# y = matmul_descriptor_persistent(a,b)
# output_check(y_ref, y.float(), mode='fp32_gemm')

# a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16).to(dtype)
# b = torch.randn((K, N), device="cuda", dtype=torch.bfloat16).to(dtype)
# # y = matmul_tma_persistent(a,b)
# y_ref = a.float() @ b.float()
# y = matmul_descriptor_persistent_nn(a,b)
# output_check(y_ref, y.float(), mode='fp32_gemm')

# a = torch.randn((K, M), device="cuda", dtype=torch.bfloat16).to(dtype)
# b = torch.randn((K, N), device="cuda", dtype=torch.bfloat16).to(dtype)
# y_ref = a.T.float() @ b.float()
# y = matmul_descriptor_persistent_tn(a, b)
# output_check(y_ref, y.float(), mode='fp32_gemm')



  