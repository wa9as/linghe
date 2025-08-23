import torch
import triton
import triton.language as tl


# for megatron 0.11 scatter_add

@triton.jit
def aligned_scatter_add_kernel(x_ptr, o_ptr, indices_ptr, weights_ptr, M,
                               N: tl.constexpr, K: tl.constexpr,
                               SCALE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, N)

    sums = tl.zeros((N,), dtype=tl.float32)
    for i in range(K):
        idx = tl.load(indices_ptr + pid * K + i)
        x = tl.load(x_ptr + idx * N + offs)
        if SCALE == 1:
            weight = tl.load(weights_ptr + idx)
            sums += x * weight
        else:
            sums += x

    tl.store(o_ptr + pid * N + offs, sums)


def triton_aligned_scatter_add(x, outputs, indices, weights=None):
    M, N = x.shape
    m = outputs.size(0)

    indices = torch.argsort(indices)
    K = M // m
    assert K * m == M
    SCALE = 1 if weights is not None else 0

    num_stages = 5
    num_warps = 8

    grid = (m,)
    aligned_scatter_add_kernel[grid](
        x, outputs,
        indices,
        weights,
        M, N, K,
        SCALE,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return outputs


# for deepep scatter_add
# atomic_add supports fp16 and fp32, but not bf16 

@triton.jit
def scatter_add_kernel(x_ptr, o_ptr, indices_ptr, M, T, N: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, N)

    for i in range(T):
        src_idx = pid * T + i
        dst_idx = tl.load(indices_ptr + src_idx, mask=src_idx < M)
        x = tl.load(x_ptr + src_idx * N + offs, mask=src_idx < M).to(tl.float32)
        tl.atomic_add(o_ptr + dst_idx * N + offs, x)


@triton.jit
def fp32_to_bf16_kernel(x_ptr, o_ptr, M, T, N: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, N)

    for i in range(T):
        idx = pid * T + i
        x = tl.load(x_ptr + idx * N + offs, mask=idx < M)
        tl.store(o_ptr + idx * N + offs, x, mask=idx < M)


def triton_scatter_add(x, outputs, indices):
    M, N = x.shape

    float_outputs = torch.zeros(outputs.shape, dtype=torch.float32,
                                device=outputs.device)

    sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    T = triton.cdiv(M, sm)

    num_stages = 5
    num_warps = 8

    grid = (sm,)
    scatter_add_kernel[grid](
        x, float_outputs,
        indices,
        M, T, N,
        num_stages=num_stages,
        num_warps=num_warps
    )

    m = outputs.shape[0]
    T = triton.cdiv(m, sm)
    grid = (sm,)
    fp32_to_bf16_kernel[grid](
        float_outputs, outputs,
        m, T, N,
        num_stages=num_stages,
        num_warps=num_warps
    )

    return outputs

@triton.jit
def unpermute_with_mask_map_kernel(grads_ptr, probs_ptr, mask_map_ptr,
                                   output_ptr, output_probs_ptr,
                                   num_experts: tl.constexpr, N: tl.constexpr,
                                   PROB: tl.constexpr):
    pid = tl.program_id(axis=0)

    sums = tl.zeros((N,), dtype=tl.float32)

    indices = tl.load(
        mask_map_ptr + pid * num_experts + tl.arange(0, num_experts))
    count = tl.sum(tl.where(indices >= 0, 1, 0))
    mask_indices = tl.where(indices < 0, 2 ** 20, indices)
    idx = tl.argmin(mask_indices, 0)
    index = tl.min(mask_indices)

    for i in range(count):

        mask = index >= 0
        sums += tl.load(grads_ptr + index * N + tl.arange(0, N), mask=mask).to(
            tl.float32)

        if PROB:
            prob = tl.load(probs_ptr + index, mask=mask)
            tl.store(output_probs_ptr + pid * num_experts + idx, prob,
                     mask=mask)

        mask_indices = tl.where(indices <= index, 2 ** 20, indices)
        idx = tl.argmin(mask_indices, 0)
        index = tl.min(mask_indices)

    tl.store(output_ptr + pid * N + tl.arange(0, N), sums)


# """
# gather and smooth quant
# inp: [num_tokens, hidden_size], rowwise_data
# row_id_map: [n_experts, num_tokens], indices
# prob: [num_out_tokens], rowwise_scale_inv
# """

def triton_unpermute_with_mask_map(
        grad: torch.Tensor,
        row_id_map: torch.Tensor,
        probs: torch.Tensor,
):
    hidden_size = grad.shape[1]
    num_tokens, num_experts = row_id_map.shape  # not transposed

    output = torch.empty((num_tokens, hidden_size), dtype=grad.dtype,
                         device="cuda")

    PROB = probs is not None
    if PROB:
        restore_probs = torch.zeros((num_tokens, num_experts),
                                    dtype=probs.dtype, device="cuda")
    else:
        restore_probs = None

    grid = (num_tokens,)
    unpermute_with_mask_map_kernel[grid](
        grad,
        probs,
        row_id_map,
        output,
        restore_probs,
        num_experts,
        hidden_size,
        PROB,
        num_stages=4,
        num_warps=4
    )
    return output, restore_probs


@triton.jit
def block_count_kernel(map_ptr, count_ptr, M, B, T: tl.constexpr,
                       b: tl.constexpr, E: tl.constexpr):
    pid = tl.program_id(axis=0)

    counts = tl.zeros((E,), dtype=tl.int32)
    offs = pid * B * E + tl.arange(0, b)[:, None] * E + tl.arange(0, E)[None, :]
    t = tl.cdiv(B, b)
    for i in range(t):
        mask = pid * B + i * b + tl.arange(0, b)[:, None] < tl.minimum(M,
                                                                       pid * B + B)
        values = tl.load(map_ptr + offs, mask=mask).to(tl.int32)
        counts += tl.sum(values, 0)
        offs += b * E

    tl.store(count_ptr + pid * E + tl.arange(0, E), counts)


@triton.jit
def accumulate_count_kernel(map_ptr, count_ptr, output_ptr, M, B, P,
                            T: tl.constexpr, b: tl.constexpr, E: tl.constexpr):
    pid = tl.program_id(axis=0)

    indices = tl.arange(0, T)[:, None] * E + tl.arange(0, E)[None, :]
    counts = tl.load(count_ptr + indices)
    sum_counts = tl.sum(counts, 0)
    sum_counts = tl.cdiv(sum_counts, P) * P
    accum_sum_counts = tl.cumsum(sum_counts, 0)

    partial_counts = tl.sum(tl.where(indices < pid * E, counts, 0), 0)

    count_offset = accum_sum_counts - sum_counts + partial_counts - 1
    offs = pid * B * E + tl.arange(0, b)[:, None] * E + tl.arange(0, E)[None, :]
    t = tl.cdiv(B, b)
    for i in range(t):
        mask = pid * B + i * b + tl.arange(0, b)[:, None] < tl.minimum(M,
                                                                       pid * B + B)
        values = tl.load(map_ptr + offs, mask=mask).to(tl.int32)
        acc = count_offset + tl.cumsum(values, 0)
        count_offset = tl.max(acc, 0)
        acc = tl.where(values == 0, -1, acc)
        tl.store(output_ptr + offs, acc, mask=mask)
        offs += b * E


# """
# make row id map, shape:[n_tokens, n_experts]
# """
def triton_make_row_id_map(
        routing_map: torch.Tensor, 
        multiple_of: int = 1
):
    n_tokens, n_experts = routing_map.shape
    T = 128
    block_counts = torch.empty((T, n_experts), dtype=torch.int32,
                               device=routing_map.device)
    output = torch.empty((n_tokens, n_experts), dtype=torch.int32,
                         device=routing_map.device)
    
    B = triton.cdiv(n_tokens, T)
    b = 16
    grid = (T,)
    block_count_kernel[grid](
        routing_map,
        block_counts,
        n_tokens,
        B,
        T,
        b,
        n_experts,
        num_stages=3,
        num_warps=8
    )

    accumulate_count_kernel[grid](
        routing_map,
        block_counts,
        output,
        n_tokens,
        B,
        multiple_of,
        T,
        b,
        n_experts,
        num_stages=3,
        num_warps=8
    )

    return output
