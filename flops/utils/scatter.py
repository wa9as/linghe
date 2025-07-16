
import math
import os
import torch
import triton
import triton.language as tl
from triton import Config


# for megatron 0.11 scatter_add

@triton.jit
def aligned_scatter_add_kernel(x_ptr, o_ptr, indices_ptr, weights_ptr, M, N: tl.constexpr, K: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, N)

    sums = tl.zeros((N,),dtype=tl.float32)
    for i in range(K):
        idx = tl.load(indices_ptr+pid*K+i)
        x = tl.load(x_ptr+idx*N+offs)
        if SCALE == 1:
            weight = tl.load(weights_ptr+idx)
            sums += x*weight
        else:
            sums += x

    tl.store(o_ptr+pid*N+offs,sums)


def triton_aligned_scatter_add(x, outputs, indices, weights=None):
    M, N = x.shape
    m = outputs.size(0)

    indices = torch.argsort(indices)
    K = M//m 
    assert K*m == M
    SCALE = 1 if weights is not None else 0

    num_stages = 5
    num_warps = 8

    grid = (m, )
    aligned_scatter_add_kernel[grid](
        x, outputs,
        indices,
        weights,
        M, N, K,
        SCALE ,
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
        src_idx = pid*T+i
        dst_idx = tl.load(indices_ptr+src_idx, mask=src_idx<M)
        x = tl.load(x_ptr+src_idx*N+offs, mask=src_idx<M).to(tl.float32)
        tl.atomic_add(o_ptr+dst_idx*N+offs,x)

@triton.jit
def fp32_to_bf16_kernel(x_ptr, o_ptr, M, T, N: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, N)

    for i in range(T):
        idx = pid*T+i
        x = tl.load(x_ptr+idx*N+offs, mask=idx<M)
        tl.store(o_ptr+idx*N+offs,x, mask=idx<M)


def triton_scatter_add(x, outputs, indices):
    M, N = x.shape

    float_outputs = torch.zeros(outputs.shape, dtype=torch.float32, device=outputs.device)

    sm = torch.cuda.get_device_properties(device).multi_processor_count
    T = triton.cdiv(M, sm)

    num_stages = 5
    num_warps = 8

    grid = (sm, )
    scatter_add_kernel[grid](
        x, float_outputs,
        indices,
        M, T, N, 
        num_stages=num_stages,
        num_warps=num_warps
    )

    m = outputs.shape[0]    
    T = triton.cdiv(m, sm)
    grid = (sm, )
    fp32_to_bf16_kernel[grid](
        float_outputs, outputs,
        m, T, N, 
        num_stages=num_stages,
        num_warps=num_warps
    )

    return outputs




@triton.jit
def scatter_add_with_count_kernel(x_ptr, o_ptr, indices_ptr, counts_ptr, accum_ptr, M, m, T, N: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, N)
    for i in range(T):
        count = tl.load(counts_ptr + pid*T+i, mask=pid*T+i<m)
        ei = tl.load(accum_ptr + pid*T+i, mask=pid*T+i<m)
        si = ei - count
        sums = tl.zeros((N,),dtype=tl.float32)
        for j in range(si, ei):
            idx = tl.load(indices_ptr+j, mask=pid*T+i<m)
            x = tl.load(x_ptr+idx*N+offs, mask=pid*T+i<m).to(tl.float32)
            sums += x 
        tl.store(o_ptr+pid*T*N+i*N+offs, sums, mask=pid*T+i<m)



def triton_scatter_add_with_count(x, outputs, indices, counts):
    M, N = x.shape
    m = outputs.size(0)

    num_stages = 3
    num_warps = 16

    indices = torch.argsort(indices)
    accum = torch.cumsum(counts, 0)
    sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    T = triton.cdiv(m, sm)
    grid = (sm, )
    scatter_add_with_count_kernel[grid](
        x, outputs,
        indices,
        counts,
        accum,
        M, m, T, N,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return outputs




# @triton.jit
# def unpermute_with_mask_map_kernel(grads_ptr, probs_ptr, mask_map_ptr, output_ptr, output_probs_ptr, num_experts: tl.constexpr, N: tl.constexpr, PROB: tl.constexpr):
#     pid = tl.program_id(axis=0)
#     M = tl.num_programs(axis=0)

#     sums = tl.zeros((N,), dtype=tl.float32)

#     for i in range(num_experts):

#         index = tl.load(mask_map_ptr+i*M+pid)
#         mask = index >= 0
#         sums  += tl.load(grads_ptr + index*N+tl.arange(0, N), mask=mask).to(tl.float32)

#         if PROB:
#             prob = tl.load(probs_ptr+index, mask=mask)
#             tl.store(output_probs_ptr+pid*num_experts+i, prob, mask=mask)

#     tl.store(output_ptr+pid*N+tl.arange(0, N), sums)



# # """
# # gather and smooth quant
# # inp: [num_tokens, hidden_size], rowwise_data
# # row_id_map: [n_experts, num_tokens], indices
# # scale: [num_tokens], rowwise_scale_inv
# # smooth_scale_ptrs: [n_experts], data_ptr
# # """

# def triton_unpermute_with_mask_map(
#     grad: torch.Tensor,
#     row_id_map: torch.Tensor,
#     probs: torch.Tensor,
#     ):
#     hidden_size = grad.shape[1]
#     num_experts, num_tokens = row_id_map.shape

#     output = torch.empty((num_tokens, hidden_size), dtype=grad.dtype, device="cuda")

#     PROB = probs is not None
#     if PROB:
#         restore_probs = torch.zeros((num_tokens, num_experts), dtype=probs.dtype, device="cuda")
#     else:
#         restore_probs = None

#     grid = (num_tokens, )
#     unpermute_with_mask_map_kernel[grid](
#         grad,
#         probs,
#         row_id_map,
#         output,
#         restore_probs,
#         num_experts,
#         hidden_size,
#         PROB,
#         num_stages=4,
#         num_warps=8
#     )
#     return output, restore_probs






@triton.jit
def unpermute_with_mask_map_kernel(grads_ptr, probs_ptr, mask_map_ptr, output_ptr, output_probs_ptr, num_experts: tl.constexpr, N: tl.constexpr, PROB: tl.constexpr):
    pid = tl.program_id(axis=0)

    sums = tl.zeros((N,), dtype=tl.float32)

    indices = tl.load(mask_map_ptr+pid*num_experts+tl.arange(0,num_experts))
    count = tl.sum(tl.where(indices>=0,1,0))
    mask_indices = tl.where(indices<0,2**20,indices)
    idx = tl.argmin(mask_indices, 0)
    index = tl.min(mask_indices)

    for i in range(count):

        mask = index >= 0
        sums += tl.load(grads_ptr + index*N+tl.arange(0, N), mask=mask).to(tl.float32)

        if PROB:
            prob = tl.load(probs_ptr+index, mask=mask)
            tl.store(output_probs_ptr+pid*num_experts+idx, prob, mask=mask)

        mask_indices = tl.where(indices<=index,2**20,indices)
        idx = tl.argmin(mask_indices, 0)
        index = tl.min(mask_indices)

    tl.store(output_ptr+pid*N+tl.arange(0, N), sums)



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

    output = torch.empty((num_tokens, hidden_size), dtype=grad.dtype, device="cuda")

    PROB = probs is not None
    if PROB:
        restore_probs = torch.zeros((num_tokens, num_experts), dtype=probs.dtype, device="cuda")
    else:
        restore_probs = None

    grid = (num_tokens, )
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
def make_row_id_map_kernel(map_ptrs, count_ptr, output_ptr, M, E: tl.constexpr, B: tl.constexpr):
    pid = tl.program_id(axis=0)
    n_experts = tl.num_programs(axis=0)

    counts = tl.load(count_ptr+tl.arange(0,E))
    counts = tl.cumsum(counts)

    mask_counts = tl.where(tl.arange(0,E) < pid, counts, 0)
    count = tl.max(mask_counts) - 1

    n = tl.cdiv(M, B)
    for i in range(n):
        mask = i*B+tl.arange(0,B)<M
        values = tl.load(map_ptrs+i*B*n_experts+tl.arange(0,B)*n_experts+pid, mask=mask).to(tl.int32)
        acc = count + tl.cumsum(values)

        count = tl.max(acc)
        acc = tl.where(values==0,-1,acc)

        tl.store(output_ptr+i*B*n_experts+tl.arange(0,B)*n_experts+pid, acc, mask=mask)





# """
# make row id map, not transposed
# """

def triton_make_row_id_map(
    routing_map: torch.Tensor
    ):
    n_tokens, n_experts = routing_map.shape
    counts = routing_map.sum(0)

    output = torch.empty((n_tokens, n_experts), dtype=torch.int32, device=routing_map.device)
    B = 128
    grid = (n_experts, )
    make_row_id_map_kernel[grid](
        routing_map,
        counts,
        output,
        n_tokens,
        n_experts,
        B,
        num_stages=3,
        num_warps=16
    )
    return output





# @triton.jit
# def make_row_id_map_kernel(map_ptrs, count_ptr, output_ptr, M, E: tl.constexpr, B: tl.constexpr):
#     pid = tl.program_id(axis=0)
#     counts = tl.load(count_ptr+tl.arange(0,E))
#     count = tl.cumsum(counts) - 1

#     n = tl.cdiv(M, B)
#     offs = tl.arange(0,B)[:,None]*E + tl.arange(0,E)[None, :]
#     for i in range(n):
#         mask = i*B+tl.arange(0,B)[:,None]<M
#         values = tl.load(map_ptrs+offs, mask=mask).to(tl.int32)
#         acc = count + tl.cumsum(values, axis=0)

#         count = tl.max(acc, 0)
#         acc = tl.where(values==0,-1,acc)

#         tl.store(output_ptr+offs, acc, mask=mask)
#         offs += E*B





# # """
# # make row id map, not transposed
# # """

# def triton_make_row_id_map(
#     routing_map: torch.Tensor
#     ):
#     n_tokens, n_experts = routing_map.shape
#     counts = routing_map.sum(0)

#     output = torch.empty((n_tokens, n_experts), dtype=torch.int32, device=routing_map.device)
#     B = 128
#     grid = (1, )
#     make_row_id_map_kernel[grid](
#         routing_map,
#         counts,
#         output,
#         n_tokens,
#         n_experts,
#         B,
#         num_stages=3,
#         num_warps=8
#     )
#     return output
