import math

import torch
import triton
import triton.language as tl
from triton import Config




@triton.jit
def index_select_kernel(x_ptr, out_ptr, scale_ptr, scale_out_ptr, index_ptr, M, T, N: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    for i in range(T):
        dst_idx = pid*T+i
        src_idx = tl.load(index_ptr+dst_idx, mask=dst_idx < M)
        x = tl.load(x_ptr+ src_idx*N+tl.arange(0, N), mask=dst_idx < M)
        tl.store(out_ptr+dst_idx*N+tl.arange(0, N), x, mask=dst_idx < M)

        if SCALE:
            scale = tl.load(scale_ptr+ src_idx, mask=dst_idx < M)
            tl.store(scale_out_ptr+dst_idx, scale,  mask=dst_idx < M)

"""
index select for quantized tensor
x: [bs, dim]
x_scale: [bs]
indices: [K]
"""
def triton_index_select(x, indices, scale=None, out=None, scale_out=None):
    # row-wise read, row-wise write
    M, N = x.shape
    E = indices.shape[0]
    device = x.device 
    if out is None:
        out = torch.empty((E, N), device=device, dtype=x.dtype)
    if scale is not None and scale_out is None:
        scale_out = torch.empty((E,), device=device, dtype=scale.dtype)
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    T = triton.cdiv(E, sm)
    SCALE = scale is not None
    grid = (sm, )
    index_select_kernel[grid](
        x,
        out,
        scale,
        scale_out,
        indices,
        E, T, N, 
        SCALE,
        num_stages=3,
        num_warps=8
    )
    return out,scale_out




# @triton.jit
# def index_from_map_kernel(x_ptr, out_ptr, scale_ptr, scale_out_ptr, index_ptr, M, T, N: tl.constexpr, SCALE: tl.constexpr):
#     pid = tl.program_id(axis=0)
#     # row-wise read, row-wise write
#     for i in range(T):
#         dst_idx = pid*T+i
#         src_idx = tl.load(index_ptr+dst_idx, mask=dst_idx < M)
#         x = tl.load(x_ptr+ src_idx*N+tl.arange(0, N), mask=dst_idx < M)
#         tl.store(out_ptr+dst_idx*N+tl.arange(0, N), x, mask=dst_idx < M)

#         if SCALE:
#             scale = tl.load(scale_ptr+ src_idx, mask=dst_idx < M)
#             tl.store(scale_out_ptr+dst_idx, scale,  mask=dst_idx < M)

# """
# x: [in_token, n_experts]
# """
# def triton_index_from_map(x):
#     M, n = x.shape
#     device = x.device 
#     xt = torch.empty((M, n), device=device, dtype=torch.int32)
#     indices = torch.empty((M, n), device=device, dtype=torch.int32)
#     N = triton.next_power_of_2(n)

#     grid = (1, )
#     index_from_map_kernel[grid](
#         x,
#         indices,

#         scale,
#         scale_out,
#         indices,
#         E, T, N, 
#         SCALE,
#         num_stages=3,
#         num_warps=8
#     )
#     return out,scale_out







# @triton.jit
# def permute_with_mask_map_kernel(grads_data_ptr, grads_scale_ptr, probs_ptr, mask_map_ptr, quant_data_ptr, quant_scale_ptr, output_probs_ptr, num_experts: tl.constexpr, N: tl.constexpr):
#     pid = tl.program_id(axis=0)
#     M = tl.num_programs(axis=0)

#     x = tl.load(grads_data_ptr + pid*N+tl.arange(0, N))
#     gs = tl.load(grads_scale_ptr + pid)

#     for i in range(num_experts):
#         index = tl.load(mask_map_ptr+i*M+pid)
#         mask = index >= 0
#         prob = tl.load(probs_ptr+pid*num_experts+i, mask=mask)

#         tl.store(quant_scale_ptr+index, gs, mask=mask)
#         tl.store(output_probs_ptr+index, prob, mask=mask)
#         tl.store(quant_data_ptr+index*N+tl.arange(0, N), x, mask=mask)



# # """
# # gather and smooth quant
# # inp: [num_tokens, hidden_size], rowwise_data
# # row_id_map: [n_experts, num_tokens], indices
# # scale: [num_tokens], rowwise_scale_inv
# # smooth_scale_ptrs: [n_experts], data_ptr
# # """

# def triton_permute_with_mask_map(
#     inp: torch.Tensor,
#     scale: torch.Tensor,
#     probs: torch.Tensor,
#     row_id_map: torch.Tensor,
#     num_out_tokens: int,
#     ):
#     num_tokens, hidden_size = inp.shape 
#     num_experts = row_id_map.size(0)

#     output = torch.empty((num_out_tokens, hidden_size), dtype=inp.dtype, device="cuda")

#     permuted_scale = torch.empty(
#         (num_out_tokens, ), dtype=scale.dtype, device="cuda"
#     )

#     permuted_probs = torch.empty(
#         (num_out_tokens, ), dtype=probs.dtype, device="cuda"
#     )

#     grid = (num_tokens, )
#     permute_with_mask_map_kernel[grid](
#         inp,
#         scale,
#         probs,
#         row_id_map,
#         output,
#         permuted_scale,
#         permuted_probs,
#         num_experts,
#         hidden_size,
#         num_stages=3,
#         num_warps=4
#     )
#     return output, permuted_scale, permuted_probs





# @triton.jit
# def permute_with_mask_map_kernel(grads_data_ptr, grads_scale_ptr, probs_ptr, mask_map_ptr, quant_data_ptr, quant_scale_ptr, output_probs_ptr, num_experts: tl.constexpr, N: tl.constexpr):
#     pid = tl.program_id(axis=0)

#     x = tl.load(grads_data_ptr + pid*N+tl.arange(0, N))
#     gs = tl.load(grads_scale_ptr + pid)

#     for i in range(num_experts):
#         index = tl.load(mask_map_ptr+pid*num_experts+i)
#         mask = index >= 0
#         prob = tl.load(probs_ptr+pid*num_experts+i, mask=mask)

#         tl.store(quant_scale_ptr+index, gs, mask=mask)
#         tl.store(output_probs_ptr+index, prob, mask=mask)
#         tl.store(quant_data_ptr+index*N+tl.arange(0, N), x, mask=mask)



# # """
# # gather and smooth quant
# # inp: [num_tokens, hidden_size], rowwise_data
# # row_id_map: [n_experts, num_tokens], indices
# # scale: [num_tokens], rowwise_scale_inv
# # smooth_scale_ptrs: [n_experts], data_ptr
# # """

# def triton_permute_with_mask_map(
#     inp: torch.Tensor,
#     scale: torch.Tensor,
#     probs: torch.Tensor,
#     row_id_map: torch.Tensor,
#     num_out_tokens: int,
#     ):
#     num_tokens, hidden_size = inp.shape 
#     num_experts = row_id_map.size(1)  # not transposed

#     output = torch.empty((num_out_tokens, hidden_size), dtype=inp.dtype, device="cuda")

#     permuted_scale = torch.empty(
#         (num_out_tokens, ), dtype=scale.dtype, device="cuda"
#     )

#     permuted_probs = torch.empty(
#         (num_out_tokens, ), dtype=probs.dtype, device="cuda"
#     )
#     grid = (num_tokens, )
#     permute_with_mask_map_kernel[grid](
#         inp,
#         scale,
#         probs,
#         row_id_map,
#         output,
#         permuted_scale,
#         permuted_probs,
#         num_experts,
#         hidden_size,
#         num_stages=3,
#         num_warps=4
#     )
#     return output, permuted_scale, permuted_probs








@triton.jit
def permute_with_mask_map_kernel(grads_data_ptr, grads_scale_ptr, probs_ptr, mask_map_ptr, quant_data_ptr, quant_scale_ptr, output_probs_ptr, num_experts: tl.constexpr, N: tl.constexpr, n: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(axis=0)
    x = tl.load(grads_data_ptr + pid*N+tl.arange(0, N))
    if GROUP:
        gs = tl.load(grads_scale_ptr + pid)
    else:
        gs = tl.load(grads_scale_ptr + pid*n + tl.arange(0, n))
    indices = tl.load(mask_map_ptr+pid*num_experts+tl.arange(0,num_experts))
    count = tl.sum(tl.where(indices>=0,1,0))
    mask_indices = tl.where(indices<0,2**20,indices)
    idx = tl.argmin(mask_indices, 0)
    index = tl.min(mask_indices)
    for i in range(count):
        prob = tl.load(probs_ptr+pid*num_experts+idx)
        if GROUP:
            tl.store(quant_scale_ptr+index*n+tl.arange(0,n), gs)
        else:
            tl.store(quant_scale_ptr+index, gs)
        tl.store(output_probs_ptr+index, prob)
        tl.store(quant_data_ptr+index*N+tl.arange(0, N), x)

        mask_indices = tl.where(indices<=index,2**20,indices)
        idx = tl.argmin(mask_indices, 0)
        index = tl.min(mask_indices)


# """
# gather and smooth quant
# inp: [num_tokens, hidden_size], rowwise_data
# row_id_map: [n_experts, num_tokens], indices
# scale: [num_tokens], rowwise_scale_inv
# smooth_scale_ptrs: [n_experts], data_ptr
# """

def triton_permute_with_mask_map(
    inp: torch.Tensor,
    scale: torch.Tensor,
    probs: torch.Tensor,
    row_id_map: torch.Tensor,
    num_out_tokens: int,
    ):
    num_tokens, hidden_size = inp.shape 
    num_experts = row_id_map.size(1)  # not transposed
    group = scale.ndim > 1
    hs = scale.shape[1] if group else 1

    output = torch.empty((num_out_tokens, hidden_size), dtype=inp.dtype, device="cuda")

    permuted_scale = torch.empty(
        (num_out_tokens, ), dtype=scale.dtype, device="cuda"
    )

    permuted_probs = torch.empty(
        (num_out_tokens, ), dtype=probs.dtype, device="cuda"
    )
    grid = (num_tokens, )
    permute_with_mask_map_kernel[grid](
        inp,
        scale,
        probs,
        row_id_map,
        output,
        permuted_scale,
        permuted_probs,
        num_experts,
        hidden_size,
        hs,
        group,
        num_stages=3,
        num_warps=8
    )
    return output, permuted_scale, permuted_probs




