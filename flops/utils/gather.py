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





@triton.jit
def permute_with_mask_map_kernel(grads_data_ptr, grads_scale_ptr, probs_ptr, mask_map_ptr, quant_data_ptr, quant_scale_ptr, output_probs_ptr, num_experts: tl.constexpr, N: tl.constexpr, hs: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(axis=0)
    x = tl.load(grads_data_ptr + pid*N+tl.arange(0, N))
    if GROUP:
        gs = tl.load(grads_scale_ptr + pid*hs + tl.arange(0, hs))
    else:
        gs = tl.load(grads_scale_ptr + pid)
    indices = tl.load(mask_map_ptr+pid*num_experts+tl.arange(0,num_experts))
    count = tl.sum(tl.where(indices>=0,1,0))
    mask_indices = tl.where(indices<0,2**20,indices)
    idx = tl.argmin(mask_indices, 0)
    index = tl.min(mask_indices)
    for i in range(count):
        prob = tl.load(probs_ptr+pid*num_experts+idx)
        if GROUP:
            tl.store(quant_scale_ptr+index*hs+tl.arange(0,hs), gs)
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





@triton.jit
def index_select_smooth_quant_kernel(x_ptr, q_ptr, ss_ptr, qs_ptr, count_ptr, accum_ptr, index_ptr, M, N: tl.constexpr, REVERSE: tl.constexpr, ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    smooth_scale = tl.load(ss_ptr+pid*N+tl.arange(0, N))
    if not REVERSE:
        smooth_scale = 1.0/smooth_scale
    count = tl.load(count_ptr+pid)
    ei = tl.load(accum_ptr+pid)
    si = ei - count
    for i in range(count):
        index = tl.load(index_ptr+si+i)
        x = tl.load(x_ptr+ index*N+tl.arange(0, N)).to(tl.float32)
        x *= smooth_scale
        x_max = tl.maximum(tl.max(tl.abs(x)), 1e-30)

        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(x_max/448.0)))
        else:
            scale = x_max/448.0

        tl.store(qs_ptr+si+i, scale)

        s = 1.0/scale
        x *= s
        xq = x.to(q_ptr.dtype.element_ty)
        tl.store(q_ptr+si*N+i*N+tl.arange(0, N), xq)

"""
select and smooth and quant
x: [bs, dim]
smooth_scales: [n_experts, dim]
indices: [n_experts*topk]
x_q: [bs*topk, dim]
x_scale: [bs*topk]
"""
def triton_index_select_smooth_quant(x, smooth_scales, token_count_per_expert, indices, x_q=None, x_scale=None, reverse=False, round_scale=False):
    # row-wise read, row-wise write
    M, N = x.shape
    n_expert = smooth_scales.shape[0]
    E = indices.shape[0]
    device = x.device 
    if x_q is None:
        x_q = torch.empty((E, N), device=device, dtype=torch.float8_e4m3fn)
    if x_scale is None:
        x_scale = torch.empty((E,), device=device, dtype=torch.float32)
    accum_token_count = torch.cumsum(token_count_per_expert, 0)
    # TODO: adapt for n_expert <= 64
    grid = (n_expert, )
    index_select_smooth_quant_kernel[grid](
        x,
        x_q,
        smooth_scales,
        x_scale,
        token_count_per_expert,
        accum_token_count,
        indices,
        M, N, 
        reverse,
        round_scale,
        num_stages=3,
        num_warps=8
    )
    return x_q,x_scale




@triton.jit
def index_select_smooth_quant_and_sum_kernel(grads_ptr, tokens_ptr, q_ptr, ss_ptr, qs_ptr, count_ptr, accum_ptr, index_ptr, sum_ptr, M, N: tl.constexpr, REVERSE: tl.constexpr, ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    smooth_scale = tl.load(ss_ptr+pid*N+tl.arange(0, N))
    if not REVERSE:
        smooth_scale = 1.0/smooth_scale
    count = tl.load(count_ptr+pid)
    ei = tl.load(accum_ptr+pid)
    si = ei - count
    for i in range(count):
        index = tl.load(index_ptr+si+i)
        x = tl.load(grads_ptr + index*N+tl.arange(0, N)).to(tl.float32)
        t = tl.load(tokens_ptr+si*N+i*N+tl.arange(0, N)).to(tl.float32)
        sums = tl.sum(x*t)
        tl.store(sum_ptr+si+i, sums)

        x *= smooth_scale
        x_max = tl.maximum(tl.max(tl.abs(x)), 1e-30)

        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(x_max/448.0)))
        else:
            scale = x_max/448.0

        tl.store(qs_ptr+si+i, scale)

        s = 1.0/scale
        x *= s
        xq = x.to(q_ptr.dtype.element_ty)
        tl.store(q_ptr+si*N+i*N+tl.arange(0, N), xq)

"""
select and smooth and quant
x: [bs, dim]
smooth_scales: [n_experts, dim]
indices: [n_experts*topk]
x_q: [bs*topk, dim]
x_scale: [bs*topk]
"""
def triton_index_select_smooth_quant_and_sum(grads, tokens, smooth_scales, token_count_per_expert, indices, x_q=None, x_scale=None, x_sum=None, reverse=False, round_scale=False):
    # row-wise read, row-wise write
    M, N = grads.shape
    n_expert, n = smooth_scales.shape
    assert N == n, f'{N=} {n=}'
    E = indices.shape[0]
    device = grads.device 
    if x_q is None:
        x_q = torch.empty((E, N), device=device, dtype=torch.float8_e4m3fn)
    if x_scale is None:
        x_scale = torch.empty((E,), device=device, dtype=torch.float32)
    if x_sum is None:
        x_sum = torch.empty((E,), device=device, dtype=grads.dtype)
    accum_token_count = torch.cumsum(token_count_per_expert, 0)
    grid = (n_expert, )
    index_select_smooth_quant_and_sum_kernel[grid](
        grads,
        tokens,
        x_q,
        smooth_scales,
        x_scale,
        token_count_per_expert,
        accum_token_count,
        indices,
        x_sum,
        M, N, 
        reverse,
        round_scale,
        num_stages=3,
        num_warps=8
    )
    return x_q,x_scale,x_sum



@triton.jit
def smooth_unpermute_backward_kernel(grads_data_ptr, grads_scale_ptr, q_ptr, ss_ptr, qs_ptr, count_ptr, accum_ptr, index_ptr, N: tl.constexpr, hs: tl.constexpr, REVERSE: tl.constexpr, ROUND: tl.constexpr, GROUP: tl.constexpr):
    eid = tl.program_id(axis=0)
    wid = tl.program_id(axis=1)
    T = tl.num_programs(axis=1)

    # row-wise read, row-wise write
    smooth_scale = tl.load(ss_ptr+eid*N+tl.arange(0, N))
    if not REVERSE:
        smooth_scale = 1.0/smooth_scale
    count = tl.load(count_ptr+eid)
    ei = tl.load(accum_ptr+eid)
    si = ei - count
    c = tl.cdiv(count, T)
    for i in range(si+wid*c, tl.minimum(si+wid*c+c,ei)):
        index = tl.load(index_ptr+i)
        x = tl.load(grads_data_ptr + index*N+tl.arange(0, N)).to(tl.float32)
        if GROUP:
            # TODO(nanxiao): index*n->index*hs
            gs = tl.load(grads_scale_ptr + index*hs + tl.arange(0,hs))
            x = tl.reshape(tl.reshape(x, (hs, N//hs))*gs[:,None], (N,))
        else:
            gs = tl.load(grads_scale_ptr + index)
            x *= gs

        x *= smooth_scale
        x_max = tl.maximum(tl.max(tl.abs(x)), 1e-33)


        scale = x_max/448.0
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))

        tl.store(qs_ptr+i, scale)

        s = 1.0/scale
        x *= s
        xq = x.to(q_ptr.dtype.element_ty)
        tl.store(q_ptr+i*N+tl.arange(0, N), xq)

"""
select and smooth and quant
grad_data: [bs, dim]
grad_scale: [bs, dim/128]
smooth_scales: [n_experts, dim]
indices: [n_experts*topk]
x_q: [bs*topk, dim]
x_scale: [bs*topk]
"""
def triton_smooth_unpermute_backward(grad_data, grad_scale, smooth_scales, token_count_per_expert, indices, x_q=None, x_scale=None, reverse=False, round_scale=False):
    assert round_scale
    # row-wise read, row-wise write
    M, N = grad_data.shape
    n_expert, n = smooth_scales.shape
    assert 128%n_expert == 0
    assert N == n
    
    group = grad_scale.ndim > 1
    hs = grad_scale.shape[1] if group else 1

    E = indices.size(0)
    device = grad_data.device 
    if x_q is None:
        x_q = torch.empty((E, N), device=device, dtype=torch.float8_e4m3fn)
    if x_scale is None:
        x_scale = torch.empty((E,), device=device, dtype=torch.float32)
    accum_token_count = torch.cumsum(token_count_per_expert, 0)
    W = 128//n_expert
    grid = (n_expert, W)
    smooth_unpermute_backward_kernel[grid](
        grad_data,
        grad_scale,
        x_q,
        smooth_scales,
        x_scale,
        token_count_per_expert,
        accum_token_count,
        indices,
        N, 
        hs,
        reverse,
        round_scale,
        group,
        num_stages=3,
        num_warps=16
    )
    return x_q,x_scale



@triton.jit
def smooth_permute_with_mask_map_kernel(grads_data_ptr, quant_data_ptr, mask_map_ptr, grads_scale_ptr, smooth_scale_ptr, quant_scale_ptr, M, T, N: tl.constexpr, hs: tl.constexpr, REVERSE: tl.constexpr, ROUND: tl.constexpr, GROUP: tl.constexpr):
    eid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    n_experts = tl.num_programs(axis=0)

    # smooth_scale_ptr = tl.load(smooth_scale_ptrs + eid).to(tl.pointer_type(tl.float32))
    smooth_scale = tl.load(smooth_scale_ptr+eid*N+tl.arange(0, N))
    if not REVERSE:
        smooth_scale = 1.0/smooth_scale
    for i in range(bid*T, tl.minimum(bid*T+T,M)):
        index = tl.load(mask_map_ptr+ i*n_experts+eid)
        mask = index >= 0
        if index >= 0:
            x = tl.load(grads_data_ptr + i*N + tl.arange(0, N), mask=mask).to(tl.float32)

            if GROUP:
                gs = tl.load(grads_scale_ptr + i*hs + tl.arange(0, hs), mask=mask)
                x = tl.reshape(tl.reshape(x,(hs,N//hs))*gs[:,None],(N,))
            else:
                gs = tl.load(grads_scale_ptr + i, mask=mask)
                x *= gs

            x *= smooth_scale
            x_max = tl.maximum(tl.max(tl.abs(x)), 1e-30)

            scale = x_max/448.0
            if ROUND:
                scale = tl.exp2(tl.ceil(tl.log2(scale)))

            tl.store(quant_scale_ptr+index, scale, mask=mask)

            x /= scale
            xq = x.to(quant_data_ptr.dtype.element_ty)
            tl.store(quant_data_ptr+index*N+tl.arange(0, N), xq, mask=mask)

# """
# gather and dequant and smooth quant
# inp: [num_tokens, hidden_size], rowwise_data
# row_id_map: [n_experts, num_tokens], indices
# scale: [num_tokens, hs], rowwise_scale_inv
# num_tokens: [n_experts]
# smooth_scale_ptrs: [n_experts, hidden_size]
# """
def triton_smooth_permute_with_mask_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    scale: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
    smooth_scales: torch.Tensor,
    reverse=True,
    round_scale=False
    ):
    assert round_scale
    assert row_id_map.shape[1] == num_experts
    output = torch.empty((num_out_tokens, hidden_size), dtype=inp.dtype, device="cuda")

    group = scale.ndim == 2
    hs = scale.shape[1] if group else 1

    permuted_scale = torch.empty(
            (num_out_tokens, ), dtype=scale.dtype, device="cuda"
        )
    sm = torch.cuda.get_device_properties(inp.device).multi_processor_count
    T = triton.cdiv(num_tokens, sm)
    grid = (num_experts, sm)
    smooth_permute_with_mask_map_kernel[grid](
        inp,
        output,
        row_id_map,
        scale,
        smooth_scales,
        permuted_scale,
        num_tokens,
        T,
        hidden_size,
        hs,
        reverse,
        round_scale,
        group
    )
    return output, permuted_scale

