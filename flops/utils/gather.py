from typing import Optional, List
import torch
import triton
import triton.language as tl





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
def make_row_id_map_kernel(map_ptr, count_ptr, output_ptr, M, B, P,
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

    make_row_id_map_kernel[grid](
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


@triton.jit
def make_row_id_map_and_indices_kernel(map_ptr, count_ptr, row_map_ptr, row_indices_ptr, M, B, P,
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
        output_acc = tl.where(values == 0, -1, acc)
        tl.store(row_map_ptr + offs, output_acc, mask=mask)

        tl.store(row_indices_ptr + acc, pid * B + i * b +  tl.arange(0, b)[:,None] + (0*tl.arange(0, E))[None, :], mask = mask & values != 0)

        offs += b * E


"""
routing map, shape:[n_tokens, n_experts]
num_out_tokens, shape:[sum(round(bs))]

row id map, shape:[n_tokens, n_experts]
row id indices, shape: [sum(n_tokens_per_experts)]
"""
def triton_make_row_id_map_and_indices(
        routing_map: torch.Tensor, 
        num_out_tokens: int,
        multiple_of: int = 1,
):
    n_tokens, n_experts = routing_map.shape
    T = 128
    block_counts = torch.empty((T, n_experts), dtype=torch.int32,
                               device=routing_map.device)
    row_id_map = torch.empty((n_tokens, n_experts), dtype=torch.int32,
                         device=routing_map.device)
    row_id_indices = torch.empty((num_out_tokens, ), dtype=torch.int32,
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

    make_row_id_map_and_indices_kernel[grid](
        routing_map,
        block_counts,
        row_id_map,
        row_id_indices,
        n_tokens,
        B,
        multiple_of,
        T,
        b,
        n_experts,
        num_stages=3,
        num_warps=8
    )
    return row_id_map, row_id_indices


@triton.jit
def index_select_kernel(x_ptr, out_ptr, scale_ptr, scale_out_ptr, index_ptr, M,
                        T, N: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    for i in range(T):
        dst_idx = pid * T + i
        src_idx = tl.load(index_ptr + dst_idx, mask=dst_idx < M)
        x = tl.load(x_ptr + src_idx * N + tl.arange(0, N), mask=dst_idx < M)
        tl.store(out_ptr + dst_idx * N + tl.arange(0, N), x, mask=dst_idx < M)

        if SCALE:
            scale = tl.load(scale_ptr + src_idx, mask=dst_idx < M)
            tl.store(scale_out_ptr + dst_idx, scale, mask=dst_idx < M)


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
    grid = (sm,)
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
    return out, scale_out


@triton.jit
def permute_with_mask_map_kernel(data_ptr, scale_ptr, probs_ptr,
                                 mask_map_ptr, 
                                 output_data_ptr, 
                                 output_scale_ptr,
                                 output_probs_ptr, 
                                 num_experts: tl.constexpr,
                                 N: tl.constexpr, 
                                 hs: tl.constexpr,
                                 SCALE: tl.constexpr, 
                                 PROB: tl.constexpr):
    pid = tl.program_id(axis=0)
    x = tl.load(data_ptr + pid * N + tl.arange(0, N))
    if SCALE == 1:
        scale = tl.load(scale_ptr + pid)
    elif SCALE == 2:
        scale = tl.load(scale_ptr + pid * hs + tl.arange(0, hs))

    indices = tl.load(
        mask_map_ptr + pid * num_experts + tl.arange(0, num_experts))
    count = tl.sum(tl.where(indices >= 0, 1, 0))
    mask_indices = tl.where(indices < 0, 2 ** 20, indices)
    idx = tl.argmin(mask_indices, 0)
    index = tl.min(mask_indices)
    for i in range(count):

        tl.store(output_data_ptr + index * N + tl.arange(0, N), x)

        if SCALE == 1:
            tl.store(output_scale_ptr + index, scale)
        elif SCALE == 2:
            tl.store(output_scale_ptr + index * hs + tl.arange(0, hs), scale)

        if PROB:
            prob = tl.load(probs_ptr + pid * num_experts + idx)
            tl.store(output_probs_ptr + index, prob)

        mask_indices = tl.where(indices <= index, 2 ** 20, indices)
        idx = tl.argmin(mask_indices, 0)
        index = tl.min(mask_indices)



@triton.jit
def fill_padded_token_with_zero_kernel(data_ptr, scale_ptr, probs_ptr,
                                 max_indices_ptr, 
                                 token_per_expert_ptr,
                                 N: tl.constexpr, 
                                 hs: tl.constexpr,
                                 SCALE: tl.constexpr, 
                                 PROB: tl.constexpr):
    pid = tl.program_id(axis=0)
    x = tl.zeros((N,), dtype=data_ptr.dtype.element_ty)
    si = tl.load(max_indices_ptr + pid)
    count = tl.load(token_per_expert_ptr + pid)
    c = tl.cdiv(count, 16) * 16 - count

    for i in range(si+1, si +1 + c):
        tl.store(data_ptr + i * N + tl.arange(0, N), x)

        if SCALE == 1:
            tl.store(scale_ptr + i, 1e-30)
        elif SCALE == 2:
            tl.store(scale_ptr + i * hs + tl.arange(0, hs), 1e-30)

        if PROB:
            tl.store(probs_ptr + i, 0.0)

"""
gather with mask map
inp: [num_tokens, hidden_size], rowwise_data
scale: [num_tokens, scale_size], rowwise_scale_inv
prob: [num_tokens], router prob
row_id_map: [n_experts, num_tokens]
    index >= 0: row index of output tensor 
    index == -1: ignore
    Note: index may not be contiguous
num_out_tokens: output token count, including padding tokens
contiguous: whether indices in row_id_map is contiguous
    False means padded
token_per_expert: [num_experts], token count per expert, non-blocking cuda tensor
"""
def triton_permute_with_mask_map(
        inp: torch.Tensor,
        scale: torch.Tensor,
        probs: torch.Tensor,
        row_id_map: torch.Tensor,
        num_out_tokens: int, 
        contiguous: bool = True,
        tokens_per_expert: Optional[torch.Tensor] = None
):
    num_tokens, hidden_size = inp.shape
    num_tokens_, num_experts = row_id_map.shape  # not transposed
    assert num_tokens == num_tokens_
    SCALE = 0 # NO SCALE
    hs = 0
    if scale is not None:
        SCALE = 2 if scale.ndim > 1 else 1
        hs = scale.shape[1] if SCALE == 2 else 1

    # use zeros to initialize if row_id_map is padded and token_per_expert is empty
    ZERO = not contiguous and tokens_per_expert is None

    if ZERO:
        output = torch.zeros((num_out_tokens, hidden_size), dtype=inp.dtype,
                            device="cuda")
    else:        
        output = torch.empty((num_out_tokens, hidden_size), dtype=inp.dtype,
                            device="cuda")
    
    if SCALE > 0:
        shape = (num_out_tokens, hs) if SCALE == 2 else (num_out_tokens, )
        if ZERO:
            permuted_scale = torch.zeros(
                    shape, 
                    dtype=scale.dtype, device="cuda"
                )
        else:
            permuted_scale = torch.empty(
                    shape, 
                    dtype=scale.dtype, device="cuda"
                )
    else:
        permuted_scale = None 

    PROB = probs is not None 
    if PROB:
        if ZERO:
            permuted_probs = torch.zeros(
                (num_out_tokens,), dtype=probs.dtype, device="cuda"
            )
        else:
            permuted_probs = torch.empty(
                (num_out_tokens,), dtype=probs.dtype, device="cuda"
            ) 
    else:
        permuted_probs = None 

    if num_tokens == 0:
        return output, permuted_scale, permuted_probs

    grid = (num_tokens,)
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
        SCALE,
        PROB,
        num_stages=3,
        num_warps=8
    )

    if not contiguous and tokens_per_expert is not None:
        max_indices = row_id_map.amax(0)
        fill_padded_token_with_zero_kernel[(num_experts,)](output, permuted_scale, permuted_probs,
                                 max_indices, 
                                 tokens_per_expert,
                                 hidden_size, 
                                 hs,
                                 SCALE, 
                                 PROB)

    return output, permuted_scale, permuted_probs




@triton.jit
def smooth_permute_with_indices_kernel(x_ptr, q_ptr, ss_ptr, qs_ptr, count_ptr,
                                       accum_ptr, index_ptr, M, N: tl.constexpr,
                                       REVERSE: tl.constexpr,
                                       ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    smooth_scale = tl.load(ss_ptr + pid * N + tl.arange(0, N))
    if not REVERSE:
        smooth_scale = 1.0 / smooth_scale
    count = tl.load(count_ptr + pid)
    ei = tl.load(accum_ptr + pid)
    si = ei - count
    for i in range(count):
        index = tl.load(index_ptr + si + i)
        x = tl.load(x_ptr + index * N + tl.arange(0, N)).to(tl.float32)
        x *= smooth_scale
        x_max = tl.max(tl.abs(x))
        scale = tl.maximum(x_max / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))

        tl.store(qs_ptr + si + i, scale)

        s = 1.0 / scale
        x *= s
        xq = x.to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + si * N + i * N + tl.arange(0, N), xq)


"""
select and smooth and quant
x: [bs, dim]
smooth_scales: [n_experts, dim]
indices: [n_experts*topk]
x_q: [bs*topk, dim]
x_scale: [bs*topk]
"""
def triton_smooth_permute_with_indices(x, smooth_scales, token_count_per_expert,
                                       indices, x_q=None, x_scale=None,
                                       reverse=False, round_scale=False):
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
    grid = (n_expert,)
    smooth_permute_with_indices_kernel[grid](
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
    return x_q, x_scale





@triton.jit
def batch_smooth_rescale_with_indices_kernel(x_ptr, scale_ptr, oss_ptr, ss_ptr, index_ptr, count_ptr,
                                       accum_ptr, q_ptr, qs_ptr,  
                                       N: tl.constexpr,
                                       E: tl.constexpr,
                                       H: tl.constexpr,
                                       W: tl.constexpr,
                                       ROUND: tl.constexpr):
    eid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)

    count = tl.load(count_ptr + eid)
    counts = tl.load(count_ptr + tl.arange(0, E))
    si = tl.load(accum_ptr + eid) - count

    pad = tl.cdiv(count, 32) * 32
    loop = tl.cdiv(pad, H)
    bias = tl.sum(tl.where(tl.arange(0, E) < eid, tl.cdiv(counts, 32), 0)) * 32 * N

    # col-wise read, row-wise write
    org_smooth_scale = tl.load(oss_ptr + cid * W + tl.arange(0, W))
    x_max = tl.zeros((H, W), dtype=tl.float32)
    for i in range(loop):
        idx = i * H + tl.arange(0, H)
        indices = tl.load(index_ptr + si + i * H + tl.arange(0, H), mask=idx<count)
        x = tl.load(x_ptr + cid * W + indices[:,None] * N + tl.arange(0, W)[None,:], mask=idx[:,None]<count).to(tl.float32)
        s = tl.load(scale_ptr + indices, mask=idx<count)[:, None]
        smooth_scale = tl.load(ss_ptr + si + i * H + tl.arange(0, H), mask=idx<count)[:, None]
        x = x * org_smooth_scale * (s * smooth_scale)
        x_max = tl.maximum(tl.abs(x), x_max)

    scale = tl.maximum(tl.max(x_max, 0) / 448.0, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    tl.store(qs_ptr + eid * N + cid * W + tl.arange(0, W), scale)

    scale = 1.0 / scale
    toffs = bias + cid * pad * W + tl.arange(0, W)[:, None] * pad + tl.arange(0, H)
    for i in range(loop):
        idx = i * H + tl.arange(0, H)
        indices = tl.load(index_ptr + si + i * H + tl.arange(0, H), mask=idx<count)
        x = tl.load(x_ptr + cid * W + indices[:,None] * N + tl.arange(0, W)[None,:], mask=idx[:,None]<count).to(tl.float32)
        s = tl.load(scale_ptr + indices, mask=idx<count)[:, None]
        smooth_scale = tl.load(ss_ptr + si + i * H + tl.arange(0, H), mask=idx<count)[:, None]
        x = x * (org_smooth_scale * scale) * (s * smooth_scale)
        xq = tl.trans(x.to(q_ptr.dtype.element_ty))
        tl.store(q_ptr + toffs, xq, mask=idx[None, :] < pad)
        toffs += H


"""
used for smooth backward in 0.12
`x` is smooth quantized dy, it should be gather, requantized, padded to multiple of 32 and tranposed
x: [bs, dim]
x: [bs]
org_smooth_scale: [dim]
smooth_scales: [n_experts, dim], reversed
token_count_per_expert: [n_experts], tensor of token count per expert
splits: [n_experts], list of token_count_per_expert
indices: [sum(tokens_per_experts)]
x_q: [sum(roundup(tokens_per_experts)) * dim]
x_scale: [sum(roundup(tokens_per_experts))]
"""
def triton_batch_smooth_rescale_with_indices(x, 
                                       scale, 
                                       org_smooth_scale, 
                                       smooth_scales, 
                                       indices, 
                                       token_count_per_expert, 
                                       splits, 
                                       x_q=None, x_scale=None,
                                       round_scale=False):
    # row-wise read, row-wise write
    M, N = x.shape
    n_expert = len(splits)
    out_tokens = sum([(x+31)//32 for x in splits])*32
    if N >= 4096:
        H = 64
        W = 64
    else:
        H = 128
        W = 32
    device = x.device
    accum_token_count = torch.cumsum(token_count_per_expert, 0)
    if x_q is None:
        # TODO(nanxiao): opt performance
        x_q = torch.empty((out_tokens * N,), device=device, dtype=torch.float8_e4m3fn)
    if x_scale is None:
        x_scale = torch.empty((n_expert, N), device=device, dtype=torch.float32)
    # import pydevd
    # pydevd.settrace(suspend=False, trace_only_current_thread=True)
    grid = (n_expert, N//W)
    batch_smooth_rescale_with_indices_kernel[grid](
        x,
        scale,
        org_smooth_scale,
        smooth_scales,
        indices,
        token_count_per_expert,
        accum_token_count,
        x_q,
        x_scale,
        N,
        n_expert,
        H, 
        W,
        round_scale,
        num_stages=3,
        num_warps=8
    )
    return x_q, x_scale



@triton.jit
def smooth_weighted_unpermute_with_indices_backward_kernel(grads_ptr,
                                                           tokens_ptr, q_ptr,
                                                           ss_ptr, qs_ptr,
                                                           count_ptr, accum_ptr,
                                                           index_ptr, sum_ptr,
                                                           M, N: tl.constexpr,
                                                           REVERSE: tl.constexpr,
                                                           ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    smooth_scale = tl.load(ss_ptr + pid * N + tl.arange(0, N))
    if not REVERSE:
        smooth_scale = 1.0 / smooth_scale
    count = tl.load(count_ptr + pid)
    ei = tl.load(accum_ptr + pid)
    si = ei - count
    for i in range(count):
        index = tl.load(index_ptr + si + i)
        x = tl.load(grads_ptr + index * N + tl.arange(0, N)).to(tl.float32)
        t = tl.load(tokens_ptr + si * N + i * N + tl.arange(0, N)).to(
            tl.float32)
        sums = tl.sum(x * t)
        tl.store(sum_ptr + si + i, sums)

        x *= smooth_scale
        x_max = tl.max(tl.abs(x))
        scale = tl.maximum(x_max / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))

        tl.store(qs_ptr + si + i, scale)

        s = 1.0 / scale
        x *= s
        xq = x.to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + si * N + i * N + tl.arange(0, N), xq)


"""
select and smooth and quant, used in 0.11 all2all moe
x: [bs, dim]
smooth_scales: [n_experts, dim]
indices: [n_experts*topk]
x_q: [bs*topk, dim]
x_scale: [bs*topk]
"""
def triton_smooth_weighted_unpermute_with_indices_backward(grads, tokens,
                                                           smooth_scales,
                                                           token_count_per_expert,
                                                           indices, x_q=None,
                                                           x_scale=None,
                                                           x_sum=None,
                                                           reverse=False,
                                                           round_scale=False):
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
    grid = (n_expert,)
    smooth_weighted_unpermute_with_indices_backward_kernel[grid](
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
    return x_q, x_scale, x_sum


@triton.jit
def smooth_unpermute_with_indices_backward_kernel(grads_data_ptr,
                                                  grads_scale_ptr, q_ptr,
                                                  ss_ptr, qs_ptr, count_ptr,
                                                  accum_ptr, index_ptr,
                                                  N: tl.constexpr,
                                                  hs: tl.constexpr,
                                                  REVERSE: tl.constexpr,
                                                  ROUND: tl.constexpr,
                                                  GROUP: tl.constexpr):
    eid = tl.program_id(axis=0)
    wid = tl.program_id(axis=1)
    T = tl.num_programs(axis=1)

    # row-wise read, row-wise write
    smooth_scale = tl.load(ss_ptr + eid * N + tl.arange(0, N))
    if not REVERSE:
        smooth_scale = 1.0 / smooth_scale
    count = tl.load(count_ptr + eid)
    ei = tl.load(accum_ptr + eid)
    si = ei - count
    c = tl.cdiv(count, T)
    for i in range(si + wid * c, tl.minimum(si + wid * c + c, ei)):
        index = tl.load(index_ptr + i)
        x = tl.load(grads_data_ptr + index * N + tl.arange(0, N)).to(tl.float32)
        if GROUP:
            gs = tl.load(grads_scale_ptr + index * hs + tl.arange(0, hs))
            x = tl.reshape(tl.reshape(x, (hs, N // hs)) * gs[:, None], (N,))
        else:
            gs = tl.load(grads_scale_ptr + index)
            x *= gs

        x *= smooth_scale
        x_max = tl.max(tl.abs(x))

        scale = tl.maximum(x_max / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))

        tl.store(qs_ptr + i, scale)

        s = 1.0 / scale
        x *= s
        xq = x.to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + i * N + tl.arange(0, N), xq)


"""
select and smooth and quant
grad_data: [bs, dim]
grad_scale: [bs, dim/128]
smooth_scales: [n_experts, dim]
indices: [n_experts*topk]
x_q: [bs*topk, dim]
x_scale: [bs*topk]
"""


def triton_smooth_unpermute_with_indices_backward(grad_data, grad_scale,
                                                  smooth_scales,
                                                  token_count_per_expert,
                                                  indices, x_q=None,
                                                  x_scale=None, reverse=False,
                                                  round_scale=False):
    # row-wise read, row-wise write
    M, N = grad_data.shape
    n_expert, n = smooth_scales.shape
    assert 128 % n_expert == 0
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
    W = 128 // n_expert
    grid = (n_expert, W)
    smooth_unpermute_with_indices_backward_kernel[grid](
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
    return x_q, x_scale


@triton.jit
def smooth_permute_with_mask_map_kernel(grads_data_ptr, quant_data_ptr,
                                        mask_map_ptr, grads_scale_ptr,
                                        smooth_scale_ptr, quant_scale_ptr, 
                                        M, T,
                                        N: tl.constexpr, 
                                        hs: tl.constexpr,
                                        REVERSE: tl.constexpr,
                                        ROUND: tl.constexpr,
                                        GROUP: tl.constexpr):
    eid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    n_experts = tl.num_programs(axis=0)

    # smooth_scale_ptr = tl.load(smooth_scale_ptrs + eid).to(tl.pointer_type(tl.float32))
    smooth_scale = tl.load(smooth_scale_ptr + eid * N + tl.arange(0, N))
    if not REVERSE:
        smooth_scale = 1.0 / smooth_scale
    for i in range(bid * T, tl.minimum(bid * T + T, M)):
        index = tl.load(mask_map_ptr + i * n_experts + eid)
        mask = index >= 0
        if index >= 0:
            x = tl.load(grads_data_ptr + i * N + tl.arange(0, N), mask=mask).to(
                tl.float32)

            if GROUP:
                gs = tl.load(grads_scale_ptr + i * hs + tl.arange(0, hs),
                             mask=mask)
                x = tl.reshape(tl.reshape(x, (hs, N // hs)) * gs[:, None], (N,))
            else:
                gs = tl.load(grads_scale_ptr + i, mask=mask)
                x *= gs

            x *= smooth_scale
            x_max = tl.max(tl.abs(x))

            scale = tl.maximum(x_max / 448.0, 1e-30)
            if ROUND:
                scale = tl.exp2(tl.ceil(tl.log2(scale)))

            tl.store(quant_scale_ptr + index, scale, mask=mask)

            x /= scale
            xq = x.to(quant_data_ptr.dtype.element_ty)
            tl.store(quant_data_ptr + index * N + tl.arange(0, N), xq,
                     mask=mask)


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
    assert row_id_map.shape[1] == num_experts
    output = torch.empty((num_out_tokens, hidden_size), dtype=inp.dtype,
                         device=inp.device)

    group = scale.ndim == 2
    hs = scale.shape[1] if group else 1

    permuted_scale = torch.empty(
        (num_out_tokens,), dtype=scale.dtype, device=inp.device
    )
    # print(f'{inp.shape=} {row_id_map.shape=} {num_tokens=} {num_out_tokens=}')
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





@triton.jit
def deprecated_smooth_permute_with_mask_map_kernel(grads_data_ptr, quant_data_ptr,
                                        mask_map_ptr,
                                        smooth_scale_ptr, quant_scale_ptr, M, T,
                                        N: tl.constexpr,
                                        REVERSE: tl.constexpr,
                                        ROUND: tl.constexpr):
    eid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    n_experts = tl.num_programs(axis=0)

    # smooth_scale_ptr = tl.load(smooth_scale_ptrs + eid).to(tl.pointer_type(tl.float32))
    smooth_scale = tl.load(smooth_scale_ptr + eid * N + tl.arange(0, N))
    if not REVERSE:
        smooth_scale = 1.0 / smooth_scale
    for i in range(bid * T, tl.minimum(bid * T + T, M)):
        index = tl.load(mask_map_ptr + i * n_experts + eid)
        mask = index >= 0
        if index >= 0:
            x = tl.load(grads_data_ptr + i * N + tl.arange(0, N), mask=mask).to(
                tl.float32)

            x *= smooth_scale
            x_max = tl.max(tl.abs(x))

            scale = tl.maximum(x_max / 448.0, 1e-30)
            if ROUND:
                scale = tl.exp2(tl.ceil(tl.log2(scale)))

            tl.store(quant_scale_ptr + index, scale, mask=mask)

            x /= scale
            xq = x.to(quant_data_ptr.dtype.element_ty)
            tl.store(quant_data_ptr + index * N + tl.arange(0, N), xq,
                     mask=mask)


# """
# gather and smooth quant
# inp: [num_tokens, hidden_size], rowwise_data
# row_id_map: [n_experts, num_tokens], indices
# num_tokens: [n_experts]
# smooth_scale_ptrs: [n_experts, hidden_size]
# """
def triton_deprecated_smooth_permute_with_mask_map(
        inp: torch.Tensor,
        row_id_map: torch.Tensor,
        num_tokens: int,
        num_experts: int,
        num_out_tokens: int,
        hidden_size: int,
        smooth_scales: torch.Tensor,
        reverse=True,
        round_scale=False
):
    assert row_id_map.shape[1] == num_experts
    output = torch.empty((num_out_tokens, hidden_size), dtype=inp.dtype,
                         device=inp.device)
    permuted_scale = torch.empty(
        (num_out_tokens,), dtype=torch.float32, device=inp.device
    )
    sm = torch.cuda.get_device_properties(inp.device).multi_processor_count
    T = triton.cdiv(num_tokens, sm)
    grid = (num_experts, sm)
    deprecated_smooth_permute_with_mask_map_kernel[grid](
        inp,
        output,
        row_id_map,
        smooth_scales,
        permuted_scale,
        num_tokens,
        T,
        hidden_size,
        reverse,
        round_scale,
    )
    return output, permuted_scale
