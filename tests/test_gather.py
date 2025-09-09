import torch

from flops.utils.gather import (triton_make_row_id_map,
                                triton_make_row_id_map_and_indices,
                                triton_index_select,
                                triton_permute_with_mask_map,
                                triton_smooth_permute_with_indices,
                                triton_smooth_permute_with_mask_map,
                                triton_smooth_unpermute_with_indices_backward,
                                triton_smooth_weighted_unpermute_with_indices_backward,
                                triton_batch_smooth_rescale_with_indices)
from flops.tools.util import (output_check,
                              torch_batch_smooth_quant,
                              torch_make_indices, 
                              torch_smooth_quant)
from flops.utils.benchmark import benchmark_func


def torch_index_select(y, indices):
    output = y.index_select(0, indices)
    return output

def torch_select_with_padded_map_mask(y, mask_map, out_tokens):
    E = mask_map.shape[1]
    if y.ndim > 1:
        output = torch.zeros((out_tokens, y.shape[1]), dtype=y.dtype, device=y.device)
    else:
        output = torch.zeros((out_tokens, ), dtype=y.dtype, device=y.device)
    for i in range(E):
        indices = mask_map[:,i]
        src_idx = torch.nonzero(indices>-1)
        dst_idx = indices[src_idx]
        output[dst_idx] = y[src_idx]
    return output

def torch_ravel_with_padded_map_mask(y, mask_map, out_tokens):
    E = mask_map.shape[1]
    output = torch.zeros((out_tokens, ), dtype=y.dtype, device=y.device)
    for i in range(E):
        indices = mask_map[:,i]
        src_idx = torch.nonzero(indices>-1)
        dst_idx = indices[src_idx]
        output[dst_idx] = y[src_idx,i]
    return output

def torch_fp16_index_select(x, scales, indices):
    return x.index_select(0, indices), scales.index_select(0, indices)


def torch_scatter(logits, routing_map, weights):
    logits[routing_map] = weights


# dequant and smooth and quant
def torch_smooth_permute_with_indices(grad_data, grad_scale, indices,
                                      smooth_scales,
                                      token_count_per_expert_list,
                                      round_scale=True):
    M, N = grad_data.shape
    B = grad_data.shape[1] // (
        1 if grad_scale.ndim == 1 else grad_scale.shape[1])
    q_refs = []
    scale_refs = []
    s = 0
    for i, c in enumerate(token_count_per_expert_list):
        c = token_count_per_expert_list[i]
        data_slice = grad_data.view(torch.uint8)[indices[s:s + c]].view(
            torch.float8_e4m3fn)
        scale_slice = grad_scale[indices[s:s + c]]
        s += c
        y_smooth = (data_slice.float().view(c, N // B, B) * scale_slice[:, :,
                                                            None]).view(c, N) / \
                   smooth_scales[i]
        scale = y_smooth.abs().amax(1) / 448
        if round_scale:
            scale = torch.exp2(torch.ceil(torch.log2(scale)))
        scale_refs.append(scale)
        q = (y_smooth / scale[:, None]).to(torch.float8_e4m3fn)
        q_refs.append(q.view(torch.uint8))
    q_ref = torch.cat(q_refs, 0).view(torch.float8_e4m3fn)
    scale_ref = torch.cat(scale_refs, 0)

    return q_ref, scale_ref



# desmooth,dequant, gather, pad, transpose, smooth, quant
def torch_batch_smooth_rescale_with_indices(x_q, x_scale, org_smooth_scale, smooth_scales,
                                      indices,
                                      token_count_per_expert_list,
                                      round_scale=True):
    M, N = x_q.shape
    q_refs = []
    scale_refs = []
    s = 0
    for i, c in enumerate(token_count_per_expert_list):
        c = token_count_per_expert_list[i]
        if c == 0:
            y_scale = torch.zeros_like(org_smooth_scale)
            scale_refs.append(y_scale.view(-1))
            continue
        N = (c + 31)//32 * 32
        data_slice = x_q[indices[s:s + c]]
        scale_slice = x_scale[indices[s:s + c]]
        y = data_slice.float() * scale_slice[:, None] * org_smooth_scale
        smooth_scale = smooth_scales[s:s+c]
        if N > c:
            y = torch.nn.functional.pad(y, (0,0,0, N-c))
            smooth_scale = torch.nn.functional.pad(smooth_scale, (0, N-c))
        y_q, y_scale, y_max= torch_smooth_quant(y.t().contiguous(), smooth_scale, reverse=True, round_scale=round_scale)
        scale_refs.append(y_scale.view(-1))
        q_refs.append(y_q.view(-1))
        s += c
    q_ref = torch.cat(q_refs, 0)
    scale_ref = torch.stack(scale_refs, 0)
    return q_ref, scale_ref


def test_make_id_map(M=4098, n_experts=32, topk=2, bias=0.0, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    logits = torch.randn((M, n_experts), dtype=torch.float32, device=device)
    probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(
        logits, topk=topk, bias=bias)

    token_count_per_expert_list = token_count_per_expert.tolist()
    out_tokens = sum(token_count_per_expert_list)


    row_id_map_output = triton_make_row_id_map(mask_map)
    assert (row_id_map - row_id_map_output).abs().sum().item() == 0

    _, row_id_indices = triton_make_row_id_map_and_indices(mask_map, out_tokens)
    assert (row_id_indices - indices).abs().sum().item() == 0



def test_triton_smooth_permute_with_indices(M=4096, N=4096, n_experts=256,
                                            topk=8, bench=False):
    device = 'cuda:0'
    y = torch.randn((M, N), dtype=torch.bfloat16, device=device)
    logits = torch.randn((M, n_experts), dtype=torch.float32, device=device)
    smooth_scales = 1 + 10 * torch.rand((n_experts, N), device=device,
                                        dtype=torch.float32)

    probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(
        logits, topk=topk, bias=0.0)

    y_q, y_scale = triton_smooth_permute_with_indices(y, smooth_scales,
                                                      token_count_per_expert,
                                                      indices, reverse=False,
                                                      round_scale=False)

    y_q_ref, y_scale_ref = torch_batch_smooth_quant(y, smooth_scales, indices,
                                                    token_count_per_expert,
                                                    reverse=False,
                                                    round_scale=False)

    output_check(y_q_ref.float(), y_q.float(), 'data')
    output_check(y_scale_ref.float(), y_scale.float(), 'scale')

    if bench:
        n_repeat = 100
        benchmark_func(torch_index_select, y, indices, n_repeat=n_repeat)
        benchmark_func(triton_smooth_permute_with_indices, y, smooth_scales,
                       token_count_per_expert, indices, reverse=False,
                       round_scale=False, n_repeat=n_repeat)


def test_triton_smooth_weighted_unpermute_with_indices_backward(M=4096, N=4096,
                                                                n_experts=256,
                                                                topk=8,
                                                                round_scale=True,
                                                                bench=False):
    device = 'cuda:0'
    reverse = True
    y = torch.randn((M, N), dtype=torch.bfloat16, device=device)
    logits = torch.randn((M, n_experts), dtype=torch.float32, device=device)
    smooth_scales = 1 + 10 * torch.rand((n_experts, N), device=device,
                                        dtype=torch.float32)
    probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(
        logits, topk=topk, bias=0.0)

    tokens = torch.randn((indices.shape[0], N), dtype=torch.bfloat16,
                         device=device)
    y_q, y_scale, y_sum = triton_smooth_weighted_unpermute_with_indices_backward(
        y, tokens, smooth_scales, token_count_per_expert, indices, x_q=None,
        x_scale=None, reverse=reverse, round_scale=round_scale)

    y_q_ref, y_scale_ref = torch_batch_smooth_quant(y, smooth_scales, indices,
                                                    token_count_per_expert,
                                                    reverse=reverse,
                                                    round_scale=round_scale)
    sum_ref = (tokens * y[indices]).sum(1)

    output_check(y_q_ref.float(), y_q.float(), 'data')
    output_check(y_scale_ref.float(), y_scale.float(), 'scale')
    output_check(sum_ref.float(), y_sum.float(), 'sum')

    if bench:
        n_repeat = 100
        benchmark_func(triton_smooth_weighted_unpermute_with_indices_backward,
                       y, tokens, smooth_scales, token_count_per_expert,
                       indices, reverse=reverse, round_scale=round_scale,
                       n_repeat=n_repeat)


def test_triton_permute_with_mask_map(M=4096, N=4096, n_experts=256, topk=8,
                                      bench=False):
    device = 'cuda:0'
    dtype = torch.bfloat16
    x = torch.randn(M, N, dtype=dtype, device=device) ** 3
    x_q = x.to(torch.float8_e4m3fn)
    scales = torch.rand(M, dtype=dtype, device=device) * 10

    logits = torch.randn((M, n_experts), dtype=torch.float32, device=device)

    probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(
        logits, topk=topk, bias=-0.01)
    out_tokens = sum(token_count_per_expert.tolist())

    x_out, scale_out = triton_index_select(x, indices, scale=scales)
    x_out_ref, scale_out_ref = torch_fp16_index_select(x, scales, indices)
    output_check(x_out_ref, x_out, 'x_out')
    output_check(scale_out_ref, scale_out, 'scale_out')

    probs_out_ref = probs.T.contiguous().masked_select(
        (probs > 0).T.contiguous())
    x_out, scale_out, probs_out = triton_permute_with_mask_map(x, scales, probs,
                                                               row_id_map,
                                                               out_tokens,
                                                               contiguous=True)
    output_check(x_out_ref, x_out, 'x_out')
    output_check(scale_out_ref, scale_out, 'scale_out')
    output_check(probs_out_ref, probs_out, 'prob_out')

    nzs = torch.sum(row_id_map>=0, 0)
    bias = torch.cumsum((nzs + 15)//16*16 - nzs, 0)
    row_id_map_clone = row_id_map.clone().detach()
    row_id_map_clone[:, 1:] += bias[:-1]
    round_row_id_map = torch.where(row_id_map>=0, row_id_map_clone, -1)
    padded_out_tokens = sum([(x+15)//16*16 for x in token_count_per_expert.tolist()])
    x_out_ref = torch_select_with_padded_map_mask(x, round_row_id_map, padded_out_tokens)
    scale_out_ref = torch_select_with_padded_map_mask(scales, round_row_id_map, padded_out_tokens)
    prob_out_ref = torch_ravel_with_padded_map_mask(probs, round_row_id_map, padded_out_tokens)
    x_out, scale_out, probs_out = triton_permute_with_mask_map(x, scales, probs,
                                                               round_row_id_map,
                                                               padded_out_tokens,
                                                               contiguous=False, 
                                                               token_per_expert=token_count_per_expert)
    output_check(x_out_ref, x_out, 'noncontiguous.x_out')
    output_check(scale_out_ref, scale_out, 'noncontiguous.scale_out')
    output_check(prob_out_ref, probs_out, 'noncontiguous.prob')

    if bench:
        n_repeat = 100
        ref_bytes = out_tokens * N * 2
        ref_time = benchmark_func(torch_fp16_index_select, x, scales, indices,
                                  n_repeat=n_repeat, ref_bytes=ref_bytes)
        benchmark_func(triton_index_select, x, indices, scale=scales,
                       n_repeat=n_repeat, ref_time=ref_time, ref_bytes=ref_bytes)
        benchmark_func(triton_permute_with_mask_map, x, scales, probs,
                       row_id_map, out_tokens, contiguous=True, n_repeat=n_repeat,
                       ref_time=ref_time, ref_bytes=ref_bytes)
        benchmark_func(triton_permute_with_mask_map, x, scales, probs,
                       row_id_map, out_tokens, contiguous=False, 
                       token_per_expert=token_count_per_expert, 
                       n_repeat=n_repeat,
                       ref_time=ref_time, ref_bytes=ref_bytes)


def test_triton_smooth_permute_with_mask_map(M=4096, N=4096, n_experts=32,
                                             topk=8, round_scale=True,
                                             bench=False):
    device = 'cuda:0'
    dtype = torch.bfloat16
    smooth_scales = 1 + 10 * torch.rand((n_experts, N), device=device,
                                        dtype=torch.float32)
    logits = torch.randn((M, n_experts), dtype=torch.float32,
                         device=device) ** 3
    probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(
        logits, topk=topk, bias=-0.01)

    token_count_per_expert_list = token_count_per_expert.tolist()
    out_tokens = sum(token_count_per_expert_list)

    B = 128
    grad_data = torch.randn((M, N), dtype=torch.bfloat16, device=device).to(
        torch.float8_e4m3fn)
    grad_scale = 1 + torch.rand((M, N // B), dtype=torch.float32, device=device)
    y_q, y_scale = triton_smooth_unpermute_with_indices_backward(grad_data,
                                                                 grad_scale,
                                                                 smooth_scales,
                                                                 token_count_per_expert,
                                                                 indices,
                                                                 x_q=None,
                                                                 x_scale=None,
                                                                 reverse=False,
                                                                 round_scale=round_scale)

    q_ref, scale_ref = torch_smooth_permute_with_indices(grad_data, grad_scale,
                                                         indices, smooth_scales,
                                                         token_count_per_expert_list,
                                                         round_scale=round_scale)

    output_check(q_ref.float(), y_q.float(), 'data')
    output_check(scale_ref.float(), y_scale.float(), 'scale')

    # smooth_scale_ptrs = torch.tensor([x.data_ptr() for x in torch.split(smooth_scales,1)], device=device)
    permuted_data, permuted_scale = triton_smooth_permute_with_mask_map(
        grad_data, row_id_map, grad_scale, M, n_experts, out_tokens, N,
        smooth_scales, reverse=False, round_scale=round_scale)
    output_check(q_ref.float(), permuted_data.float(), 'data')
    output_check(scale_ref.float(), permuted_scale.float(), 'scale')

    if bench:
        benchmark_func(triton_smooth_unpermute_with_indices_backward, grad_data,
                       grad_scale, smooth_scales, token_count_per_expert,
                       indices, round_scale=round_scale, n_repeat=100,
                       ref_bytes=out_tokens * N * 2)
        benchmark_func(triton_smooth_permute_with_mask_map, grad_data,
                       row_id_map, grad_scale, M, n_experts, out_tokens, N,
                       smooth_scales, reverse=False, round_scale=round_scale,
                       n_repeat=100, ref_bytes=out_tokens * N * 2)





def test_triton_batch_smooth_rescale_with_indices(M=1024, N=2048, n_experts=32, topk=8, bench=False):

    device = 'cuda:0'
    if True:
        logits = torch.randn((M, n_experts), dtype=torch.float32,
                            device=device) ** 3
        logits[:,0] -= 1000
        logits[:,2] -= 100
        probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(
            logits, topk=topk, bias=-0.01)

        token_count_per_expert_list = token_count_per_expert.tolist()
        out_tokens = sum(token_count_per_expert_list)

        x = torch.randn((M, N), dtype=torch.bfloat16, device=device).to(
            torch.float8_e4m3fn)
        scale = torch.rand((M,), dtype=torch.float32, device=device) + 0.1
        org_smooth_scale = torch.rand((N,), dtype=torch.float32, device=device) + 0.1
        smooth_scales = torch.rand((out_tokens, ), dtype=torch.float32, device=device) + 0.1
    else:
        #         torch.save({"x":x, "scale":scale, "org_smooth_scale":org_smooth_scale,"smooth_scales":smooth_scales, "indices":indices, "token_count_per_expert":token_count_per_expert,"splits":splits}, '/tmp/debug.bin')
        state = torch.load('/tmp/debug.bin')
        x = state['x']
        scale = state['scale']
        org_smooth_scale = state['org_smooth_scale']
        smooth_scales = state['smooth_scales']
        indices = state['indices']
        token_count_per_expert = state['token_count_per_expert']
        token_count_per_expert_list = state['splits']
        out_tokens = sum(token_count_per_expert_list)


    x_q_ref, x_scale_ref = torch_batch_smooth_rescale_with_indices(x, scale, org_smooth_scale, smooth_scales,
                                      indices,
                                      token_count_per_expert_list,
                                      round_scale=True)

    x_q, x_scale = triton_batch_smooth_rescale_with_indices(x, scale, org_smooth_scale, smooth_scales, 
                                       indices, 
                                       token_count_per_expert, token_count_per_expert_list, 
                                       round_scale=True)
    output_check(x_q_ref.float(), x_q.float(), 'data')
    output_check(x_scale_ref.float(), x_scale.float(), 'scale')

    if bench:
        benchmark_func(torch_batch_smooth_rescale_with_indices, x, scale, org_smooth_scale, smooth_scales,
                                      indices,
                                      token_count_per_expert_list,
                                      round_scale=True, 
                       ref_bytes=out_tokens * N * 2)
        benchmark_func(triton_batch_smooth_rescale_with_indices, x, scale, org_smooth_scale, smooth_scales, 
                                       indices, 
                                       token_count_per_expert, token_count_per_expert_list, 
                                       round_scale=True,
                                       ref_bytes=out_tokens * N * 2)



if __name__ == '__main__':
    test_make_id_map(M=4098, n_experts=32, topk=2, bias=0.0, bench=False)
    test_triton_smooth_permute_with_indices(M=4096, N=4096, n_experts=32,
                                            topk=8)
    test_triton_smooth_weighted_unpermute_with_indices_backward(M=4096, N=4096,
                                                                n_experts=32,
                                                                topk=8)
    test_triton_permute_with_mask_map(M=16384, N=2048, n_experts=32, topk=8, bench=False)
    test_triton_permute_with_mask_map(M=8192, N=4096, n_experts=32, topk=8, bench=False)
    test_triton_permute_with_mask_map(M=7628, N=2048, n_experts=32, topk=8, bench=False)

    test_triton_smooth_permute_with_mask_map(M=4096, N=4096, n_experts=32,
                                             topk=8)
    test_triton_smooth_permute_with_mask_map(M=7628, N=2048, n_experts=32,
                                             topk=8)

    test_triton_batch_smooth_rescale_with_indices(M=16384, N=2048, n_experts=32, topk=2, bench=False)
    test_triton_batch_smooth_rescale_with_indices(M=8192, N=4096, n_experts=32, topk=2, bench=False)
