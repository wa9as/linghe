import torch

from flops.utils.gather import (triton_make_row_id_map,
                                triton_make_row_id_map_and_indices,
                                triton_index_select,
                                triton_permute_with_mask_map)
from flops.tools.util import (output_check,
                              torch_make_indices)
from flops.tools.benchmark import benchmark_func


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
                                                               tokens_per_expert=token_count_per_expert)
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


if __name__ == '__main__':
    test_make_id_map(M=4098, n_experts=32, topk=2, bias=0.0, bench=False)
    test_triton_permute_with_mask_map(M=16384, N=2048, n_experts=32, topk=8, bench=False)
    test_triton_permute_with_mask_map(M=8192, N=4096, n_experts=32, topk=8, bench=False)
    test_triton_permute_with_mask_map(M=7628, N=2048, n_experts=32, topk=8, bench=False)