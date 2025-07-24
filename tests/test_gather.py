

import torch

import time
import os
import random
from flops.utils.util import *
from flops.utils.gather import *
from flops.quant.smooth.reused_smooth import *
from flops.utils.benchmark import benchmark_func

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def torch_index_select(y, indices):
    output = y.index_select(0, indices)
    return output

def torch_fp16_index_select(x, scales, indices):
    return x.index_select(0, indices), scales.index_select(0, indices)

def torch_scatter(logits,routing_map,weights):
    logits[routing_map] = weights

# dequant and smooth and quant
def torch_smooth_permute_with_indices(grad_data, grad_scale, indices, smooth_scales, token_count_per_expert_list, round_scale=True):
    M, N = grad_data.shape 
    B = grad_data.shape[1]//(1 if grad_scale.ndim==1 else grad_scale.shape[1]) 
    q_refs = [] 
    scale_refs = []
    s = 0
    for i, c in enumerate(token_count_per_expert_list):
        c = token_count_per_expert_list[i]
        data_slice = grad_data.view(torch.uint8)[indices[s:s+c]].view(torch.float8_e4m3fn)
        scale_slice = grad_scale[indices[s:s+c]]
        s += c
        y_smooth = (data_slice.float().view(c,N//B,B)*scale_slice[:,:,None]).view(c,N)/smooth_scales[i]
        scale = y_smooth.abs().amax(1)/448
        if round_scale:
            scale = torch.exp2(torch.ceil(torch.log2(scale)))
        scale_refs.append(scale)
        q = (y_smooth/scale[:,None]).to(torch.float8_e4m3fn) 
        q_refs.append(q.view(torch.uint8))
    q_ref = torch.cat(q_refs, 0).view(torch.float8_e4m3fn)
    scale_ref = torch.cat(scale_refs,0)

    return q_ref, scale_ref


def torch_index_select_and_vec_dot(y, indices, tokens):
    output = y.index_select(0, indices)
    sums = (output*tokens).sum(1)
    return output, sums





def test_triton_smooth_permute_with_indices(M=4096, N=4096, n_experts=256, topk=8, bench=False):
    device='cuda:0'
    y = torch.randn((M, N), dtype=torch.bfloat16, device=device) 
    logits = torch.randn((M, n_experts), dtype=torch.float32, device=device) 
    smooth_scales = 1+10*torch.rand((n_experts,N),device=device,dtype=torch.float32)

    probs, mask_map,token_count_per_expert, indices, row_id_map = torch_make_indices(logits, topk=topk, bias=0.0)

    y_q,y_scale = triton_smooth_permute_with_indices(y, smooth_scales, token_count_per_expert, indices, reverse=False, round_scale=False)

    y_q_ref, y_scale_ref = torch_batch_smooth_quant(y, smooth_scales, indices, token_count_per_expert, reverse=False, round_scale=False)

    output_check(y_q_ref.float(), y_q.float(), 'data')
    output_check(y_scale_ref.float(), y_scale.float(), 'scale')

    if bench:
        n_repeat = 100
        benchmark_func(torch_index_select, y, indices, n_repeat=n_repeat)
        benchmark_func(triton_smooth_permute_with_indices, y, smooth_scales, token_count_per_expert, indices, reverse=False, round_scale=False, n_repeat=n_repeat)



def test_triton_smooth_weighted_unpermute_with_indices_backward(M=4096, N=4096, n_experts=256, topk=8, round_scale=True, bench=False):
    device='cuda:0'
    reverse = True
    y = torch.randn((M, N), dtype=torch.bfloat16, device=device) 
    logits = torch.randn((M, n_experts), dtype=torch.float32, device=device) 
    smooth_scales = 1+10*torch.rand((n_experts,N),device=device,dtype=torch.float32)
    probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(logits, topk=topk, bias=0.0)

    tokens = torch.randn((indices.shape[0],N), dtype=torch.bfloat16, device=device)
    y_q,y_scale,y_sum = triton_smooth_weighted_unpermute_with_indices_backward(y, tokens, smooth_scales, token_count_per_expert, indices, x_q=None, x_scale=None, reverse=reverse, round_scale=round_scale)

    y_q_ref, y_scale_ref = torch_batch_smooth_quant(y, smooth_scales, indices, token_count_per_expert, reverse=reverse, round_scale=round_scale)
    sum_ref = (tokens*y[indices]).sum(1)

    output_check(y_q_ref.float(), y_q.float(), 'data')
    output_check(y_scale_ref.float(), y_scale.float(), 'scale')
    output_check(sum_ref.float(), y_sum.float(), 'sum')

    if bench:
        n_repeat = 100
        benchmark_func(torch_index_select_and_vec_dot, y, indices, tokens, n_repeat=n_repeat)
        benchmark_func(triton_smooth_weighted_unpermute_with_indices_backward, y, tokens, smooth_scales, token_count_per_expert, indices, reverse=reverse, round_scale=round_scale, n_repeat=n_repeat)



def test_triton_permute_with_mask_map(M=4096, N=4096, n_experts=256, topk=8, bench=False):
    device='cuda:0'
    dtype = torch.bfloat16
    x = torch.randn(M, N, dtype=dtype, device=device)**3
    x_q = x.to(torch.float8_e4m3fn)
    scales = torch.rand(M, dtype=dtype, device=device)*10

    logits = torch.randn((M,n_experts), dtype=torch.float32, device=device)
    
    probs, mask_map,token_count_per_expert, indices, row_id_map = torch_make_indices(logits, topk=topk, bias=-0.01)
    out_tokens = sum(token_count_per_expert.tolist())

    x_out, scale_out = triton_index_select(x, indices, scale=scales)
    x_out_ref, scale_out_ref = torch_fp16_index_select(x, scales, indices)
    output_check(x_out_ref,x_out,'x_out')
    output_check(scale_out_ref,scale_out,'scale_out')

    probs_out_ref = probs.T.contiguous().masked_select((probs>0).T.contiguous())
    x_out, scale_out, probs_out = triton_permute_with_mask_map(x,scales,probs,row_id_map,out_tokens)
    output_check(x_out_ref,x_out,'x_out')
    output_check(scale_out_ref,scale_out,'scale_out')
    output_check(probs_out_ref,probs_out,'prob_out')


    if bench:
        n_repeat = 100
        ref_time = benchmark_func(torch_fp16_index_select,x, scales, indices, n_repeat=n_repeat)
        benchmark_func(triton_index_select,x, indices, scale=scales, n_repeat=n_repeat, ref_time=ref_time)
        benchmark_func(triton_permute_with_mask_map,x,scales,probs,row_id_map,out_tokens, n_repeat=n_repeat, ref_time=ref_time)

        

def test_triton_smooth_permute_with_mask_map(M=4096, N=4096, n_experts=32, topk=8, round_scale=True, bench=False):
    device='cuda:0'
    dtype = torch.bfloat16
    smooth_scales = 1+10*torch.rand((n_experts, N),device=device,dtype=torch.float32)
    logits = torch.randn((M,n_experts), dtype=torch.float32, device=device)**3
    probs, mask_map,token_count_per_expert, indices, row_id_map = torch_make_indices(logits, topk=topk, bias=-0.01)

    token_count_per_expert_list = token_count_per_expert.tolist()
    out_tokens = sum(token_count_per_expert_list)

    B = 128
    grad_data = torch.randn((M,N), dtype=torch.bfloat16,device=device).to(torch.float8_e4m3fn)
    grad_scale = 1+torch.rand((M,N//B), dtype=torch.float32,device=device)
    y_q, y_scale = triton_smooth_unpermute_with_indices_backward(grad_data, grad_scale, smooth_scales, token_count_per_expert, indices, x_q=None, x_scale=None, reverse=False, round_scale=round_scale)


    q_ref, scale_ref= torch_smooth_permute_with_indices(grad_data, grad_scale, indices, smooth_scales, token_count_per_expert_list, round_scale=round_scale)

    output_check(q_ref.float(), y_q.float(), 'data')
    output_check(scale_ref.float(), y_scale.float(), 'scale')


    # smooth_scale_ptrs = torch.tensor([x.data_ptr() for x in torch.split(smooth_scales,1)], device=device)
    permuted_data, permuted_scale = triton_smooth_permute_with_mask_map(grad_data,row_id_map,grad_scale,M, n_experts, out_tokens,N,smooth_scales, reverse=False,round_scale=round_scale)
    output_check(q_ref.float(), permuted_data.float(), 'data')
    output_check(scale_ref.float(), permuted_scale.float(), 'scale')

    if bench:
        benchmark_func(triton_smooth_unpermute_with_indices_backward, grad_data, grad_scale, smooth_scales, token_count_per_expert, indices, round_scale=round_scale, n_repeat=100, ref_bytes=out_tokens*N*2)
        benchmark_func(triton_smooth_permute_with_mask_map, grad_data,row_id_map,grad_scale,M,n_experts,out_tokens,N,smooth_scales,reverse=False,round_scale=round_scale, n_repeat=100, ref_bytes=out_tokens*N*2)


if __name__ == '__main__':

    test_triton_smooth_permute_with_indices(M=4096, N=4096, n_experts=32, topk=8)
    test_triton_smooth_weighted_unpermute_with_indices_backward(M=4096, N=4096, n_experts=32, topk=8)

    test_triton_permute_with_mask_map(M=4096, N=4096, n_experts=32, topk=8)
    test_triton_permute_with_mask_map(M=7628, N=2048, n_experts=32, topk=8)

    test_triton_smooth_permute_with_mask_map(M=4096, N=4096, n_experts=32, topk=8)
    test_triton_smooth_permute_with_mask_map(M=7628, N=2048, n_experts=32, topk=8)


    