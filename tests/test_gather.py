

import torch

import time
import os
import random
from flops.utils.util import *
from flops.utils.gather import *
from flops.utils.benchmark import benchmark_func
import transformer_engine.pytorch.triton.permutation as triton_permutation

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def torch_fp16_index_select(x, scales, indices):
    return x.index_select(0, indices), scales.index_select(0, indices)

def torch_scatter(logits,routing_map,weights):
    logits[routing_map] = weights


M, N = 8192*4, 8192

dtype = torch.bfloat16
device = 'cuda:0'
n_experts = 32
topk = 2




if False:
    from flops.quant.smooth.reused_smooth import *
    n_expert = 256
    topk = 8
    smooth_scales = 1+10*torch.rand((n_expert,N),device=device,dtype=torch.float32)
    token_count_per_expert_list = [M*topk//n_expert]*(n_expert-1)
    token_count_per_expert_list.append(M*topk-sum(token_count_per_expert_list))

    token_count_per_expert = torch.tensor(token_count_per_expert_list, dtype=torch.int32, device=device)
    indices = (torch.arange(M*topk, device=device,dtype=torch.int32)//topk)[torch.argsort(torch.randn(M*topk,dtype=torch.float32,device=device))]
    y_q,y_scale = triton_index_select_smooth_quant(y, smooth_scales, token_count_per_expert, indices, x_q=None, x_scale=None, reverse=False, round_scale=False)
    y_q_refs = []
    y_scale_refs = []
    s = 0
    for i in range(n_expert):
        c = token_count_per_expert_list[i]
        idx = indices[s:s+c]
        y_slice = y[idx]
        y_smooth = y_slice/smooth_scales[i]
        y_max = y_smooth.abs().amax(1)/448
        y_q_refs.append((y_smooth/y_max[:,None]).to(torch.float8_e4m3fn))
        y_scale_refs.append(y_max)
        s += c
    y_q_ref = torch.cat(y_q_refs,0)
    y_scale_ref = torch.cat(y_scale_refs, 0)
    output_check(y_q_ref.float(), y_q.float(), 'data')
    output_check(y_scale_ref.float(), y_scale.float(), 'scale')

    def torch_index_select(y, indices):
        # output = torch.empty((indices.shape[0], y.shape[1]),dtype=y.dtype,device=y.device)
        output = y.index_select(0, indices)
        return output

    n_repeat = 100
    benchmark_func(torch_index_select, y, indices, n_repeat=n_repeat)
    benchmark_func(triton_index_select_smooth_quant, y, smooth_scales, token_count_per_expert, indices, reverse=False, round_scale=False, n_repeat=n_repeat)



if False:
    from flops.quant.smooth.reused_smooth import *
    n_expert = 256
    topk = 8
    reverse = True
    smooth_scales = 1+10*torch.rand((n_expert,N),device=device,dtype=torch.float32)
    token_count_per_expert_list = [M*topk//n_expert]*(n_expert-1)
    token_count_per_expert_list.append(M*topk-sum(token_count_per_expert_list))

    tokens = torch.randn((M*topk,N),device=device,dtype=torch.float32)

    token_count_per_expert = torch.tensor(token_count_per_expert_list, dtype=torch.int32, device=device)
    indices = (torch.arange(M*topk, device=device,dtype=torch.int32)//topk)[torch.argsort(torch.randn(M*topk,dtype=torch.float32,device=device))]
    y_q,y_scale,y_sum = triton_index_select_smooth_quant_and_sum(y, tokens, smooth_scales, token_count_per_expert, indices, x_q=None, x_scale=None, reverse=reverse, round_scale=False)
    y_q_refs = []
    y_scale_refs = []
    s = 0
    for i in range(n_expert):
        c = token_count_per_expert_list[i]
        idx = indices[s:s+c]
        y_slice = y[idx]
        y_smooth = y_slice*smooth_scales[i] if reverse else y_slice/smooth_scales[i]
        y_max = y_smooth.abs().amax(1)/448
        y_q_refs.append((y_smooth/y_max[:,None]).to(torch.float8_e4m3fn))
        y_scale_refs.append(y_max)
        s += c
    y_q_ref = torch.cat(y_q_refs,0)
    y_scale_ref = torch.cat(y_scale_refs, 0)
    sum_ref = (tokens*y[indices]).sum(1)

    output_check(y_q_ref.float(), y_q.float(), 'data')
    output_check(y_scale_ref.float(), y_scale.float(), 'scale')
    output_check(sum_ref.float(), y_sum.float(), 'scale')

    def torch_index_select_and_sum(y, indices, tokens):
        # output = torch.empty((indices.shape[0], y.shape[1]),dtype=y.dtype,device=y.device)
        output = y.index_select(0, indices)
        sums = (output*tokens).sum(1)
        return output, sums

    n_repeat = 100
    benchmark_func(torch_index_select_and_sum, y, indices, tokens, n_repeat=n_repeat)
    benchmark_func(triton_index_select_smooth_quant_and_sum, y, tokens, smooth_scales, token_count_per_expert, indices, reverse=False, round_scale=False, n_repeat=n_repeat)



if False:
    x = torch.randn(M, N, dtype=dtype, device=device)
    x_q = x.to(torch.float8_e4m3fn)
    scales = torch.randn(M, dtype=dtype, device=device)

    logits = torch.randn((M,n_experts), dtype=torch.float32, device=device)
    topk_values, topk_indices = torch.topk(logits, 1, dim=-1, sorted=True)
    logits[logits<topk_values[:,-1:]] = -1000000
    probs = torch.nn.Softmax(dim=1)(logits)
    route_map = probs>0
    token_count_per_expert = route_map.sum(0)
    token_count_per_expert_list = token_count_per_expert.tolist()
    out_tokens = sum(token_count_per_expert_list)

    token_indices = (
        torch.arange(M, device=device).unsqueeze(0).expand(n_experts, -1)
    )
    indices = token_indices.masked_select(route_map.T.contiguous())
    row_id_map = torch.reshape(torch.cumsum(route_map.T.contiguous().view(-1), 0),(n_experts, M)) - 1
    row_id_map[torch.logical_not(route_map.T)] = -1

    x_out, scale_out = triton_index_select(x, indices, scale=scales)
    x_out_ref, scale_out_ref = torch_fp16_index_select(x, scales, indices)
    output_check(x_out_ref,x_out,'x_out')
    output_check(scale_out_ref,scale_out,'scale_out')

    probs_out_ref = probs.T.contiguous().masked_select(route_map.T.contiguous())
    x_out, scale_out, probs_out = triton_permute_with_mask_map(x,scales,probs,row_id_map.T.contiguous(),out_tokens)
    output_check(x_out_ref,x_out,'x_out')
    output_check(scale_out_ref,scale_out,'scale_out')
    output_check(probs_out_ref,probs_out,'prob_out')


    n_repeat = 100
    ref_time = benchmark_func(torch_fp16_index_select,x, scales, indices, n_repeat=n_repeat)
    benchmark_func(triton_index_select,x, indices, scale=scales, n_repeat=n_repeat, ref_time=ref_time)
    benchmark_func(triton_permute_with_mask_map,x,scales,probs,row_id_map.T.contiguous(),out_tokens, n_repeat=n_repeat, ref_time=ref_time)


    outputs = triton_permutation.permute_with_mask_map(x,row_id_map,probs,scales.view(-1,1),M,n_experts,out_tokens,N,1)
    benchmark_func(triton_permutation.permute_with_mask_map,x,row_id_map,probs,scales.view(-1,1),M,n_experts,out_tokens,N,1, n_repeat=n_repeat, ref_time=ref_time)

    # benchmark_func(torch_scatter,logits.to(dtype), routing_map.T.contiguous(), weights, n_repeat=n_repeat)

    

if True:
    
    n_experts = 4
    topk = 2
    B = 128
    smooth_scales = 1+10*torch.rand((n_experts, N),device=device,dtype=torch.float32)
    logits = torch.randn((M,n_experts), dtype=torch.float32, device=device)
    logits = torch.nn.Softmax(dim=1)(logits)
    route_map = logits>=1/n_experts
    token_count_per_expert = route_map.sum(0)
    token_count_per_expert_list = token_count_per_expert.tolist()
    out_tokens = sum(token_count_per_expert_list)

    token_indices = (
        torch.arange(M, device=device).unsqueeze(0).expand(n_experts, -1)
    )
    indices = token_indices.masked_select(route_map.T.contiguous())
    row_id_map = torch.reshape(torch.cumsum(route_map.T.contiguous().view(-1), 0),(n_experts, M)) - 1
    row_id_map[torch.logical_not(route_map.T)] = -1
    grad_data = torch.randn((M,N), dtype=torch.bfloat16,device=device).to(torch.float8_e4m3fn)
    grad_scale = 1+torch.rand((M,N//B), dtype=torch.float32,device=device)
    y_q, y_scale = triton_smooth_unpermute_backward(grad_data, grad_scale, smooth_scales, token_count_per_expert, indices, x_q=None, x_scale=None, reverse=False, round_scale=True)

    q_refs = [] 
    scale_refs = []
    s = 0
    for i in range(n_experts):
        c = token_count_per_expert_list[i]
        data_slice = grad_data.view(torch.uint8)[indices[s:s+c]].view(torch.float8_e4m3fn)
        scale_slice = grad_scale[indices[s:s+c]]
        s += c
        y_smooth = (data_slice.float().view(c,N//B,B)*scale_slice[:,:,None]).view(c,N)/smooth_scales[i]
        scale = y_smooth.abs().amax(1)/448
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
        scale_refs.append(scale)
        q = (y_smooth/scale[:,None]).to(torch.float8_e4m3fn) 
        q_refs.append(q.view(torch.uint8))
    q_ref = torch.cat(q_refs, 0).view(torch.float8_e4m3fn)
    scale_ref = torch.cat(scale_refs,0)
    output_check(q_ref.float(), y_q.float(), 'data')
    output_check(scale_ref.float(), y_scale.float(), 'scale')


    # smooth_scale_ptrs = torch.tensor([x.data_ptr() for x in torch.split(smooth_scales,1)], device=device)
    permuted_data, permuted_scale = triton_smooth_permute_with_mask_map(grad_data,row_id_map.T.contiguous(),grad_scale,M, n_experts, out_tokens,N,smooth_scales, reverse=False,round_scale=True)
    output_check(q_ref.float(), permuted_data.float(), 'data')
    output_check(scale_ref.float(), permuted_scale.float(), 'scale')

    benchmark_func(triton_smooth_unpermute_backward, grad_data, grad_scale, smooth_scales, token_count_per_expert, indices, round_scale=True, n_repeat=100, ref_bytes=out_tokens*N*2)
    benchmark_func(triton_smooth_permute_with_mask_map, grad_data,row_id_map.T.contiguous(),grad_scale,M,n_experts,out_tokens,N,smooth_scales,reverse=False,round_scale=True, n_repeat=100, ref_bytes=out_tokens*N*2)

