

import torch

import time
import os
import random
from flops.utils.util import *
from flops.utils.gather import *
from flops.utils.benchmark import benchmark_func

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

if True:
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

    # benchmark_func(torch_scatter,logits.to(dtype), routing_map.T.contiguous(), weights, n_repeat=n_repeat)

    