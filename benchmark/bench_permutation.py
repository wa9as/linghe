
import torch

import time
import os
import random
from flops.utils.util import *   # noqa: F403
from flops.utils.gather import *   # noqa: F403
from flops.utils.scatter import *   # noqa: F403

from flops.utils.benchmark import benchmark_func

import transformer_engine.pytorch.triton.permutation as triton_permutation


def torch_index_select(y, indices):
    output = y.index_select(0, indices)
    return output

def torch_fp16_index_select(x, scales, indices):
    return x.index_select(0, indices), scales.index_select(0, indices)




def torch_fp16_scatter_add(x, outputs, indices, weights):
    if weights is not None:
        x = x*weights[:,None]
    dim = x.size(1)
    outputs.scatter_add_(0, indices.unsqueeze(1).expand(-1, dim), x)
    return outputs


def bench_triton_permute_with_mask_map(M=4096, N=4096, n_experts=256, topk=8):
    device='cuda:0'
    dtype = torch.bfloat16
    x = torch.randn(M, N, dtype=dtype, device=device)
    x_q = x.to(torch.float8_e4m3fn)
    scales = torch.randn(M, dtype=dtype, device=device)

    logits = torch.randn((M,n_experts), dtype=torch.float32, device=device)
    
    probs, mask_map,token_count_per_expert, indices, row_id_map = torch_make_indices(logits, topk=topk, bias=0.0)
    out_tokens = sum(token_count_per_expert.tolist())

    n_repeat = 100
    ref_time = benchmark_func(torch_fp16_index_select,x, scales, indices, n_repeat=n_repeat)
    benchmark_func(triton_permute_with_mask_map,x,scales,probs,row_id_map,out_tokens, n_repeat=n_repeat, ref_time=ref_time)

    benchmark_func(triton_permutation.permute_with_mask_map,x,row_id_map.T.contiguous(),probs,scales.view(-1,1),M,n_experts,out_tokens,N,1, n_repeat=n_repeat, ref_time=ref_time)





def bench_triton_unpermute_with_mask_map(M=4098, N=4096, n_experts=32, topk=2):

    dtype = torch.bfloat16
    device = 'cuda:0'

    weights = torch.randn(M*topk, dtype=dtype,device=device)
    logits = torch.randn((M,n_experts),dtype=torch.float32,device=device)
    probs, mask_map,token_count_per_expert, indices, row_id_map = torch_make_indices(logits, topk=topk, bias=0.0)
    
    token_count_per_expert_list = token_count_per_expert.tolist()
    out_tokens = sum(token_count_per_expert_list)

    x = torch.randn(out_tokens, N, dtype=dtype, device=device)

    outputs = torch.zeros((M, N), dtype=dtype, device=device)
    counts = mask_map.sum(1)

    n_repeat = 100
    ref_time=benchmark_func(triton_scatter_add,x, outputs, indices, n_repeat=n_repeat)
    benchmark_func(triton_scatter_add_with_count,x, outputs, indices, counts, n_repeat=n_repeat,ref_time=ref_time)
    benchmark_func(triton_unpermute_with_mask_map,x, row_id_map.T.contiguous(), probs, n_repeat=n_repeat,ref_time=ref_time)
    benchmark_func(triton_permutation.make_row_id_map,mask_map.T.contiguous(), M, n_experts, n_repeat=n_repeat,ref_time=ref_time)
    benchmark_func(triton_make_row_id_map,mask_map,n_repeat=n_repeat,ref_time=ref_time)

    benchmark_func(triton_permutation.unpermute_with_mask_map,x,row_id_map,probs,None,M,n_experts,N)


if __name__ == '__main__':
    bench_triton_permute_with_mask_map(M=4098, N=4096, n_experts=32, topk=2)
    bench_triton_unpermute_with_mask_map(M=4098, N=4096, n_experts=32, topk=2)