

import torch

import time
import os
import random
from flops.utils.util import *
from flops.utils.scatter import *
from flops.utils.benchmark import benchmark_func

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def torch_fp16_scatter_add(x, outputs, indices, weights):
    if weights is not None:
        x = x*weights[:,None]
    dim = x.size(1)
    outputs.scatter_add_(0, indices.unsqueeze(1).expand(-1, dim), x)
    return outputs

M, N = 8192*2, 8192
n_experts = 32
topk = 2

dtype = torch.bfloat16
device = 'cuda:0'


outputs = torch.zeros(M, N, dtype=dtype, device=device)
weights = torch.randn(M*topk, dtype=dtype,device=device)
logits = torch.randn((M,n_experts),dtype=torch.float32,device=device)
double_logits = logits.to(torch.float64)*(1+torch.arange(n_experts, device=device, dtype=torch.float64)*1e-12)
double_top_values, top_indices = torch.topk(double_logits, topk, dim=-1, largest=True, sorted=True)
routing_map = double_logits>=double_top_values[:,-1:]
logits[torch.logical_not(routing_map)] = -1e6
probs = torch.nn.Softmax(dim=1)(logits)
counts = routing_map.sum(-1)
out_tokens = counts.sum().item()

x = torch.randn(out_tokens, N, dtype=dtype, device=device)

routing_map = routing_map.bool().T.contiguous()
dummy_indices = torch.arange(M, device=routing_map.device).unsqueeze(0).expand(n_experts, -1)
indices = dummy_indices.masked_select(routing_map)

row_id_map = torch.reshape(torch.cumsum(routing_map.view(-1), 0),(n_experts, M)) - 1
row_id_map[torch.logical_not(routing_map)] = -1




# sums = triton_aligned_scatter_add(x, outputs.clone(), indices, weights=weights)
# sums_ref = torch_fp16_scatter_add(x, outputs.clone(), indices, weights)
# output_check(sums_ref,sums,'aligned_scatter_add')

sums_ref = torch_fp16_scatter_add(x, outputs.clone(), indices, None)
# sums = triton_scatter_add(x, outputs.clone(), indices)
sums_split = triton_scatter_add_with_count(x, outputs.clone(), indices, counts)
unpermuted_prob = probs.T.contiguous().masked_select(routing_map)

# output_check(sums_ref,sums,'scatter_add')
output_check(sums_ref, sums_split,'split_scatter_add')

sums_unpermute, output_prob = triton_unpermute_with_mask_map(x, row_id_map.T.contiguous(), unpermuted_prob)
output_check(sums_ref, sums_unpermute,'unpermute_data')
output_check(probs, output_prob,'unpermute_prob')

import transformer_engine.pytorch.triton.permutation as triton_permutation
row_id_map_ref = triton_permutation.make_row_id_map(routing_map.T.contiguous(), M, n_experts)
row_id_map_output = triton_make_row_id_map(routing_map.T.contiguous())
output_check(row_id_map_ref.float(), row_id_map_output.T.float(),'row_id_map')

n_repeat = 100
ref_time = benchmark_func(torch_fp16_scatter_add,x, outputs, indices, weights,n_repeat=n_repeat)
# benchmark_func(triton_aligned_scatter_add,x, outputs, indices, weights=weights, n_repeat=n_repeat,ref_time=ref_time)
benchmark_func(triton_scatter_add,x, outputs, indices, n_repeat=n_repeat,ref_time=ref_time)
benchmark_func(triton_scatter_add_with_count,x, outputs, indices, counts, n_repeat=n_repeat,ref_time=ref_time)
benchmark_func(triton_unpermute_with_mask_map,x, row_id_map.T.contiguous(), probs, n_repeat=n_repeat,ref_time=ref_time)
benchmark_func(triton_permutation.make_row_id_map,routing_map.T.contiguous(), M, n_experts, n_repeat=n_repeat,ref_time=ref_time)
benchmark_func(triton_make_row_id_map,routing_map.T.contiguous(),n_repeat=n_repeat,ref_time=ref_time)
