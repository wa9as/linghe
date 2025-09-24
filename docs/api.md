# API Reference


```
flops.utils.norm.triton_rms_norm_and_block_quant_forward(x, weight, eps=1e-6, out=None, scale=None, rms=None, calibrate=False, output_rms=False, round_scale=False, output_mode=2) -> torch.Tensor
```

Computes the forward pass of RMSNorm and block quantization.

**Parameters:**  
- x(*torch.Tensor*) - Input tensor. [M, N]
- weight(*torch.Tensor*) - RMSNorm weight. [N]
- eps(float, default = 1e-6) -  epsilon value for L2 normalization.
- round_scale(*bool*) - Set whether to force power of 2 scales.
- rms(*torch.Tensor*) - Reciprocal of the root mean square of the input calculated over the last dimension.[N]
- output_mode - (*int*,  {0, 1, 2}, default = 2) 0 only output non-transpose tensor, 1 only output transposed tensor, 2 return both.

**`
Class flops.facade.rope.QkNormHalfRopeFunction
`**

```
forward(qkv:, q_norm_weight, k_norm_weight, freqs, H, h, eps=1e-6)
```
Split qkv, and apply L2 nrom and ROPE on q and k.

**Parameters:**  
- qkv(*torch.Tensor*) - size of [S, B, dim]
- freqs(*torch.Tensor*) - freqs matrix based on half dim.
- H(int) - number of attention heads.
- h(int) - number of query groups.

```
backward(grad_q, grad_k, grad_v)
```
**Parameters:**  
- grad_q(*torch.Tensor*) grad of q tensor.
- grad_k(*torch.Tensor*) grad of k tensor.
- grad_v(*torch.Tensor*) gard of v tensor.


**`
Class flops.facade.fp32_linear.FusedFp32GEMM
`**

Optimized fp32 gemm in router gate function. Convert bf16 input and weight to float32 during the gemm operation.

```
forward(input, weight)->torch.Tensor
```
**Parameters:**  
- input(*torch.Tensor*) - Input tensor with [B, S, dim], dtype of bf16.
- weight(*torch.Tensor*) - Weight tensor of router.

```
backward(grad_output)->torch.Tensor
```
**Parameters:**  
- grad_output(*torch.Tensor*) - gradient of the activation.


```
flops.utils.gather.triton_permute_with_mask_map(inp, scale, probs, row_id_map, num_out_tokens, contiguous, tokens_per_expert)
```
Permute the tokens and probs based on the routing_map. Index indicates row index of the output tensor(-1 means not selected). Perform well even when inp.size(0) < expert padding number.

**Parameters:**  
- inp(*torch.Tensor*) - [num_tokens, hidden_size]
- scale(*torch.Tensor*) - [num_tokens, scale_size] 
- prob(*torch.Tensor*) - [num_tokens] router prob.
- row_id_map(*torch.Tensor*) - [n_experts, num_tokens] 
- num_out_tokens(int) - output token count, including padding tokens.
- contiguous(bool) - whether indices in row_id_map is contiguous, should be False if padded.
- token_per_expert(bool) - [num_experts], token count per expert, non-blocking cuda tensor

```
flops.utils.scatter.triton_unpermute_with_mask_map(grad, row_id_map, probs)
```
Unpermute a tensor with permuted tokens.

**Parameters:**  
- inp(*torch.Tensor*) - [num_tokens, hidden_size] permuted tokens.
- row_id_map(*torch.Tensor*) - [n_experts, num_tokens] routing map to unpermute the tokens.
- prob(*torch.Tensor*) - [num_out_tokens] permuted probs.

```
flops.util.silu.triton_silu_and_block_quant_forward(x, out=None, scale=None, round_scale=False, output_mode=2)
```

Applies the forward pass of Sigmoid Linear Unit(SiLU) element-wise and block quant.(used in shared expert layers.)

**Parameters:**  
- x(torch.Tensor) - Input tensor to be quanted.
- round_scale - Set whether to force power of 2 scales.
- output_mode - (int,  {0, 1, 2}, default = 2) 0 only output non-transpose tensor, 1 only output transposed tensor, 2 return both.

```
flops.util.silu.triton_silu_and_block_quant_backward(g, x, round_scale=False)
```
**Parameters:**  
- g(*torch.Tensor*) - Gradient tensor to be quanted.
- x(*torch.Tensor*) - Input tensor.
- round_scale - Set whether to force power of 2 scales. Default to False.


```
flops.util.silu.triton_batch_weighted_silu_and_block_quant_forward(x, weight, counts, splits=None ,output_mode, **kwargs)
```

Fused op for batched weighted SiLU and block quant.

**Parameters:**  
- x(*torch.Tensor*) - Input tensor.
- weight(*torch.Tensor*)  - Permuted probs
- couts(*torch.Tensor*)  - Tokens per expert cuda tensor.
- splits[List] - List of tokens per expert. If compute in batch mode should not be None.


```
flops.util.silu.triton_batch_weighted_silu_and_block_quant_backward(g, x, weight, counts, splits=None, round_scale=False)
```
**Parameters:**  
- g(*torch.Tensor*) - Input gradient tensor.
- x(*torch.Tensor*) - Input tensor.
- weight(*torch.Tensor*)  - permuted probs
- couts(*torch.Tensor*)  - tokens per expert cuda tensor.
- splits[List] - list of tokens per expert. If compute in batch mode should not be None.

**`
Class  flops.facade.loss.SoftmaxCrossEntropyFunction
`**

```
forward(logits, labels) -> torch.Tensor
```

Fast impl of softmax cross entropy.

**Parameters:**  
- logits(*torch.Tensor*) - Input logits.
- labels(*torch.Tensor*) - Input labels.

```
backward(grad_output) -> torch.Tensor
```

**Parameters:**  
- grad_output(*torch.Tensor*) - Gradient tensor.

```
flops.util.reduce.triton_batch_sum_with_ord(xs, ord) -> torch.Tensor
```
Square sum the gards of all the experts. All the experts grads are applied simultaneously.

**Parameters:**  
- xs(List)[*torch.Tensor*] - Grads lists
- ord(int) -Sum type. 1 for abs add and 2 for square add.

```
flops.util.reduce.triton_batch_count_zero(xs) -> torch.Tensor
```
Prallel cout zeros in all the given grads lists.
**Parameters:**  
- xs(List)[*torch.Tensor*] - Grads lists