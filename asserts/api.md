# API Reference


```
linghe.utils.norm.triton_rms_norm_and_block_quant_forward(x, weight, eps:Optional[float]=1e-6, out:Optional[torch.Tensor]=None, scale:Optional[torch.Tensor]=None, rms:Optional[torch.Tensor]=None, round_scale: Optional[bool]=False, output_mode:Optional[int]=2)
```

Computes the forward pass of RMSNorm and block quantization.

**Parameters:**  
- x(*torch.Tensor*) - Input tensor. [M, N]
- weight(*torch.Tensor*) - RMSNorm weight. [N]
- eps(*float*) -  epsilon value for L2 normalization.
- round_scale(*bool*) - Set whether to force power of 2 scales.
- rms(*torch.Tensor*) - Reciprocal of the root mean square of the input calculated over the last dimension.[N]
- output_mode - (*int*,  {0, 1, 2}, default = 2) 0 only output non-transpose tensor, 1 only output transposed tensor, 2 return both.

---

**`
Class linghe.facade.rope.QkNormHalfRopeFunction
`**

```
forward(qkv:, q_norm_weight, k_norm_weight, freqs, H, h, eps:Optional[float]=1e-6)
```
Split qkv, and apply L2 nrom and ROPE on q and k.

**Parameters:**  
- qkv(*torch.Tensor*) - QKV tensor with size of [S, B, dim]
- freqs(*torch.Tensor*) - Freqs matrix based on half dim.
- H(*int*) - Number of attention heads.
- h(*int*) - Number of query groups.
- eps(*float*) -  epsilon value for L2 normalization.

```
backward(grad_q, grad_k, grad_v)
```
**Parameters:**  
- grad_q(*torch.Tensor*) Grad of q tensor.
- grad_k(*torch.Tensor*) Grad of k tensor.
- grad_v(*torch.Tensor*) Gard of v tensor.

---

**`
Class linghe.facade.fp32_linear.FusedFp32GEMM
`**

Optimized fp32 gemm in router gate function. Convert bf16 input and weight to float32 during the gemm operation.

```
forward(input, weight)
```
**Parameters:**  
- input(*torch.Tensor*) - Input tensor with [B, S, dim], dtype of bf16.
- weight(*torch.Tensor*) - Weight tensor of router.

```
backward(grad_output)
```
**Parameters:**  
- grad_output(*torch.Tensor*) - Gradient of the activation.

---

```
linghe.utils.gather.triton_permute_with_mask_map(inp, scale, probs, row_id_map, num_out_tokens, contiguous, tokens_per_expert)
```
Permute the tokens and probs based on the routing map. Index indicates row index of the output tensor(-1 means not selected). Perform well even when inp.size(0) < expert padding number, do not need extra explict padding.

**Parameters:**  
- inp(*torch.Tensor*) - Input hidden.[num_tokens, hidden_size]
- scale(*torch.Tensor*) - [num_tokens, scale_size] 
- prob(*torch.Tensor*) - [num_tokens] Router prob.
- row_id_map(*torch.Tensor*) - [n_experts, num_tokens] Index indicates row index of the output tensor.
- num_out_tokens(*int*) - Output token count, including padding tokens.
- contiguous(*bool*) - Whether indices in row_id_map is contiguous, should be False if padded.
- token_per_expert(bool) - [num_experts] Token count per expert, non-blocking cuda tensor.

---

```
linghe.utils.scatter.triton_unpermute_with_mask_map(grad, row_id_map, probs)
```
Unpermute a tensor with permuted tokens with router mapping.

**Parameters:**  
- inp(*torch.Tensor*) - [num_tokens, hidden_size] Permuted tokens.
- row_id_map(*torch.Tensor*) - [n_experts, num_tokens] Routing map to unpermute the tokens.
- prob(*torch.Tensor*) - [num_out_tokens] Permuted probs.

---

```
linghe.util.silu.triton_silu_and_block_quant_forward(x, out:Optional[torch.Tensor]=None, scale:Optional[torch.Tensor]=None, round_scale:Optional[bool]=False, output_mode:Optional[int]=2)
```

Applies the forward pass of Sigmoid Linear Unit(SiLU) element-wise and block quant.(used in shared expert layers.)

**Parameters:**  
- x(*torch.Tensor*) - Input tensor to be quanted.
- round_scale(*bool*) - Set whether to force power of 2 scales.
- output_mode - (*int*,  {0, 1, 2}, default = 2) 0 only output non-transpose tensor, 1 only output transposed tensor, 2 return both.

---

```
linghe.util.silu.triton_silu_and_block_quant_backward(g, x, round_scale:Optional[bool]=False)
```
**Parameters:**  
- g(*torch.Tensor*) - Gradient tensor to be quanted.
- x(*torch.Tensor*) - Input tensor.
- round_scale(*bool*) - Set whether to force power of 2 scales. Default to False.

---

```
linghe.util.silu.triton_batch_weighted_silu_and_block_quant_forward(x, weight, counts, splits:Optional[List]=None ,out:Optional[torch.Tensor]=None, scale:Optional[torch.Tensor]=None, round_scale:Optional[bool]=False, output_mode:Optional[int]=2)
```

Fused op for batched weighted SiLU and block quant.

**Parameters:**  
- x(*torch.Tensor*) - Input tensor.
- weight(*torch.Tensor*)  - Permuted probs
- couts(*torch.Tensor*)  - Tokens per expert cuda tensor.
- splits(*List[int]*) - List of tokens per expert. If compute in batch mode should not be None.
- output_mode - (*int*,  {0, 1, 2}, default = 2) 0 only output non-transpose tensor, 1 only output transposed tensor, 2 return both.

---

```
linghe.util.silu.triton_batch_weighted_silu_and_block_quant_backward(g, x, weight, counts, splits:Optional[List]=None, round_scale:Optional[bool]=False)
```
Return blockwise quantized gradient of silu backward.
Quantized tensor is A tuple of ()
**Parameters:**  
- g(*torch.Tensor*) - Input gradient tensor.
- x(*torch.Tensor*) - Input tensor.
- weight(*torch.Tensor*)  - Permuted probs, 
- counts(*torch.Tensor*)  - Tokens per expert, it is a CUDA tensor.
- splits(*List[int]*) - Tokens per expert, it is a list of int.
- round_scale(bool) - round  scale to integer pow of 2
---

**`
Class  linghe.facade.loss.SoftmaxCrossEntropyFunction
`**

SoftmaxCrossEntropy.

```
forward(logits, labels, inplace: Optional[bool]=False) 
```

Fast impl of softmax cross entropy.

**Parameters:**  
- logits(*torch.Tensor*) - Input logits.
- labels(*torch.Tensor*) - Input labels.
- inplace(*bool*) - reuse the `logits` tensor as gradient tensor if inplace=True, else allocate a new tensor.

```


```
linghe.util.reduce.triton_batch_sum_with_ord(xs, ord:Optional[int]=2) 
```
return sum(abs(x)**ord).

**Parameters:**  
- xs(*List[torch.Tensor]*) - Tensor lists.
- ord(*int*) - the order of tensor.

--- 

```
linghe.util.reduce.triton_batch_count_zero(xs) 
```
Parallel count zeros in the given tensor lists, return the total zero number.

**Parameters:**  
- xs(*List[torch.Tensor]*) - Tensor lists.

--- 

**`
Class linghe.facade.norm.GroupNormGateFunction
`**
Fused operation of group RMSNorm and sigmoid gate function.

```
forward(x, gate, weight, eps:Optional[float]=1e-6, group_size:Optional[int]=4)
```
Note that the output shape is transposed [S, B, dim]

**Parameters:**  

- x(*torch.Tensor*) - [B, S, dim], output tensor of attention kernel.
- gate(*torch.Tensor*) - [S, B, dim], gate tensor. 
- weight(*torch.Tensor*) - [dim], RMSNorm weight tensor.
- group_size(int) - group size of RMSNorm
```
