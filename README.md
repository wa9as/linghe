

<h1 align="center">FLOPS</h1>

  
<p align="center">
   A collection of high-performance kernels for LLM training.
</p>



## Roadmap ##

- Support more shapes and various GPU archs.
- Release our fp8 training kernels beyond blockwise quantization.

## *News or Update* 🔥

- [2025/07] We implement multiple kernels for fp8 training with `Megatron-LM` blockwise quantization. 


## Introduction

Our repo, FLOPS, is designed for LLM training. It provides 3 main categories of kernels:

- ** Fused quantization kernels **: fuse quantization with previous layer, e.g., RMS norm and Silu.
- ** Memory-friendly kernels **: use dtype cast in kernels instead of casting out kernels, e.g., softmax cross entropy and moe router gemm.
- ** Fused  kernels **: fuse multiple IO-itensive operations, e.g., ROPE with qk-norm and transpose, permute and padding, group RMS norm with sigmoid gate.


## Benchmark

We benchmark on H800 with batch size 8192, hidden size 2048, num experts 256, activation experts 8.

| kernel | baseline(us) | flops(us) | speedup | desc |
|--------|--------------|-----------|---------|------|
| RMSNorm+Quantization(forward) | 159.3 us | 72.4 us | 2.2 | recomputation-optimized layernorm & quantization kernel |
| Split+qk-norm+rope+transpose(forward) | 472 us | 59.1 us | 7.99 | split qkv (layout[length, bs, dim]) to q/k/v, rmsnorm with qk heads, half rope with qk heads and transpose to layout[bs,length, num_heads, head_dim] |
| Split+qk-norm+rope+transpose(backward) | 645 us | 107.5 us | 6.0 | |
| Fp32 router gemm(forward) | 242.3 us | 61.6 us | 3.931 | read bf16 input and cast to fp32 in register to save activation memory and accelerate IO. |
| Fp32 router gemm(backward) | 232.7 us | 78.1 us | 2.979 | |
| Permute with padded indices | 388 us | 229.4 us | 1.69 | pad indices instead of activations, reuse padding indices of permutation |
| Unpermute with padding indices | 988.6 us | 806.9 us | 1.23 | |
| Batch Silu+quantization(forward) | 6241.7 us | 1181.7 us | 5.28 | Used in routed experts silu activation (with expert# = 32) |
| Batch Silu+quantization(backward) | 7147.7 us | 2317.9 us | 3.08 | |
| Silu+quantization(forward) | 144.9 us | 58.2 us | 2.48 | Used in shared expert silu activation |
| Silu+quantization(backward) | 163.4 us | 74.2 us | 2.2 | |
| fused linear gate(forward) | 160.4 us | 46.9 us | 3.42 | used in linear attention |
| fused linear gate(backward) | 572.9 us | 81.1 us | 7.06 | |
| Cross entropy(forward) | 2780.8 us | 818.2 us | 3.4 | Use bf16 input and inplace upgrade for backward (with vocab size 157184) |
| Cross entropy(backward) | 7086.3 us | 1781.0 us | 3.98 | |
| batch grad norm | 1733.7 us | 1413.7 us | 1.23 | Used in calculate grad norm |
| Batch count zero | 4997.9 us | 746.8 us | 6.69 | Used in calculate grad zero |


## examples

Examples can be found in tests.