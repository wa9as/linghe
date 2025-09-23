

<h1 align="center">FLood OPS(FLOPS) </h1>

  
<p align="center">
   A collection of high-performance kernels for LLM training.
</p>



## Roadmap ##

- All more shapes support and optional auto tuning with our kernels.
- Release our fp8 training kernels beyond blockwise quantization.

## *News or Update* 🔥

- [2025/07] We implement multiple kernels for fp8 training with `Megatron-LM` blockwise quantization. 


## Introduction

Our repo, FLOPS, is designed for LLM training. It provides 3 main categories of kernels:

- ** Fused quantization kernels **: fuse quantization with previous layer, e.g., RMS norm and Silu.
- ** Memory-friendly kernels **: use dtype cast in kernels instead of casting out kernels, e.g., softmax cross entropy and moe router gemm.
- ** Fused multi-operation kernels **: fuse multiple IO-itensive operations, e.g., ROPE with qk-norm and transpose, permute and padding, group RMS norm with sigmoid gate.







