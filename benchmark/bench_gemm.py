import torch

from flops.gemm.blockwise_fp8_gemm import blockwise_fp8_gemm
from flops.gemm.channelwise_fp8_gemm import triton_scaled_mm
from flops.gemm.fp8_gemm import (persistent_fp8_gemm,
                                 trival_fp8_gemm)
from flops.utils.add import triton_block_add
from flops.tools.benchmark import benchmark_func




def separate_gemm(x_q, w_q, x_scales, w_scales, c=None, accum=True):
    bf16_out = torch._scaled_mm(x_q,
                                w_q.t(),
                                scale_a=x_scales.view(-1, 1),
                                scale_b=w_scales.view(1, -1),
                                out_dtype=torch.bfloat16,
                                use_fast_accum=True)
    triton_block_add(c, bf16_out, accum=accum)
    return c


def test_fp8_gemm(M=4096, N=4096, K=4096):

    dtype = torch.bfloat16
    device = 'cuda:0'
    n_repeat = 100

    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)
    y = torch.randn(M, N, dtype=dtype, device=device)

    x_q = x.to(torch.float8_e4m3fn)
    w_q = w.to(torch.float8_e4m3fn)
    y_q = y.to(torch.float8_e4m3fn)

    ref_flops = M * N * K * 2
    benchmark_func(trival_fp8_gemm, x_q, w_q, torch.bfloat16, n_repeat=n_repeat,
                   ref_flops=ref_flops)
    benchmark_func(persistent_fp8_gemm, x_q, w_q.t(), torch.bfloat16,
                   n_repeat=n_repeat, ref_flops=ref_flops)


def test_blockwise_gemm(M=4096, N=4096, K=4096):
    dtype = torch.bfloat16
    device = 'cuda:0'
    n_repeat = 100

    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)
    y = torch.randn(M, N, dtype=dtype, device=device)

    x_q = x.to(torch.float8_e4m3fn)
    w_q = w.to(torch.float8_e4m3fn)
    y_q = y.to(torch.float8_e4m3fn)

    B = 64
    ref_flops = M * N * K * 2
    x_scales = torch.randn((M // B, K // B), dtype=torch.float32, device=device)
    w_scales = torch.randn((N // B, K // B), dtype=torch.float32, device=device)
    blockwise_fp8_gemm(x_q, x_scales, w_q, w_scales, dtype)
    benchmark_func(blockwise_fp8_gemm, x_q, w_q, x_scales, w_scales, out_dtype=dtype,
                   n_repeat=n_repeat, ref_flops=ref_flops)



def test_channelwise_gemm(M=4096, N=4096, K=4096):

    dtype = torch.bfloat16
    device = 'cuda:0'
    n_repeat = 100

    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)
    y = torch.randn(M, N, dtype=dtype, device=device)

    x_q = x.to(torch.float8_e4m3fn)
    w_q = w.to(torch.float8_e4m3fn)
    y_q = y.to(torch.float8_e4m3fn)

    ref_flops = M * N * K * 2
    x_scales = torch.randn((M,), dtype=torch.float32, device=device)
    w_scales = torch.randn((N,), dtype=torch.float32, device=device)

    y_fp32 = torch.zeros(M, N, dtype=torch.float32, device=device)
    y_fp16 = torch.zeros(M, N, dtype=torch.float16, device=device)


    benchmark_func(triton_scaled_mm, x_q, w_q, x_scales, w_scales, c=y,
                   accum=True, n_repeat=n_repeat, ref_flops=ref_flops)
    benchmark_func(triton_scaled_mm, x_q, w_q, x_scales, w_scales, c=y_fp16,
                   accum=True, n_repeat=n_repeat, ref_flops=ref_flops)
    benchmark_func(triton_scaled_mm, x_q, w_q, x_scales, w_scales, c=y_fp32,
                   accum=True, n_repeat=n_repeat, ref_flops=ref_flops)

    benchmark_func(separate_gemm, x_q, w_q, x_scales, w_scales, c=y, accum=True,
                   n_repeat=n_repeat, ref_flops=ref_flops)
    benchmark_func(separate_gemm, x_q, w_q, x_scales, w_scales, c=y_fp16,
                   accum=True, n_repeat=n_repeat, ref_flops=ref_flops)
    benchmark_func(separate_gemm, x_q, w_q, x_scales, w_scales, c=y_fp32,
                   accum=True, n_repeat=n_repeat, ref_flops=ref_flops)


def test_te_blockwise_gemm(M=4096, N=4096, K=4096):

    # layout == 'TN':  # forward, y=x@w
    # x_q = B._rowwise_data
    # x_scale = B._rowwise_scale_inv
    # w_q = A._rowwise_data 
    # w_scale = A._rowwise_scale_inv

    # import transformer_engine_torch as tex
    import transformer_engine as te
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor
    from transformer_engine.pytorch.module.base import get_workspace
    from transformer_engine.pytorch.constants import TE_DType
    row_data = torch.randn((M,K), device='cuda:0').to(torch.float8_e4m3fn)
    row_scales = torch.randn((K//128,M), device='cuda:0')
    x = Float8BlockwiseQTensor(shape=(M,K),
                                dtype=torch.bfloat16,
                                fp8_dtype=TE_DType[torch.float8_e4m3fn],
                                rowwise_data=row_data,
                                rowwise_scale_inv=row_scales,
                                columnwise_data=None,
                                columnwise_scale_inv=None,
                                quantizer=None,
                                requires_grad=False,
                                is_2D_scaled=False
                            )
    
    row_data = torch.randn((N,K), device='cuda:0').to(torch.float8_e4m3fn)
    row_scales = torch.randn((K//128,N//128), device='cuda:0')
    w = Float8BlockwiseQTensor(shape=(N,K),
                                dtype=torch.bfloat16,
                                fp8_dtype=TE_DType[torch.float8_e4m3fn],
                                rowwise_data=row_data,
                                rowwise_scale_inv=row_scales,
                                columnwise_data=None,
                                columnwise_scale_inv=None,
                                quantizer=None,
                                requires_grad=False,
                                is_2D_scaled=True
                            )
    A = w 
    transa = True 
    B = x 
    transb = False 
    out = None 
    quantization_params = None 
    out_dtype = TE_DType[torch.bfloat16]
    bias = None 
    bias_dtype = TE_DType[torch.bfloat16]
    gelu = False 
    gelu_in = None 
    grad = False 
    workspace = get_workspace()
    workspace_size = workspace.shape[0]
    accumulate = False 
    use_split_accumulator = True 
    args = (
            A,
            transa,  # transa
            B,
            transb,  # transb
            out,
            quantization_params,
            out_dtype,
            bias,
            bias_dtype,
            gelu,
            gelu_in,
            grad,  # grad
            workspace,
            workspace_size,
            accumulate,
            use_split_accumulator,
        )
    out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args)

    ref_flops = M * N * K * 2
    ref_bytes = M * K + N * K + M * N *2 
    benchmark_func(tex.generic_gemm, *args,
                   n_repeat=100, ref_flops=ref_flops, ref_bytes=ref_bytes)


if __name__ == '__main__':
    test_te_blockwise_gemm(M=1024-16, N=8192, K=8192)
