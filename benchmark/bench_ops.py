import torch

import time
import os
import random
from flops.utils.benchmark import benchmark_func
from flops.gemm.fp8_gemm import *
from flops.gemm.blockwise_fp8_gemm import *
from flops.utils.util import *
from flops.utils.transpose import *
from flops.quant.channel.channel import *
from flops.quant.block.block import *
from flops.quant.block.group import *
from flops.quant.smooth.seperate_smooth import *
from flops.utils.chunk import *




# M, N, K = 8192, 10240, 8192  # max qkv
# M, N, K = 8192, 8192, 8192  # max out
# M, N, K = 2048, 4096, 8192  # max gate_up
# M, N, K = 2048, 8192, 2048  # max down
M, N, K = 8192, 8192, 8192
# M, N, K = 2048, 8192, 8192

# M, N, K = M-1, N-1, K-1

dtype = torch.bfloat16
device = 'cuda:0'
n_repeat = 100

x = torch.randn(M, K, dtype=dtype, device=device)
w = torch.randn(N, K, dtype=dtype, device=device)
y = torch.randn(M, N, dtype=dtype, device=device)

x_q = x.to(torch.float8_e4m3fn)
w_q = w.to(torch.float8_e4m3fn)
y_q = y.to(torch.float8_e4m3fn)

mode = 'd2h'
if mode == 'gemm':

    benchmark_func(trival_fp8_gemm, x_q, w_q, torch.bfloat16, n_repeat=n_repeat)
    benchmark_func(persistent_fp8_gemm, x_q, w_q.t(), torch.bfloat16, n_repeat=n_repeat)
    benchmark_func(fp8_gemm_nn, x_q,w_q,torch.bfloat16, n_repeat=n_repeat)

elif mode == 'bb':
    B = 64
    ref_flops = M * N * K * 2
    x_scales = torch.randn((M//B, K//B),dtype=torch.float32, device=device)
    w_scales = torch.randn((N//B, K//B),dtype=torch.float32, device=device)
    fp8_gemm(x_q, x_scales, w_q, w_scales, dtype)
    benchmark_func(fp8_gemm, x_q, w_q, x_scales, w_scales, out_dtype=dtype, n_repeat=n_repeat, ref_flops=ref_flops)

elif mode == 'quant':
    benchmark_func(block_quant, x,n_repeat=n_repeat)
    benchmark_func(stupid_group_quant, x, n_repeat=n_repeat)

    y_ref, s_ref = stupid_group_quant(x)
    y_opt, s_opt = group_quant(x)

    torch.testing.assert_close(y_opt.float(), y_ref.float(),
                                    rtol=0.02, atol=0.02)
    torch.testing.assert_close(s_ref.float(), s_opt.float(),
                                    rtol=0.02, atol=0.02)
    benchmark_func(stupid_group_quant, x, n_repeat=n_repeat)
    benchmark_func(group_quant, x, n_repeat=n_repeat)
    benchmark_func(persist_group_quant, x, n_repeat=n_repeat)

elif mode == 'transpose':
    # benchmark_func(fp16_transpose, x, n_repeat=n_repeat)
    # benchmark_func(triton_transpose,x, n_repeat=n_repeat)
    # benchmark_func(triton_block_transpose,x, n_repeat=n_repeat)
    # benchmark_func(triton_opt_transpose,x, n_repeat=n_repeat)

    benchmark_func(fp8_transpose, x_q, n_repeat=n_repeat)
    # benchmark_func(triton_transpose,x_q, n_repeat=n_repeat)
    benchmark_func(triton_block_transpose,x_q, n_repeat=n_repeat)
    # benchmark_func(triton_block_pad_transpose,x_q,pad=True, n_repeat=n_repeat)
    # benchmark_func(triton_opt_transpose,x_q, n_repeat=n_repeat)

elif mode == 'split':

    def torch_chunk_and_cat(x,counts,indices):
        chunks = torch.split(x,counts)
        chunks = [chunks[indices[i]] for i in range(256)]
        return torch.cat(chunks,dim=0)

    chunks = torch.split(x,[M//256]*256)
    chunks = [chunks[i//32+i%8*32] for i in range(256)]

    counts = torch.tensor([M//256]*256,device=device)
    indices = torch.tensor([i//32+i%8*32 for i in range(256)],device=device)

    benchmark_func(torch.split,x,[M//256]*256, n_repeat=n_repeat)
    benchmark_func(torch.cat,chunks,dim=0, n_repeat=n_repeat)
    benchmark_func(torch_chunk_and_cat,x,counts.tolist(),indices.tolist(), n_repeat=n_repeat)
    benchmark_func(triton_chunk_and_cat,x,counts,indices, n_repeat=n_repeat)

elif mode == 'zero':
    benchmark_func(torch.zeros,(M,K),dtype=dtype,device=device, n_repeat=n_repeat, ref_bytes=M*K*2)

elif mode == 'd2h':

    def d2h(x,device):
        return x.to(device)
    counts = torch.tensor([0]*32,dtype=torch.int64)
    benchmark_func(d2h,counts,device=device, n_repeat=n_repeat, ref_bytes=M*K*2)








