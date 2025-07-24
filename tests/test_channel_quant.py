import math
import torch 
from flops.utils.util import *
from flops.quant.channel.channel import *
from flops.utils.benchmark import benchmark_func



def test_row_quant(M=4096,N=4096, round_scale=True, bench=False):
    device = 'cuda:0'
    dtype = torch.bfloat16
    x = torch.randn((M,N),dtype=dtype,device=device)**3

    x_q_ref, x_scale_ref = torch_row_quant(x, round_scale=round_scale)

    x_q, x_scale = triton_row_quant(x,round_scale=round_scale)
    output_check(x_q_ref.float(), x_q.float(), mode='data')
    output_check(x_scale_ref, x_scale, mode='scale')

    x_q, x_scale = triton_tokenwise_row_quant(x, round_scale=round_scale)
    output_check(x_q_ref.float(), x_q.float(), mode='data')
    output_check(x_scale_ref, x_scale, mode='scale')

    if bench:
        ref_time = benchmark_func(torch_row_quant, x, n_repeat=100, ref_bytes=M*N*3)
        benchmark_func(triton_row_quant, x, n_repeat=100, ref_bytes=M*N*3, ref_time=ref_time)
        benchmark_func(triton_deprecated_tokenwise_row_quant, x, n_repeat=100, ref_bytes=M*N*3, ref_time=ref_time)
        benchmark_func(triton_tokenwise_row_quant, x, n_repeat=100, ref_bytes=M*N*3, ref_time=ref_time)


if __name__ == '__main__':
    test_row_quant(M=4096,N=4096, round_scale=False)
    test_row_quant(M=4090,N=4096, round_scale=True)
    test_row_quant(M=4096,N=8192, round_scale=True)
    test_row_quant(M=3456,N=2048, round_scale=True)
    test_row_quant(M=1,N=2048, round_scale=True)