import math
import torch 
from flops.utils.util import *   # noqa: F403
from flops.quant.channel.channel import *   # noqa: F403
from flops.utils.benchmark import benchmark_func



def test_row_quant(M=4096,N=4096):
    device = 'cuda:0'
    dtype = torch.bfloat16
    x = torch.randn((M,N),dtype=dtype,device=device)

    x_q_ref, x_scale_ref = torch_row_quant(x, dtype=torch.float8_e4m3fn)

    x_q, x_scale = triton_row_quant(x)
    output_check(x_q_ref.float(), x_q.float(), mode='data')
    output_check(x_scale_ref, x_scale, mode='scale')

    x_q, x_scale = triton_tokenwise_row_quant(x)
    output_check(x_q_ref.float(), x_q.float(), mode='data')
    output_check(x_scale_ref, x_scale, mode='scale')

    ref_time = benchmark_func(torch_row_quant, x, n_repeat=100, ref_bytes=M*N*3)
    benchmark_func(triton_row_quant, x, n_repeat=100, ref_bytes=M*N*3, ref_time=ref_time)
    benchmark_func(triton_deprecated_tokenwise_row_quant, x, n_repeat=100, ref_bytes=M*N*3, ref_time=ref_time)
    benchmark_func(triton_tokenwise_row_quant, x, n_repeat=100, ref_bytes=M*N*3, ref_time=ref_time)


if __name__ == '__main__':
    test_row_quant(M=4096,N=4096)