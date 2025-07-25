import torch 
from flops.utils.util import ( fp16_backward,
                               fp16_f_and_b,
                               fp16_forward,
                               fp16_update )
from flops.utils.benchmark import benchmark_func
from flops.quant.channel.channel import ( channel_quant_backward,
                                          channel_quant_forward,
                                          channel_quant_update )

# 5b: hidden_size:4k  seq_length:8K shape:(M,N,K)
# qkv: 8192, 6144, 4096
# out: 8192, 4096, 4096
# up/gate: 8192, 13312, 4096
# down: 8192, 4096, 13312   # benchmark setting


# 80b: hidden_size:8k  seq_length:8K shape:(M,N,K)
# qkv: 8192, 10240, 8192
# out: 8192, 8192, 8192
# up/gate: 8192, 34048, 8192
# down: 8192, 4096, 13312


M, N, K = 8192, 4096, 4096

def benchmark_with_shape(shape):
    M, N, K = shape
    device = 'cuda:0'
    dtype = torch.bfloat16
    qtype = torch.float8_e4m3fn  # torch.float8_e5m2
    n_repeat = 100
    gpu = torch.cuda.get_device_properties(0).name


    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)
    y = torch.randn(M, N, dtype=dtype, device=device)
    x_q = x.to(qtype)
    w_q = w.to(qtype)
    y_q = y.to(qtype)

    org_out = fp16_forward(x, w.t())
    print(f'\ndevice:{gpu} M:{M} N:{N} K:{K}')

    # y = x @ w
    # dx = y @ wT
    # dwT = yT @ x

    ref_time = benchmark_func(fp16_forward, x, w.t(), n_repeat=n_repeat, ref_flops=M*K*N*2)
    benchmark_func(channel_quant_forward, x, w, n_repeat=n_repeat, ref_flops=M*K*N*2, ref_time=ref_time)
    ref_time = benchmark_func(fp16_backward, y, w, n_repeat=n_repeat, ref_flops=M*K*N*2)
    benchmark_func(channel_quant_backward, y, w, n_repeat=n_repeat, ref_flops=M*K*N*2, ref_time=ref_time)
    ref_time = benchmark_func(fp16_update, y, x, n_repeat=n_repeat, ref_flops=M*K*N*2)
    benchmark_func(channel_quant_update, y, x, n_repeat=n_repeat, ref_flops=M*K*N*2, ref_time=ref_time)

    ref_time = benchmark_func(fp16_f_and_b, x, w, y, n_repeat=n_repeat, ref_flops=M*K*N*6)
    benchmark_func(fp8_channel_f_and_b, x, w, y, n_repeat=n_repeat, ref_time=ref_time,ref_flops=M*K*N*6)



# 5b: hidden_size:4k  seq_length:8K shape:(M,N,K)
# qkv: 8192, 6144, 4096
# out: 8192, 4096, 4096
# up/gate: 8192, 13312, 4096
# down: 8192, 4096, 13312   # benchmark setting


# 80b: hidden_size:8k  seq_length:8K shape:(M,N,K)
# qkv: 8192, 10240, 8192
# out: 8192, 8192, 8192
# up/gate: 8192, 34048, 8192
# down: 8192, 4096, 34048


benchmark_with_shape([8192, 4096, 13312])

# for shape in [[8192, 6144, 4096], [8192, 4096, 4096], [8192, 13312, 4096], [8192, 4096, 13312],
#             [8192, 10240, 8192],[8192, 8192, 8192],[8192, 34048, 8192],[8192, 4096, 34048]]:
#     benchmark_with_shape(shape)
