import torch 
from flops.utils.util import ( fp16_f_and_b,
                               fp16_forward )
from flops.utils.benchmark import benchmark_func
from flops.quant.hadamard.naive_hadamard import ( fp8_hadamard_f_and_b,
                                                  hadamard_matrix,
                                                  triton_hadamard_quant_nn,
                                                  triton_hadamard_quant_nt,
                                                  triton_hadamard_quant_tn )
from flops.quant.hadamard.seperate_hadamard import ( fp8_hadamard_f_and_b_megatron,
                                                     triton_hadamard_quant_nn_megatron,
                                                     triton_hadamard_quant_nt_megatron,
                                                     triton_hadamard_quant_tn_megatron )
from flops.quant.hadamard.fused_hadamard import fp8_fused_hadamard_f_and_b
from flops.quant.hadamard.duplex_hadamard import fp8_duplex_hadamard_f_and_b

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



def benchmark_with_shape(shape):
    M, N, K = shape
    device = 'cuda:0'
    dtype = torch.bfloat16
    qtype = torch.float8_e4m3fn  # torch.float8_e5m2
    n_repeat = 1000
    gpu = torch.cuda.get_device_properties(0).name


    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)
    y = torch.randn(M, N, dtype=dtype, device=device)
    x_q = x.to(qtype)
    w_q = w.to(qtype)
    y_q = y.to(qtype)
    B = 32
    hm = hadamard_matrix(B, dtype=dtype, device=device)

    org_out = fp16_forward(x, w.t())
    print(f'\ndevice:{gpu} M:{M} N:{N} K:{K}')

    # y = x @ w
    # dx = y @ wT
    # dwT = yT @ x

    # benchmark_func(triton_hadamard_nt, x, w, hm, n_repeat=n_repeat)
    # benchmark_func(triton_row_quant, x, n_repeat=n_repeat)
    # benchmark_func(triton_row_quant, w, n_repeat=n_repeat)

    benchmark_func(triton_hadamard_quant_nt, x, w, hm, n_repeat=n_repeat)
    benchmark_func(triton_hadamard_quant_nt_megatron, x, w, hm, n_repeat=n_repeat)
    benchmark_func(triton_hadamard_quant_nn, y, w, hm, n_repeat=n_repeat)
    benchmark_func(triton_hadamard_quant_nn_megatron, y, w, hm, n_repeat=n_repeat)
    benchmark_func(triton_hadamard_quant_tn, y, x, hm, n_repeat=n_repeat)
    benchmark_func(triton_hadamard_quant_tn_megatron, y, x, hm, n_repeat=n_repeat)

    # benchmark_func(triton_fused_hadamard, x, hm, hm_side=1, op_side=0)
    # benchmark_func(triton_fused_transpose_hadamard, x, hm, hm_side=1, op_side=0)
    # benchmark_func(triton_fused_hadamard_quant_nt, x,w,hm, n_repeat=n_repeat)
    # benchmark_func(triton_fused_hadamard_quant_nn, y,x,hm, n_repeat=n_repeat)
    # benchmark_func(triton_fused_hadamard_quant_tn, y,w,hm, n_repeat=n_repeat)
    
    # ref_time=benchmark_func(triton_hadamard_quant_nt_nn_tn, x,w,y,hm, n_repeat=n_repeat)
    # benchmark_func(triton_fused_hadamard_quant_nt_nn_tn, x,w,y,hm, n_repeat=n_repeat,ref_time=ref_time)


    # benchmark_func(triton_bit_hadamard_nt, x, w, hm, n_repeat=n_repeat)
    # benchmark_func(triton_bit_hadamard_quant_nt, x, w, hm, n_repeat=n_repeat)
    # benchmark_func(triton_bit_hadamard_quant_nn, y, w.t().contiguous(), hm, n_repeat=n_repeat)
    # benchmark_func(triton_bit_hadamard_quant_tn, y.t().contiguous(), x.t().contiguous(), hm, n_repeat=n_repeat)


    # ref_time = benchmark_func(fp16_forward, x, w.t(), n_repeat=n_repeat, ref_flops=M*K*N*2)
    # benchmark_func(hadamard_quant_forward, x, w, hm, n_repeat=n_repeat, ref_flops=M*K*N*2, ref_time=ref_time)
    # ref_time = benchmark_func(fp16_backward, y, w, n_repeat=n_repeat, ref_flops=M*K*N*2)
    # benchmark_func(hadamard_quant_backward, y, w, hm, n_repeat=n_repeat, ref_flops=M*K*N*2, ref_time=ref_time)
    # ref_time = benchmark_func(fp16_update, y, x, n_repeat=n_repeat, ref_flops=M*K*N*2)
    # benchmark_func(hadamard_quant_update, y, x, hm, n_repeat=n_repeat, ref_flops=M*K*N*2, ref_time=ref_time)

    ref_time = benchmark_func(fp16_f_and_b, x, w, y, n_repeat=n_repeat, ref_flops=M*K*N*6)
    benchmark_func(fp8_hadamard_f_and_b, x, w, y, hm, n_repeat=n_repeat, ref_time=ref_time,ref_flops=M*K*N*6)
    benchmark_func(fp8_fused_hadamard_f_and_b, x, w, y, hm, n_repeat=n_repeat, ref_time=ref_time,ref_flops=M*K*N*6)
    benchmark_func(fp8_hadamard_f_and_b_megatron, x, w, y, hm, n_repeat=n_repeat, ref_time=ref_time,ref_flops=M*K*N*6)
    benchmark_func(fp8_duplex_hadamard_f_and_b, x, w, y, hm, n_repeat=n_repeat, ref_time=ref_time,ref_flops=M*K*N*6)



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


benchmark_with_shape([8192, 6144, 4096])

# for shape in [[8192, 6144, 4096], [8192, 4096, 4096], [8192, 13312, 4096], [8192, 4096, 13312],
#             [8192, 10240, 8192],[8192, 8192, 8192],[8192, 34048, 8192],[8192, 4096, 34048]]:
#     benchmark_with_shape(shape)
