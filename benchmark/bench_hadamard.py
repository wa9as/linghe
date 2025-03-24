import torch 
from flops.utils.util import *
from flops.utils.benchmark import benchmark_func
from flops.quant.hadamard import *
from flops.quant.quantize import *


batch_size = 4096  # M
out_dim = 4096  # N
in_dim = 4096  # K
# batch_size = 8192  # M
# out_dim = 4096  # N
# in_dim = 13312  # K
# batch_size = 8192  # M
# out_dim = 8192  # N
# in_dim = 8192  # K
device = 'cuda:0'
dtype = torch.bfloat16
qtype = torch.float8_e4m3fn  # torch.float8_e5m2
n_repeat = 1000

x = torch.randn(batch_size, in_dim, dtype=dtype, device=device)
w = torch.randn(out_dim, in_dim, dtype=dtype, device=device)
y = torch.randn(batch_size, out_dim, dtype=dtype, device=device)
x_f8 = x.to(qtype)
w_f8 = w.to(qtype)
y_f8 = y.to(qtype)
B = 32
hm = hadamard_matrix(B, dtype=dtype, device=device)

org_out = fp16_forward(x, w.t())

# benchmark_func(fp8_transpose, x_f8, n_repeat=n_repeat)
# benchmark_func(fp16_transpose, x, n_repeat=n_repeat)

benchmark_func(triton_fused_hadamard, x, hm, hm_side=1, op_side=0)
benchmark_func(triton_fused_transpose_hadamard, x, hm, hm_side=1, op_side=0)

benchmark_func(triton_fused_hadamard_quant_nt, x,w,hm, n_repeat=n_repeat)



benchmark_func(triton_hadamard_nt, x,w,hm, n_repeat=n_repeat)
benchmark_func(triton_row_quant, x, n_repeat=n_repeat)
benchmark_func(triton_hadamard_quant_nt, x,w,hm, n_repeat=n_repeat)
benchmark_func(triton_hadamard_quant_tn, y,x,hm, n_repeat=n_repeat)
benchmark_func(triton_hadamard_quant_nn, y,w,hm, n_repeat=n_repeat)

# benchmark_func(trival_fp8_gemm, x_f8, w_f8, torch.bfloat16, n_repeat=n_repeat)
# benchmark_func(persistent_fp8_gemm, x_f8, w_f8.t(), torch.bfloat16, n_repeat=n_repeat)
# benchmark_func(nt_fp8_gemm, x_f8,w_f8,torch.bfloat16, n_repeat=n_repeat)

# ref_time = benchmark_func(fp16_forward, x, w.t(), n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
# benchmark_func(hadamard_quant_forward, x, w, hm, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)
# ref_time = benchmark_func(fp16_update, y, x, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
# benchmark_func(hadamard_quant_update, y, x, hm, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)
# ref_time = benchmark_func(fp16_backward, y, w, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
# benchmark_func(hadamard_quant_backward, y, w, hm, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)

# ref_time = benchmark_func(fp16_f_and_b, x,w,y, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*6)
# benchmark_func(fp8_hadamard_f_and_b, x,w,y, hm, n_repeat=n_repeat, ref_time=ref_time,ref_flops=batch_size*in_dim*out_dim*6)