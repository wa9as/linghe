
import torch 
from flops.utils.util import *
from flops.utils.benchmark import benchmark_func
from flops.quant.smooth import *
from flops.quant.quantize import *



batch_size = 4096
in_dim = 4096
out_dim = 4096
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



