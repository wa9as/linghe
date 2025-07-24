import torch

from flops.utils.util import *

qtype = torch.float8_e4m3fn
device = 'cuda:0'
dtype = torch.bfloat16
B = 128
M, N, K = 4096, 4096, 8192

x = torch.randn((M, K), dtype=dtype, device=device)
w = torch.randn((N, K), dtype=dtype, device=device)
xq, x_scale = torch_group_quant(x, B)
wq, w_scale = torch_block_quant(x, B)
x_scales = torch.repeat_interleave(x_scale, B, 1)
w_scales = torch.repeat_interleave(torch.repeat_interleave(w_scale, B, 0), B, 1)
xdq = xq.to(torch.float32) * x_scales
wdq = wq.to(torch.float32).t() * w_scales
opt_out = xdq @ wdq
org_out = fp16_forward(x, w.t())
quant_check(org_out, xq, wq, opt_out, 'BLOCK')
