import torch 

from flops.quant.hadamard import *
from flops.utils.util import *
from flops.quant.block.group import *
from flops.quant.block.block import *


qtype = torch.float8_e4m3fn
device = 'cuda:0'
dtype = torch.bfloat16

def group_block_quant(x, w):
    x_q, x_scale = group_quant(x)
    w_q, w_scale = block_quant(w)
    # output = torch._scaled_mm(x_q,
    #                             w_q.t(),
    #                             scale_a=x_scale,
    #                             scale_b=w_scale,
    #                             out_dtype=x.dtype,
    #                             use_fast_accum=True)
    output = torch.randn([8192, 7168], device=x.device)
    return output,x_q,w_q,x_scale,w_scale
    

# x,w,y= read_and_tile('/mntnlp/nanxiao/dataset/flops/down_fb_1.pkl', tile=True)
x,w,y= read_and_tile('/mntnlp/nanxiao/dataset/tmp_flops/forward_1.pkl', tile=True)


B = 128
hm = hadamard_matrix(B, dtype=dtype, device=device, norm=True)

batch_size, in_dim = x.shape 
out_dim, in_dim = w.shape

org_out = fp16_forward(x, w.t())

mode = 'torch'

if mode == 'torch':
  xq,wq,x_scale,w_scale = torch_tile_block_quant(x,w,B,qtype)
  x_scales = torch.repeat_interleave(x_scale, B, 1)
  w_scales = torch.repeat_interleave(torch.repeat_interleave(w_scale, B, 0), B, 1)
  xdq = xq.to(torch.float32)*x_scales
  wdq = wq.to(torch.float32).t()*w_scales
  opt_out = xdq@wdq
  quant_check(org_out, xq, wq, opt_out, mode)
elif mode == 'triton':
  pass


