import math
import torch 
from flops.utils.util import *


device = 'cuda:0'
dtype = torch.bfloat16

x,w,y= read_and_tile('down_fb_1.pkl', tile=True)

batch_size, in_dim = x.shape 
out_dim, in_dim = w.shape

org_out = fp16_forward(x, w.t())

modes = ['tensor','channel']
for mode in modes:
    if mode == 'tensor':
        xq, wq, x_scale, w_scale = torch_tensor_quant(x,w,torch.float8_e4m3fn)
        opt_out =  (xq.to(dtype)@wq.to(dtype).t())*(x_scale*w_scale).to(dtype)
        quant_check(org_out, xq, wq, opt_out,mode)

    elif mode == 'channel':
        xq, wq, x_scale, w_scale = torch_channel_quant(x,w,torch.float8_e4m3fn)
        opt_out =  (xq.to(dtype)@wq.to(dtype).t())*x_scale.to(dtype)*w_scale.to(dtype)[:,0]
        quant_check(org_out, xq, wq, opt_out,mode)