import math
import torch 
from flops.utils.util import *


device = 'cuda:0'
dtype = torch.bfloat16

x,w,y= read_and_tile('/mntnlp/nanxiao/dataset/flops/down_fb_1.pkl', tile=True)

batch_size, in_dim = x.shape 
out_dim, in_dim = w.shape

org_out = fp16_forward(x, w.t())

modes = ['tensor','channel']
if 'tensor' in modes:
    xq, wq, x_scale, w_scale = torch_tensor_quant(x,w,torch.float8_e4m3fn)
    opt_out =  (xq.to(dtype)@wq.to(dtype).t())*(x_scale*w_scale).to(dtype)
    quant_check(org_out, xq, wq, opt_out,'tensor')

if 'channel' in modes:
    xq, wq, x_scale, w_scale = torch_channel_quant(x,w,torch.float8_e4m3fn)
    opt_out =  (xq.to(dtype)@wq.to(dtype).t())*x_scale.to(dtype)*w_scale.to(dtype)[:,0]
    quant_check(org_out, xq, wq, opt_out,'channel')

if 'channels' in modes:
    xq,wq,yq,ytq,o, dx, dw = torch_channel_quant_f_and_b(x,w,y)
    ref_o, ref_dx, ref_dw = fp16_f_and_b(x,w,y)
    mode = 'channels'
    output_check(ref_o, o, mode)
    output_check(ref_dx,dx,  mode)
    output_check(ref_dw, dw,  mode)
    quant_check(ref_o, xq, wq, o,mode)
    quant_check(ref_dx, yq, wq, dx,mode)
    quant_check(ref_dw, ytq, xq, dw,mode)