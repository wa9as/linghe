import torch 
from flops.quant.hadamard import *
from flops.quant.quantize import *
from flops.utils.util import *
from flops.utils.benchmark import benchmark_func



qtype = torch.float8_e4m3fn
device = 'cuda:0'
dtype = torch.bfloat16

x,w,y= read_and_tile('down_fb_1.pkl', tile=True)


batch_size, in_dim = x.shape 
out_dim, in_dim = w.shape

org_out = fp16_forward(x, w.t())

print(f'\n{batch_size=} {in_dim=} {out_dim=} {dtype=} ' \
        f'x.max={x.abs().max().item():.3f} x.mean={x.abs().mean().item():.3f} w.max={w.abs().max().item():.3f} '\
        f'w.mean={w.abs().mean().item():.3f} y.max={y.abs().max().item():.3f}')

modes = ['direct','global','channel']
for mode in modes:
    if mode == 'direct':
        xq, wq, scale = torch_smooth_direct_quant(x,w,qtype)
        opt_out =  (xq.to(dtype)@wq.to(dtype).t())
        quant_check(org_out, xq, wq, opt_out,mode)

    elif mode == 'global':
        xq, wq, scale, rescale = torch_smooth_tensor_quant(x,w,qtype)
        opt_out =  (xq.to(dtype)@wq.to(dtype).t())/(rescale**2).to(dtype)
        quant_check(org_out, xq, wq, opt_out,mode)

    else:
        xq, wq, x_scale, w_scale = torch_smooth_quant(x,w,qtype)
        xdq = xq.to(dtype)*x_scale
        wdq = wq.to(dtype)*w_scale
        opt_out = xdq@wdq.t()
        quant_check(org_out, xq, wq, opt_out,mode)
