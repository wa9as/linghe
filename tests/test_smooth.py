import torch 
from flops.quant.hadamard import *
from flops.quant.quantize import *
from flops.utils.util import *
from flops.utils.benchmark import benchmark_func



qtype = torch.float8_e4m3fn
device = 'cuda:0'
dtype = torch.bfloat16


d = torch.load('down_fb_1.pkl', weights_only=True)
x = d['x'][0].to(dtype).to(device)
w = d['w'].to(dtype).to(device)
y = d['y'][0].to(dtype).to(device)


reshape = True
if reshape:
    x = torch.cat([x]*2,0)[:128,:2048].contiguous()
    y = torch.cat([y]*2,0)[:128,:4096].contiguous()
    w = w[:4096,:2048].contiguous()
    # x = x*0.0+torch.arange(128,dtype=dtype,device=device)[:,None]+1

batch_size, in_dim = x.shape 
out_dim, in_dim = w.shape

org_out = fp16_forward(x, w.t())

print(f'\n{batch_size=} {in_dim=} {out_dim=} {dtype=} ' \
        f'x.max={x.abs().max().item():.3f} x.mean={x.abs().mean().item():.3f} w.max={w.abs().max().item():.3f} '\
        f'w.mean={w.abs().mean().item():.3f} y.max={y.abs().max().item():.3f}')

modes = ['smooth_v0','smooth_v1','smooth_v2','smooth_v3']
for mode in modes:
    if mode == 'smooth_v0':
        xq, wq, scale = torch_smooth_quant_v0(x,w,qtype)
        opt_out =  (xq.to(dtype)@wq.to(dtype).t())
        quant_check(org_out, xq, wq, opt_out,mode)

    elif mode == 'smooth_v1':
        xq, wq, scale, rescale = torch_smooth_quant_v1(x,w,qtype)
        opt_out =  (xq.to(dtype)@wq.to(dtype).t())/(rescale**2).to(dtype)
        quant_check(org_out, xq, wq, opt_out,mode)

    elif mode == 'smooth_v2':

        xq, wq, w_scale, scale, rescale = torch_smooth_quant_v2_deprecated(x,w,qtype)
        opt_out =  (xq.to(dtype)@wq.to(dtype).t())/(rescale * rescale)*w_scale.t()
        quant_check(org_out, xq, wq, opt_out,mode)

    elif mode == 'smooth_v3':

        xq, wq, x_scale, w_scale = torch_smooth_quant_v3(x,w,qtype)
        xdq = xq.to(dtype)*x_scale
        wdq = wq.to(dtype)*w_scale
        opt_out = xdq@wdq.t()
        quant_check(org_out, xq, wq, opt_out,mode)
