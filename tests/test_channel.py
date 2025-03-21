import torch 
from flops.utils.util import fp16_forward, quant_check, tensor_quant, channel_quant



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

modes = ['tensor','channel']
for mode in modes:
    if mode == 'tensor':
        xq, wq, x_scale, w_scale = tensor_quant(x,w,qtype)
        opt_out =  (xq.to(dtype)@wq.to(dtype).t())*(x_scale*w_scale).to(dtype)
        quant_check(org_out, xq, wq, opt_out,mode)

    elif mode == 'channel':
        xq, wq, x_scale, w_scale = channel_quant(x,w,qtype)
        opt_out =  (xq.to(dtype)@wq.to(dtype).t())*x_scale.to(dtype)*w_scale.to(dtype)[:,0]
        quant_check(org_out, xq, wq, opt_out,mode)