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
B = 32
hm = hadamard_matrix(B, dtype=dtype, device=device)

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

modes = ['ht_v1']
for mode in modes:
    if mode == 'ht_v0':
        xq,wq,x_scale,w_scale=torch_ht_quant_v0(x,w,hm,qtype)
        opt_out =  (xq.to(dtype)@wq.to(dtype).t())*(x_scale*w_scale).to(dtype)

        # print('x',x[:4,:4])
        # print('w',w[:4,:4])
        # print('xq',xq[:4,:4])
        # print('wq',wq[:4,:4])
        # print('org',org_out[:4,:4])
        # print('opt',opt_out[:4,:4])

        quant_check(org_out, xq, wq, opt_out,mode)

    elif mode == 'ht_v1':

        # xb, wb = triton_ht_nt(x,w,hm)
        # opt_out = xb@wb.t()
        # quant_check(org_out, xb, wb, opt_out, mode)

        # xq,wq,x_scale,w_scale=torch_ht_quant_v1(x,w,hm,qtype)
        # opt_out = ((xq.to(dtype)*x_scale)@((wq.to(dtype)*w_scale).t())).to(dtype)
        # quant_check(org_out, xq, wq, opt_out, mode)

        opt_out,xq,wq,x_scale,w_scale = ht_quant_forward(x,w,hm)
        quant_check(org_out, xq, wq, opt_out, mode)

        opt_out,yq,xq,y_scale,x_scale = ht_quant_update(y,x,hm)
        quant_check(y.t()@x, yq, xq, opt_out, mode)

        opt_out,yq,wq,y_scale,w_scale = ht_quant_backward(y,w,hm)
        quant_check(y@w, yq, wq, opt_out, mode)



