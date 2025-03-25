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
    bs = x.size(0)
    m = 2**(int(math.log2(bs)+1))
    x = torch.cat([x]*2,0)[:m].contiguous()
    y = torch.cat([y]*2,0)[:m].contiguous()


B = 64
hm = hadamard_matrix(B, dtype=dtype, device=device, norm=True)

batch_size, in_dim = x.shape 
out_dim, in_dim = w.shape

org_out = fp16_forward(x, w.t())

print(f'\n{batch_size=} {in_dim=} {out_dim=} {dtype=} ' \
        f'x.max={x.abs().max().item():.3f} x.mean={x.abs().mean().item():.3f} ' \
        f'w.max={w.abs().max().item():.3f} '\
        f'w.mean={w.abs().mean().item():.3f} y.max={y.abs().max().item():.3f}')

modes = ['tensor', 'channel']
for mode in modes:
    if mode == 'tensor':
        xq,wq,x_scale,w_scale = torch_hadamard_tensor_quant(x,w,hm,qtype)
        xdq = xq.to(torch.float32)*x_scale
        wdq = wq.to(torch.float32).t()*w_scale
        opt_out = xdq@wdq

        # print('x',x[:4,:4])
        # print('w',w[:4,:4])
        # print('xq',xq[:4,:4])
        # print('wq',wq[:4,:4])
        # print('org',org_out[:4,:4])
        # print('opt',opt_out[:4,:4])

        quant_check(org_out, xq, wq, opt_out, mode)

    elif mode == 'channel':

        xq,wq,x_scale,w_scale=torch_hadamard_channel_quant(x,w,hm,qtype)
        xdq = xq.to(torch.float32)*x_scale
        wdq = wq.to(torch.float32).t()*w_scale
        opt_out = xdq@wdq
        quant_check(org_out, xq, wq, opt_out, mode)


        # xb, wb = triton_hadamard_nt(x,w,hm)
        # opt_out = xb@wb.t()
        # quant_check(org_out, xb, wb, opt_out, mode)


        opt_out,xq,wq,x_scale,w_scale = hadamard_quant_forward(x,w,hm)
        quant_check(org_out, xq, wq, opt_out, 'hadamard_quant_forward')

        opt_out,yq,wq,y_scale,w_scale = hadamard_quant_backward(y,w,hm)
        quant_check(y@w, yq, wq, opt_out, 'hadamard_quant_backward')

        opt_out,yq,xq,y_scale,x_scale = hadamard_quant_update(y,x,hm)
        quant_check(y.t()@x, yq, xq, opt_out, 'hadamard_quant_update')


