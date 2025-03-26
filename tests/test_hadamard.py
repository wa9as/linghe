import torch 

from flops.quant.hadamard import *
from flops.quant.quantize import *
from flops.utils.util import *
from flops.utils.benchmark import benchmark_func



qtype = torch.float8_e4m3fn
device = 'cuda:0'
dtype = torch.bfloat16


x,w,y= read_and_tile('down_fb_1.pkl', tile=True)


B = 64
hm = hadamard_matrix(B, dtype=dtype, device=device, norm=True)

batch_size, in_dim = x.shape 
out_dim, in_dim = w.shape

org_out = fp16_forward(x, w.t())

modes = ['direct', 'tensor', 'channel']
for mode in modes:
    if mode == 'direct':
        xq,wq,x_scale,w_scale = torch_hadamard_direct_quant(x,w,hm,qtype)
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

    elif mode == 'tensor':
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


