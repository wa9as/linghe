import torch 

from flops.quant.hadamard import *
from flops.utils.util import *


qtype = torch.float8_e4m3fn
device = 'cuda:0'
dtype = torch.bfloat16


x,w,y= read_and_tile('/mntnlp/nanxiao/dataset/flops/down_fb_1.pkl', tile=True)


B = 64
hm = hadamard_matrix(B, dtype=dtype, device=device, norm=True)

batch_size, in_dim = x.shape 
out_dim, in_dim = w.shape

org_out = fp16_forward(x, w.t())

# modes = ['direct', 'tensor', 'channel']
modes = ['channel']
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

        impl = 'seperate'
        if impl == 'seperate':
            opt_out,xq,wq,x_scale,w_scale = hadamard_quant_forward_debug(x,w,hm)
            quant_check(org_out, xq, wq, opt_out, 'hadamard_quant_forward')

            opt_out,yq,wq,y_scale,w_scale = hadamard_quant_backward_debug(y,w,hm)
            quant_check(y@w, yq, wq, opt_out, 'hadamard_quant_backward')

            opt_out,yq,xq,y_scale,x_scale = hadamard_quant_update_debug(y,x,hm)
            quant_check(y.t()@x, yq, xq, opt_out, 'hadamard_quant_update')
        elif impl == 'fused':
            output,x_q,x_s,w_q,w_s = fused_hadamard_quant_forward_debug(x, w, hm)
            quant_check(org_out, x_q, w_q, output, 'fuse_hadamard_quant_forward')

            output,y_q,y_s,w_q,w_s = fused_hadamard_quant_backward_debug(y, w, hm)
            quant_check(y@w, y_q, w_q, output, 'fuse_hadamard_quant_backward')

            output,y_q,y_s,x_q,x_s = fused_hadamard_quant_update_debug(y,x, hm)
            quant_check(y.t()@x, y_q, x_q, output, 'fuse_hadamard_quant_update')

        elif impl == 'bit':
            output,x_bt,w_bt,x_q,w_q,x_scale,w_scale = bit_hadamard_quant_forward_debug(x, w, hm)
            quant_check(org_out, x_q, w_q, output, 'bit_hadamard_quant_forward')

            output,y_bt,y_q,w_q,y_scale,w_scale=bit_hadamard_quant_backward_debug(y, w_bt, hm)
            quant_check(y@w, y_q, w_q, output, 'bit_hadamard_quant_backward')

            output,y_q,x_q,y_scale,x_scale=bit_hadamard_quant_update_debug(y_bt,x_bt, hm)
            quant_check(y.t()@x, y_q, x_q, output, 'bit_hadamard_quant_update')



