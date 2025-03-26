import torch 
from flops.quant.hadamard import *
from flops.quant.quantize import *
from flops.utils.util import *
from flops.utils.benchmark import *



qtype = torch.float8_e4m3fn
device = 'cuda:0'
dtype = torch.bfloat16

x,w,y= read_and_tile('/ossfs/workspace/flops/tests/down_fb_1.pkl', tile=True)


batch_size, in_dim = x.shape 
out_dim, in_dim = w.shape

org_out = fp16_forward(x, w.t())

# modes = ['direct','global','channel','dynamic','reuse']
modes = ['dynamic','reuse']
for mode in modes:
    if mode == 'direct':
        xq, wq, scale = torch_smooth_direct_quant(x,w,qtype)
        opt_out =  (xq.to(dtype)@wq.to(dtype).t())
        quant_check(org_out, xq, wq, opt_out,mode)

    elif mode == 'global':
        xq, wq, scale, rescale = torch_smooth_tensor_quant(x,w,qtype)
        opt_out =  (xq.to(dtype)@wq.to(dtype).t())/(rescale**2).to(dtype)
        quant_check(org_out, xq, wq, opt_out,mode)

    elif mode == 'channel':
        xq, wq, x_scale, w_scale = torch_smooth_quant(x,w,qtype)
        xdq = xq.to(dtype)*x_scale
        wdq = wq.to(dtype)*w_scale
        opt_out = xdq@wdq.t()
        quant_check(org_out, xq, wq, opt_out,mode)

    elif mode == 'dynamic':
        xq,wq,yq,ytq,o, dx, dw = dynamic_quant_f_and_b(x,w,y)
        ref_o, ref_dx, ref_dw = fp16_f_and_b(x,w,y)
        output_check(ref_o, o, mode)
        output_check(ref_dx,dx,  mode)
        output_check(ref_dw, dw,  mode)
        quant_check(ref_o, xq, wq, o,mode)
        quant_check(ref_dx, yq, wq, dx,mode)
        quant_check(ref_dw, ytq, xq, dw,mode)

    elif mode == 'reuse':
        xq,wq,yq,ytq,o, dx, dw = reuse_smooth_quant_f_and_b(x,w,y)
        ref_o, ref_dx, ref_dw = fp16_f_and_b(x,w,y)
        output_check(ref_o, o, mode)
        output_check(ref_dx,dx,  mode)
        output_check(ref_dw, dw,  mode)
        quant_check(ref_o, xq, wq, o,mode)
        quant_check(ref_dx, yq, wq, dx,mode)
        quant_check(ref_dw, ytq, xq, dw,mode)



