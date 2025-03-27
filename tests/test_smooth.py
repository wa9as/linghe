import torch 
from flops.quant.smooth import *
from flops.utils.util import *
from flops.utils.benchmark import *



device = 'cuda:0'
dtype = torch.bfloat16

x,w,y= read_and_tile('/ossfs/workspace/flops/tests/down_fb_1.pkl', tile=True)


batch_size, in_dim = x.shape 
out_dim, in_dim = w.shape

org_out = fp16_forward(x, w.t())

# modes = ['direct','global','channel','dynamic','reuse']
modes = ['reuse']
for mode in modes:
    if mode == 'direct':
        xq, wq, scale = torch_smooth_direct_quant(x,w,torch.float8_e4m3fn)
        opt_out =  (xq.to(dtype)@wq.to(dtype).t())
        quant_check(org_out, xq, wq, opt_out,mode)

    elif mode == 'global':
        xq, wq, scale, rescale = torch_smooth_tensor_quant(x,w,torch.float8_e4m3fn)
        opt_out =  (xq.to(dtype)@wq.to(dtype).t())/(rescale**2).to(dtype)
        quant_check(org_out, xq, wq, opt_out,mode)

    elif mode == 'channel':
        xq, wq, x_scale, w_scale = torch_smooth_quant(x,w,torch.float8_e4m3fn)
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
        xq,wq,yq,ytq,o, dx, dw = torch_reuse_smooth_quant_f_and_b(x,w,y)
        ref_o, ref_dx, ref_dw = fp16_f_and_b(x,w,y)
        output_check(ref_o, o, mode)
        output_check(ref_dx,dx,  mode)
        output_check(ref_dw, dw,  mode)
        quant_check(ref_o, xq, wq, o,mode)
        quant_check(ref_dx, yq, wq, dx,mode)
        quant_check(ref_dw, ytq, xq, dw,mode)

        impl = 'reuse'
        if impl == 'reuse':
            o, dx, dw = reused_smooth_quant_f_and_b(x,w,y)
            output_check(ref_o, o, mode)
            output_check(ref_dx,dx,  mode)
            output_check(ref_dw, dw,  mode)

            # print(f'{ref_dx=}')
            # print(f'{dx=}')

            # sm = torch.rand((out_dim,),dtype=torch.float32,device=device)
            # y_q,y_s = triton_slide_smooth_quant(y, sm)
            # y_max = (y.float()*sm).abs().amax(dim=-1,keepdim=True)
            # y_s_ref= y_max/448.0 
            # y_q_ref = y.float()/y_s_ref
            # print(f'{y_s[:,0]=} {y_s_ref[:,0]=}')
            # print(f'{y_q=} {y_q_ref=}')

            # xqt = triton_transpose(xq)
            # print((xqt.float().t().contiguous()-xq.float()).abs().mean()/x.float().abs().mean())

