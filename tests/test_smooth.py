import torch 
from flops.utils.util import *
from flops.utils.benchmark import *


device = 'cuda:0'
dtype = torch.bfloat16

# x,w,y= read_and_tile('/mntnlp/nanxiao/dataset/flops/down_fb_1.pkl', tile=True)
x,w,y= read_and_tile('/mntnlp/nanxiao/dataset/tmp_flops/forward_1.pkl', tile=False)


M, K = x.shape 
N, K = w.shape

org_out = fp16_forward(x, w.t())

modes = ['rescale']


if 'torch_tensor' in modes:
    xq, wq, scale, rescale = torch_smooth_tensor_quant(x,w,torch.float8_e4m3fn)
    opt_out =  (xq.to(dtype)@wq.to(dtype).t())/(rescale**2).to(dtype)
    quant_check(org_out, xq, wq, opt_out,'tensor')

if 'torch_channel' in modes:
    xq, wq, x_scale, w_scale = torch_smooth_quant(x,w,torch.float8_e4m3fn)
    # print(f"x.size() {x.size()}")
    # print(f"w.size() {w.size()}")
    # print(f"x_scale.size() {x_scale.size()}")
    # print(f"w_scale.size() {w_scale.size()}")

    xdq = xq.to(dtype)*x_scale
    wdq = wq.to(dtype)*w_scale
    opt_out = xdq@wdq.t()
    quant_check(org_out, xq, wq, opt_out,'torch_channel')

if 'torch_channel_backward'  in modes:
    # print(f"y.size() {y.size()}")
    # print(f"w.size() {w.size()}")
    yq, wq, y_scale, w_scale = torch_smooth_quant(y,w.t(),torch.float8_e4m3fn)
    ydq = yq.to(dtype)*y_scale
    wdq = wq.to(dtype)*w_scale
    # print(f"ydq.size() {ydq.size()}")
    # print(f"wdq.size() {wdq.size()}")
    opt_out = ydq@wdq.t()
    quant_check(y@w, yq, wq, opt_out, 'torch_channel_backward')

if 'torch_channel_update' in modes:
    # print(f"y.size() {y.size()}")
    # print(f"x.size() {x.size()}")
    yq, xq, y_scale, x_scale = torch_smooth_quant(y.t(),x.t(),torch.float8_e4m3fn)

    quant_check(org_out, yq, xq, org_out,'torch_channel_update')


if 'torch_reuse' in modes:
    xq,wq,yq,ytq,o, dx, dw = torch_reuse_smooth_quant_f_and_b(x,w,y)
    ref_o, ref_dx, ref_dw = fp16_f_and_b(x,w,y)
    mode = 'torch_reuse'
    output_check(ref_o, o, mode)
    output_check(ref_dx,dx,  mode)
    output_check(ref_dw, dw,  mode)
    quant_check(ref_o, xq, wq, o,mode)
    quant_check(ref_dx, yq, wq, dx,mode)
    quant_check(ref_dw, ytq, xq, dw,mode)



if 'rescale' in modes:
    from flops.quant.smooth.reused_smooth import *
    org_smooth_scale = torch.ones((K,),device=device,dtype=torch.float32)
    y_q,y_scale = triton_reused_smooth_quant(y, org_smooth_scale, reverse=True, pad_scale=False, round_scale=True)
    transpose_smooth_scale = torch.ones((M,),device=device,dtype=torch.float32)
    yt_q,yt_scale = triton_reused_transpose_pad_rescale_smooth_quant(y_q, org_smooth_scale, y_scale, transpose_smooth_scale, reverse=True, pad=False)
    y_tmp = (y.to(torch.float32)*transpose_smooth_scale[:,None]).t().contiguous()
    yt_scale_ref = torch.exp2(torch.ceil(torch.log2(y_tmp.amax(dim=1)/448)))
    yt_q_ref = (y_tmp/yt_scale_ref[:,None]).to(torch.float8_e4m3fn).to(torch.float32)
    output_check(yt_q_ref, yt_q.float(), 'rescale_data')
    output_check(yt_scale_ref, yt_scale.float(), 'rescale_scale')


# modes = ['direct','global','channel','channel_backward', 'channel_update', 'dynamic','reuse']
# modes = ['channel', 'channel_backward', 'channel_update']
modes = ['seperate']
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
        # print(f"x.size() {x.size()}")
        # print(f"w.size() {w.size()}")
        # print(f"x_scale.size() {x_scale.size()}")
        # print(f"w_scale.size() {w_scale.size()}")
    
        xdq = xq.to(dtype)*x_scale
        wdq = wq.to(dtype)*w_scale
        opt_out = xdq@wdq.t()
        quant_check(org_out, xq, wq, opt_out,mode)

    elif mode == 'channel_backward':
        # print(f"y.size() {y.size()}")
        # print(f"w.size() {w.size()}")
        yq, wq, y_scale, w_scale = torch_smooth_quant(y,w.t(),torch.float8_e4m3fn)
        ydq = yq.to(dtype)*y_scale
        wdq = wq.to(dtype)*w_scale
        # print(f"ydq.size() {ydq.size()}")
        # print(f"wdq.size() {wdq.size()}")
        opt_out = ydq@wdq.t()
        quant_check(y@w, yq, wq, opt_out, mode)

    elif mode == 'channel_update':
        # print(f"y.size() {y.size()}")
        # print(f"x.size() {x.size()}")
        yq, xq, y_scale, x_scale = torch_smooth_quant(y.t(),x.t(),torch.float8_e4m3fn)

        quant_check(org_out, yq, xq, org_out,mode)
    
    elif mode == 'seperate':
        from flops.quant.smooth.seperate_smooth import *

        opt_out,xq,wq,x_scale,w_scale = seperate_smooth_quant_forward(x,w)
        quant_check(org_out, xq, wq, opt_out, 'seperate_smooth_quant_forward')

        opt_out,yq,wq,y_scale,w_scale = seperate_smooth_quant_backward(y,w)
        quant_check(y@w, yq, wq, opt_out, 'seperate_smooth_quant_backward')

        opt_out,yq,xq,y_scale,x_scale = seperate_smooth_quant_update(y,x)
        quant_check(y.t()@x, yq, xq, opt_out, 'seperate_smooth_quant_update')

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

        impl = 'none'
        if impl == 'reuse':
            o, dx, dw, smooth_scale = reused_smooth_quant_f_and_b(x,w,y)
            output_check(ref_o, o, mode)
            output_check(ref_dx,dx,  mode)
            output_check(ref_dw, dw,  mode)

            o, dx, dw, smooth_scale = reused_smooth_quant_f_and_b(x,w,y, smooth_scale=smooth_scale)
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

