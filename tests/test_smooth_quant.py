import torch 
from flops.utils.util import *
from flops.utils.benchmark import *


device = 'cuda:0'
dtype = torch.bfloat16

if False:
    # x,w,y= read_and_tile('/mntnlp/nanxiao/dataset/flops/down_fb_1.pkl', tile=True)
    x,w,y= read_and_tile('/mntnlp/nanxiao/dataset/tmp_flops/forward_1.pkl', tile=False)
    M, K = x.shape 
    N, K = w.shape
else:
    M,N,K=4096*4,8192,8192
    x = torch.randn((M,K),dtype=dtype,device=device)*3
    w = torch.randn((N,K),dtype=dtype,device=device)
    y = torch.randn((M,N),dtype=dtype,device=device)

org_out = fp16_forward(x, w.t())



if False:
    xq, wq, scale, rescale = torch_smooth_tensor_quant(x,w,torch.float8_e4m3fn)
    opt_out =  (xq.to(dtype)@wq.to(dtype).t())/(rescale**2).to(dtype)
    quant_check(org_out, xq, wq, opt_out,'tensor')

if False:
    xq, wq, x_scale, w_scale = torch_smooth_quant(x,w,torch.float8_e4m3fn)
    # print(f"x.size() {x.size()}")
    # print(f"w.size() {w.size()}")
    # print(f"x_scale.size() {x_scale.size()}")
    # print(f"w_scale.size() {w_scale.size()}")

    xdq = xq.to(dtype)*x_scale
    wdq = wq.to(dtype)*w_scale
    opt_out = xdq@wdq.t()
    quant_check(org_out, xq, wq, opt_out,'torch_channel')

if False:
    # print(f"y.size() {y.size()}")
    # print(f"w.size() {w.size()}")
    yq, wq, y_scale, w_scale = torch_smooth_quant(y,w.t(),torch.float8_e4m3fn)
    ydq = yq.to(dtype)*y_scale
    wdq = wq.to(dtype)*w_scale
    # print(f"ydq.size() {ydq.size()}")
    # print(f"wdq.size() {wdq.size()}")
    opt_out = ydq@wdq.t()
    quant_check(y@w, yq, wq, opt_out, 'torch_channel_backward')

if False:
    # print(f"y.size() {y.size()}")
    # print(f"x.size() {x.size()}")
    yq, xq, y_scale, x_scale = torch_smooth_quant(y.t(),x.t(),torch.float8_e4m3fn)

    quant_check(org_out, yq, xq, org_out,'torch_channel_update')


if False:
    xq,wq,yq,ytq,o, dx, dw = torch_reuse_smooth_quant_f_and_b(x,w,y)
    ref_o, ref_dx, ref_dw = fp16_f_and_b(x,w,y)
    mode = 'torch_reuse'
    output_check(ref_o, o, mode)
    output_check(ref_dx,dx,  mode)
    output_check(ref_dw, dw,  mode)
    quant_check(ref_o, xq, wq, o,mode)
    quant_check(ref_dx, yq, wq, dx,mode)
    quant_check(ref_dw, ytq, xq, dw,mode)

if True:
    from flops.quant.smooth.reused_smooth import *
    from flops.quant.smooth.seperate_smooth import *
    smooth_scale = torch.randn((K,),device=device,dtype=torch.float32).abs()
    tmp = x.float()/smooth_scale
    scales = tmp.abs().amax(1)/448
    scales_ref = torch.exp2(torch.ceil(torch.log2(scales)))
    x_q_ref = (tmp/scales_ref[:,None]).to(torch.float8_e4m3fn)
    x_q, x_scale = triton_depracated_tokenwise_reused_smooth_quant(x, smooth_scale, reverse=False, round_scale=True)
    output_check(x_q_ref.float(), x_q.float(), 'data')
    output_check(scales_ref, x_scale, 'scale')
    # output_check(x_q_ref[-1], x_q.float()[-1], 'data[-1]')
    # print(x_scale[-1])

    x_q_, x_scale_ = triton_reused_smooth_quant(x, smooth_scale, reverse=False, round_scale=True)
    output_check(x_q_.float(), x_q.float(), 'data')
    output_check(x_scale_, x_scale, 'scale')

if False:

    from flops.quant.smooth.reused_smooth import *
    from flops.quant.smooth.seperate_smooth import *
    if True:
        dic = torch.load('/ossfs/workspace/tmp/yst.bin')
        y, smooth_scale, transpose_smooth_scale = dic['y'], dic['smooth_scale'], dic['transpose_smooth_scale']
        M, N = y.shape
    yt_q,yt_scale = triton_reused_transpose_pad_smooth_quant(y, transpose_smooth_scale, reverse=True, pad=True)
    yt_q_smooth = y*transpose_smooth_scale[:,None]
    yt_scale_ref = yt_q_smooth.abs().amax(0)/448
    yt_q_ref = (yt_q_smooth/yt_scale_ref).t().contiguous().to(torch.float8_e4m3fn)
    output_check(yt_q_ref.float(), yt_q.float()[:,:M], 'data')
    output_check(yt_scale_ref.float(), yt_scale.float(), 'scale')




if False:
    from flops.quant.smooth.reused_smooth import *
    from flops.quant.smooth.seperate_smooth import *
    org_smooth_scale = torch.ones((N,),device=device,dtype=torch.float32)
    y_q, y_scale = triton_reused_smooth_quant(y, org_smooth_scale, reverse=True, pad_scale=False, round_scale=True)


    transpose_smooth_scale = torch.ones((M,),device=device,dtype=torch.float32)
    yt_q,yt_scale = triton_reused_transpose_pad_rescale_smooth_quant(y_q, org_smooth_scale, y_scale, transpose_smooth_scale, reverse=True, pad=False)
    y_tmp = (y.to(torch.float32)*transpose_smooth_scale[:,None]).t().contiguous()
    yt_scale_ref = torch.exp2(torch.ceil(torch.log2(y_tmp.abs().amax(dim=1)/448+1e-30)))
    yt_q_ref = (y_tmp/yt_scale_ref[:,None]).to(torch.float8_e4m3fn).to(torch.float32)
    output_check(yt_q_ref, yt_q.float(), 'rescale_data')
    output_check(yt_scale_ref, yt_scale.float(), 'rescale_scale')


if False:
    from flops.quant.smooth.reused_smooth import *

    n_expert = 32
    smooth_scales = 1+10*torch.rand((n_expert,K),device=device,dtype=torch.float32)
    token_count_per_expert_list = [M//n_expert]*n_expert
    token_count_per_expert = torch.tensor(token_count_per_expert_list, device=device)
    x_q,x_scale,x_maxs = triton_batch_smooth_quant(x, smooth_scales, token_count_per_expert, reverse=False, round_scale=False, calibrate=True)

    def torch_split_smooth_quant(x_split,smooth_scales,calibrate=False):
        x_qs = []
        x_scales = []
        x_maxs = []
        for i, x_ in enumerate(x_split):
            x_maxs.append(x_.amax(0))
            x_smooth = x_/smooth_scales[i]
            x_scale_ = x_smooth.float().abs().amax(1)/448
            x_q_ = (x_smooth/x_scale_[:,None]).to(torch.float8_e4m3fn)
            x_qs.append(x_q_)
            x_scales.append(x_scale_)
        x_maxs = torch.stack(x_maxs, 0)
        return x_qs,x_scales,x_maxs

    def triton_split_smooth_quant(x_split,smooth_scales):
        x_qs = []
        x_scales = []
        for i, x_ in enumerate(x_split):
            x_q_,x_scale_ = triton_reused_smooth_quant(x_,smooth_scales[i])
            x_qs.append(x_q_)
            x_scales.append(x_scale_)
        return x_qs,x_scales

    x_split = torch.split(x, token_count_per_expert_list)
    x_q_ref, x_scale_ref, x_maxs_ref = torch_split_smooth_quant(x_split,smooth_scales)
    x_q_ref = torch.cat([x.view(torch.uint8) for x in x_q_ref], 0).view(torch.float8_e4m3fn)
    x_scale_ref = torch.cat(x_scale_ref, 0)
    output_check(x_q_ref.float(), x_q.float(), 'data')
    output_check(x_scale_ref.float(), x_scale.float(), 'scale')
    output_check(x_maxs_ref.float(), x_maxs.float(), 'maxs')

    n_repeat = 100
    ref_time = benchmark_func(triton_split_smooth_quant, x_split, smooth_scales, n_repeat=n_repeat)
    benchmark_func(triton_batch_smooth_quant, x, smooth_scales, token_count_per_expert, reverse=False, round_scale=False, n_repeat=n_repeat, ref_time=ref_time)
    benchmark_func(triton_batch_smooth_quant, x, smooth_scales, token_count_per_expert, reverse=False, round_scale=False, calibrate=True, n_repeat=n_repeat, ref_time=ref_time)

