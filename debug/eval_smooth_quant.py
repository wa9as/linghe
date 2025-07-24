import torch

from flops.utils.util import *

device = 'cuda:0'
dtype = torch.bfloat16

if False:
    # x,w,y= read_and_tile('/mntnlp/nanxiao/dataset/flops/down_fb_1.pkl', tile=True)
    x, w, y = read_and_tile('/mntnlp/nanxiao/dataset/tmp_flops/forward_1.pkl',
                            tile=False)
    M, K = x.shape
    N, K = w.shape
else:
    M, N, K = 4096 * 4, 8192, 8192
    x = torch.randn((M, K), dtype=dtype, device=device) * 3
    w = torch.randn((N, K), dtype=dtype, device=device)
    y = torch.randn((M, N), dtype=dtype, device=device)

org_out = fp16_forward(x, w.t())

if False:
    xq, wq, scale, rescale = torch_smooth_tensor_quant(x, w,
                                                       torch.float8_e4m3fn)
    opt_out = (xq.to(dtype) @ wq.to(dtype).t()) / (rescale ** 2).to(dtype)
    quant_check(org_out, xq, wq, opt_out, 'tensor')

if False:
    xq, wq, x_scale, w_scale = torch_smooth_quant(x, w, torch.float8_e4m3fn)
    # print(f"x.size() {x.size()}")
    # print(f"w.size() {w.size()}")
    # print(f"x_scale.size() {x_scale.size()}")
    # print(f"w_scale.size() {w_scale.size()}")

    xdq = xq.to(dtype) * x_scale
    wdq = wq.to(dtype) * w_scale
    opt_out = xdq @ wdq.t()
    quant_check(org_out, xq, wq, opt_out, 'torch_channel')

if False:
    # print(f"y.size() {y.size()}")
    # print(f"w.size() {w.size()}")
    yq, wq, y_scale, w_scale = torch_smooth_quant(y, w.t(), torch.float8_e4m3fn)
    ydq = yq.to(dtype) * y_scale
    wdq = wq.to(dtype) * w_scale
    # print(f"ydq.size() {ydq.size()}")
    # print(f"wdq.size() {wdq.size()}")
    opt_out = ydq @ wdq.t()
    quant_check(y @ w, yq, wq, opt_out, 'torch_channel_backward')

if False:
    # print(f"y.size() {y.size()}")
    # print(f"x.size() {x.size()}")
    yq, xq, y_scale, x_scale = torch_smooth_quant(y.t(), x.t(),
                                                  torch.float8_e4m3fn)

    quant_check(org_out, yq, xq, org_out, 'torch_channel_update')

if False:
    xq, wq, yq, ytq, o, dx, dw = torch_reuse_smooth_quant_f_and_b(x, w, y)
    ref_o, ref_dx, ref_dw = fp16_f_and_b(x, w, y)
    mode = 'torch_reuse'
    output_check(ref_o, o, mode)
    output_check(ref_dx, dx, mode)
    output_check(ref_dw, dw, mode)
    quant_check(ref_o, xq, wq, o, mode)
    quant_check(ref_dx, yq, wq, dx, mode)
    quant_check(ref_dw, ytq, xq, dw, mode)
