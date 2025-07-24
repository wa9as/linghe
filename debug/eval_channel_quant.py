import torch

from flops.utils.util import *

device = 'cuda:0'
dtype = torch.bfloat16
M, N, K = 8192, 8192, 8192

if False:
    x, w, y = read_and_tile('/mntnlp/nanxiao/dataset/flops/down_fb_1.pkl',
                            tile=True)
    M, K = x.shape
    N, K = w.shape
else:
    x = torch.randn((M, K), dtype=dtype, device=device)
    w = torch.randn((N, K), dtype=dtype, device=device)
    y = torch.randn((M, N), dtype=dtype, device=device)

org_out = fp16_forward(x, w.t())

if False:
    xq, x_scale = torch_row_quant(x, dtype=torch.float8_e4m3fn)
    wq, w_scale = torch_row_quant(w, dtype=torch.float8_e4m3fn)
    opt_out = (xq.to(dtype) @ wq.to(dtype).t()) * x_scale.to(
        dtype) * w_scale.to(dtype)[:, 0]
    quant_check(org_out, xq, wq, opt_out, 'channel')

if False:
    xq, wq, yq, ytq, o, dx, dw = torch_channel_quant_f_and_b(x, w, y)
    ref_o, ref_dx, ref_dw = fp16_f_and_b(x, w, y)
    mode = 'channels'
    output_check(ref_o, o, mode)
    output_check(ref_dx, dx, mode)
    output_check(ref_dw, dw, mode)
    quant_check(ref_o, xq, wq, o, mode)
    quant_check(ref_dx, yq, wq, dx, mode)
    quant_check(ref_dw, ytq, xq, dw, mode)
