import torch

from flops.utils.util import (fp16_forward,
                              output_check,
                              torch_smooth_quant)

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
    x = torch.randn((M, K), dtype=dtype, device=device) ** 3
    w = torch.randn((N, K), dtype=dtype, device=device)
    y = torch.randn((M, N), dtype=dtype, device=device)

org_out = fp16_forward(x, w.t())

if False:
    xq, wq, x_scale, w_scale = torch_smooth_quant(x, w, torch.float8_e4m3fn)
    xdq = xq.to(dtype) * x_scale
    wdq = wq.to(dtype) * w_scale
    opt_out = xdq @ wdq.t()
    quant_check(org_out, xq, wq, opt_out, 'torch_channel')

if False:
    yq, wq, y_scale, w_scale = torch_smooth_quant(y, w.t(), torch.float8_e4m3fn)
    ydq = yq.to(dtype) * y_scale
    wdq = wq.to(dtype) * w_scale
    opt_out = ydq @ wdq.t()
    quant_check(y @ w, yq, wq, opt_out, 'torch_channel_backward')

if False:
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

if True:
    # compare rescale and direct quant
    round_scale = False
    smooth_scale = torch.rand((N,), dtype=torch.float32, device='cuda:0') * 10
    if round_scale:
        smooth_scale = torch.exp2(torch.ceil(torch.log2(smooth_scale)))
    transpose_smooth_scale = torch.rand((M,), dtype=torch.float32,
                                        device='cuda:0') * 10
    if round_scale:
        transpose_smooth_scale = torch.exp2(
            torch.ceil(torch.log2(transpose_smooth_scale)))

    y_q_ref, y_scales_ref = torch_smooth_quant(y, smooth_scale, reverse=True,
                                               round_scale=round_scale)
    y_t_q_ref, y_t_scales_ref = torch_smooth_quant(y.t(),
                                                   transpose_smooth_scale,
                                                   reverse=True,
                                                   round_scale=round_scale)

    B = 128
    y_q, y_scales = torch_group_quant(y, B=B, round_scale=round_scale)
    yf = torch.reshape(y_q.float().view(M, N // B, B) * y_scales[:, :, None],
                       (M, N))
    yf_q, yf_scales = torch_smooth_quant(yf, smooth_scale, reverse=True,
                                         round_scale=round_scale)
    yf_t_q, yf_t_scales = torch_smooth_quant(yf.t(), transpose_smooth_scale,
                                             reverse=True,
                                             round_scale=round_scale)
    y_onetime_dequant = (yf_t_q.float() * yf_t_scales[:,
                                          None] / transpose_smooth_scale).t()

    yff = yf_q.float() * yf_scales[:, None] / smooth_scale
    yf_q_output, yf_scales_output = torch_smooth_quant(yff.t(),
                                                       transpose_smooth_scale,
                                                       reverse=True,
                                                       round_scale=round_scale)
    y_rescale_dequant = (yf_q_output.float() * yf_scales_output[:,
                                               None] / transpose_smooth_scale).t()

    output_check(y_t_q_ref, yf_t_q, 'onetime.data')
    output_check(y_t_scales_ref, yf_t_scales, 'onetime.scale')

    output_check(y_t_q_ref, yf_q_output, 'rescale.data')
    output_check(y_t_scales_ref, yf_scales_output, 'rescale.scale')

    output_check(y.float(), y_onetime_dequant, 'onetime')
    output_check(y.float(), y_rescale_dequant, 'rescale')
