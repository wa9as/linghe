import torch 

from flops.utils.util import *
from flops.quant.block.group import *
from flops.quant.block.block import *
from flops.utils.benchmark import benchmark_func

qtype = torch.float8_e4m3fn
device = 'cuda:0'
dtype = torch.bfloat16
B = 128

if False:
    # x,w,y= read_and_tile('/mntnlp/nanxiao/dataset/flops/down_fb_1.pkl', tile=True)
    x,w,y= read_and_tile('/mntnlp/nanxiao/dataset/tmp_flops/forward_1.pkl', tile=True)
    M, K = x.shape 
    N, K = w.shape 
else:
    M, N, K = 8192, 8192, 8192
    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)
    y = torch.randn(M, N, dtype=dtype, device=device)




if False:
    xq, x_scle = torch_group_quant(x, B)
    wq, w_scale = torch_block_quant(x, B)
    x_scales = torch.repeat_interleave(x_scale, B, 1)
    w_scales = torch.repeat_interleave(torch.repeat_interleave(w_scale, B, 0), B, 1)
    xdq = xq.to(torch.float32)*x_scales
    wdq = wq.to(torch.float32).t()*w_scales
    opt_out = xdq@wdq
    org_out = fp16_forward(x, w.t())
    quant_check(org_out, xq, wq, opt_out, mode)

if True:

    xq_ref, x_scale_ref = torch_group_quant(x, B)
    xq, x_scale = triton_group_quant(x, group_size=B, round_scale=False)
    output_check(xq_ref.float(), xq.float(), mode='data')
    output_check(x_scale_ref.float(), x_scale.float(), mode='scale')

    xq, x_scale = triton_persist_group_quant(x, group_size=B, round_scale=False)
    output_check(xq_ref.float(), xq.float(), mode='data')
    output_check(x_scale_ref.float(), x_scale.float(), mode='scale')

    # torch.testing.assert_close(xq_ref.float(), xq.float(), rtol=0.02, atol=0.02)

    n_repeat = 100

    ref_time = benchmark_func(triton_group_quant, x, group_size=B, n_repeat=n_repeat,  ref_bytes=M*K*3)
    benchmark_func(triton_persist_group_quant, x, group_size=B, n_repeat=n_repeat,ref_time=ref_time, ref_bytes=M*K*3)
