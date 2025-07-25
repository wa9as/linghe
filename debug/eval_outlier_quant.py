import torch

from flops.utils.util import ( fp16_forward,
                               quant_check,
                               torch_outlier_quant )

qtype = torch.float8_e4m3fn
device = 'cuda:0'
dtype = torch.bfloat16

d = torch.load('down_fb_1.pkl', weights_only=True)
x = d['x'][0].to(dtype).to(device)
w = d['w'].to(dtype).to(device)
y = d['y'][0].to(dtype).to(device)

reshape = True
if reshape:
    x = torch.cat([x] * 2, 0)[:128, :2048].contiguous()
    y = torch.cat([y] * 2, 0)[:128, :4096].contiguous()
    w = w[:4096, :2048].contiguous()
    # x = x*0.0+torch.arange(128,dtype=dtype,device=device)[:,None]+1

batch_size, in_dim = x.shape
out_dim, in_dim = w.shape

org_out = fp16_forward(x, w.t())

print(f'\n{batch_size=} {in_dim=} {out_dim=} {dtype=} ' \
      f'x.max={x.abs().max().item():.3f} x.mean={x.abs().mean().item():.3f} w.max={w.abs().max().item():.3f} ' \
      f'w.mean={w.abs().mean().item():.3f} y.max={y.abs().max().item():.3f}')

xq, wq, x_scale, w_scale, max_idx, x_outlier = torch_outlier_quant(x, w, qtype)
opt_base = xq.to(dtype) @ wq.to(dtype).t()
opt_addon = (wq.to(dtype)[:, max_idx] @ x_outlier.to(dtype).t()).t()
opt_out = opt_base * (x_scale * w_scale).to(dtype)
opt_out += opt_addon * w_scale.to(dtype)
quant_check(org_out, xq, wq, opt_out, 'os')
