# import torch

# from flops.quant.hadamard.naive_hadamard import *   # noqa: F403
# from flops.tools.util import *   # noqa: F403



# def torch_hadamard_tensor_quant(x, w, hm, dtype):
#     fmax = torch.finfo(dtype).max
#     x = x.clone()
#     w = w.clone()
#     M, K = x.shape
#     N, K = w.shape
#     B = hm.size(0)

#     xp = torch.reshape(x, (M // B, B, K // B, B)).permute(0, 2, 1, 3)

#     xp = xp @ hm
#     xp = xp.permute(0, 2, 1, 3)
#     xp = torch.reshape(xp, (M, K))

#     # print(f'x.max:{x.abs().max().item()} x.mean:{x.abs().mean().item()} xp.max:{xp.abs().max().item()} xp.mean:{xp.abs().mean().item()}')
#     x_scale = torch.max(torch.abs(xp).float()) / fmax
#     xq = (xp / x_scale).to(dtype)

#     wp = torch.reshape(w.t().contiguous(), (K // B, B, N // B, B)).permute(0, 2,
#                                                                            1, 3)
#     wp = hm @ wp
#     wp = wp.permute(0, 2, 1, 3)
#     wp = torch.reshape(wp, (K, N)).t().contiguous()
#     # print(f'w.max:{w.abs().max().item()} w.mean:{w.abs().mean().item()} wp.max:{wp.abs().max().item()} wp.mean:{wp.abs().mean().item()}')
#     w_scale = torch.max(torch.abs(wp).float()) / fmax
#     wq = (wp / w_scale).to(dtype)

#     return xq, wq, x_scale, w_scale


# def torch_hadamard_block_quant(x, w, hm, dtype):
#     fmax = torch.finfo(dtype).max
#     x = x.clone()
#     w = w.clone()
#     M, K = x.shape
#     N, K = w.shape
#     B = hm.size(0)

#     xp = torch.reshape(x, (M // B, B, K // B, B)).permute(0, 2, 1, 3)

#     xp = xp @ hm

#     x_scale = torch.amax(torch.amax(torch.abs(xp).float(), dim=2), dim=2) / fmax
#     # print(f'x.max:{x.abs().max().item()} x.mean:{x.abs().mean().item()} xp.max:{xp.abs().max().item()} xp.mean:{xp.abs().mean().item()}')
#     xq = (xp / x_scale[:, :, None, None]).to(dtype)

#     xq = xq.view(torch.int8).permute(0, 2, 1, 3)
#     xq = torch.reshape(xq, (M, K)).view(torch.float8_e4m3fn)

#     wp = torch.reshape(w.t().contiguous(), (K // B, B, N // B, B)).permute(0, 2,
#                                                                            1, 3)
#     wp = hm @ wp

#     w_scale = torch.amax(torch.amax(torch.abs(wp).float(), dim=2), dim=2) / fmax
#     # print(f'w.max:{w.abs().max().item()} w.mean:{w.abs().mean().item()} wp.max:{wp.abs().max().item()} wp.mean:{wp.abs().mean().item()}')
#     wq = (wp / w_scale[:, :, None, None]).to(dtype)

#     wq = wq.view(torch.int8).permute(0, 2, 1, 3)
#     wq = torch.reshape(wq, (K, N)).t().contiguous().view(torch.float8_e4m3fn)

#     return xq, wq, x_scale, w_scale




# def torch_hadamard_quant(x, w, hm, dtype):
#     fmax = torch.finfo(dtype).max
#     x = x.clone()
#     w = w.clone()
#     M, K = x.shape
#     N, K = w.shape
#     B = hm.size(0)
#     xp = torch.reshape(x, (M // B, B, K // B, B)).permute(0, 2, 1, 3)

#     xp = xp @ hm
#     xp = xp.permute(0, 2, 1, 3)
#     xp = torch.reshape(xp, (M, K))
#     # print(f'{x.abs().mean()=} {xp.abs().mean()=} {x.abs().max()=} {xp.abs().max()=}')
#     x_scale = torch.amax(torch.abs(xp).float(), dim=1, keepdim=True) / fmax
#     xq = (xp / x_scale).to(dtype)

#     wp = torch.reshape(w.t().contiguous(), (K // B, B, N // B, B)).permute(0, 2,
#                                                                            1, 3)
#     wp = hm @ wp
#     wp = wp.permute(0, 2, 1, 3)
#     wp = torch.reshape(wp, (K, N)).t().contiguous()
#     w_scale = torch.amax(torch.abs(wp).float(), dim=1, keepdim=True) / fmax
#     wq = (wp / w_scale).to(dtype)

#     return xq, wq, x_scale, w_scale.view(1, -1)


# qtype = torch.float8_e4m3fn
# device = 'cuda:0'
# dtype = torch.bfloat16

# # x,w,y= read_and_tile('/mntnlp/nanxiao/dataset/flops/down_fb_1.pkl', tile=True)
# x, w, y = read_and_tile('/mntnlp/nanxiao/dataset/tmp_flops/forward_1.pkl',
#                         tile=True)

# B = 64
# hm = hadamard_matrix(B, dtype=dtype, device=device, norm=True)

# batch_size, in_dim = x.shape
# out_dim, in_dim = w.shape

# org_out = fp16_forward(x, w.t())

# modes = ['channel']
# megatron = True

# if 'torch_tensor' in modes:
#     xq, wq, x_scale, w_scale = torch_hadamard_tensor_quant(x, w, hm, qtype)
#     xdq = xq.to(torch.float32) * x_scale
#     wdq = wq.to(torch.float32).t() * w_scale
#     opt_out = xdq @ wdq

#     # print('x',x[:4,:4])
#     # print('w',w[:4,:4])
#     # print('xq',xq[:4,:4])
#     # print('wq',wq[:4,:4])
#     # print('org',org_out[:4,:4])
#     # print('opt',opt_out[:4,:4])

#     quant_check(org_out, xq, wq, opt_out, 'torch_tensor')

# if 'torch_block' in modes:
#     xq, wq, x_scale, w_scale = torch_hadamard_block_quant(x, w, hm, qtype)
#     x_scales = torch.repeat_interleave(torch.repeat_interleave(x_scale, B, 0),
#                                        B, 1)
#     w_scales = torch.repeat_interleave(torch.repeat_interleave(w_scale, B, 0),
#                                        B, 1)
#     xdq = xq.to(torch.float32) * x_scales
#     wdq = wq.to(torch.float32).t() * w_scales
#     opt_out = xdq @ wdq

#     quant_check(org_out, xq, wq, opt_out, 'torch_block')

# if 'torch_channel' in modes:
#     xq, wq, x_scale, w_scale = torch_hadamard_channel_quant(x, w, hm, qtype)
#     xdq = xq.to(torch.float32) * x_scale
#     wdq = wq.to(torch.float32).t() * w_scale
#     opt_out = xdq @ wdq
#     quant_check(org_out, xq, wq, opt_out, 'torch_channel')

# if 'megatron' in modes:
#     from flops.quant.hadamard.seperate_hadamard import *   # noqa: F403

#     opt_out, xq, wq, x_scale, w_scale = hadamard_quant_forward_debug_megatron(x,
#                                                                               w,
#                                                                               hm)
#     quant_check(org_out, xq, wq, opt_out, 'hadamard_quant_forward_megatron')

#     opt_out, yq, wq, y_scale, w_scale = hadamard_quant_backward_debug_megatron(
#         y, w, hm)
#     quant_check(y @ w, yq, wq, opt_out, 'hadamard_quant_backward_megatron')

#     opt_out, yq, xq, y_scale, x_scale = hadamard_quant_update_debug_megatron(y,
#                                                                              x,
#                                                                              hm)
#     quant_check(y.t() @ x, yq, xq, opt_out, 'hadamard_quant_update_megatron')

# if 'naive' in modes:
#     opt_out, xq, wq, x_scale, w_scale = hadamard_quant_forward_debug(x, w, hm)
#     print(opt_out.size())
#     quant_check(org_out, xq, wq, opt_out, 'hadamard_quant_forward')

#     opt_out, yq, wq, y_scale, w_scale = hadamard_quant_backward_debug(y, w, hm)
#     quant_check(y @ w, yq, wq, opt_out, 'hadamard_quant_backward')

#     opt_out, yq, xq, y_scale, x_scale = hadamard_quant_update_debug(y, x, hm)
#     quant_check(y.t() @ x, yq, xq, opt_out, 'hadamard_quant_update')

# if 'fused' in modes:
#     from flops.quant.hadamard.fused_hadamard import *   # noqa: F403

#     output, x_q, x_s, w_q, w_s = fused_hadamard_quant_forward_debug(x, w, hm)
#     quant_check(org_out, x_q, w_q, output, 'fuse_hadamard_quant_forward')

#     output, y_q, y_s, w_q, w_s = fused_hadamard_quant_backward_debug(y, w, hm)
#     quant_check(y @ w, y_q, w_q, output, 'fuse_hadamard_quant_backward')

#     output, y_q, y_s, x_q, x_s = fused_hadamard_quant_update_debug(y, x, hm)
#     quant_check(y.t() @ x, y_q, x_q, output, 'fuse_hadamard_quant_update')

# if 'duplex' in modes:
#     from flops.quant.hadamard.duplex_hadamard import *   # noqa: F403

#     output, x_bt, w_bt, x_q, w_q, x_scale, w_scale = duplex_hadamard_quant_forward_debug(
#         x, w, hm)
#     quant_check(org_out, x_q, w_q, output, 'bit_hadamard_quant_forward')

#     output, y_bt, y_q, w_q, y_scale, w_scale = duplex_hadamard_quant_backward_debug(
#         y, w_bt, hm)
#     quant_check(y @ w, y_q, w_q, output, 'bit_hadamard_quant_backward')

#     output, y_q, x_q, y_scale, x_scale = duplex_hadamard_quant_update_debug(
#         y_bt, x_bt, hm)
#     quant_check(y.t() @ x, y_q, x_q, output, 'bit_hadamard_quant_update')
