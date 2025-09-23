import math
import torch


def round_up(x, b=16):
    return ((x - 1) // b + 1) * b


def torch_tensor_quant(x, dtype=torch.float8_e4m3fn, round_scale=False):
    fmax = torch.finfo(dtype).max
    scale = torch.max(torch.abs(x)) / fmax
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    x_q = (x / scale).to(dtype)
    return x_q, scale


def torch_row_quant(x, dtype=torch.float8_e4m3fn, round_scale=False):
    fmax = torch.finfo(dtype).max
    scale = torch.abs(x).amax(1) / fmax
    scale = torch.maximum(scale, 1e-30 * torch.ones((1,), dtype=torch.float32,
                                                    device=x.device))
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    x_q = (x / scale[:, None]).to(dtype)
    return x_q, scale


def torch_column_quant(x, dtype=torch.float8_e4m3fn, round_scale=False):
    fmax = torch.finfo(dtype).max
    scale = torch.abs(x).amax(0) / fmax
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    x_q = (x / scale).to(dtype)
    return x_q, scale


def torch_group_quant(x, B=128, dtype=torch.float8_e4m3fn, round_scale=False):
    # fmax = torch.finfo(dtype).max
    fmax = 448
    x = x.clone()
    M, K = x.shape
    P = K
    if K % B != 0:
        x = torch.nn.functional.pad(x, (0, B - K % B))
        P = x.shape[1]
    
    xp = torch.reshape(x.contiguous(), (M, P // B, B))
    scale = torch.amax(torch.abs(xp).float(), dim=2) / fmax 
    scaoe = torch.maximum(scale, 1e-30*torch.ones((1,),dtype=torch.float32,device=x.device))
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    xq = (xp / scale[:, :, None]).to(dtype)
    xq = torch.reshape(xq, (M, P)).contiguous()
    xq = xq[:, :K].contiguous()
    return xq, scale


def torch_block_quant(w, B=128, dtype=torch.float8_e4m3fn, round_scale=False):
    fmax = torch.finfo(dtype).max
    w = w.clone()
    N, K = w.shape

    wp = torch.reshape(w.t().contiguous(), (K // B, B, N // B, B)).permute(0, 2,
                                                                           1, 3)
    scale = torch.amax(torch.amax(torch.abs(wp).float(), dim=2), dim=2) / fmax
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    wq = (wp / scale[:, :, None, None]).to(dtype)
    wq = wq.permute(0, 2, 1, 3)
    wq = torch.reshape(wq, (K, N)).t().contiguous()

    return wq, scale



def torch_make_indices(logits, topk=8, bias=-0.01):
    M, n_experts = logits.shape
    device = logits.device
    logits = logits.to(torch.float64) + 1e-10 * torch.arange(n_experts,
                                                             device=device).to(
        torch.float32)
    topk_values, topk_indices = torch.topk(logits, topk, dim=-1, sorted=True)
    logits[logits < topk_values[:, -1:] + bias] = -1000000
    probs = torch.nn.Softmax(dim=1)(logits)
    route_map = probs > 0
    token_count_per_expert = route_map.sum(0)
    # token_count_per_expert_list = token_count_per_expert.tolist()
    # out_tokens = sum(token_count_per_expert_list)

    token_indices = (
        torch.arange(M, device=logits.device).unsqueeze(0).expand(n_experts, -1)
    )
    indices = token_indices.masked_select(route_map.T.contiguous())
    row_id_map = torch.reshape(
        torch.cumsum(route_map.T.contiguous().view(-1), 0), (n_experts, M)) - 1
    row_id_map[torch.logical_not(route_map.T)] = -1
    row_id_map = row_id_map.T.contiguous()
    return probs, route_map, token_count_per_expert, indices, row_id_map


def fp16_forward(x, w):
    return x @ w


def fp16_update(y, x):
    return y.t() @ x


def fp16_backward(y, w):
    return y @ w


def fp8_transpose(x):
    return x.t().contiguous()


def fp16_transpose(x):
    return x.t().contiguous()


def fp16_f_and_b(x, w, y):
    o = x @ w.t()
    dw = y.t() @ x
    dx = y @ w
    return o, dx, dw


def output_check(org_out, opt_out, mode='', rtol=None, atol=None):
    assert org_out.shape == opt_out.shape, f"ref:{org_out.shape} != out:{opt_out.shape}"
    dtype = org_out.dtype
    assert opt_out.dtype == dtype or dtype == torch.float32, f"ref:{dtype} != out:{opt_out.dtype}"
    if org_out.numel() == 0:
        return 

    if dtype != torch.float32:
        org_out = org_out.float()
        opt_out = opt_out.float()
    if dtype == torch.float8_e4m3fn:
        rtol = 0.1
    abs_error = (opt_out - org_out).abs().mean().item()
    rel_error = abs_error / max(org_out.abs().mean().item(), 1e-38)
    if rel_error >= 0.005:
        rel_err_str = f"\033[91m {rel_error:.6f}\033[00m"
    else:
        rel_err_str = f"{rel_error:.6f}"
    org_max = org_out.abs().max()
    org_mean = org_out.abs().mean()
    opt_max = opt_out.abs().max()
    opt_mean = opt_out.abs().mean()
    print(f'\n{mode:<16}  rel:{rel_err_str}  abs:{abs_error:.6f}  ' \
          f'org:{org_max:.3f}/{org_mean:.3f} ' \
          f'opt:{opt_max:.3f}/{opt_mean:.3f} ')
    if rtol is not None and atol is not None:
        torch.testing.assert_close(opt_out, org_out, rtol=rtol, atol=atol)


def quant_check(org_out, xq, wq, opt_out, mode):
    abs_error = (opt_out.float() - org_out.float()).abs().mean().item()
    rel_error = abs_error / org_out.float().abs().mean().item()
    x_underflow = (xq == 0.0).sum().item() / xq.numel()
    w_underflow = (wq == 0.0).sum().item() / wq.numel()
    x_overflow = (torch.isnan(xq)).sum().item()
    w_overflow = (torch.isnan(wq)).sum().item()
    print(f'\n{mode}  rel:{rel_error:.3f}  abs:{abs_error:.3f}  ' \
          f'org:{org_out.abs().max():.3f}/{org_out.abs().mean():.3f} ' \
          f'opt:{opt_out.abs().max():.3f}/{opt_out.abs().mean():.3f} ' \
          f'x_underflow:{x_underflow:.5f} w_underflow:{w_underflow:.5f} ' \
          f'x_overflow:{x_overflow} w_overflow:{w_overflow}')


def read_and_tile(filename, tile=True):
    device = 'cuda:0'
    dtype = torch.bfloat16
    d = torch.load(filename, weights_only=True)
    # x = d['x'][0].to(dtype).to(device)
    # w = d['w'].to(dtype).to(device)
    # y = d['y'][0].to(dtype).to(device)
    x = d['x']
    y = d['y']
    x = x.view(-1, x.size(2)).to(dtype).to(device)
    w = d['w'].to(dtype).to(device)
    y = y.view(-1, y.size(2)).to(dtype).to(device)

    if tile:
        min_block = 256
        indices = y.abs().float().sum(-1) > 0
        x = x[indices]
        y = y[indices]

        bs = x.size(0)
        m = max(2 ** (int(math.log2(bs) + 1)), min_block)
        rep = (m - 1) // bs + 1
        x = torch.cat([x] * rep, 0)[:m].contiguous()
        y = torch.cat([y] * rep, 0)[:m].contiguous()

        if x.size(1) % min_block != 0:
            xs = x.size(1) // min_block * min_block
            x = x[:, :xs].contiguous()
            w = w[:, :xs].contiguous()
        if y.size(1) % min_block != 0:
            ys = y.size(1) // min_block * min_block
            y = y[:, :ys].contiguous()
            w = w[:ys].contiguous()

    batch_size, in_dim = x.shape
    out_dim, in_dim = w.shape
    print(f'\ndataset: {batch_size=} {in_dim=} {out_dim=} ' \
          f'x.max={x.abs().max().item():.3f} x.mean={x.abs().mean().item():.3f} ' \
          f'w.max={w.abs().max().item():.3f} w.mean={w.abs().mean().item():.3f} ' \
          f'y.max={y.abs().max().item():.3f} y.mean={y.abs().mean().item():.3f}')

    return x, w, y
