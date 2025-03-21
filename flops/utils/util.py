import math
import torch 



def tensor_quant(x,w, dtype):
    fmax = torch.finfo(dtype).max
    x_scale = torch.max(torch.abs(x))/fmax
    w_scale = torch.max(torch.abs(w))/fmax
    x_q = (x/x_scale).to(dtype)
    w_q = (w/w_scale).to(dtype)
    return x_q,w_q,x_scale,w_scale

def channel_quant(x,w, dtype):
    fmax = torch.finfo(dtype).max
    x_scale = (torch.max(torch.abs(x),dim=1, keepdims=True)[0]+1e-6)/fmax
    w_scale = (torch.max(torch.abs(w),dim=1, keepdims=True)[0]+1e-6)/fmax
    x_q = (x/x_scale).to(dtype)
    w_q = (w/w_scale).to(dtype)
    return x_q,w_q,x_scale,w_scale

# direct quant without scaling to 448
def torch_smooth_quant_v0(x, w, dtype):
    # w:[bs, in]  w:[out, in]
    x = x.clone()
    w = w.clone()
    fmax = torch.finfo(dtype).max
    x_max = torch.max(torch.abs(x).float(), dim=0, keepdim=True)[0]
    w_max = torch.max(torch.abs(w).float(), dim=0, keepdim=True)[0]
    scale = (x_max/w_max)**0.5
    x_max_ = x_max/scale
    w_max_ = w_max*scale
    # print(f'x_max:{x_max.max().item()} w_max:{w_max.max().item()} scale:{scale.max().item()} x_smooth_max:{x_max_.max().item()} w_smooth_max:{w_max_.max().item()} x_scale:{x_scale.max().item()} w_scale:{w_scale.max().item()}')
    x_q = (x*(1.0/scale).to(x.dtype)).to(dtype)
    w_q = (w*(scale).to(x.dtype)).to(dtype)

    return x_q, w_q, scale

# quant with scaling to 448
def torch_smooth_quant_v1(x, w, dtype):
    # w:[bs, in]  w:[out, in]
    x = x.clone()
    w = w.clone()
    fmax = torch.finfo(dtype).max
    x_max = torch.max(torch.abs(x).float(), dim=0, keepdim=True)[0]
    w_max = torch.max(torch.abs(w).float(), dim=0, keepdim=True)[0]
    scale = (x_max/w_max)**0.5
    x_max_ = x_max/scale
    w_max_ = w_max*scale
    x_scale = x_max_/fmax
    w_scale = w_max_/fmax
    rescale = fmax/torch.maximum(x_max_.max(),w_max_.max())
    # print(f'x_max:{x_max.max().item()} w_max:{w_max.max().item()} scale:{scale.max().item()} x_smooth_max:{x_max_.max().item()} w_smooth_max:{w_max_.max().item()} x_scale:{x_scale.max().item()} w_scale:{w_scale.max().item()}')
    x_q = (x*(rescale/scale).to(x.dtype)).to(dtype)
    w_q = (w*(scale*rescale).to(x.dtype)).to(dtype)

    return x_q, w_q, scale, rescale

# output-col wise w scale and quant with scaling to 448
def torch_smooth_quant_v2_deprecated(x, w, dtype):
    # w:[bs, in]  w:[out, in]
    fmax = torch.finfo(dtype).max
    x = x.clone()
    w = w.clone()
    w_max = torch.max(torch.abs(w).float(), dim=1, keepdim=True)[0]
    w_scale = w_max
    w = w/w_scale

    x_max = torch.max(torch.abs(x).float(), dim=0, keepdim=True)[0]
    maxs = x_max**0.5
    rescale = fmax/maxs.max()
    # print(f'x_max:{x_max.max().item()} w_max:{w_max.max().item()} maxs:{maxs.max().item()} w_scale:{w_scale.max().item()}')
    x_q = (x*(rescale/maxs).to(x.dtype)).to(dtype)
    w_q = (w*(rescale*maxs).to(x.dtype)).to(dtype)

    return x_q, w_q, w_scale, maxs, rescale

def torch_smooth_quant_v3(x, w, dtype):
    # w:[bs, in]  w:[out, in]
    x = x.clone()
    w = w.clone()
    fmax = torch.finfo(dtype).max
    x_max = torch.max(torch.abs(x).float(), dim=0, keepdim=True)[0]
    w_max = torch.max(torch.abs(w).float(), dim=0, keepdim=True)[0]
    maxs = (x_max*w_max)**0.5
    x_scale = x_max/maxs
    w_scale = w_max/maxs  # reciprocal of x_scale
    x_smooth = x/x_scale 
    w_smooth = w/w_scale
    x_max = torch.max(torch.abs(x_smooth).float(), dim=1, keepdim=True)[0]
    w_max = torch.max(torch.abs(w_smooth).float(), dim=1, keepdim=True)[0]
    x_scale = x_max/448.0
    w_scale = w_max/448.0
    # print(f'x_max:{x_max.max().item()} w_max:{w_max.max().item()} scale:{scale.max().item()} x_smooth_max:{x_max_.max().item()} w_smooth_max:{w_max_.max().item()} x_scale:{x_scale.max().item()} w_scale:{w_scale.max().item()}')
    x_q = (x_smooth*(1.0/x_scale).to(x.dtype)).to(dtype)
    w_q = (w_smooth*(1.0/w_scale).to(x.dtype)).to(dtype)

    return x_q, w_q, x_scale,  w_scale

def torch_os_quant_v0(x,w,dtype):
    x = x.clone()
    w = w.clone()
    fmax = torch.finfo(dtype).max
    max_val, max_idx = torch.topk(x.abs().float().max(dim=0)[0], 5)
    # print(max_idx)
    x_outlier = x[:,max_idx[:4]]
    x[:,max_idx[:4]] = 0.0
    x_scale = max_val[-1]/fmax
    xq = (x/x_scale.to(x.dtype)).to(dtype)
    w_max = w.abs().float().max()
    w_scale = w_max/fmax
    wq = (w/w_scale.to(x.dtype)).to(dtype)
    return xq, wq, x_scale, w_scale, max_idx[:4], x_outlier


def hadamard_matrix(n, device='cuda:0', dtype=torch.bfloat16):
    m2 = torch.tensor([[1,1],[1,-1]],device=device,dtype=dtype)
    m = m2
    for i in range(int(round(math.log2(n)-1))):
        m = torch.kron(m,m2)
    return m.to(dtype)

def torch_ht(x, hm):
    x = x.clone()
    hm = hm.clone()
    M, K = x.shape 
    B = hm.size(0)
    xp = torch.reshape(x,(M//B,B,K//B,B)).permute(0,2,1,3).contiguous()
    # print('xp',xp)
    # pm = hm/B
    pm = hm
    xp = xp@pm
    # print('xp@pm',xp)
    xp = xp.permute(0,2,1,3)
    # print('xp.permute',xp)
    xp = torch.reshape(xp,(M,K))
    return xp 

def torch_ht_quant_v0(x,w,hm,dtype):
    fmax = torch.finfo(dtype).max
    x = x.clone()
    w = w.clone()
    M, K = x.shape 
    N, K = w.shape
    B = hm.size(0)
    xp = torch.reshape(x,(M//B,B,K//B,B)).permute(0,2,1,3)

    # pm = 1.0*((torch.randn((B,B),dtype=x.dtype,device=x.device)>0).to(x.dtype)-0.5)
    pm = hm/B
    xp = xp@pm
    xp = xp.permute(0,2,1,3)
    xp = torch.reshape(xp,(M,K))
    # print(f'{x.abs().mean()=} {xp.abs().mean()=} {x.abs().max()=} {xp.abs().max()=}')
    x_scale = torch.max(torch.abs(xp).float())/fmax
    xq = (xp/x_scale).to(dtype)

    wp = torch.reshape(w.t().contiguous(),(K//B,B,N//B,B)).permute(0,2,1,3)
    # pmi = torch.linalg.inv(pm.float()).to(x.dtype)
    pmi = hm
    wp = pmi@wp
    wp = wp.permute(0,2,1,3)
    wp = torch.reshape(wp,(K,N)).t().contiguous()
    w_scale = torch.max(torch.abs(wp).float())/fmax
    wq = (wp/w_scale).to(dtype)

    return xq, wq, x_scale, w_scale

def torch_ht_quant_v1(x,w,hm,dtype):
    fmax = torch.finfo(dtype).max
    x = x.clone()
    w = w.clone()
    M, K = x.shape 
    N, K = w.shape
    B = hm.size(0)
    xp = torch.reshape(x,(M//B,B,K//B,B)).permute(0,2,1,3)

    # pm = 1.0*((torch.randn((B,B),dtype=x.dtype,device=x.device)>0).to(x.dtype)-0.5)
    pm = hm/B
    xp = xp@pm
    xp = xp.permute(0,2,1,3)
    xp = torch.reshape(xp,(M,K))
    # print(f'{x.abs().mean()=} {xp.abs().mean()=} {x.abs().max()=} {xp.abs().max()=}')
    x_scale = torch.amax(torch.abs(xp).float(),dim=1,keepdim=True)/fmax
    xq = (xp/x_scale).to(dtype)

    wp = torch.reshape(w.t().contiguous(),(K//B,B,N//B,B)).permute(0,2,1,3)
    # pmi = torch.linalg.inv(pm.float()).to(x.dtype)
    pmi = hm
    wp = pmi@wp
    wp = wp.permute(0,2,1,3)
    wp = torch.reshape(wp,(K,N)).t().contiguous()
    w_scale = torch.amax(torch.abs(wp).float(),dim=1,keepdim=True)/fmax
    wq = (wp/w_scale).to(dtype)

    return xq, wq, x_scale, w_scale


def torch_col_col_quant_v1(x, w, dtype):
    dtype = x.dtype 
    w_max = torch.max(torch.abs(w), dim=0, keepdim=True)[0]  # col
    x_max = torch.max(torch.abs(x), dim=1, keepdim=True)[0]  # col as well
    w_scale = w_max/dtype
    x_scale = x_max/dtype
    w_q = (w/w_scale).to(dtype)
    x_q = (x/x_scale).to(dtype)
    return x_q, w_q, w_scale, x_scale




def fp16_forward(x,w):
    return x @ w

def fp16_update(y,x):
    return y.t() @ x

def fp16_backward(y,w):
    return y @ w

def fp8_transpose(x):
    return x.t().contiguous()

def fp16_transpose(x):
    return x.t().contiguous()

def fp16_f_and_b(x,w,y):
    y = x@w.t()
    dw = y.t()@x
    dx = y@w



def quant_check(org_out, xq, wq, opt_out, mode):
    abs_error = (opt_out.float() - org_out.float()).abs().mean().item()
    rel_error = abs_error/org_out.float().abs().mean().item()
    x_underflow = (xq==0.0).sum().item()/xq.numel()
    w_underflow = (wq==0.0).sum().item()/wq.numel()
    x_overflow = (torch.isnan(xq)).sum().item()
    w_overflow = (torch.isnan(wq)).sum().item()
    print(f'\nmode:{mode} abs_error:{abs_error:.3f} rel_error:{rel_error:.3f} ' \
            f'org:{org_out.abs().max():.3f}/{org_out.abs().mean():.3f} ' \
            f'opt:{opt_out.abs().max():.3f}/{opt_out.abs().mean():.3f} ' \
            f'x_underflow:{x_underflow:.5f} w_underflow:{w_underflow:.5f} ' \
            f'x_overflow:{x_overflow} w_overflow:{w_overflow}')

