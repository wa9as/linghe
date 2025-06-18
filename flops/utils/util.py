import math
import torch 


def round_up(x, b=16):
    assert b==32
    return ((x-1)//b+1)*b


def torch_tensor_quant(x,w, dtype):
    fmax = torch.finfo(dtype).max
    x_scale = torch.max(torch.abs(x))/fmax
    w_scale = torch.max(torch.abs(w))/fmax
    x_q = (x/x_scale).to(dtype)
    w_q = (w/w_scale).to(dtype)
    return x_q,w_q,x_scale,w_scale

def torch_channel_quant(x,w, dtype):
    fmax = torch.finfo(dtype).max
    x_scale = (torch.max(torch.abs(x),dim=1, keepdims=True)[0]+1e-6)/fmax
    w_scale = (torch.max(torch.abs(w),dim=1, keepdims=True)[0]+1e-6)/fmax
    x_q = (x/x_scale).to(dtype)
    w_q = (w/w_scale).to(dtype)
    return x_q,w_q,x_scale,w_scale

def torch_tile_block_quant(x,w,B,dtype):
    fmax = torch.finfo(dtype).max
    x = x.clone()
    w = w.clone()
    M, K = x.shape 
    N, K = w.shape

    xp = torch.reshape(x.contiguous(),(M,K//B,B))
    x_scale = torch.amax(torch.abs(xp).float(),dim=2)/fmax
    xq = (xp/x_scale[:,:,None]).to(dtype)
    xq = torch.reshape(xq,(M,K)).contiguous()

    wp = torch.reshape(w.t().contiguous(),(K//B,B,N//B,B)).permute(0,2,1,3)
    w_scale = torch.amax(torch.amax(torch.abs(wp).float(),dim=2),dim=2)/fmax
    wq = (wp/w_scale[:,:,None,None]).to(dtype)
    wq = wq.permute(0,2,1,3)
    wq = torch.reshape(wq,(K,N)).t().contiguous()

    return xq,wq,x_scale,w_scale

# quant with scaling to 448
def torch_smooth_tensor_quant(x, w, dtype):
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
    x_q = (x*(rescale/scale).to(x.dtype)).to(dtype)
    w_q = (w*(scale*rescale).to(x.dtype)).to(dtype)

    return x_q, w_q, scale, rescale

def torch_smooth_quant(x, w, dtype):
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
    x_q = (x_smooth*(1.0/x_scale).to(x.dtype)).to(dtype)
    w_q = (w_smooth*(1.0/w_scale).to(x.dtype)).to(dtype)

    return x_q, w_q, x_scale,  w_scale

def torch_outlier_quant(x,w,dtype):
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


def torch_hadamard_transform(x, hm):
    x = x.clone()
    hm = hm.clone()
    M, K = x.shape 
    B = hm.size(0)
    xp = torch.reshape(x,(M//B,B,K//B,B)).permute(0,2,1,3).contiguous()
    xp = xp@hm
    xp = xp.permute(0,2,1,3)
    xp = torch.reshape(xp,(M,K))
    return xp 

def torch_hadamard_tensor_quant(x,w,hm,dtype):
    fmax = torch.finfo(dtype).max
    x = x.clone()
    w = w.clone()
    M, K = x.shape 
    N, K = w.shape
    B = hm.size(0)

    xp = torch.reshape(x,(M//B,B,K//B,B)).permute(0,2,1,3)

    xp = xp@hm
    xp = xp.permute(0,2,1,3)
    xp = torch.reshape(xp,(M,K))

    # print(f'x.max:{x.abs().max().item()} x.mean:{x.abs().mean().item()} xp.max:{xp.abs().max().item()} xp.mean:{xp.abs().mean().item()}')
    x_scale = torch.max(torch.abs(xp).float())/fmax
    xq = (xp/x_scale).to(dtype)

    wp = torch.reshape(w.t().contiguous(),(K//B,B,N//B,B)).permute(0,2,1,3)
    wp = hm@wp
    wp = wp.permute(0,2,1,3)
    wp = torch.reshape(wp,(K,N)).t().contiguous()
    # print(f'w.max:{w.abs().max().item()} w.mean:{w.abs().mean().item()} wp.max:{wp.abs().max().item()} wp.mean:{wp.abs().mean().item()}')
    w_scale = torch.max(torch.abs(wp).float())/fmax
    wq = (wp/w_scale).to(dtype)

    return xq, wq, x_scale, w_scale


def torch_hadamard_block_quant(x,w,hm,dtype):
    fmax = torch.finfo(dtype).max
    x = x.clone()
    w = w.clone()
    M, K = x.shape 
    N, K = w.shape
    B = hm.size(0)

    xp = torch.reshape(x,(M//B,B,K//B,B)).permute(0,2,1,3)

    xp = xp@hm

    x_scale = torch.amax(torch.amax(torch.abs(xp).float(),dim=2),dim=2)/fmax
    # print(f'x.max:{x.abs().max().item()} x.mean:{x.abs().mean().item()} xp.max:{xp.abs().max().item()} xp.mean:{xp.abs().mean().item()}')
    xq = (xp/x_scale[:,:,None,None]).to(dtype)

    xq = xq.view(torch.int8).permute(0,2,1,3)
    xq = torch.reshape(xq,(M,K)).view(torch.float8_e4m3fn)


    wp = torch.reshape(w.t().contiguous(),(K//B,B,N//B,B)).permute(0,2,1,3)
    wp = hm@wp

    w_scale = torch.amax(torch.amax(torch.abs(wp).float(),dim=2),dim=2)/fmax
    # print(f'w.max:{w.abs().max().item()} w.mean:{w.abs().mean().item()} wp.max:{wp.abs().max().item()} wp.mean:{wp.abs().mean().item()}')
    wq = (wp/w_scale[:,:,None,None]).to(dtype)

    wq = wq.view(torch.int8).permute(0,2,1,3)
    wq = torch.reshape(wq,(K,N)).t().contiguous().view(torch.float8_e4m3fn)


    return xq, wq, x_scale, w_scale

def torch_hadamard_channel_quant(x,w,hm,dtype):
    fmax = torch.finfo(dtype).max
    x = x.clone()
    w = w.clone()
    M, K = x.shape 
    N, K = w.shape
    B = hm.size(0)
    xp = torch.reshape(x,(M//B,B,K//B,B)).permute(0,2,1,3)

    xp = xp@hm
    xp = xp.permute(0,2,1,3)
    xp = torch.reshape(xp,(M,K))
    # print(f'{x.abs().mean()=} {xp.abs().mean()=} {x.abs().max()=} {xp.abs().max()=}')
    x_scale = torch.amax(torch.abs(xp).float(),dim=1,keepdim=True)/fmax
    xq = (xp/x_scale).to(dtype)

    wp = torch.reshape(w.t().contiguous(),(K//B,B,N//B,B)).permute(0,2,1,3)
    wp = hm@wp
    wp = wp.permute(0,2,1,3)
    wp = torch.reshape(wp,(K,N)).t().contiguous()
    w_scale = torch.amax(torch.abs(wp).float(),dim=1,keepdim=True)/fmax
    wq = (wp/w_scale).to(dtype)

    return xq, wq, x_scale, w_scale.view(1,-1)



# token-wise and channel-wise
def torch_channel_quant_f_and_b(x,w,y):
    M,K = x.shape 
    N,K = w.shape 
    M,N = y.shape 
    x_scale = x.abs().float().amax(dim=1, keepdim=True)/448.0  # [M,1]
    w_scale = w.abs().float().amax(dim=1, keepdim=True)/448.0  # [N,1]
    xq = (x/x_scale).to(torch.float8_e4m3fn)
    wq = (w/w_scale).to(torch.float8_e4m3fn)
    o = torch._scaled_mm(xq,
                            wq.t(),
                            scale_a=x_scale.view(-1,1),
                            scale_b=w_scale.view(1,-1),
                            out_dtype=torch.bfloat16,
                            use_fast_accum=True)
    
    # dx = y @ wT
    # absort w quant scale to y
    ys = y*w_scale.view(1,N)
    y_scale = ys.abs().float().amax(dim=1, keepdim=True)/448.0+1e-9
    yq = (ys/y_scale).to(torch.float8_e4m3fn)
    w_dummy_scale = torch.ones((1,K),dtype=torch.float32, device=x.device)
    dx = torch._scaled_mm(yq,
                            wq.t().contiguous().t(),
                            scale_a=y_scale,
                            scale_b=w_dummy_scale,
                            out_dtype=torch.bfloat16,
                            use_fast_accum=True)

    # dw = yT@x
    yt = y.t().contiguous()
    yts = yt*x_scale.view(1,M)
    yt_scale = yts.abs().float().amax(dim=1, keepdim=True)/448.0+1e-9
    ytq = (yts/yt_scale).to(torch.float8_e4m3fn)
    dw = torch._scaled_mm(ytq,
                                    xq.t().contiguous().t(),
                                    scale_a=yt_scale.view(-1,1),
                                    scale_b=w_dummy_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return xq,wq,yq,ytq,o,dx,dw




# smooth and token-wise/channel-wise
def torch_reuse_smooth_quant_f_and_b(x,w,y):
    x = x.clone()
    w = w.clone()
    y = y.clone()
    M,K = x.shape 
    N,K = w.shape 
    M,N = y.shape 
    x_smooth_max = torch.amax(torch.abs(x).float(), dim=0, keepdim=True)
    w_smooth_max = torch.amax(torch.abs(w).float(), dim=0, keepdim=True)
    maxs = (x_smooth_max*w_smooth_max)**0.5
    x_smooth_scale = x_smooth_max/maxs  # [K, 1]
    w_smooth_scale = w_smooth_max/maxs  # [K, 1] reciprocal of x_scale
    x_smooth = x/x_smooth_scale 
    w_smooth = w/w_smooth_scale

    x_quant_max = torch.amax(torch.abs(x_smooth).float(), dim=1, keepdim=True)
    w_quant_max = torch.amax(torch.abs(w_smooth).float(), dim=1, keepdim=True)

    x_quant_scale = x_quant_max/448.0  # [M, 1]
    w_quant_scale = w_quant_max/448.0  # [N, 1]
    xq = (x_smooth/x_quant_scale).to(torch.float8_e4m3fn)
    wq = (w_smooth/w_quant_scale).to(torch.float8_e4m3fn)

    o = torch._scaled_mm(xq,
                            wq.t(),
                            scale_a=x_quant_scale.view(-1,1),
                            scale_b=w_quant_scale.view(1,-1),
                            out_dtype=torch.bfloat16,
                            use_fast_accum=True)

    # print(f'{x_smooth_scale=} {x_quant_scale[:,0]=} {w_quant_scale=}')

    # dx = y @ wT
    # absort w quant scale to y
    ys = y*w_quant_scale.view(1,N)
    y_scale = ys.abs().float().amax(dim=1, keepdim=True)/448.0+1e-9
    yq = (ys/y_scale).to(torch.float8_e4m3fn)
    dx = torch._scaled_mm(yq,
                            wq.t().contiguous().t(),
                            scale_a=y_scale,
                            scale_b=w_smooth_scale.view(1,-1),
                            out_dtype=torch.bfloat16,
                            use_fast_accum=True)

    # dw = yT@x
    yt = y.t().contiguous()  # [N, M]
    yts = yt*x_quant_scale.view(1, M)
    yt_scale = yts.abs().amax(dim=1, keepdim=True)/448.0+1e-9
    ytq = (yts/yt_scale).to(torch.float8_e4m3fn)
    dw = torch._scaled_mm(ytq,
                                    xq.t().contiguous().t(),
                                    scale_a=yt_scale.view(-1,1),
                                    scale_b=x_smooth_scale.view(1,-1),
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)

    return xq,wq,yq,ytq,o, dx, dw


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
    o = x@w.t()
    dw = y.t()@x
    dx = y@w
    return o, dx, dw



def output_check(org_out, opt_out, mode=''):
    abs_error = (opt_out.float() - org_out.float()).abs().mean().item()
    rel_error = abs_error/org_out.float().abs().mean().item()
    print(f'\nmode:{mode} abs_error:{abs_error:.3f} rel_error:{rel_error:.3f} ' \
            f'org:{org_out.abs().max():.3f}/{org_out.abs().mean():.3f} ' \
            f'opt:{opt_out.abs().max():.3f}/{opt_out.abs().mean():.3f} ')


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
        indices = y.abs().float().sum(-1)>0
        x = x[indices]
        y = y[indices]

        bs = x.size(0)
        m = max(2**(int(math.log2(bs)+1)),min_block)
        rep = (m-1)//bs+1
        x = torch.cat([x]*rep,0)[:m].contiguous()
        y = torch.cat([y]*rep,0)[:m].contiguous()

        if x.size(1) % min_block != 0:
            xs = x.size(1)//min_block*min_block
            x = x[:,:xs].contiguous()
            w = w[:,:xs].contiguous()
        if y.size(1) % min_block != 0:
            ys = y.size(1)//min_block*min_block
            y = y[:,:ys].contiguous()
            w = w[:ys].contiguous()

    batch_size, in_dim = x.shape 
    out_dim, in_dim = w.shape
    print(f'\ndataset: {batch_size=} {in_dim=} {out_dim=} ' \
        f'x.max={x.abs().max().item():.3f} x.mean={x.abs().mean().item():.3f} ' \
        f'w.max={w.abs().max().item():.3f} w.mean={w.abs().mean().item():.3f} ' \
        f'y.max={y.abs().max().item():.3f} y.mean={y.abs().mean().item():.3f}')

    return x,w,y


def torch_fp16_vector_scaled_mm(x, weight, x_scale, weight_scale):
    output = torch._scaled_mm(x,
                                    weight,
                                    scale_a=x_scale,
                                    scale_b=weight_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output

def torch_fp32_vector_scaled_mm(x, weight, x_scale, weight_scale, ones, out=None):
    output = torch._scaled_mm(x,
                                    weight,
                                    scale_a=ones,
                                    scale_b=ones,
                                    out_dtype=torch.float32,
                                    use_fast_accum=True,
                                    out=out)
    return output*x_scale*weight_scale

def torch_fp16_scaler_scaled_mm(x, weight, x_scale, weight_scale):
    output = torch._scaled_mm(x,
                                weight,
                                scale_a=x_scale,
                                scale_b=weight_scale,
                                out_dtype=torch.bfloat16,
                                use_fast_accum=True)
    return output


def torch_fp32_scaler_scaled_mm(x, weight, x_scale, weight_scale):
    output = torch._scaled_mm(x,
                                weight,
                                scale_a=x_scale,
                                scale_b=weight_scale,
                                out_dtype=torch.float32,
                                use_fast_accum=True)
    return output
