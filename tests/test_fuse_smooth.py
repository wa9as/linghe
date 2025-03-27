import torch 

from flops.quant.hadamard import *
from flops.utils.util import *
from bench_h800_smooth import smooth_quant_forward, smooth_quant_backward, smooth_quant_update

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
    #  np.random.seed(seed)
    #  random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(100)

qtype = torch.float8_e4m3fn
device = 'cuda:0'
dtype = torch.bfloat16

# x,w,y= read_and_tile('down_fb_1.pkl', tile=True)
batch_size, out_dim, in_dim = [8192, 6144, 4096]

x = torch.randn(batch_size, in_dim, dtype=dtype, device=device)
w = torch.randn(out_dim, in_dim, dtype=dtype, device=device)
y = torch.randn(batch_size, out_dim, dtype=dtype, device=device)
M,K = x.shape 
N,K = w.shape 

ref_o, ref_dx, ref_dw = fp16_f_and_b(x, w, y)


def torch_nt(x, w):
    x = x.clone()
    w = w.clone()
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
    return xq, wq, x_quant_scale, w_quant_scale, x_smooth_scale, w_smooth_scale

def torch_nn(y, w_quant_scale):
    y = y.clone()
    ys = y*w_quant_scale.view(1,N)
    y_scale = ys.abs().float().amax(dim=1, keepdim=True)/448.0+1e-9
    yq = (ys/y_scale).to(torch.float8_e4m3fn)
    # print(y[0, :])
    # print(ys[0,:])
    # print(y_scale[0])
    # yq = (ys/y_scale)
    # print(y_scale[0])
    # print(ys[0,:])
    return ys, y_scale, yq, w_quant_scale

def torch_tn(y, x_quant_scale):
    # print(y.size())
    # print(x_quant_scale.size())
    y = y.clone()
    yt = y.t().contiguous()  # [N, M]
    yts = yt*x_quant_scale.view(1, M)
    print(yt[0, :])
    print(x_quant_scale[:, :8])
    print(yts[0, :])
    yt_scale = yts.abs().amax(dim=1, keepdim=True)/448.0+1e-9
    # print(yt_scale.size())
    ytq = (yts/yt_scale).to(torch.float8_e4m3fn)
    return yts, yt_scale, ytq, x_quant_scale

def abs_error(a, b):
  return (a.float() - b.float()).abs().mean().item()

### single part test ###

### smooth_quant_forward ###

# xqt, wqt, x_quant_scale_t, w_quant_scale_t, x_smooth_scale_t, w_smooth_scale_t = torch_nt(x, w)
opt_out,xq,wq,x_quant_scale,w_quant_scale,x_smooth_scale,w_smooth_scale = smooth_quant_forward(x,w)
# quant_check(ref_o, xq, wq, opt_out, 'smooth_quant_forward')

# print(f"x_quant_scale abs error:{abs_error(x_quant_scale_t, x_quant_scale)}")
# print(f"x_smooth_scale abs error:{abs_error(x_smooth_scale_t, x_smooth_scale)}")
# print(f"w_quant_scale abs error:{abs_error(w_quant_scale_t, w_quant_scale)}")
# print(f"w_smooth_scale abs error :{abs_error(w_smooth_scale_t, w_smooth_scale)}")

### smooth_quant_backward ###

# print(w_quant_scale.size())
# print(w_quant_scale)
# wqt_t = wq.clone().t().contiguous().t()

y_s, y_scale_t, yqt, w_quant_scale_t = torch_nn(y, w_quant_scale)
opt_dx,yq,wq_t,y_scale, w_quant_scale  = smooth_quant_backward(y,wq,w_quant_scale,w_smooth_scale)

# print(y_quant_scale_t[:10])
# print(y_quant_scale[:10])

# print(f"wq abs error :{abs_error(wqt_t, wq_t)}") # pass
# print(f"y_quant_scale abs error :{abs_error(y_scale_t, y_scale)}") # pass
# print(f"y_q abs error :{abs_error(yqt, yq)}") # pass
# quant_check(ref_dx, yq, wq, opt_dx, 'smooth_quant_backward')


### smooth_quant_update ###
yts, yt_scale_t, ytq_t, x_quant_scale_t = torch_tn(y, x_quant_scale)
opt_dw,yq,xq_t,y_scale,x_smooth_scale, yt_s = smooth_quant_update(y, xq, x_quant_scale, x_smooth_scale)

# xqt_t = xq.clone().t().contiguous().t()
# print(f"wq abs error :{abs_error(xqt_t, xq_t)}") # pass
print(f"yts abs error :{abs_error(yts, yt_s)}") # pass


# print(y_quant_scale_t[:10, :])
# print(y_quant_scale[:10, :])
# print(yqt)
# print(yq)
# quant_check(ref_dx, yq, wq, opt_dx, 'smooth_quant_backward')
# print(f"y_quant_scale abs error:{abs_error(y_quant_scale_t, y_quant_scale)}")
# print(f"w_quant_scale abs error:{abs_error(w_quant_scale_t, w_quant_scale)}")






#     output,x_q,x_s,w_q,w_s = fuse_hadamard_quant_forward(x, w, hm)
#     quant_check(org_out, x_q, w_q, output, 'fuse_hadamard_quant_forward')

#     output,y_q,y_s,w_q,w_s = fuse_hadamard_quant_backward(y, w, hm)
#     quant_check(y@w, y_q, w_q, output, 'fuse_hadamard_quant_backward')

#     output,y_q,y_s,x_q,x_s = fuse_hadamard_quant_update(y,x, hm)
#     quant_check(y.t()@x, y_q, x_q, output, 'fuse_hadamard_quant_update')
