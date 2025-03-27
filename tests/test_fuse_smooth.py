import torch 

from flops.quant.hadamard import *
from flops.utils.util import *
from bench_h800_smooth import smooth_quant_forward

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

org_out = fp16_forward(x, w.t())


def torch_nt(x, w):
    x = x.clone()
    w = w.clone()
    # y = y.clone()
    M,K = x.shape 
    N,K = w.shape 
    # M,N = y.shape 
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

def abs_error(a, b):
  return (a.float() - b.float()).abs().mean().item()

xqt, wqt, x_quant_scale_t, w_quant_scale_t, x_smooth_scale_t, w_smooth_scale_t = torch_nt(x, w)
opt_out,xq,wq,x_quant_scale,w_quant_scale,x_smooth_scale,w_smooth_scale = smooth_quant_forward(x,w)

print(f"x_quant_scale abs error:{abs_error(x_quant_scale_t, x_quant_scale)}")
print(f"x_smooth_scale abs error:{abs_error(x_smooth_scale_t, x_smooth_scale)}")
print(f"w_quant_scale abs error:{abs_error(w_quant_scale_t, w_quant_scale)}")
print(f"w_smooth_scale abs error :{abs_error(w_smooth_scale_t, w_smooth_scale)}")

print(x_smooth_scale.size())
print(x_smooth_scale_t[:10])
print(x_smooth_scale[:10])
print(x_quant_scale_t[:10])
print(x_smooth_scale[:10])


# quant_check(org_out, xq, wq, opt_out, 'smooth_quant_forward')

#     opt_out,yq,wq,y_scale,w_scale = hadamard_quant_backward(y,w,hm)
#     quant_check(y@w, yq, wq, opt_out, 'hadamard_quant_backward')

#     opt_out,yq,xq,y_scale,x_scale = hadamard_quant_update(y,x,hm)
#     quant_check(y.t()@x, yq, xq, opt_out, 'hadamard_quant_update')
# elif impl == 'fuse':
#     output,x_q,x_s,w_q,w_s = fuse_hadamard_quant_forward(x, w, hm)
#     quant_check(org_out, x_q, w_q, output, 'fuse_hadamard_quant_forward')

#     output,y_q,y_s,w_q,w_s = fuse_hadamard_quant_backward(y, w, hm)
#     quant_check(y@w, y_q, w_q, output, 'fuse_hadamard_quant_backward')

#     output,y_q,y_s,x_q,x_s = fuse_hadamard_quant_update(y,x, hm)
#     quant_check(y.t()@x, y_q, x_q, output, 'fuse_hadamard_quant_update')

# elif impl == 'bit':
#     output,x_bt,w_bt,x_q,w_q,x_scale,w_scale = bit_hadamard_quant_forward(x, w, hm)
#     quant_check(org_out, x_q, w_q, output, 'bit_hadamard_quant_forward')

#     output,y_bt,y_q,w_q,y_scale,w_scale=bit_hadamard_quant_backward(y, w_bt, hm)
#     quant_check(y@w, y_q, w_q, output, 'bit_hadamard_quant_backward')

#     output,y_q,x_q,y_scale,x_scale=bit_hadamard_quant_update(y_bt,x_bt, hm)
#     quant_check(y.t()@x, y_q, x_q, output, 'bit_hadamard_quant_update')



