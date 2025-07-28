
import torch 

d = torch.load("/ossfs/workspace/tmp/vis/backward.bin")
dy = d['dy']
dys = d['dys']
w = d['w']
wt = d['wt']
ws = d['w_smooth_scale']

err = w.float()-wt.float().t()
print(f'{err.abs().sum().item()=}')

out = torch._scaled_mm(dy, wt.t(), dys.view(-1,1), ws.view(1,-1) , use_fast_accum=True, out_dtype=torch.bfloat16)

print(f'{ws.norm()=} {out.max()=}')