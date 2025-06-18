import torch
from flops.quant.hadamard.fused_hadamard import triton_fused_hadamard



def triton_fused_hadamard_quant_nt(x, w, hm):
    stream = torch.cuda.Stream(device=0)
    x_q,x_s = triton_fused_hadamard(x, hm, hm_side=1, op_side=0)
    # stream.wait_stream(torch.cuda.default_stream(x.device))  # STILL REQUIRED!
    with torch.cuda.stream(stream):
        w_q,w_s = triton_fused_hadamard(w, hm, hm_side=1, op_side=1)
    # torch.cuda.current_stream().wait_stream(stream)

    return x_q,x_s,w_q,w_s



M,N,K = 8192,8192,8192
device = 'cuda:0'
dtype = torch.bfloat16
x = torch.empty((M,K),dtype=dtype,device=device)
w = torch.empty((N,K),dtype=dtype,device=device)
hm = torch.empty((64,64),dtype=dtype,device=device)
# hm = hm.to('cuda:0')

x_q,x_s,w_q,w_s = triton_fused_hadamard_quant_nt(x, w, hm)
x_q,x_s,w_q,w_s = triton_fused_hadamard_quant_nt(x, w, hm)


print(x_s,w_s)