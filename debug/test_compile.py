import torch 
import torch.nn.functional as F
from flops.tools.benchmark import benchmark_func



@torch.compile 
def gate_attn_output(attn_output, gate, weight, eps, group_size):
    bs, length, nh, dim = attn_output.shape
    d = nh*dim//group_size
    attn_output = attn_output.view(bs, length, group_size, d).transpose(0, 1)
    outputs = []
    for i in range(group_size):
        outputs.append(F.rms_norm(attn_output[:,:,i], [d], weight=weight[i*d:(i+1)*d], eps=eps))
    outputs = torch.stack(outputs, 2)
    outputs = outputs.view(length, bs, nh*dim)
    gate = F.sigmoid(gate) 
    return outputs * gate


bs, length, nh, dim = 4, 4096, 16, 128
group_size = 4
x = torch.randn((bs, length, nh, dim), dtype=torch.bfloat16, device='cuda:0')
gate = torch.randn((length, bs, nh*dim), dtype=torch.bfloat16, device='cuda:0')
weight = torch.randn((nh*dim, ), dtype=torch.bfloat16, device='cuda:0')
eps = 1e-6 
benchmark_func(gate_attn_output,x,gate,weight,eps,group_size,ref_bytes=bs* length* nh* dim*6)