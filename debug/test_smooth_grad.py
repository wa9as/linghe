
import torch 
from flops.tools.util import torch_smooth_quant,output_check


M,N,K = 8192,2048,4096
y = torch.randn((M,N),dtype=torch.bfloat16,device='cuda:0')

w = torch.randn((N,K),dtype=torch.bfloat16,device='cuda:0')
w_smooth_scale = torch.randn((K,),dtype=torch.float32,device='cuda:0').abs()+0.1
w_quant_scale = torch.randn((N,),dtype=torch.float32,device='cuda:0').abs()+0.1


ref_dx = y@w 
