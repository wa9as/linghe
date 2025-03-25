

import torch 
from flops.core.hadamard_quant_linear import HadamardQuantLinear


device = 'cuda:0'
dtype = torch.bfloat16
M, N, K = 8192, 4096, 13312

layer = HadamardQuantLinear(in_features=K,
        out_features=N,
        bias = False,
        device = device,
        dtype = dtype,
        hadamard_matrix_size=64)


x = torch.randn((M,K), dtype=dtype, device=device) 
x = torch.nn.parameter.Parameter(x, requires_grad=True)  # to mock input in training

y=layer(x)
loss = (y**2).sum()
loss.backward()

print("x",x)
print("y",y)
print("grad", layer.weight.grad)