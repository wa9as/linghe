

import torch 
from flops.core.hadamard_quant_linear import QuantLinear as HadamardQuantLinear
from flops.core.smooth_quant_linear import QuantLinear as SmoothQuantLinear


device = 'cuda:0'
dtype = torch.bfloat16
M, N, K = 8192, 4096, 13312

layer = HadamardQuantLinear(in_features=K,
        out_features=N,
        bias = False,
        device = device,
        dtype = dtype,
        impl='bit')


# layer = SmoothQuantLinear(in_features=K,
#         out_features=N,
#         bias = False,
#         device = device,
#         dtype = dtype,
#         impl='reused')


x = torch.randn((M,K), dtype=dtype, device=device) 
x = torch.nn.parameter.Parameter(x, requires_grad=True)  # to mock input in training

y=layer(x)
loss = (y**2).sum()
loss.backward()

print("x",x)
print("y",y)
print("grad", layer.weight.grad)