import torch

from flops.tools.util import output_check

M, N = 2049, 8192
n = 64

xs = [torch.randn((M, N), dtype=torch.float32, device='cuda:0') for _ in
      range(n)]

output_float32 = sum(xs)
output_float16 = sum([x.half() for x in xs])
output_bfloat16 = sum([x.to(torch.bfloat16) for x in xs])
output_mix = torch.zeros((M, N), dtype=torch.float32, device='cuda:0')
for x in xs:
    output_mix += x.to(torch.bfloat16)

output_mix_2 = torch.zeros((M, N), dtype=torch.float16, device='cuda:0')
for x in xs:
    output_mix_2 += x.to(torch.bfloat16).to(torch.float16)

output_check(output_float32, output_float16, mode='float16')
output_check(output_float32, output_bfloat16, mode='bfloat16')
output_check(output_float32, output_mix, mode='mix')
output_check(output_float32, output_mix_2, mode='mix_2')
