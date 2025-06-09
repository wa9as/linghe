
import numpy as np

import torch
import triton
import triton.language as tl
from triton import Config


@triton.jit
def reused_smooth_quant_tma_kernel(x_desc_ptr, q_desc_ptr, ss_ptr, qs_ptr, M, N, H: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    # offs = pid*W*N + tl.arange(0, W)[:,None]*N + tl.arange(0, H)[None,:]
    offs = pid*W*N
    soffs = tl.arange(0, H)
    x_max = tl.zeros((W,),dtype=tl.float32) + 5.27e-36  # torch.finfo(torch.float32).tiny*448
    n = tl.cdiv(N, H)
    for i in range(n):
        # x = tl.load(x_ptr+offs)
        offs_w = i*H
        x = tl._experimental_descriptor_load(x_desc_ptr, [offs, offs_w], [W, H], tl.float16)
        scale = tl.load(ss_ptr+soffs)
        x = x.to(tl.float32) * scale
        x_max = tl.maximum(tl.max(tl.abs(x), axis=1),x_max)
        # offs += H 
        soffs += H

    scale = x_max/448.0
    tl.store(qs_ptr+pid*W+tl.arange(0, W), scale)

    s = (1.0/scale)[:,None]
    # offs = pid*W*N + tl.arange(0, W)[:,None]*N + tl.arange(0, H)[None,:]
    offs = pid*W*N 
    soffs = tl.arange(0, H)
    for i in range(n):
        # x = tl.load(x_ptr+offs)
        offs_w = i*H
        x = tl._experimental_descriptor_load(x_desc_ptr, [offs, offs_w], [W, H], tl.float16)
        smooth_scale = tl.load(ss_ptr+soffs)
        xq = (x.to(tl.float32) * smooth_scale * s).to(tl.float8e4nv)
        # tl.store(q_ptr+offs, xq)
        tl._experimental_descriptor_store(q_desc_ptr, xq, [offs, offs_w])
        # offs += H 
        soffs += H


def triton_reused_smooth_quant_tma(x, smooth_scale):
    M, N = x.shape
    device = x.device 
    x_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((M,1), device=device, dtype=torch.float32)
    # H = 1024 if N%1024 == N else 256
    H = 32
    W = 16


    TMA_SIZE = 128
    desc_x = np.empty(TMA_SIZE, dtype=np.int8)
    desc_xq = np.empty(TMA_SIZE, dtype=np.int8)

    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(x.data_ptr(), M, N, W, H, x.element_size(),
                                                              desc_x)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(x_q.data_ptr(), M, N, W, H, x_q.element_size(),
                                                              desc_xq)

    desc_x = torch.tensor(desc_x, device=device)
    desc_xq = torch.tensor(desc_xq, device=device)
    
    grid = lambda META: (M//W, )
    reused_smooth_quant_tma_kernel[grid](
        desc_x,
        desc_xq,
        smooth_scale,
        x_scale,
        M, N,
        H, W,
        num_stages=5,
        num_warps=4
    )

    return x_q,x_scale

