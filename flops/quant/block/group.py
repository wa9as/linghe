
from enum import IntEnum
from typing import Tuple

import torch
import triton
import triton.language as tl
from triton import Config



@triton.jit
def group_quant_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr, K: tl.constexpr, ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * N + tl.arange(0, K*BLOCK_SIZE)
    n = tl.cdiv(N, K*BLOCK_SIZE)
    soffs = pid * n * K + tl.arange(0, K)
    for i in range(n):
        x = tl.load(x_ptr + offs).to(tl.float32)
        x = tl.reshape(x, (K, BLOCK_SIZE), can_reorder=False)
        s = tl.maximum(tl.max(tl.abs(x),1), 1e-30) / 448.0
        if ROUND:
            s = tl.exp2(tl.floor(tl.log2(s) + 0.5))
        y = x / s[:,None]
        y = y.to(y_ptr.dtype.element_ty)
        y = tl.reshape(y, (K*BLOCK_SIZE,), can_reorder=False)
        tl.store(y_ptr + offs, y)
        tl.store(s_ptr + soffs, s)
        offs += K*BLOCK_SIZE
        soffs += K



def triton_group_quant(
    x: torch.Tensor, dtype=torch.float8_e4m3fn, group_size: int = 128, round_scale = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert round_scale 
    M, N= x.shape
    K = 16
    assert N % group_size == 0 and N%(group_size*K) == 0
    assert x.is_contiguous()

    y = torch.empty((M,N), device=x.device, dtype=dtype)
    s = torch.empty(M, N // group_size, device=x.device, dtype=torch.float32)
    grid = (M,)  # noqa: E731
    group_quant_kernel[grid](x, 
                             y, 
                             s, 
                             M, 
                             N, 
                             group_size, 
                             K, 
                             round_scale, 
                             num_stages=5, 
                             num_warps=4)
    return y, s



@triton.jit
def persist_group_quant_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr, B:tl.constexpr,  K: tl.constexpr, ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * B * N + tl.arange(0, B)[:,None] * N + tl.arange(0, K*BLOCK_SIZE)[None,:]
    n = tl.cdiv(N, K*BLOCK_SIZE)
    soffs = pid * B * n * K + tl.arange(0, B)[:,None] * n * K + tl.arange(0, K)[None,:]

    for j in range(n):
        x = tl.load(x_ptr + offs).to(tl.float32)
        x = tl.reshape(x, (B, K, BLOCK_SIZE))

        s = tl.maximum(tl.max(tl.abs(x),2), 1e-30) / 448.0
        if ROUND:
            s = tl.exp2(tl.floor(tl.log2(s)))
        y = x / s[:,:,None]
        y = y.to(y_ptr.dtype.element_ty)
        y = tl.reshape(y, (B, K*BLOCK_SIZE))
        tl.store(y_ptr + offs, y)
        tl.store(s_ptr + soffs, s)
        offs += K*BLOCK_SIZE
        soffs += K


def triton_persist_group_quant(
    x: torch.Tensor, dtype=torch.float8_e4m3fn, group_size: int = 128, round_scale = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, N= x.shape
    device = x.device
    K = 8
    B = 8
    assert N % group_size == 0 and N%(group_size*K) == 0
    assert x.is_contiguous()

    y = torch.empty((M,N), dtype=dtype, device=device)
    s = torch.empty(M, N // group_size, device=x.device, dtype=torch.float32)

    grid = (M//B,)  # noqa: E731
    persist_group_quant_kernel[grid](x, 
                                     y, 
                                     s, 
                                     M, 
                                     N, 
                                     group_size, 
                                     B, 
                                     K, 
                                     round_scale, 
                                     num_stages=3, 
                                     num_warps=8)
    return y, s
