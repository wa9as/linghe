import random

import torch

from flops.utils.rope import triton_half_rope_forward, triton_half_rope_backward, triton_qk_norm_and_half_rope_forward, triton_qk_norm_and_half_rope_backward
from flops.tools.benchmark import benchmark_func
from flops.tools.util import output_check


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids][:,:,None]
    sin = sin[position_ids][:,:,None]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rope_freqs(length, dim, rope_theta=10000.0):
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device='cuda:0').float() / dim))
    t = torch.arange(length, device='cuda:0', dtype=torch.int64).float()
    freqs = torch.outer(t, inv_freq)
    return freqs

def torch_half_rope(q,k, freqs, rope_theta=10000.0):
    L, B, H, D = q.shape 
    d = D//2
    h = k.shape[2]
    cos = freqs.cos().to(q.dtype)
    sin = freqs.sin().to(q.dtype)
    position_ids = torch.arange(L, device='cuda:0')[:,None].expand(-1,B)
    qr, kr = apply_rotary_pos_emb(q[:,:,:,:d], k[:,:,:,:d], cos, sin, position_ids)
    qo = torch.cat([qr,q[:,:,:,d:]],dim=-1)
    ko = torch.cat([kr,k[:,:,:,d:]],dim=-1)
    return qo,ko

def torch_qk_norm(q,k, qw, kw, eps=1e-6):
    dtype = q.dtype
    L, B, H, D = q.shape 
    rms = torch.sqrt(q.float().square().mean(-1)+eps)
    q = q/rms[:,:,:,None]
    q = q*qw
    rms = torch.sqrt(k.float().square().mean(-1)+eps)
    k = k/rms[:,:,:,None]
    k = k*kw
    return q.to(dtype),k.to(dtype)

def torch_qk_norm_and_half_rope(qkv,qw,kw,freqs, rope_theta=10000.0,H=32,h=4, eps=1e-6,interleave=True):
    length, bs, dim = qkv.shape
    qkv = qkv.float()
    qw = qw.float()
    kw = kw.float()
    D = dim//(H+2*h)
    if interleave:
        qkv = qkv.view(length, bs, h, (2+H//h)*D)
        q,k,v = torch.split(qkv, [H//h*D, D, D], 3) 
        q = torch.reshape(q, (length, bs, H, D))
    else:
        qkv = qkv.view(length, bs, H+2*h, D)
        q,k,v = torch.split(qkv, [H,h,h], dim=2)
    q, k = torch_qk_norm(q, k, qw, kw, eps=eps)
    q, k = torch_half_rope(q, k, freqs, rope_theta=rope_theta)
    q = q.transpose(0,1)
    k = k.transpose(0,1)
    v = v.transpose(0,1)
    return q,k,v


def test_half_rope(B=2,L=4096,H=32,h=8,D=128,rope_theta=10000.0, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'
    q = torch.randn(L,B,H,D,dtype=dtype,device=device)
    k = torch.randn(L,B,h,D,dtype=dtype,device=device)
    freqs = rope_freqs(L, D//2, rope_theta=rope_theta)
    freqs = torch.cat([freqs,freqs], -1)

    q_ref,k_ref = torch_half_rope(q,k,freqs,rope_theta=rope_theta)
    qo,ko = triton_half_rope_forward(q,k,freqs)
    output_check(q_ref,qo, mode='q')
    output_check(k_ref,ko, mode='k')

    q_grad = torch.randn(L,B,H,D,dtype=dtype,device=device)
    k_grad = torch.randn(L,B,h,D,dtype=dtype,device=device)
    q_ref = q.detach().clone().requires_grad_()
    k_ref = k.detach().clone().requires_grad_()
    qo_ref, ko_ref = torch_half_rope(q_ref,freqs,k_ref,rope_theta=rope_theta)
    qo_ref.backward(gradient=q_grad)
    ko_ref.backward(gradient=k_grad)
    dq_ref = q_ref.grad 
    dk_ref = k_ref.grad 

    dq, dk = triton_half_rope_backward(q_grad, k_grad, freqs, inplace=True)
    output_check(dq_ref,dq, mode='dq')
    output_check(dk_ref,dk, mode='dk')

    if bench:
        benchmark_func(triton_half_rope_forward, q,k,freqs, ref_bytes=L*B*(H+h)*D*4, 
                       n_profile=0)



def test_qk_norm_and_half_rope(B=2,L=4096,H=32,h=8,D=128,rope_theta=10000.0,interleave=False, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'
    qkv = torch.randn(L,B,(H+2*h)*D,dtype=dtype,device=device)
    qw = torch.randn(D,dtype=dtype,device=device)
    kw = torch.randn(D,dtype=dtype,device=device)
    freqs = rope_freqs(L, D//2, rope_theta=rope_theta)
    freqs = torch.cat([freqs,freqs], -1)
    q_ref,k_ref,v_ref = torch_qk_norm_and_half_rope(qkv,qw,kw,freqs,rope_theta, H=H,h=h, eps=1e-6,interleave=interleave)
    qo,ko,vo = triton_qk_norm_and_half_rope_forward(qkv,qw,kw,freqs,H=H,h=h,eps=1e-6,transpose=True,interleave=interleave)
    output_check(q_ref,qo, mode='q')
    output_check(k_ref,ko, mode='k')
    output_check(v_ref,vo, mode='v')


    q_grad = torch.randn(B,L,H,D,dtype=dtype,device=device)
    k_grad = torch.randn(B,L,h,D,dtype=dtype,device=device)
    v_grad = torch.randn(B,L,h,D,dtype=dtype,device=device)
    qkv_ref = qkv.detach().clone().requires_grad_()
    qw_ref = qw.detach().clone().requires_grad_()
    kw_ref = kw.detach().clone().requires_grad_()
    qo_ref, ko_ref, vo_ref = torch_qk_norm_and_half_rope(qkv_ref,qw_ref,kw_ref,freqs,rope_theta=rope_theta, H=H, h=h, eps=1e-6)
    qo_ref.backward(gradient=q_grad)
    ko_ref.backward(gradient=k_grad)
    vo_ref.backward(gradient=v_grad)

    dqkv_ref = qkv_ref.grad 
    dqw_ref = qw_ref.grad
    dkw_ref = kw_ref.grad

    dqkv, dqw, dkw = triton_qk_norm_and_half_rope_backward(q_grad, k_grad, v_grad, qkv, qw, kw, freqs, eps=1e-6, transpose=True)
    output_check(dqkv_ref,dqkv, mode='dqkv')
    output_check(dqw_ref,dqw, mode='dqw')
    output_check(dkw_ref,dkw, mode='dkw')

    if bench:
        benchmark_func(triton_qk_norm_and_half_rope_forward, qkv,qw,kw,freqs,H=H,h=h,eps=1e-6,transpose=True, ref_bytes=L*B*(H+2*h)*D*4, 
                       n_profile=0)
        benchmark_func(triton_qk_norm_and_half_rope_backward, q_grad, k_grad, v_grad, qkv, qw, kw, freqs,eps=1e-6, transpose=True, ref_bytes=L*B*(H+2*h)*D*6, 
                       n_profile=0)

if __name__ == '__main__':
    test_half_rope(B=2,L=4096,H=32,h=8,D=128,rope_theta=10000.0, bench=False)
    test_qk_norm_and_half_rope(B=2,L=4096,H=16,h=16,D=128,rope_theta=10000.0,interleave=True,bench=False)
    test_qk_norm_and_half_rope(B=4,L=4096,H=16,h=4,D=128,rope_theta=10000.0,interleave=True,bench=False)
    test_qk_norm_and_half_rope(B=4,L=4096,H=32,h=8,D=128,rope_theta=10000.0,interleave=True,bench=False)
    test_qk_norm_and_half_rope(B=4,L=4096,H=32,h=8,D=128,rope_theta=10000.0,interleave=False,bench=False)