import time 
from enum import IntEnum
from typing import Tuple
import math
import torch
import triton
import triton.language as tl
from triton import Config



def fp16_forward(x,w):
    return x @ w

def fp16_update(y,x):
    return y.t() @ x

def fp16_backward(y,w):
    return y @ w

@triton.jit
def smooth_kernel_nt(x_ptr, xs_ptr, xs_max_ptr, w_ptr, ws_ptr, ws_max_ptr, M, N, K, eps, BLOCK_SIZE: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)

    offs = pid*BLOCK_K + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_K)[None,:]
    m = tl.cdiv(M, BLOCK_SIZE)
    x_max = tl.zeros((BLOCK_K,),dtype=tl.float32)
    x_ptrs = x_ptr + offs
    for i in range(m):
        x = tl.load(x_ptrs)
        x_max = tl.maximum(tl.max(tl.abs(x), axis=0),x_max)
        x_ptrs += BLOCK_SIZE*K

    # offs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_K)[None,:]
    n = tl.cdiv(N, BLOCK_SIZE)
    w_max = tl.zeros((BLOCK_K,),dtype=tl.float32)
    w_ptrs = w_ptr + offs
    for i in range(n):
        w = tl.load(w_ptrs)
        w_max = tl.maximum(tl.max(tl.abs(w), axis=0),x_max)
        w_ptrs += BLOCK_SIZE*K

    scale = tl.sqrt(x_max*w_max)
    x_scale = x_max/scale
    w_scale = w_max/scale

    x_ptrs = x_ptr + offs
    xs_ptrs = xs_ptr + offs
    xs_offs = tl.arange(0, BLOCK_SIZE)
    xs_max_ptrs =  xs_max_ptr + pid*M + xs_offs
    # xs_max_ptrs =  xs_max_ptr + pid + tl.arange(0, BLOCK_SIZE)*m #2D
    for i in range(m):
        x = tl.load(x_ptrs)
        x = x / x_scale
        xs_max = tl.maximum(tl.max(tl.abs(x), axis=1), eps)
        tl.store(xs_ptrs, x)
        tl.store(xs_max_ptrs, xs_max)
        x_ptrs += BLOCK_SIZE*K
        xs_ptrs += BLOCK_SIZE*K
        xs_max_ptrs += BLOCK_SIZE

    w_ptrs = w_ptr + offs
    ws_ptrs = ws_ptr + offs
    ws_offs = tl.arange(0, BLOCK_SIZE)
    ws_max_ptrs = ws_max_ptr + pid*N + ws_offs
    for i in range(n):
        w = tl.load(w_ptrs)
        ws = w / w_scale
        ws_max = tl.maximum(tl.max(tl.abs(ws), axis=1),eps)
        tl.store(ws_ptrs, ws)
        tl.store(ws_max_ptrs, ws_max)
        w_ptrs += BLOCK_SIZE*K
        ws_ptrs += BLOCK_SIZE*K
        ws_max_ptrs += BLOCK_SIZE


@triton.jit
def row_quant_sm_kernel(x_ptr, q_ptr, s_ptr,  M, K, quant_scale_ptr, BLOCK_SIZE: tl.constexpr, BLOCK_K: tl.constexpr, SCALE_K: tl.constexpr):
    pid = tl.program_id(0)

    offs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_K)[None,:]
    n_block = SCALE_K 
    
    # offs_scale = pid * (M // BLOCK_SIZE) + tl.arange(0, 8)[None,:] #需要加边界 K // BLOCK_K
    offs_scale = pid * BLOCK_SIZE * SCALE_K +  tl.arange(0, BLOCK_SIZE)[:,None]* SCALE_K + tl.arange(0, SCALE_K)[None,:]
    quant_scale_ptrs = quant_scale_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    scale_off = tl.load(s_ptr+offs_scale)
    scale = tl.max(scale_off, axis=1) / 448.0
    tl.store(quant_scale_ptrs, scale)
    scale = scale.expand_dims(-1)
    # if pid == 0:
    #     tl.device_print(scale)

    x_ptrs = x_ptr + offs
    q_ptrs = q_ptr + offs
    for j in range(n_block):
        x = tl.load(x_ptrs)
        # if pid == 0:
        #     tl.device_print(x)
        y = x.to(tl.float32) / scale
        y = y.to(q_ptr.dtype.element_ty)
        tl.store(q_ptrs, y)
        x_ptrs += BLOCK_K
        q_ptrs += BLOCK_K

@triton.jit
def smooth_kernel_tn(y_ptr, yq_ptr,x_ptr, xq_ptr, x_quant_scale_ptr, M, N, K, eps, BLOCK_SIZE: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    offs = pid*BLOCK_SIZE*N + tl.arange(0, BLOCK_SIZE)[:,None]*N + tl.arange(0, BLOCK_K)[None,:] 
    toffs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_K)[:,None]*M + tl.arange(0, BLOCK_SIZE)[None,:]
    scale_ptrs = x_quant_scale_ptr + pid * BLOCK_SIZE +tl.arange(0, BLOCK_SIZE)[None,:]
    scale = tl.load(scale_ptrs)
    n = tl.cdiv(N, BLOCK_K)
    for i in range(n):
        y = tl.trans(tl.load(y_ptr+offs))
        o = (y*scale).to(yq_ptr.dtype.element_ty)
        tl.store(yq_ptr+toffs, o)
        offs += BLOCK_K
        toffs += BLOCK_K*M
        # scale_ptrs += BLOCK_K

    offs = pid*BLOCK_SIZE*K + tl.arange(0, BLOCK_SIZE)[:,None]*K + tl.arange(0, BLOCK_K)[None,:]
    toffs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_K)[:,None]*M + tl.arange(0, BLOCK_SIZE)[None,:]
    k = tl.cdiv(K, BLOCK_K)
    for j in range(k):
        x = tl.trans(tl.load(x_ptr+offs))
        tl.store(xq_ptr+toffs, x)
        offs += BLOCK_K
        toffs += BLOCK_K*M

@triton.jit
def smooth_kernel_nn(y_ptr, yq_ptr,w_ptr, wq_ptr, w_quant_scale_ptr, M, N, K, eps, BLOCK_SIZE: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # offs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:,None]*N + tl.arange(0, BLOCK_N)[None,:] 
    offs = pid*BLOCK_SIZE*N + tl.arange(0, BLOCK_SIZE)[:,None]*N + tl.arange(0, BLOCK_N)[None,:] 
    scale_ptrs = w_quant_scale_ptr + tl.arange(0, BLOCK_N)[None,:]
    scale = tl.load(scale_ptrs)
    # n = tl.cdiv(M, BLOCK_SIZE)
    n = tl.cdiv(N, BLOCK_N)
    for i in range(n):
        y = tl.load(y_ptr+offs)
        o = (y*scale).to(yq_ptr.dtype.element_ty)
        tl.store(yq_ptr+offs, o)
        # offs += BLOCK_SIZE * N
        offs += BLOCK_N
        scale_ptrs += BLOCK_N

@triton.jit
def smooth_kernel_wq(w_ptr, wq_ptr, M, N, K, eps, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    offs = pid*BLOCK_N*K + tl.arange(0, BLOCK_N)[:,None]*K + tl.arange(0, BLOCK_K)[None,:]
    toffs = pid*BLOCK_N + tl.arange(0, BLOCK_K)[:,None]*N + tl.arange(0, BLOCK_N)[None,:]
    k = tl.cdiv(K, BLOCK_K)
    for j in range(k):
        x = tl.trans(tl.load(w_ptr+offs))
        tl.store(wq_ptr+toffs, x)
        offs += BLOCK_K
        toffs += BLOCK_K*N

# v3: smooth + token/channel
def triton_sm_quant_nt(x, w):
    eps = 1e-10
    M, K = x.shape
    N, K = w.shape
    device = x.device 
    x_s = torch.empty((M, K), device=device, dtype=x.dtype)
    w_s = torch.empty((N, K), device=device, dtype=w.dtype)
    x_q = torch.empty((M, K), device=device, dtype=torch.float8_e4m3fn)
    w_q = torch.empty((N, K), device=evice, dtype=torch.float8_e4m3fn)
    x_quant_scale = torch.empty((M,1), device=x.device, dtype=torch.float32)
    w_quant_scale = torch.empty((N,1), device=x.device, dtype=torch.float32)
    # xs_max_tmp = torch.empty(M, device=x.device, dtype=torch.float32)
    # ws_max_tmp = torch.empty(N, device=w.device, dtype=torch.float32)
    
    BLOCK_SIZE = 512 
    BLOCK_K = 64

    # xs_max_tmp = torch.empty((M, K//BLOCK_K), device=x.device, dtype=torch.float32)
    # ws_max_tmp = torch.empty((N, K//BLOCK_K), device=w.device, dtype=torch.float32)
    
    xs_max_tmp = torch.empty(M*(K//BLOCK_K), device=x.device, dtype=torch.float32)
    ws_max_tmp = torch.empty(N*(K//BLOCK_K), device=w.device, dtype=torch.float32)
    # print(f"grid: {K//BLOCK_SIZE}")

    grid = lambda META: (K//BLOCK_K, )
    smooth_kernel_nt[grid](
        x, x_s, xs_max_tmp,
        w, w_s, ws_max_tmp,
        M,N,K,
        eps, 
        BLOCK_SIZE,
        BLOCK_K,
        num_stages=6,
        num_warps=16
    )

    xs_max_tmp = xs_max_tmp.view(M,-1)
    ws_max_tmp = ws_max_tmp.view(N,-1)
    
    # print(x_s)
    # print(torch.sum(x_s==0))
    # print(torch.nonzero(x_s)[:514])
    #70 us for bellow
    # xs_max = xs_max_tmp.view(M, -1).max(1)[0]/448.0
    # ws_max = ws_max_tmp.view(N, -1).max(1)[0]/448.0

    BLOCK_SIZE = 16
    BLOCK_K = 512
    SCALE_K = K // BLOCK_K
    grid = lambda META: (M//BLOCK_SIZE, )
    row_quant_sm_kernel[grid](
        x_s, x_q, xs_max_tmp,
        M,K,
        x_quant_scale,
        BLOCK_SIZE,
        BLOCK_K,
        SCALE_K,
        num_stages=6,
        num_warps=32
    )

    # BLOCK_SIZE = 4096
    # BLOCK_SIZE = 4096
    grid = lambda META: (N//BLOCK_SIZE, )
    row_quant_sm_kernel[grid](
        w_s, w_q, ws_max_tmp,
        N,K,
        w_quant_scale, 
        BLOCK_SIZE,
        BLOCK_K,
        SCALE_K,
        num_stages=6,
        num_warps=32
    )

    return x_q, w_q, x_quant_scale, w_quant_scale


def triton_sm_quant_tn(y, x):
    eps = 1e-10
    M, N = y.shape
    M, K = x.shape
    device = x.device 
    # x_s = torch.empty((M, K), device=x.device, dtype=x.dtype)
    # w_s = torch.empty((N, K), device=w.device, dtype=w.dtype)
    y_q = torch.empty((N, M), device=device, dtype=torch.float8_e4m3fn)
    x_q = torch.empty((K, M), device=x.device, dtype=torch.float8_e4m3fn)
    x_quant_scale = torch.empty((M,1), device=x.device, dtype=torch.float32)#pass from outside 
    
    BLOCK_SIZE = 64
    BLOCK_K = 64 #128

    grid = lambda META: (M//BLOCK_SIZE, )
    smooth_kernel_tn[grid](
        y, y_q,
        x, x_q, x_quant_scale,
        M,N,K,
        eps, 
        BLOCK_SIZE,
        BLOCK_K,
        num_stages=6,
        num_warps=32
    )

def triton_sm_quant_nn(y, w):
    eps = 1e-10
    M, N = y.shape
    N, K = w.shape
    device = w.device 
    # x_s = torch.empty((M, K), device=x.device, dtype=x.dtype)
    # w_s = torch.empty((N, K), device=w.device, dtype=w.dtype)
    y_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    w_q = torch.empty((K, N), device=device, dtype=torch.float8_e4m3fn)
    w_quant_scale = torch.empty((1,N), device=device, dtype=torch.float32)#pass from outside 
    
    BLOCK_SIZE = 64
    BLOCK_N = 512

    # grid = lambda META: (N//BLOCK_N, )
    grid = lambda META: (M//BLOCK_SIZE, )
    smooth_kernel_nn[grid](
        y, y_q,
        w, w_q, w_quant_scale,
        M,N,K,
        eps, 
        BLOCK_SIZE,
        BLOCK_N,
        num_stages=6,
        num_warps=32
    )

    BLOCK_K = 64
    BLOCK_N = 64
    grid = lambda META: (N//BLOCK_N, )
    smooth_kernel_wq[grid](
        w, w_q,
        M,N,K,
        eps, 
        BLOCK_K,
        BLOCK_N,
        num_stages=6,
        num_warps=32
    )


def smooth_quant_forward(x,w,hm):
    x_q, w_q, x_quant_scale, w_quant_scale = triton_sm_quant_nt(x, w)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_quant_scale,
                                    scale_b=w_quant_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,x_q,w_q,x_quant_scale,w_quant_scale

def smooth_quant_backward(y,w,w_quant_scale):

    y_q,w_q,y_scale,w_scale = triton_sm_quant_nn(y, w, w_quant_scale)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_scale,
                                    scale_b=w_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_q,w_q,y_scale,w_scale


def hadamard_quant_update(y,x,x_quant_scale):
    y_q,x_q,y_scale,x_scale = triton_sm_quant_tn(y, x, hm)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_scale,
                                    scale_b=x_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_q,x_q,y_scale,x_scale


def triton_hadamard_quant_nt_nn_tn(x,w,y,hm):
    triton_hadamard_quant_nt(x, w, hm)
    triton_hadamard_quant_nn(y, w, hm)
    triton_hadamard_quant_tn(y, x, hm)


def fp8_hadamard_f_and_b(x,w,y,hm):
    hadamard_quant_forward(x, w, hm)
    hadamard_quant_backward(y, w, hm)
    hadamard_quant_update(y,x, hm)


# y = x @ w
def triton_fused_hadamard_quant_nt(x, w, hm):
    stream = torch.cuda.Stream(device=0)
    x_q,x_s = triton_fused_hadamard(x, hm, hm_side=1, op_side=0)
    with torch.cuda.stream(stream):
        w_q,w_s = triton_fused_hadamard(w, hm, hm_side=1, op_side=1)
    torch.cuda.current_stream().wait_stream(stream)
    return x_q,x_s,w_q,w_s


# dx = y @ wT
def triton_fused_hadamard_quant_nn(y, w, hm):
    stream = torch.cuda.Stream(device=0)
    y_q,y_s = triton_fused_hadamard(y, hm, hm_side=1, op_side=0)
    with torch.cuda.stream(stream):
        w_q,w_s = triton_fused_transpose_hadamard(w, hm, hm_side=1, op_side=1)
    torch.cuda.current_stream().wait_stream(stream)
    return y_q,y_s,w_q,w_s


# dwT = yT @ x
def triton_fused_hadamard_quant_tn(y, x, hm):
    stream = torch.cuda.Stream(device=0)
    y_q,y_s = triton_fused_transpose_hadamard(y, hm, hm_side=1, op_side=0)
    with torch.cuda.stream(stream):
        x_q,x_s = triton_fused_transpose_hadamard(x, hm, hm_side=1, op_side=1)
    torch.cuda.current_stream().wait_stream(stream)
    return y_q,y_s,x_q,x_s

def triton_fuse_hadamard_quant_nt_nn_tn(x,w,y,hm):
    triton_fused_hadamard_quant_nt(x, w, hm)
    triton_fused_hadamard_quant_nn(y, w, hm)
    triton_fused_hadamard_quant_tn(y, x, hm)

def fp8_fuse_hadamard_f_and_b(x,w,y,hm):
    fuse_hadamard_quant_forward(x, w, hm)
    fuse_hadamard_quant_backward(y, w, hm)
    fuse_hadamard_quant_update(y,x, hm)





def fuse_hadamard_quant_forward(x,w,hm):

    x_q,x_s,w_q,w_s = triton_fused_hadamard_quant_nt(x, w, hm)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_s,
                                    scale_b=w_s,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,x_q,x_s,w_q,w_s

def fuse_hadamard_quant_backward(y,w,hm):

    y_q,y_s,w_q,w_s = triton_fused_hadamard_quant_nn(y, w, hm)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_s,
                                    scale_b=w_s,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_q,y_s,w_q,w_s



def fuse_hadamard_quant_update(y,x,hm):
    y_q,y_s,x_q,x_s = triton_fused_hadamard_quant_tn(y, x, hm)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_s,
                                    scale_b=x_s,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_q,y_s,x_q,x_s


def bit_hadamard_quant_forward(x,w,hm):

    x_bt,w_bt, x_q,w_q,x_scale,w_scale = triton_bit_hadamard_quant_nt(x, w, hm)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_scale,
                                    scale_b=w_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,x_bt,w_bt,x_q,w_q,x_scale,w_scale

def bit_hadamard_quant_backward(y,w,hm):

    y_bt,y_q,w_q,y_scale,w_scale = triton_bit_hadamard_quant_nn(y, w, hm)
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_scale,
                                    scale_b=w_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_bt,y_q,w_q,y_scale,w_scale



def bit_hadamard_quant_update(y,x,hm):
    y_q,x_q,y_scale,x_scale = triton_bit_hadamard_quant_tn(y, x, hm)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_scale,
                                    scale_b=x_scale,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_q,x_q,y_scale,x_scale



def fp8_bit_hadamard_f_and_b(x,w,y,hm):
    output,x_bt,w_bt,x_q,w_q,x_scale,w_scale = bit_hadamard_quant_forward(x, w, hm)
    output,y_bt,y_q,w_q,y_scale,w_scale=bit_hadamard_quant_backward(y, w_bt, hm)
    output,y_q,x_q,y_scale,x_scale=bit_hadamard_quant_update(y_bt,x_bt, hm)



@triton.jit
def row_quant_kernel(x_ptr, q_ptr, s_ptr,  M, N,  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    n_block = tl.cdiv(N, BLOCK_SIZE)
    indices = tl.arange(0, BLOCK_SIZE)
    max_val = 1e-6
    for j in range(n_block):
        offs = pid*N + j*BLOCK_SIZE + indices
        x = tl.load(x_ptr + offs, mask=j*BLOCK_SIZE + indices<N, other=0)
        max_val = tl.maximum(tl.max(tl.abs(x.to(tl.float32))), max_val)
    scale = max_val/448.0
    tl.store(s_ptr + pid, scale)
    for j in range(n_block):
        offs = pid*N + j*BLOCK_SIZE + indices
        x = tl.load(x_ptr + offs, mask=j*BLOCK_SIZE + indices<N, other=0)
        y = x.to(tl.float32) / scale
        y = y.to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + offs, y, mask=j*BLOCK_SIZE + indices<N)


def triton_row_quant(x):
    M, N = x.shape 
    BLOCK_SIZE = 4096
    x_q = torch.empty((M,N),dtype=torch.float8_e4m3fn,device=x.device)
    x_scale = torch.empty((M,1),dtype=torch.float32,device=x.device)
    grid = lambda META: (M, )
    row_quant_kernel[grid](
        x, x_q, x_scale,
        M,N,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=8
    )
    return x_q, x_scale



def fp16_f_and_b(x,w,y):
    y = x@w.t()
    dw = y.t()@x
    dx = y@w
    return y, dw, dx



def benchmark_func(fn, *args, n_repeat=1000, ref_flops=None, ref_time=None, name='', **kwargs):
    func_name = fn.__name__

    for i in range(100):
        fn(*args,**kwargs)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    
    ts = time.time()
    for i in range(n_repeat):
        start_events[i].record()
        fn(*args,**kwargs)
        end_events[i].record()
    
    torch.cuda.synchronize() 
    te = time.time()
    
    # times = sum([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
    # average_event_time = times * 1000 / n_repeat

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times = sorted(times)
    clip = max(1,n_repeat//100)
    times = sum(times[clip:-clip])
    
    average_event_time = times * 1000 / (n_repeat - 2*clip)
    
    fs = ''
    if ref_flops is not None:
        flops = ref_flops/1e12/(average_event_time/1e6)
        fs = f'FLOPS:{flops:.2f}T'
    ss = ''
    if ref_time is not None:
        ss = f'speedup:{ref_time/average_event_time:.3f}'
    print(f'{func_name} {name} time:{average_event_time:.1f} us {fs} {ss}')
    return average_event_time


# 5b: hidden_size:4k  seq_length:8K shape:(M,N,K)
# qkv: 8192, 6144, 4096
# out: 8192, 4096, 4096
# up/gate: 8192, 13312, 4096
# down: 8192, 4096, 13312   # benchmark setting


# 80b: hidden_size:8k  seq_length:8K shape:(M,N,K)
# qkv: 8192, 10240, 8192
# out: 8192, 8192, 8192
# up/gate: 8192, 34048, 8192
# down: 8192, 4096, 13312



def benchmark_with_shape(shape):
    batch_size, out_dim, in_dim = shape
    device = 'cuda:0'
    dtype = torch.bfloat16
    qtype = torch.float8_e4m3fn  # torch.float8_e5m2
    n_repeat = 1000
    gpu = torch.cuda.get_device_properties(0).name


    x = torch.randn(batch_size, in_dim, dtype=dtype, device=device)
    w = torch.randn(out_dim, in_dim, dtype=dtype, device=device)
    y = torch.randn(batch_size, out_dim, dtype=dtype, device=device)
    x_f8 = x.to(qtype)
    w_f8 = w.to(qtype)
    y_f8 = y.to(qtype)
    B = 64
    hm = hadamard_matrix(B, dtype=dtype, device=device)

    org_out = fp16_forward(x, w.t())
    print(f'\ndevice:{gpu} M:{batch_size} N:{out_dim} K:{in_dim}')

    # y = x @ w
    # dx = y @ wT
    # dwT = yT @ x

    # benchmark_func(triton_hadamard_nt, x, w, hm, n_repeat=n_repeat)
    # benchmark_func(triton_row_quant, x, n_repeat=n_repeat)
    # benchmark_func(triton_row_quant, w, n_repeat=n_repeat)

    benchmark_func(triton_hadamard_quant_nt, x, w, hm, n_repeat=n_repeat)
    benchmark_func(triton_hadamard_quant_tn, y, x, hm, n_repeat=n_repeat)
    benchmark_func(triton_hadamard_quant_nn, y, w, hm, n_repeat=n_repeat)

    benchmark_func(triton_fused_hadamard, x, hm, hm_side=1, op_side=0)
    benchmark_func(triton_fused_transpose_hadamard, x, hm, hm_side=1, op_side=0)
    benchmark_func(triton_fused_hadamard_quant_nt, x,w,hm, n_repeat=n_repeat)
    benchmark_func(triton_fused_hadamard_quant_nn, y,x,hm, n_repeat=n_repeat)
    benchmark_func(triton_fused_hadamard_quant_tn, y,w,hm, n_repeat=n_repeat)
    
    benchmark_func(triton_hadamard_quant_nt_nn_tn, x,w,y,hm, n_repeat=n_repeat)
    benchmark_func(triton_fuse_hadamard_quant_nt_nn_tn, x,w,y,hm, n_repeat=n_repeat)


    # benchmark_func(triton_bit_hadamard_nt, x, w, hm, n_repeat=n_repeat)
    # benchmark_func(triton_bit_hadamard_quant_nt, x, w, hm, n_repeat=n_repeat)
    # benchmark_func(triton_bit_hadamard_quant_nn, y, w.t().contiguous(), hm, n_repeat=n_repeat)
    # benchmark_func(triton_bit_hadamard_quant_tn, y.t().contiguous(), x.t().contiguous(), hm, n_repeat=n_repeat)


    # ref_time = benchmark_func(fp16_forward, x, w.t(), n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
    # benchmark_func(hadamard_quant_forward, x, w, hm, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)
    # ref_time = benchmark_func(fp16_backward, y, w, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
    # benchmark_func(hadamard_quant_backward, y, w, hm, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)
    # ref_time = benchmark_func(fp16_update, y, x, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2)
    # benchmark_func(hadamard_quant_update, y, x, hm, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*2, ref_time=ref_time)

    ref_time = benchmark_func(fp16_f_and_b, x, w, y, n_repeat=n_repeat, ref_flops=batch_size*in_dim*out_dim*6)
    benchmark_func(fp8_hadamard_f_and_b, x, w, y, hm, n_repeat=n_repeat, ref_time=ref_time,ref_flops=batch_size*in_dim*out_dim*6)
    benchmark_func(fp8_fuse_hadamard_f_and_b, x, w, y, hm, n_repeat=n_repeat, ref_time=ref_time,ref_flops=batch_size*in_dim*out_dim*6)
    benchmark_func(fp8_bit_hadamard_f_and_b, x, w, y, hm, n_repeat=n_repeat, ref_time=ref_time,ref_flops=batch_size*in_dim*out_dim*6)



# 5b: hidden_size:4k  seq_length:8K shape:(M,N,K)
# qkv: 8192, 6144, 4096
# out: 8192, 4096, 4096
# up/gate: 8192, 13312, 4096
# down: 8192, 4096, 13312   # benchmark setting


# 80b: hidden_size:8k  seq_length:8K shape:(M,N,K)
# qkv: 8192, 10240, 8192
# out: 8192, 8192, 8192
# up/gate: 8192, 34048, 8192
# down: 8192, 4096, 34048


# benchmark_with_shape([8192, 4096, 13312])

for shape in [[8192, 6144, 4096], [8192, 4096, 4096], [8192, 13312, 4096], [8192, 4096, 13312],
            [8192, 10240, 8192],[8192, 8192, 8192],[8192, 34048, 8192],[8192, 4096, 34048]]:
    benchmark_with_shape(shape)