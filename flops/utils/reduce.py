import torch
import triton
import triton.language as tl
from triton import Config





@triton.jit
def update_weight_smooth_scale_kernel(x_ptr, smooth_scale_ptr, M, N, H: tl.constexpr, W: tl.constexpr, EVEN: tl.constexpr, ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    # col-wise read, col-wise write
    x_max = tl.zeros((W,),dtype=tl.float32)
    m = tl.cdiv(M, H)
    offs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)
    for i in range(m):
        if EVEN:
            x = tl.load(x_ptr + offs).to(tl.float32)
        else:
            x = tl.load(x_ptr + offs, mask=i * H + tl.arange(0,H)[:,None] < M).to(tl.float32)
        
        x_max = tl.maximum(x_max, tl.max(tl.abs(x), axis=0))
        offs += H*N
    
    scale = x_max
    # scale = 1.0/tl.sqrt(tl.maximum(x_max,1.0))
    # if ROUND:
    #     scale = tl.exp2(tl.ceil(tl.log2(scale)))
    
    tl.store(smooth_scale_ptr + pid * W + tl.arange(0, W), scale)

# update weight smooth scale for next step with x input 
def triton_update_weight_smooth_scale(x, round_scale=False):
    
    N = x.size(-1)
    M = x.numel()//N
    device = x.device 
    weight_smooth_scale = torch.empty((N,), device=device, dtype=torch.float32)
    H = 512
    W = 16
    assert N%W == 0
    EVEN = M%H == 0
    grid = (triton.cdiv(N, W), )
    update_weight_smooth_scale_kernel[grid](
        x,
        weight_smooth_scale,
        M, N,
        H, W, 
        EVEN,
        round_scale,
        num_stages=2,
        num_warps=4
    )
    return weight_smooth_scale





@triton.jit
def batch_count_zero_kernel(input_ptrs, size_ptr, count_ptr, B: tl.constexpr):
    tid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    sm = tl.num_programs(axis=1)
    count = 0

    size = tl.load(size_ptr+tid) 
    input_ptr = tl.load(input_ptrs+tid).to(tl.pointer_type(tl.float32))
    t = tl.cdiv(size, B*sm)
    offs = bid*t*B + tl.arange(0, B)
    for i in range(t):
        x = tl.load(input_ptr + offs, mask=offs < size, other=1).to(tl.float32)
        count += tl.sum(tl.where(x==0,1,0))
        offs += B
    
    tl.store(count_ptr + tid * sm + bid, count)


def triton_batch_count_zero(xs):
    device = xs[0].device
    sizes = torch.tensor([x.numel() for x in xs], dtype=torch.int64, device=device)
    ptrs = torch.tensor([x.data_ptr() for x in xs], dtype=torch.int64, device=device)

    sm = torch.cuda.get_device_properties(device).multi_processor_count
    tensor_count = len(xs)
    counts = torch.empty((tensor_count, sm), device=device, dtype=torch.int64)
    B = 4096
    grid = (tensor_count, sm)
    batch_count_zero_kernel[grid](
        ptrs,
        sizes,
        counts,
        B,
        num_stages=2,
        num_warps=4
    )
    count = counts.sum()
    return count




@triton.jit
def batch_sum_with_ord_kernel(input_ptrs, size_ptr, count_ptr, B: tl.constexpr, ORD: tl.constexpr):
    tid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    sm = tl.num_programs(axis=1)
    sums = 0.0

    size = tl.load(size_ptr+tid) 
    input_ptr = tl.load(input_ptrs+tid).to(tl.pointer_type(tl.float32))
    t = tl.cdiv(size, B*sm)
    offs = bid*t*B + tl.arange(0, B)
    for i in range(t):
        x = tl.load(input_ptr + offs, mask=offs < size, other=0).to(tl.float32)
        if ORD == 2:
            sums += tl.sum(x*x)
        elif ORD == 1:
            sums += tl.sum(tl.abs(x))
        offs += B
    
    tl.store(count_ptr + tid * sm + bid, sums)


def triton_batch_sum_with_ord(xs, ord=2):
    assert ord in (1, 2)
    device = xs[0].device
    sizes = torch.tensor([x.numel() for x in xs], dtype=torch.int64, device=device)
    ptrs = torch.tensor([x.data_ptr() for x in xs], dtype=torch.int64, device=device)

    sm = torch.cuda.get_device_properties(device).multi_processor_count
    tensor_count = len(xs)
    sums = torch.empty((tensor_count, sm), device=device, dtype=torch.float32)
    B = 4096
    grid = (tensor_count, sm)
    batch_sum_with_ord_kernel[grid](
        ptrs,
        sizes,
        sums,
        B,
        ord,
        num_stages=2,
        num_warps=4
    )
    sums = sums.sum()
    return sums

