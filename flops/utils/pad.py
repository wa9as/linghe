import math
import random 
import os
import torch
import triton
import triton.language as tl
from triton import Config


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


@triton.jit
def slice_and_pad_kernel(x_ptr, index_ptr, out_ptr, DIM: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    indices = tl.arange(0, BLOCK)
    b = DIM//BLOCK 
    for j in range(b):
        index = tl.load(index_ptr+pid)
        x = tl.load(x_ptr + index*DIM + j*BLOCK + indices)
        tl.store(out_ptr+pid*DIM + j*BLOCK + indices, x)


def triton_slice_and_pad(x, indices, block=32):
    bs, dim = x.shape
    size = indices.size(0)
    device = x.device
    output = torch.empty((((size-1)//block+1)*block, dim), device=device, dtype=x.dtype)
    num_stages = 5
    num_warps = 8
    grid = lambda META: (size, )
    slice_and_pad_kernel[grid](
        x, 
        indices,
        output,
        DIM=dim, 
        BLOCK=1024,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return output


@triton.jit
def batch_slice_and_pad_kernel(x_ptr, indices_ptrs, size_ptr, out_ptrs, GROUP: tl.constexpr, DIM: tl.constexpr, BLOCK: tl.constexpr):

    pid = tl.program_id(axis=0)
    indices = tl.arange(0, BLOCK)
    for g in range(GROUP):
        size = tl.load(size_ptr+g)
        if pid < size:
            i_ptr = tl.load(indices_ptrs + g).to(tl.pointer_type(tl.int32))
            o_ptr = tl.load(out_ptrs + g).to(tl.pointer_type(x_ptr.dtype.element_ty))
            b = DIM//BLOCK 
            idx_off = tl.load(i_ptr + pid)
            for j in range(b):
                x = tl.load(x_ptr + idx_off*DIM + j*BLOCK + indices)
                tl.store(o_ptr+pid*DIM + j*BLOCK + indices, x)



def triton_batch_slice_and_pad(x, indices, expert=64, block=32):
    bs, dim = x.shape
    device = x.device
    assert len(indices) == expert
    sizes = [x.size(0) for x in indices]
    max_size = max(sizes)
    outputs = [torch.empty((((size-1)//block+1)*block, dim), device=device, dtype=x.dtype) for size in sizes]
    indices_ptrs = torch.tensor([i.data_ptr() for i in indices], device=x.device)
    out_ptrs = torch.tensor([i.data_ptr() for i in outputs], device=x.device)
    sizes = torch.tensor(sizes, device=x.device, dtype=torch.int32)
    num_stages = 5
    num_warps = 8
    grid = lambda META: (max_size, )
    batch_slice_and_pad_kernel[grid](
        x, 
        indices_ptrs,
        sizes,
        out_ptrs,
        GROUP=len(indices),
        DIM=dim, 
        BLOCK=1024,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return outputs


def batch_index_expert(logits, min_val=0.0):
    length, expert = logits.shape 
    indices = []
    for i in range(expert):
        index = torch.where(logits[:,i] > min_val)
        indices.append(index)
    return indices


def batch_slice_and_pad(x, indices, block=32):
    ys = []
    for index in indices:
        y = triton_slice_and_pad(x, index, block=block)
        ys.append(y)
    return ys




if __name__ == '__main__':

    from flops.utils.benchmark import benchmark_func

    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    # torch.backends.cudnn.deterministic = True

    device = 'cuda:0'
    dtype = torch.bfloat16 
    expert = 64 
    bs, dim = 8192, 4096 
    n_act = 4

    x = torch.rand((bs,dim),dtype=dtype,device=device)
    logits = torch.rand((bs, expert), dtype=torch.float32, device=device)

    indices = []
    counts = [int(n_act*bs/expert*(0.5+random.random())) for _ in range(expert)]
    rate = sum(counts)/(bs*n_act)
    counts = [int(rate*counts[i]) for i in range(expert)]
    counts = counts[:-1] + [(bs*n_act-sum(counts[:-1]))]
    # print(sum(counts), counts)
    for i in range(expert):
        index = list(range(bs))
        random.shuffle(index)
        index = torch.tensor(index[:counts[i]], dtype=torch.int32, device=device)
        indices.append(index)

    # benchmark_func(batch_index_expert, logits, min_val=0.95)
    # benchmark_func(triton_slice_and_pad, x, indices[0], block=32)
    benchmark_func(batch_slice_and_pad, x, indices, block=32)

    # raise error when benchmark
    pads = triton_batch_slice_and_pad(x, indices, expert=expert, block=32)
    benchmark_func(triton_slice_and_pad, x, indices, expert=expert, block=32)
