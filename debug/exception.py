import torch
import triton
import triton.language as tl


# for megatron 0.11 scatter_add

@triton.jit
def address_exception_kernel(x_ptr, M, W, N: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, N)

    sums = tl.zeros((N,), dtype=tl.float32)
    for i in range(W):
        x = tl.load(x_ptr + pid * W * N + i * N + offs)
        sums += x

    tl.store(x_ptr + pid * N + offs, sums)


def triton_address_exception(x):
    M, N = x.shape
    W = 1600

    num_stages = 5
    num_warps = 8

    grid = lambda META: (M,)
    address_exception_kernel[grid](
        x,
        M, W, N,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return x


if __name__ == '__main__':
    for i in range(4):
        x = torch.randn((128, 4096), dtype=torch.bfloat16, device=f'cuda:{i}')
        triton_address_exception(x)
    torch.cuda.synchronize()
