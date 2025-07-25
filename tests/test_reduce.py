import torch

from flops.utils.benchmark import benchmark_func
from flops.utils.reduce import (triton_abs_max,
                                triton_batch_count_zero,
                                triton_batch_sum_with_ord)
from flops.utils.util import output_check


def torch_sum(xs):
    return sum([x.square().sum() for x in xs])


def torch_count_zero(xs):
    count = torch.tensor([0], dtype=torch.int64, device='cuda')
    for x in xs:
        count += x.numel() - torch.count_nonzero(x)
    return count


def test_triton_abs_max(M=4096, N=4096, bench=False):
    x = 100 * torch.randn(M, 1, N, dtype=torch.bfloat16, device='cuda:0')

    # scales = 1.0/torch.sqrt(torch.maximum(x[:,0].abs().float().amax(0), torch.ones(M,N,dtype=dtype,device=device)) )
    # smooth_scale_ref = torch.exp2(torch.ceil(torch.log2(scales)))
    maxs_ref = x.abs().amax(0).view(N)

    maxs = triton_abs_max(x)
    output_check(maxs_ref, maxs)
    if bench:
        benchmark_func(triton_abs_max, x, n_repeat=100, ref_bytes=M * N * 2)


def test_count_zero(M=4096, N=8192, k=32, bench=False):
    xs = [torch.randn(M, N, dtype=torch.float32, device='cuda:0').to(
        torch.float8_e4m3fn).to(torch.float32) for i in range(k)]

    ref_bytes = sum([x.numel() for x in xs]) * 4

    count_ref = torch_count_zero(xs)
    count = triton_batch_count_zero(xs)
    print(f'{count_ref=} {count=}')
    assert count_ref.item() - count.item() == 0

    sum_ref = torch_sum(xs)
    sums = triton_batch_sum_with_ord(xs)
    output_check(sum_ref, sums)

    if bench:
        n_repeat = 100
        ref_time = benchmark_func(torch_count_zero, xs, n_repeat=n_repeat,
                                  ref_bytes=ref_bytes)
        benchmark_func(triton_batch_count_zero, xs, n_repeat=n_repeat,
                       ref_bytes=ref_bytes, ref_time=ref_time)


def test_ord_sum(M=4096, N=8192, k=32, bench=False):
    xs = [torch.randn(M, N, dtype=torch.float32, device='cuda:0').to(
        torch.float8_e4m3fn).to(torch.float32) for i in range(k)]

    ref_bytes = sum([x.numel() for x in xs]) * 4

    sum_ref = torch_sum(xs)
    sums = triton_batch_sum_with_ord(xs)
    output_check(sum_ref, sums)

    if bench:
        n_repeat = 100
        ref_time = benchmark_func(torch_sum, xs, n_repeat=n_repeat,
                                  ref_bytes=ref_bytes)
        benchmark_func(triton_batch_sum_with_ord, xs, n_repeat=n_repeat,
                       ref_bytes=ref_bytes, ref_time=ref_time)


if __name__ == '__main__':
    test_triton_abs_max(M=4096, N=4096)
    test_count_zero(M=4096, N=8192, k=32)
    test_ord_sum(M=4096, N=8192, k=32)
