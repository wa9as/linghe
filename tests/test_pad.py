

import random
import torch
from flops.tools.benchmark import benchmark_func
from flops.utils.pad import triton_slice_and_pad
from flops.utils.pad import triton_batch_slice_and_pad




def batch_index_expert(logits, min_val=0.0):
    length, expert = logits.shape
    indices = []
    for i in range(expert):
        index = torch.where(logits[:, i] > min_val)
        indices.append(index)
    return indices


def batch_slice_and_pad(x, indices, block=32):
    ys = []
    for index in indices:
        y = triton_slice_and_pad(x, index, block=block)
        ys.append(y)
    return ys



if __name__ == '__main__':


    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    # torch.backends.cudnn.deterministic = True

    device = 'cuda:0'
    dtype = torch.bfloat16
    expert = 64
    bs, dim = 8192, 4096
    n_act = 4

    x = torch.rand((bs, dim), dtype=dtype, device=device)
    logits = torch.rand((bs, expert), dtype=torch.float32, device=device)

    indices = []
    counts = [int(n_act * bs / expert * (0.5 + random.random())) for _ in
              range(expert)]
    rate = sum(counts) / (bs * n_act)
    counts = [int(rate * counts[i]) for i in range(expert)]
    counts = counts[:-1] + [(bs * n_act - sum(counts[:-1]))]
    # print(sum(counts), counts)
    for i in range(expert):
        index = list(range(bs))
        random.shuffle(index)
        index = torch.tensor(index[:counts[i]], dtype=torch.int32,
                             device=device)
        indices.append(index)

    # benchmark_func(batch_index_expert, logits, min_val=0.95)
    # benchmark_func(triton_slice_and_pad, x, indices[0], block=32)
    benchmark_func(batch_slice_and_pad, x, indices, block=32)

    # raise error when benchmark
    pads = triton_batch_slice_and_pad(x, indices, expert=expert, block=32)
    benchmark_func(triton_slice_and_pad, x, indices, expert=expert, block=32)
