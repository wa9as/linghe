from datetime import timedelta
import os
import torch
import torch.distributed as dist


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        """Forward function."""
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(input)
        else:
            # Unequal split (all2all-v)
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.size()[1:]),
                dtype=input.dtype,
                device=torch.cuda.current_device(),
            )
        torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        """Backward function."""
        return (
            None,
            _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes,
                            ctx.output_split_sizes),
            None,
            None,
        )


def all_to_all(group, input_, output_split_sizes_=None, input_split_sizes=None):
    """Wrapper for autograd function"""
    return _AllToAll.apply(group, input_, output_split_sizes_,
                           input_split_sizes)


"""
OMP_NUM_THREADS=4 torchrun --nnodes=1 --nproc-per-node=2 test_dist_grad.py 
"""

if __name__ == '__main__':
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    init_method = "env://"
    # print(f'{rank=} {world_size=} {init_method=}')
    dist.init_process_group(backend='nccl', init_method=init_method,
                            world_size=world_size, rank=rank,
                            timeout=timedelta(seconds=30))
    torch.cuda.set_device(rank)

    length, dim, ims, expert, n_act = 4096, 2048, 512, 64, 8  # mini
    exp = expert // world_size
    device = f'cuda:{rank}'
    dtype = torch.bfloat16
    ref_flops = length * n_act * dim * ims * 3 * 2
    n_repeat = 10

    logits = torch.randn((length, expert), dtype=torch.float32, device=device,
                         requires_grad=True)
    hidden_states = torch.ones((length, dim), dtype=dtype, device=device,
                               requires_grad=True) * (rank + 1)
    output_states = torch.zeros((length, dim), dtype=dtype, device=device,
                                requires_grad=True) * (rank + 1)
    dist.all_to_all_single(output_states, hidden_states)
    loss = output_states.sum()
    loss.backward()
    print(f'{hidden_states.grad=}')
