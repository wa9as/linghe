
from datetime import timedelta
import os 
import time 
import torch 
import torch.distributed as dist






def tp_moe(hs, logits, ws, vs, n_act=4, rank=0, world_size=2):
    pass


"""

"""

def ep_moe(hs, logits, ws, vs, n_act=4, rank=0, world_size=2):
    # hs: [bs, length, dim]
    # logits: [bs, length, expert]
    # ws: [exp, im, dim]
    assert logits.dtype == torch.float32
    bs, length, dim = hs.shape 
    device = hs.device
    dtype = hs.dtype 
    hs = hs.view(bs*length, dim)
    exp, dim, ims = vs.shape 
    expert = logits.size(-1)
    logits = logits.view(bs*length, expert)
    double_logits = logits.to(torch.float64)*(1+torch.arange(expert, device=device, dtype=torch.float64)*1e-12)
    double_top_values, top_indices = torch.topk(double_logits, n_act, dim=-1, largest=True, sorted=True)
    mask_logits = torch.where(double_logits>=double_top_values[:,-1:], logits, -10000.0)
    probs = torch.softmax(double_top_values, -1).to(hs.dtype)
    expert_float_indices = []
    expert_indices = []
    outputs = []
    mask_logits_transpose = mask_logits.t().contiguous()
    gather_token_counts = []
    for i in range(exp):
        to_scatter_list = []
        token_counts = []
        for j in range(world_size):
            idx = i + j * exp 
            expert_index = torch.where(mask_logits_transpose[idx]>-5000)[0]
            expert_indices.append(expert_index)
            expert_float_indices.append(expert_index.float() + 1e-6*idx)
            h = hs[expert_index]
            to_scatter_list.append(h)
            token_counts.append(expert_index.numel())
        
        token_counts_tensor = torch.tensor(token_counts, device=device)
        output_token_counts = torch.zeros((world_size,), dtype=torch.int64, device=device)
        dist.all_to_all_single(output_token_counts, token_counts_tensor)

        gather_token_count = output_token_counts.sum().item()
        gather_token_counts.append(gather_token_count)
        to_scatter_tensor = torch.cat(to_scatter_list, 0)
        output_tensor = torch.empty((gather_token_count, dim), device=device, dtype=dtype)
        dist.all_to_all_single(output_tensor, to_scatter_tensor, output_split_sizes=output_token_counts.tolist(), input_split_sizes=token_counts)

        gus = output_tensor@ws[i].t()
        act = torch.nn.functional.silu(gus[:,:ims])*gus[:,ims:]
        out = act@vs[i].t()
        outputs.append(out)


    outputs = torch.cat(outputs, dim=0)
    ts =  outputs.size(0)

    token_counts_tensor = torch.tensor([ts]*world_size, device=device)
    output_token_counts = torch.zeros((world_size,), dtype=torch.int64, device=device)
    dist.all_to_all_single(output_token_counts, token_counts_tensor)

    expert_outputs = torch.empty((expert*n_act, dim), dtype=dtype, device=device)
    to_scatter_blocks = torch.reshape(outputs.view(ts, world_size, dim//world_size).permute(1,0,2), (ts*world_size, dim//world_size))
    dist.all_to_all_single(expert_outputs, to_scatter_blocks, output_split_sizes=output_token_counts.tolist(), input_split_sizes=[ts]*world_size)

    expert_indices = torch.cat(expert_float_indices, dim=0)
    expert_indices_outputs = [torch.empty((tc, ), dtype=torch.float32, device=device) for tc in output_token_counts.tolist()]
    dist.all_gather(expert_indices_outputs, expert_indices)
    expert_indices = torch.cat(expert_indices_outputs, dim=0)

    gather_indices = torch.argsort(expert_indices, dim=-1, descending=False, stable=False)
    outputs = expert_outputs[gather_indices]
    outputs = outputs.view(bs * length, n_act, dim//world_size)
    outputs = torch.einsum("bno,bn->bo",outputs,probs)

    weight_outputs = torch.empty((bs*length*world_size, dim//world_size), dtype=dtype, device=device)
    dist.all_to_all_single(weight_outputs, outputs)
    weight_outputs = weight_outputs.view(world_size, bs*length, dim//world_size).permute(1,0,2)

    return weight_outputs.view(bs, length, dim)


#torchrun --nnodes=1 --nproc-per-node=2 test_dist_moe.py 

if __name__ == '__main__':
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f'{rank=} {local_rank=} {world_size=}')
    dist.init_process_group(backend='nccl', init_method=None, world_size=world_size, rank=rank, timeout=timedelta(seconds=3))

    bs, length, dim, ims, expert, n_act = 1, 4096, 4096, 4096, 32, 2
    exp = expert//world_size
    device = f'cuda:{rank}'
    dtype = torch.bfloat16
    logits = torch.randn((bs, length, expert), dtype=torch.float32, device=device)
    hidden_states = torch.randn((bs, length, dim), dtype=dtype, device=device)
    gate_weights = torch.randn((exp,ims,dim), dtype=dtype, device=device)
    up_weights = torch.randn((exp,ims,dim), dtype=dtype, device=device)
    down_weights = torch.randn((exp,dim, ims), dtype=dtype, device=device)
    gate_up_weights = torch.cat([gate_weights, up_weights], dim=1)

    # output = ep_moe(hidden_states, logits, gate_up_weights, down_weights, n_act=4, rank=0, world_size=2)



    if False:
        tokens = 4096*32
        inputs = torch.randn((tokens,dim), dtype=dtype, device=device)*(rank+1)
        outputs = torch.zeros((tokens,dim), dtype=dtype, device=device)
        torch.cuda.synchronize()
        ts = time.time()
        dist.all_to_all_single(outputs, inputs, output_split_sizes=[tokens//2,tokens//2], input_split_sizes=[tokens//2,tokens//2])
        torch.cuda.synchronize()
        te = time.time()
        elapse = te-ts 
        bulk_size = tokens*dim/2*2
        bandwidth = bulk_size/2**30/elapse
        print(f'first {rank=} elpase:{elapse*1000:.3f}ms bandwidth:{bandwidth:.3f}GiB/s {outputs[:,0]}')

        inputs *= 2.0
        outputs *= 0.0
        torch.cuda.synchronize()
        ts = time.time()
        dist.all_to_all_single(outputs, inputs, output_split_sizes=[tokens//2,tokens//2], input_split_sizes=[tokens//2,tokens//2])
        torch.cuda.synchronize()
        te = time.time()
        elapse = te-ts 
        bandwidth = bulk_size/2**30/elapse
        print(f'second {rank=} elpase:{elapse*1000:.3f}ms bandwidth:{bandwidth:.3f}GiB/s {outputs[:,0]}')

        inputs *= 2.0
        outputs *= 0.0
        torch.cuda.synchronize()
        ts = time.time()
        dist.all_to_all_single(outputs, inputs, output_split_sizes=[tokens//2,tokens//2], input_split_sizes=[tokens//2,tokens//2])
        torch.cuda.synchronize()
        te = time.time()
        elapse = te-ts 
        bandwidth = bulk_size/2**30/elapse
        print(f'third {rank=} elpase:{elapse*1000:.3f}ms bandwidth:{bandwidth:.3f}GiB/s {outputs[:,0]}')


        # pad_outputs = torch.zeros((tokens*2,dim), dtype=dtype, device=device)
        # ts = time.time()
        # dist.all_to_all_single(pad_outputs, inputs, output_split_sizes=[tokens,tokens],input_split_sizes=[4,4])
        # te = time.time()
        # print(f'pad {rank=} elpase:{(te-ts)*1000:.3f}ms {pad_outputs[:,0]}')