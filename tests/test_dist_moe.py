
from datetime import timedelta
import os 
import time 
import torch 
import torch.distributed as dist
from flops.utils.benchmark import benchmark_func





def moe(hs, logits, ws, vs, n_act=4):
    # hs: [bs, length, dim]
    # logits: [bs, length, expert]
    # w1s: [expert, im, dim]
    assert logits.dtype == torch.float32
    bs, length, dim = hs.shape 
    hs = hs.view(bs*length, dim)
    expert, dim, ims = vs.shape 
    logits = logits.view(bs*length, expert)
    double_logits = logits.to(torch.float64)*(1+torch.arange(expert, device=hs.device, dtype=torch.float64)*1e-12)
    double_top_values, top_indices = torch.topk(double_logits, n_act, dim=-1, largest=True, sorted=True)
    mask_logits = torch.where(double_logits>=double_top_values[:,-1:], logits, -10000.0)
    mask_probs = torch.softmax(mask_logits, -1).to(hs.dtype)
    expert_float_indices = []
    outputs = []
    for i in range(expert):
        expert_index, = torch.where(mask_logits[:,i]>-5000)
        expert_float_indices.append(expert_index.float() + 0.001*i)
        expert_prob = mask_probs[expert_index,i:i+1]
        h = hs[expert_index]
        gus = h@ws[i].t()
        act = torch.nn.functional.silu(gus[:,:ims])*gus[:,ims:]
        out = act@vs[i].t()
        out *= expert_prob
        outputs.append(out)
    outputs = torch.cat(outputs, dim=0)
    expert_float_indices = torch.cat(expert_float_indices, dim=0)
    gather_indices = torch.argsort(expert_float_indices, dim=-1, descending=False, stable=False)
    outputs = outputs[gather_indices]
    outputs = outputs.view(bs * length, n_act, dim)
    outputs = outputs.sum(1)
    return outputs.view(bs, length, dim)



def tp_moe_ref(hs, logits, ws, vs, n_act=4, rank=0, world_size=2):

    assert logits.dtype == torch.float32
    bs, length, dim = hs.shape 
    expert, dim, ims = vs.shape 
    dtype = hs.dtype

    wss = [torch.empty((expert, ims*2, dim), dtype=dtype, device=device) for _ in range(world_size)]
    dist.all_gather(wss, ws)
    ws = torch.stack(wss, dim=1).view(expert, ims*2*world_size, dim)

    vss = [torch.empty((expert, dim, ims), dtype=dtype, device=device) for _ in range(world_size)]
    dist.all_gather(vss, vs)
    vs = torch.cat(vss, dim=2)

    outputs = moe(hs, logits, ws, vs, n_act=n_act)
    return outputs



def tp_moe(hs, logits, ws, vs, n_act=4, rank=0, world_size=2):
    # hs: [bs, length, dim]
    # logits: [bs, length, expert]
    # w1s: [expert, im, dim]
    assert logits.dtype == torch.float32
    bs, length, dim = hs.shape 
    expert, dim, ims = vs.shape 
    dtype = hs.dtype
    hs = hs.view(bs*length, dim)
    logits = logits.view(bs*length, expert)

    hss = torch.empty((bs*length*world_size,dim), dtype=dtype, device=device)
    dist.all_gather_into_tensor(hss, hs)
    hs = hss

    logitss = torch.empty((bs*length*world_size,expert), dtype=torch.float32, device=device)
    dist.all_gather_into_tensor(logitss, logits)
    logits = logitss

    double_logits = logits.to(torch.float64)*(1+torch.arange(expert, device=hs.device, dtype=torch.float64)*1e-12)
    double_top_values, top_indices = torch.topk(double_logits, n_act, dim=-1, largest=True, sorted=True)
    mask_logits = torch.where(double_logits>=double_top_values[:,-1:], logits, -10000.0)
    mask_probs = torch.softmax(mask_logits, -1).to(hs.dtype)
    expert_indices = []
    outputs = []
    for i in range(expert):
        expert_index = torch.where(mask_logits[:,i]>-5000)[0]
        expert_indices.append(expert_index.float() + 0.001*i)
        expert_prob = mask_probs[expert_index,i:i+1]
        h = hs[expert_index]
        gus = h@ws[i].t()
        act = torch.nn.functional.silu(gus[:,:ims])*gus[:,ims:]
        out = act@vs[i].t()
        out *= expert_prob
        outputs.append(out)
    outputs = torch.cat(outputs, dim=0)
    expert_indices = torch.cat(expert_indices, dim=0)
    gather_indices = torch.argsort(expert_indices, dim=-1, descending=False, stable=False)
    outputs = outputs[gather_indices]
    outputs = outputs.view(bs * length * world_size, n_act, dim)
    outputs = outputs.sum(1)

    for i in range(world_size):
        dist.reduce(outputs[i*bs*length:(i+1)*bs*length], dst=i)

    outputs = outputs[rank*bs*length:(rank+1)*bs*length]
    return outputs.view(bs, length, dim)



def ep_moe_ref(hs, logits, ws, vs, n_act=4, rank=0, world_size=2):

    assert logits.dtype == torch.float32
    bs, length, dim = hs.shape 
    exp, dim, ims = vs.shape 
    expert = exp * world_size

    wss = [torch.empty((exp, ims*2, dim), dtype=dtype, device=device) for _ in range(world_size)]
    dist.all_gather(wss, ws)
    ws = torch.cat(wss, dim=0)

    vss = [torch.empty((exp, dim, ims), dtype=dtype, device=device) for _ in range(world_size)]
    dist.all_gather(vss, vs)
    vs = torch.cat(vss, dim=0)

    outputs = moe(hs, logits, ws, vs, n_act=n_act)
    return outputs


def ep_moe(hs, logits, ws, vs, n_act=4, rank=0, world_size=2, step=0):
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
    mask_probs = torch.softmax(mask_logits, -1).to(hs.dtype)
    expert_float_indices = []
    expert_indices = []
    outputs = []
    total_dst_token_counts = []
    for i in range(exp):
        src_h_list = []
        src_token_count_list = []
        expert_probs = []

        # 准备发给每个gpu的数据,每个GPU上的每个专家分开处理便于后面优化为多流的形式
        # i=0, GPU0:[exp0,exp32] GPU1:[exp0,exp32] -> GPU0:[exp0,exp0], GPU1:[exp32,exp32]
        # i=1, GPU0:[exp1,exp33] GPU1:[exp1,exp33] -> GPU0:[exp1,exp1], GPU1:[exp33,exp33]
        for j in range(world_size):
            idx = i + j * exp 
            expert_index = torch.where(mask_logits[:,idx]>-5000)[0]
            expert_indices.append(expert_index)
            expert_probs.append(mask_probs[expert_index,idx])
            expert_float_indices.append(expert_index.float() + 1e-6*idx)
            h = hs[expert_index]
            src_h_list.append(h)
            src_token_count_list.append(expert_index.numel())
        
        expert_probs = torch.cat(expert_probs, 0)

        # 通信每个专家的token个数, 用来申请空间
        if True:
            src_token_counts = torch.tensor(src_token_count_list, dtype=torch.int32, device=device)
            dst_token_counts = torch.zeros((world_size,), dtype=torch.int32, device=device)
            dist.all_to_all_single(dst_token_counts, src_token_counts,output_split_sizes=[1]*world_size, input_split_sizes=[1]*world_size)
            dst_token_count_list = dst_token_counts.tolist()
            dst_token_count = sum(dst_token_count_list)
            # print(f' \n************* {step=} {rank=} {i=} {src_token_count_list=} {dst_token_count_list=} ********* \n')
        else:
            dst_token_count_list = [None for _ in range(world_size)]
            for j in range(world_size):
                dist.gather_object(src_token_count_list[j], dst_token_count_list if rank == j else None, dst=j)
            dst_token_count = sum(dst_token_count_list)
            print(f' ************* {step=} {rank=} {i=} {src_token_count_list=} {dst_token_count_list=} ********* ')

        # 通信每个专家需要的加权系数
        dst_expert_probs = torch.empty([dst_token_count], dtype=dtype, device=device)
        dist.all_to_all_single(dst_expert_probs, expert_probs, output_split_sizes=dst_token_count_list, input_split_sizes=src_token_count_list)

        # 通信每个专家需要的hidden_states
        # print(f' ************* {step=} {rank=} {i=} {src_token_count_list=} {dst_token_count_list=}')
        total_dst_token_counts.append(dst_token_count)
        src_h_tensor = torch.cat(src_h_list, 0)
        dst_h_tensor = torch.empty((dst_token_count, dim), device=device, dtype=dtype)
        dist.all_to_all_single(dst_h_tensor, src_h_tensor, output_split_sizes=dst_token_count_list, input_split_sizes=src_token_count_list)

        gus = dst_h_tensor@ws[i].t()
        act = torch.nn.functional.silu(gus[:,:ims])*gus[:,ims:]
        out = act@vs[i].t()
        out *= dst_expert_probs[:,None]
        outputs.append(out)

    outputs = torch.cat(outputs, dim=0)
    expert_float_indices = torch.cat(expert_float_indices, dim=0)
    token_counts =  outputs.size(0)
    assert token_counts == sum(total_dst_token_counts)

    # 通信每个GPU处理的总token个数
    token_count_per_gpu = [None for _ in range(world_size)]
    dist.all_gather_object(token_count_per_gpu, token_counts)

    # 通信一个shard的数据
    # print(f'{step=} {rank=} start comm shard {token_counts=} {token_count_per_gpu=}')
    src_shard_outputs = torch.reshape(outputs.view(token_counts, world_size, dim//world_size).permute(1,0,2), (token_counts*world_size, dim//world_size))
    dst_shard_outputs = torch.empty((n_act*bs*length*world_size, dim//world_size), dtype=dtype, device=device)
    dist.all_to_all_single(dst_shard_outputs, src_shard_outputs, output_split_sizes=token_count_per_gpu, input_split_sizes=[token_counts]*world_size)

    # 通信专家索引
    # print(f'{step=} {rank=} start comm expert indices {expert_float_indices.shape=}')
    expert_indices_outputs = torch.empty((n_act*bs*length*world_size, ), dtype=torch.float32, device=device)
    dist.all_gather_into_tensor(expert_indices_outputs, expert_float_indices)

    # shard加权和
    # print(f'{step=} {rank=} start calc weighted act')
    gather_indices = torch.argsort(expert_indices_outputs, dim=-1, descending=False, stable=False)
    outputs = dst_shard_outputs[gather_indices]
    outputs = outputs.view(bs * length * world_size, n_act, dim//world_size)
    outputs = outputs.sum(1)  # [bs * length * world_size, dim//world_size]

    # shard组装为完整的dim
    # print(f'{step=} {rank=} start unshard ')
    weighted_outputs = torch.empty((bs*length*world_size, dim//world_size), dtype=dtype, device=device)
    dist.all_to_all_single(weighted_outputs, outputs)
    weighted_outputs = weighted_outputs.view(world_size, bs*length, dim//world_size).permute(1,0,2)
    weighted_outputs = torch.reshape(weighted_outputs, (bs, length, dim))

    return weighted_outputs

"""
OMP_NUM_THREADS=4 torchrun --nnodes=1 --nproc-per-node=2 test_dist_moe.py 
"""

if __name__ == '__main__':
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    init_method = "env://" # f'tcp://127.0.0.1:4673'
    print(f'{rank=} {world_size=} {init_method=}')
    dist.init_process_group(backend='nccl', init_method=init_method, world_size=world_size, rank=rank, timeout=timedelta(seconds=30))
    torch.cuda.set_device(rank) 

    mode = 'ep'
    if mode == 'ep':
        # bs, length, dim, ims, expert, n_act = 1, 4096, 4096, 4096, 32, 2
        bs, length, dim, ims, expert, n_act = 1, 4096, 7168, 5376, 64, 2
        exp = expert//world_size
        device = f'cuda:{rank}'
        dtype = torch.bfloat16
        ref_flops = bs*length*n_act*dim*ims*3*2
        n_repeat = 100

        logits = torch.randn((bs, length, expert), dtype=torch.float32, device=device)
        hidden_states = 0.1*torch.randn((bs, length, dim), dtype=dtype, device=device)
        gate_weights = 0.1*torch.randn((exp,ims,dim), dtype=dtype, device=device)
        up_weights = 0.1*torch.randn((exp,ims,dim), dtype=dtype, device=device)
        down_weights = 0.1*torch.randn((exp,dim, ims), dtype=dtype, device=device)
        gate_up_weights = torch.cat([gate_weights, up_weights], dim=1)

        org_output = ep_moe_ref(hidden_states, logits, gate_up_weights, down_weights, n_act=n_act, rank=rank, world_size=world_size)

        opt_output = ep_moe(hidden_states, logits, gate_up_weights, down_weights, n_act=n_act, rank=rank, world_size=world_size, step=0)
        # print(f'{rank=} {org_output[0,0]=} {opt_output[0,0]=}')
        # torch.testing.assert_close(org_output,opt_output,rtol=0.05,atol=0.05)
        benchmark_func(ep_moe, hidden_states, logits, gate_up_weights, down_weights, n_act=n_act, rank=rank, world_size=world_size, ref_flops=ref_flops, n_warmup=10,  n_repeat=n_repeat)


    elif mode == 'tp':
        # bs, length, dim, ims, expert, n_act = 1, 4096, 4096, 4096, 32, 2
        bs, length, dim, ims, expert, n_act = 1, 4096, 7168, 5376, 64, 2
        exp = expert//world_size
        device = f'cuda:{rank}'
        dtype = torch.bfloat16
        ref_flops = bs*length*n_act*dim*ims*3*2
        n_repeat = 100

        logits = torch.randn((bs, length, expert), dtype=torch.float32, device=device)
        hidden_states = 0.1*torch.randn((bs, length, dim), dtype=dtype, device=device)
        shard_gate_up_weights = 0.1*torch.randn((expert,ims//world_size*2,dim), dtype=dtype, device=device)
        shard_down_weights = 0.1*torch.randn((expert, dim, ims//world_size), dtype=dtype, device=device)

        org_output = tp_moe_ref(hidden_states, logits, shard_gate_up_weights, shard_down_weights, n_act=n_act, rank=rank, world_size=world_size)
        opt_output = tp_moe(hidden_states, logits, shard_gate_up_weights, shard_down_weights, n_act=n_act, rank=rank, world_size=world_size)
        # print(f'{rank=} {org_output[0,0]=} {opt_output[0,0]=}')
        # torch.testing.assert_close(org_output,opt_output,rtol=0.05,atol=0.05)


        benchmark_func(tp_moe, hidden_states, logits, shard_gate_up_weights, shard_down_weights, n_act=n_act, rank=rank, world_size=world_size, ref_flops=ref_flops, n_warmup=10,  n_repeat=n_repeat)


    elif mode == 'pad':

        dim = 4
        tokens = 4
        device = f'cuda:{rank}'
        dtype = torch.bfloat16

        inputs = torch.ones((tokens,dim), dtype=dtype, device=device)*(rank+1)
        outputs = torch.zeros((tokens*2,dim), dtype=dtype, device=device)
        torch.cuda.synchronize()
        ts = time.time()
        dist.all_to_all_single(outputs, inputs, output_split_sizes=[tokens,tokens], input_split_sizes=[tokens//2,tokens//2])
        torch.cuda.synchronize()
        te = time.time()
        elapse = te-ts 
        bulk_size = tokens*dim/2*2
        bandwidth = bulk_size/2**30/elapse
        print(f'{rank=} elpase:{elapse*1000:.3f}ms outputs:{outputs[:,0]}')

    elif mode == 'bandwidth':
        dim = 4096
        device = f'cuda:{rank}'
        dtype = torch.bfloat16
        for i in range(8):
            tokens = 2**(i+12)
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
            print(f'{rank=} size:[{tokens},{dim}] elpase:{elapse*1000:.3f}ms bandwidth:{bandwidth:.3f}GiB/s')
            time.sleep(2.0)
            print('')