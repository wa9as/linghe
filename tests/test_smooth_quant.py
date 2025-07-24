import torch 
from flops.utils.util import *
from flops.utils.benchmark import *
from flops.quant.smooth.reused_smooth import *
from flops.quant.smooth.seperate_smooth import *




def torch_split_smooth_quant(x_split,smooth_scales, round_scale=False):
    x_qs = []
    x_scales = []
    x_maxs = []
    for i, x_ in enumerate(x_split):
        x_maxs.append(x_.amax(0))
        x_smooth = x_/smooth_scales[i]
        x_scale_ = x_smooth.float().abs().amax(1)/448
        if round_scale:
            x_scale_ = torch.exp2(torch.ceil(torch.log2(x_scale_)))
        x_q_ = (x_smooth/x_scale_[:,None]).to(torch.float8_e4m3fn)
        x_qs.append(x_q_)
        x_scales.append(x_scale_)
    x_maxs = torch.stack(x_maxs, 0)
    return x_qs,x_scales,x_maxs

def triton_split_smooth_quant(x_split,smooth_scales):
    x_qs = []
    x_scales = []
    for i, x_ in enumerate(x_split):
        x_q_,x_scale_ = triton_reused_smooth_quant(x_,smooth_scales[i])
        x_qs.append(x_q_)
        x_scales.append(x_scale_)
    return x_qs,x_scales


def test_triton_reused_smooth_quant(M=4096, N=4096):
    device = 'cuda:0'
    x = torch.randn((M,N),dtype=torch.bfloat16,device=device)
    smooth_scale = torch.randn((N,),device=device,dtype=torch.float32).abs()
    x_q_ref, scales_ref = torch_smooth_quant(x,smooth_scale,reverse=False,round_scale=True)

    x_q, x_scale = triton_reused_smooth_quant(x, smooth_scale, reverse=False, round_scale=True)
    output_check(x_q_ref.float(), x_q.float(), 'data')
    output_check(scales_ref, x_scale, 'scale')


def test_triton_reused_transpose_pad_smooth_quant(M=4096,N=4096):
    device = 'cuda:0'
    y = torch.randn((M,N),dtype=torch.bfloat16,device=device)
    transpose_smooth_scale = torch.randn((M,),device=device,dtype=torch.float32).abs()+1
    yt_q,yt_scale = triton_reused_transpose_smooth_quant(y, transpose_smooth_scale, reverse=True, pad=True)
    q_ref, scale_ref = torch_smooth_quant(y.T.contiguous(),transpose_smooth_scale,reverse=True,round_scale=True)
    q_ref = q_ref.T.float()

    output_check(q_ref, yt_q.float()[:,:M], 'data')
    output_check(scale_ref.float(), yt_scale.float(), 'scale')


def test_triton_reused_transpose_rescale_smooth_quant(M=4096,N=4096, round_scale=False):
    device = 'cuda:0'
    y = torch.randn((M,N),dtype=torch.bfloat16,device=device)
    org_smooth_scale = torch.randn((N,),device=device,dtype=torch.float32).abs()+1
    y_q, y_scale = triton_reused_smooth_quant(y, org_smooth_scale, reverse=True, round_scale=True)

    transpose_smooth_scale = torch.ones((M,),device=device,dtype=torch.float32)
    yt_q,yt_scale = triton_reused_transpose_rescale_smooth_quant(y_q, org_smooth_scale, y_scale, transpose_smooth_scale, reverse=True, pad=False)

    y_tmp = (y.to(torch.float32)*transpose_smooth_scale[:,None]).t().contiguous()
    yt_scale_ref = y_tmp.abs().amax(dim=1)/448+1e-30
    if round_scale:
        yt_scale_ref = torch.exp2(torch.ceil(torch.log2(yt_scale_ref)))
    yt_q_ref = (y_tmp/yt_scale_ref[:,None]).to(torch.float8_e4m3fn).to(torch.float32)

    output_check(yt_q_ref, yt_q.float(), 'data')
    output_check(yt_scale_ref, yt_scale.float(), 'scale')


def test_triton_batch_smooth_quant(M=4096,N=4096, n_experts=32, topk=8, round_scale=False,bench=False):
    device = 'cuda:0'

    smooth_scales = 1+10*torch.rand((n_experts,N),device=device,dtype=torch.float32)

    logits = torch.randn((M,n_experts), dtype=torch.float32, device=device)
    probs, mask_map,token_count_per_expert, indices, row_id_map = torch_make_indices(logits, topk=topk, bias=0.0)
    token_count_per_expert_list = token_count_per_expert.tolist()
    x = torch.randn((sum(token_count_per_expert_list), N), dtype=torch.bfloat16,device=device)

    x_q,x_scale,x_maxs = triton_batch_smooth_quant(x, smooth_scales, token_count_per_expert, reverse=False, round_scale=round_scale, calibrate=True)

    x_split = torch.split(x, token_count_per_expert_list)
    x_q_ref, x_scale_ref, x_maxs_ref = torch_split_smooth_quant(x_split,smooth_scales)
    x_q_ref = torch.cat([x.view(torch.uint8) for x in x_q_ref], 0).view(torch.float8_e4m3fn)
    x_scale_ref = torch.cat(x_scale_ref, 0)
    output_check(x_q_ref.float(), x_q.float(), 'data')
    output_check(x_scale_ref.float(), x_scale.float(), 'scale')
    output_check(x_maxs_ref.float(), x_maxs.float(), 'maxs')

    if bench:
        n_repeat = 100
        ref_time = benchmark_func(triton_split_smooth_quant, x_split, smooth_scales, n_repeat=n_repeat)
        benchmark_func(triton_batch_smooth_quant, x, smooth_scales, token_count_per_expert, reverse=False, round_scale=round_scale, n_repeat=n_repeat, ref_time=ref_time)
        benchmark_func(triton_batch_smooth_quant, x, smooth_scales, token_count_per_expert, reverse=False, round_scale=round_scale, calibrate=True, n_repeat=n_repeat, ref_time=ref_time)



if __name__ == '__main__':
    test_triton_reused_smooth_quant(M=4096, N=4096)
    test_triton_reused_transpose_pad_smooth_quant(M=4096,N=4096)
    test_triton_reused_transpose_rescale_smooth_quant(M=4096,N=4096, round_scale=False)
    test_triton_batch_smooth_quant(M=4096,N=4096, n_experts=32, topk=8, round_scale=False)
