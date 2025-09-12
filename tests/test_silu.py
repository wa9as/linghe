import random

import torch

from flops.utils.benchmark import benchmark_func
from flops.utils.silu import (triton_batch_weighted_silu_and_quant_backward,
                              triton_batch_weighted_silu_and_quant_forward,
                              triton_silu_and_quant_backward,
                              triton_silu_and_quant_forward)
from flops.tools.util import output_check, torch_smooth_quant, torch_group_quant


def torch_silu(x):
    M, N = x.shape
    x1, x2 = torch.split(x, N // 2, dim=1)
    y = torch.sigmoid(x1) * x1 * x2
    return y


def torch_weighted_silu(x, weight):
    M, N = x.shape
    x1, x2 = torch.split(x, N // 2, dim=1)
    y = torch.sigmoid(x1) * x1 * x2 * weight
    return y


def torch_weighted_silu_backward(dy, x, weight):
    x = x.clone().detach().requires_grad_()
    weight = weight.clone().detach().requires_grad_()
    y = torch_weighted_silu(x, weight)
    y.backward(gradient=dy)
    return x.grad, weight.grad


def torch_silu_and_quant_forward(x, smooth_scale=None, round_scale=True):
    M, N = x.shape
    x1, x2 = torch.split(x, N // 2, dim=1)
    y = torch.sigmoid(x1) * x1 * x2
    if smooth_scale is None:
        # blockwise
        y_q, y_scale = torch_group_quant(y, round_scale=round_scale)
        yt_q, yt_scale = torch_group_quant(y.t(), round_scale=round_scale)
        x_maxs = None
    else:
        # smooth
        y_q, y_scale, x_maxs = torch_smooth_quant(y, smooth_scale, reverse=False, round_scale=round_scale)
        # y_smooth = y / smooth_scale
        # x_maxs = y.abs().float().amax(0)
        # y_scale = y_smooth.abs().amax(1) / 448
        # if round_scale:
        #     y_scale = torch.exp2(torch.ceil(torch.log2(y_scale)))
        # y_q = (y_smooth / y_scale[:, None]).to(torch.float8_e4m3fn)
        yt_q = None 
        yt_scale = None
    return y_q, y_scale, x_maxs, yt_q, yt_scale


def torch_silu_and_quant_backward(grad, x, smooth_scale=None, transpose_smooth_scale=None, round_scale=True, reverse=True):
    x = x.detach().clone().requires_grad_()
    y = torch_silu(x)
    y.backward(gradient=grad)
    dx = x.grad
    if smooth_scale is None:
        # blockwise
        q, dx_scale = torch_group_quant(dx, round_scale=round_scale)
        yt_q, yt_scale = torch_group_quant(dx.t(), round_scale=round_scale)
    else:
        q, dx_scale, ms = torch_smooth_quant(dx, smooth_scale, reverse=reverse,
                                        round_scale=round_scale)
        yt_q, yt_scale, ms = torch_smooth_quant(dx.t().contiguous(), transpose_smooth_scale, reverse=reverse,
                                        round_scale=round_scale)
    return q, dx_scale, yt_q, yt_scale


def torch_batch_weighted_silu_and_quant_forward(xs, weight,
                                                counts, 
                                                smooth_scales=None,
                                                round_scale=True,
                                                reverse=False):
    counts = counts.tolist()
    N = xs.shape[1]
    if sum(counts) == 0:
        device = xs.device
        qs = torch.empty((0,N//2),device=device,dtype=torch.float8_e4m3fn)
        scales = torch.empty((0,),device=device,dtype=torch.float32)
        maxs = torch.zeros((len(counts),N),device=device,dtype=torch.float32)
        qts = torch.empty((0,),device=device,dtype=torch.float8_e4m3fn)
        qtscales = torch.zeros((0,),device=device,dtype=torch.float32)
        return qs, scales, maxs, qts, qtscales
    qs = []
    scales = []
    maxs = []
    qts = []
    qtscales = []
    s = 0
    for i, c in enumerate(counts):
        x = xs[s:s + c]
        y = torch_weighted_silu(x, weight[s:s + c])
        if smooth_scales is None:
            q, scale = torch_group_quant(y, round_scale=round_scale)
            qt, qtscale = torch_group_quant(y.t(), round_scale=round_scale)
            qs.append(q)
            scales.append(scale.t().contiguous().view(-1)) 
            qts.append(qt.view(-1))
            qtscales.append(qtscale.t().contiguous().view(-1))
        else:
            q, scale, ms = torch_smooth_quant(y, smooth_scales[i], reverse=reverse,
                                        round_scale=round_scale)
            qs.append(q)
            scales.append(scale) 
            maxs.append(ms)

        s += c
    qs = torch.cat(qs, 0)
    scales = torch.cat(scales, 0) 
    if smooth_scales is None:
        qts = torch.cat(qts, 0)
        qtscales = torch.cat(qtscales, 0) 
    else:
        maxs = torch.cat(maxs, 0)
    return qs, scales, maxs, qts, qtscales


def torch_batch_weighted_silu_and_quant_backward(grad_output, x, weight,
                                                 counts,
                                                 smooth_scales=None,
                                                 transpose_smooth_scale=None,
                                                 round_scale=True,
                                                 reverse=False):
    if sum(counts) == 0:
        device = x.device
        N = x.shape[1]
        dx_q = torch.empty((0,N),device=device,dtype=torch.float8_e4m3fn)
        dx_scale = torch.empty((0,),device=device,dtype=torch.float32)
        dw = torch.empty_like(weight)
        qts = torch.empty((0,),device=device,dtype=torch.float8_e4m3fn)
        qtscales = torch.zeros((N*len(counts),),device=device,dtype=torch.float32)
        return dx_q, dx_scale, dw, qts, qtscales

    dx, dw = torch_weighted_silu_backward(grad_output, x, weight)
    qs = []
    scales = []
    qts = []
    qtscales = []
    s = 0
    for i, c in enumerate(counts):
        if smooth_scales is None:
            q, scale = torch_group_quant(dx[s:s+c], round_scale=round_scale)
            qt, qtscale = torch_group_quant(dx[s:s+c].t(), round_scale=round_scale)
            qs.append(q)
            scales.append(scale.t().contiguous().view(-1)) 
            qts.append(qt.view(-1))
            qtscales.append(qtscale.t().contiguous().view(-1))
        else:
            q, scale, dx_max = torch_smooth_quant(dx[s:s + c], smooth_scales[i],
                                        reverse=reverse, round_scale=round_scale)
            dxt = dx[s:s + c].t().contiguous()
            dxt_s = transpose_smooth_scale[s:s+c]
            padding_size = (c+31)//32*32-c 
            if padding_size > 0:
                dxt = torch.nn.functional.pad(dxt, (0,padding_size,0,0))
                dxt_s = torch.nn.functional.pad(dxt_s, (0,padding_size))
            qt, t_scale, dx_max = torch_smooth_quant(dxt, dxt_s,
                                        reverse=reverse, round_scale=round_scale)
            
            qs.append(q)
            scales.append(scale)
            qts.append(qt.view(-1))
            qtscales.append(t_scale.view(-1))
        s += c
    dx_q = torch.cat(qs, 0)
    dx_scale = torch.cat(scales, 0)
    qts = torch.cat(qts, 0)
    qtscales = torch.cat(qtscales, 0)
    return dx_q, dx_scale, dw, qts, qtscales



def test_silu_and_smooth_quant(M=4096, N=4096, bench=False):
    if True:
        x = torch.randn((M, N), dtype=torch.bfloat16, device='cuda:0')
        x = (x*10).clone().detach().requires_grad_()
        grad_output = torch.randn((M, N // 2), dtype=torch.bfloat16,
                                device='cuda:0')
        smooth_scale = 1 + torch.rand((N // 2,), dtype=torch.float32,
                                    device='cuda:0')
        grad_smooth_scale = 1 + torch.rand((N,), dtype=torch.float32,
                                        device='cuda:0')
        transpose_grad_smooth_scale = 1 + torch.rand((M,), dtype=torch.float32,
                                        device='cuda:0')
    else:   
        d = torch.load('/ossfs/workspace/tmp/vis/silu.bin')
        x = d['x'].clone().detach().to('cuda:0').requires_grad_()
        grad_output = d['g'].to('cuda:0')
        grad_smooth_scale = d['smooth_scale'].to('cuda:0')
        N = x.shape[-1]
        M = x.shape[0]
        smooth_scale = 1 + torch.rand((N // 2,), dtype=torch.float32,
                                    device='cuda:0')

    y_q_ref, y_scale_ref, y_maxs_ref, _, _ = torch_silu_and_quant_forward(x, smooth_scale=smooth_scale)
    y_q, y_scale, y_maxs, _, _ = triton_silu_and_quant_forward(x, 
                                                    smooth_scale=smooth_scale,
                                                    round_scale=True,
                                                    calibrate=True)
    output_check(y_q_ref.float(), y_q.float(), 'smooth.y_q')
    output_check(y_scale_ref, y_scale, 'smooth.y_scale')
    output_check(y_maxs_ref, y_maxs, 'smooth.y_max')


    dx_q_ref, dx_scale_ref, dxt_q_ref, dxt_scale_ref = torch_silu_and_quant_backward(grad_output, x,
                                                           smooth_scale=grad_smooth_scale,
                                                           transpose_smooth_scale=transpose_grad_smooth_scale,
                                                           reverse=True,
                                                           round_scale=True)
    dx_q, dx_scale, dxt_q, dxt_scale = triton_silu_and_quant_backward(grad_output, x,
                                                    smooth_scale=grad_smooth_scale,
                                                    transpose_smooth_scale=transpose_grad_smooth_scale,
                                                    reverse=True,
                                                    round_scale=True)

    output_check(dx_q_ref.float(), dx_q.float(), 'smooth.dx_data')
    output_check(dx_scale_ref, dx_scale, 'smooth.dx_scale')
    output_check(dxt_q_ref.float(), dxt_q.float(), 'smooth.dxt_data')
    output_check(dxt_scale_ref, dxt_scale, 'smooth.dxt_scale')


    if bench:
        benchmark_func(torch_silu_and_quant_forward, x, smooth_scale=smooth_scale,
                       n_repeat=100, ref_bytes=M * N * 2.5)
        benchmark_func(triton_silu_and_quant_forward, x, smooth_scale=smooth_scale,
                       n_repeat=100, ref_bytes=M * N * 2.5)
        benchmark_func(triton_silu_and_quant_backward, grad_output, x, 
                       smooth_scale=grad_smooth_scale,
                       transpose_smooth_scale=transpose_grad_smooth_scale,
                       n_repeat=100, ref_bytes=M * N * 5)




def test_silu_and_block_quant(M=4096, N=4096, bench=False):
    if True:
        x = torch.randn((M, N), dtype=torch.bfloat16, device='cuda:0')
        x = (x*10).clone().detach().requires_grad_()
        grad_output = torch.randn((M, N // 2), dtype=torch.bfloat16,
                                device='cuda:0')
    else:   
        d = torch.load('/ossfs/workspace/tmp/vis/silu.bin')
        x = d['x'].clone().detach().to('cuda:0').requires_grad_()
        grad_output = d['g'].to('cuda:0')
        N = x.shape[-1]
        M = x.shape[0]


    y_q_ref, y_scale_ref, _, yt_q_ref, yt_scale_ref  = torch_silu_and_quant_forward(x, smooth_scale=None)
    y_q, y_scale, y_maxs, yt_q, yt_scale = triton_silu_and_quant_forward(x, 
                                                    smooth_scale=None,
                                                    round_scale=True,
                                                    calibrate=False,
                                                    output_mode=2)
    output_check(y_q_ref.float(), y_q.float(), 'block.y_q')
    output_check(y_scale_ref, y_scale.t(), 'block.y_scale')
    output_check(yt_q_ref.float(), yt_q.float(), 'block.yt_q')
    output_check(yt_scale_ref, yt_scale.t(), 'block.yt_scale')

    y_q, y_scale, y_maxs, yt_q, yt_scale = triton_silu_and_quant_forward(x, 
                                                    smooth_scale=None,
                                                    round_scale=True,
                                                    calibrate=False,
                                                    output_mode=0)
    output_check(y_q_ref.float(), y_q.float(), 'block.y_q')
    output_check(y_scale_ref, y_scale.t(), 'block.y_scale')

    y_q, y_scale, y_maxs, yt_q, yt_scale = triton_silu_and_quant_forward(x, 
                                                    smooth_scale=None,
                                                    round_scale=True,
                                                    calibrate=False,
                                                    output_mode=1)
    output_check(yt_q_ref.float(), yt_q.float(), 'block.yt_q')
    output_check(yt_scale_ref, yt_scale.t(), 'block.yt_scale')

    dx_q_ref, dx_scale_ref, dxt_q_ref, dxt_scale_ref = torch_silu_and_quant_backward(grad_output, x,
                                                           smooth_scale=None,
                                                           round_scale=True)
    dx_q, dx_scale, dxt_q, dxt_scale = triton_silu_and_quant_backward(grad_output, x,
                                                    smooth_scale=None,
                                                    reverse=False,
                                                    round_scale=True)
    output_check(dx_q_ref.float(), dx_q.float(), 'block.dx_q')
    output_check(dx_scale_ref.t(), dx_scale, 'block.dx_scale')
    output_check(dxt_q_ref.float(), dxt_q.float(), 'block.dxt_q')
    output_check(dxt_scale_ref.t(), dxt_scale, 'block.dxt_scale')

    if bench:
        benchmark_func(triton_silu_and_quant_forward, x, smooth_scale=None,
                       n_repeat=100, ref_bytes=M * N * 3)
        benchmark_func(triton_silu_and_quant_backward, grad_output, x, smooth_scale=None,
                       n_repeat=100, ref_bytes=M * N * 5)


def test_triton_batch_weighted_silu_and_smooth_quant(M=1024, N=4096, n_experts=32,
                                              bench=False):
    if True:
        count_list = [random.randint(M // 2, M // 2 * 3)//16*16 for _ in range(n_experts)]
        counts = torch.tensor(count_list, device='cuda:0', dtype=torch.int32)
        bs = sum(count_list)

        x = torch.randn((bs, N), dtype=torch.bfloat16, device='cuda:0') ** 3 / 4
        weight = torch.randn((bs, 1), dtype=torch.float32, device='cuda:0')
        smooth_scales = 1+torch.rand((n_experts,N//2),dtype=torch.float32,device='cuda:0')*10
    else:
        d = torch.load('/ossfs/workspace/Megatron-LM/silu.bin')
        counts = d['counts'].cuda()
        x = d['x'].cuda()
        weight = d['weight'].cuda()
        smooth_scales = d['smooth_scale'].cuda()
        bs = sum(counts.tolist())
        N = x.shape[-1]
        n_experts = counts.shape[0]

    grad_output = torch.randn((bs, N // 2), dtype=torch.bfloat16,
                              device='cuda:0') ** 3
    grad_smooth_scales = 1 + torch.rand((n_experts, N), dtype=torch.float32,
                                        device='cuda:0') * 10
    transpose_grad_smooth_scales = 1 + torch.rand((bs,), dtype=torch.float32,
                                        device='cuda:0') * 10
    round_scale = True


    x_q_ref, x_scale_ref, x_max_ref, _, _ = torch_batch_weighted_silu_and_quant_forward(x,
                                                                       weight,
                                                                       counts,
                                                                       smooth_scales=smooth_scales,
                                                                       round_scale=round_scale,
                                                                       reverse=False)
    x_q, x_scale,_, _, _ = triton_batch_weighted_silu_and_quant_forward(x, weight,
                                                                   counts,
                                                                   smooth_scale=smooth_scales,
                                                                   round_scale=round_scale,
                                                                   reverse=False)
    output_check(x_q_ref.float(), x_q.float(), 'smooth.data')
    output_check(x_scale_ref, x_scale, 'smooth.scale')


    dx_ref, dx_scale_ref, dw_ref, dxt_ref, dxt_scale_ref = torch_batch_weighted_silu_and_quant_backward(
        grad_output, x, weight, count_list, 
        smooth_scales=grad_smooth_scales,
        transpose_smooth_scale=transpose_grad_smooth_scales,
        round_scale=round_scale, reverse=False)
    dx, dx_scale, dw, dxt, dxt_scale = triton_batch_weighted_silu_and_quant_backward(
        grad_output, x, weight, counts, 
        smooth_scale=grad_smooth_scales,
        transpose_smooth_scale=transpose_grad_smooth_scales,
        splits=count_list,
        round_scale=round_scale, 
        reverse=False)
    output_check(dx_ref.float(), dx.float(), 'smooth.dx')
    output_check(dx_scale_ref, dx_scale, 'smooth.dx_scale')
    output_check(dw_ref, dw, 'smooth.dw')
    output_check(dxt_ref.float(), dxt.float(), 'smooth.dxt')
    output_check(dxt_scale_ref, dxt_scale.view(-1), 'smooth.dxt_scale')

    if bench:
        ref_time = None
        benchmark_func(triton_batch_weighted_silu_and_quant_forward, x, weight,
                    counts, smooth_scale=smooth_scales, round_scale=True, 
                       ref_bytes=n_experts * M * N * 2.5, ref_time=ref_time)
        benchmark_func(triton_batch_weighted_silu_and_quant_backward,
                       grad_output, x, weight, counts, 
                       smooth_scale=smooth_scales,
                       transpose_smooth_scale=transpose_grad_smooth_scales,
                       splits=count_list,
                       round_scale=True, 
                       ref_bytes=n_experts * M * N * 4, ref_time=ref_time)



def test_triton_batch_weighted_silu_and_block_quant(M=1024, N=4096, n_experts=32,
                                              bench=False):
    if True:
        count_list = [random.randint(M // 2, M // 2 * 3)//16*16 for _ in range(n_experts)]
        counts = torch.tensor(count_list, device='cuda:0', dtype=torch.int32)
        bs = sum(count_list)

        x = torch.randn((bs, N), dtype=torch.bfloat16, device='cuda:0') ** 3 / 4
        weight = torch.randn((bs, 1), dtype=torch.float32, device='cuda:0')
    else:
        d = torch.load('/ossfs/workspace/Megatron-LM/silu.bin')
        counts = d['counts'].cuda()
        x = d['x'].cuda()
        weight = d['weight'].cuda()
        smooth_scales = d['smooth_scale'].cuda()
        bs = sum(counts.tolist())
        N = x.shape[-1]
        n_experts = counts.shape[0]

    grad_output = torch.randn((bs, N // 2), dtype=torch.bfloat16,
                              device='cuda:0') ** 3
    round_scale = True



    x_q_ref, x_scale_ref,x_max_ref, xt_q_ref, xt_scale_ref = torch_batch_weighted_silu_and_quant_forward(x,
                                                                       weight,
                                                                       counts,
                                                                       smooth_scales=None,
                                                                       round_scale=round_scale,
                                                                       reverse=False)
    x_q, x_scale, _, xt_q, xt_scale = triton_batch_weighted_silu_and_quant_forward(x, weight,
                                                                   counts,
                                                                   smooth_scale=None,
                                                                   round_scale=round_scale,
                                                                   splits=count_list,
                                                                   reverse=False)
    output_check(x_q_ref.float(), x_q.float(), 'block.q')
    output_check(x_scale_ref, x_scale, 'block.scale')
    output_check(xt_q_ref.float(), xt_q.float(), 'block.qt')
    output_check(xt_scale_ref, xt_scale, 'block.t_scale')


    dx_ref, dx_scale_ref, dw_ref, dxt_ref, dxt_scale_ref = torch_batch_weighted_silu_and_quant_backward(
        grad_output, x, weight, counts, smooth_scales=None,
        round_scale=round_scale, reverse=False)
    dx, dx_scale, dw, dxt, dxt_scale = triton_batch_weighted_silu_and_quant_backward(
        grad_output, x, weight, counts, smooth_scale=None, splits=count_list,
        round_scale=round_scale, reverse=False)
    output_check(dx_ref.float(), dx.float(), 'block.dx')
    output_check(dx_scale_ref, dx_scale, 'block.dx_scale')
    output_check(dw_ref, dw, 'block.dw')
    output_check(dxt_ref.float(), dxt.float(), 'block.dxt')
    output_check(dxt_scale_ref, dxt_scale, 'block.dxt_scale')

    if bench:
        ref_time = None
        benchmark_func(triton_batch_weighted_silu_and_quant_forward, x, weight,
                    counts, smooth_scale=None, round_scale=True, splits=count_list, 
                    output_mode=0, n_repeat=100,
                       ref_bytes=n_experts * M * N * 2.5, ref_time=ref_time)
        benchmark_func(triton_batch_weighted_silu_and_quant_forward, x, weight,
                    counts, smooth_scale=None, round_scale=True, splits=count_list, 
                    output_mode=1, n_repeat=100,
                       ref_bytes=n_experts * M * N * 2.5, ref_time=ref_time)
        benchmark_func(triton_batch_weighted_silu_and_quant_forward, x, weight,
                    counts, smooth_scale=None, round_scale=True, splits=count_list, 
                    output_mode=2, n_repeat=100,
                       ref_bytes=n_experts * M * N * 3, ref_time=ref_time)
        benchmark_func(triton_batch_weighted_silu_and_quant_backward,
                       grad_output, x, weight, counts, smooth_scale=None,
                       round_scale=True, splits=count_list, n_repeat=100,
                       ref_bytes=n_experts * M * N * 4, ref_time=ref_time)

if __name__ == '__main__':
    # test_silu_and_smooth_quant(M=16384, N=1024, bench=True)
    # test_silu_and_smooth_quant(M=8192, N=2048, bench=False)
    # test_silu_and_smooth_quant(M=4096, N=10240, bench=False)
    # test_silu_and_smooth_quant(M=4096, N=5120, bench=False)

    # test_silu_and_block_quant(M=16384, N=1024, bench=True)


    # test_triton_batch_weighted_silu_and_smooth_quant(M=2048, N=2048, n_experts=32, bench=False)

    # test_triton_batch_weighted_silu_and_smooth_quant(M=800, N=2048, n_experts=32, bench=False)
    test_triton_batch_weighted_silu_and_smooth_quant(M=0, N=2048, n_experts=32, bench=False)

    # test_triton_batch_weighted_silu_and_block_quant(M=4096, N=2048, n_experts=32, bench=False)
    # test_triton_batch_weighted_silu_and_block_quant(M=800, N=2048, n_experts=32, bench=False)
