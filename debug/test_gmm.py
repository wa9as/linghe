import torch

from flops.facade.smooth_quant_linear import QuantLinear
from flops.quant.smooth import triton_smooth_quant_nt, smooth_quant_forward, \
    triton_slide_smooth_quant
from flops.utils.benchmark import benchmark_func
from flops.utils.pad import triton_batch_slice_and_pad

torch.manual_seed(7)
torch.cuda.manual_seed_all(7)


# torch.backends.cudnn.deterministic = True
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def get_expert_indices(logits, n_act=4):
    assert logits.dtype == torch.float32
    bs, length, expert = logits.shape
    logits = logits.view(bs * length, expert)
    double_logits = logits.to(torch.float64) * (
                1 + torch.arange(expert, device=logits.device,
                                 dtype=torch.float64) * 1e-12)
    double_top_values, top_indices = torch.topk(double_logits, n_act, dim=-1,
                                                largest=True, sorted=True)
    mask_logits = torch.where(double_logits >= double_top_values[:, -1:],
                              logits, -10000.0)
    expert_indices = []
    for i in range(expert):
        expert_index = torch.where(mask_logits[:, i] > -5000)[0]
        expert_indices.append(expert_index)
    return expert_indices


def gmm(hs, ws):
    # hs: [v, dim]*bs
    # ws: [expert, im, dim]
    expert, ims, dim = ws.shape
    outputs = []
    for i in range(expert):
        h = hs[i]
        out = h @ ws[i].t()
        outputs.append(out)
    # outputs = torch.cat(outputs, dim=0)
    return outputs


def slice_and_gmm(hs, expert_indices, ws):
    # hs: [bs, length, dim]
    # expert_indices: [indices]*n_act
    # ws: [expert, im, dim]
    bs, length, dim = hs.shape
    hs = hs.view(bs * length, dim)
    expert, ims, dim = ws.shape
    outputs = []
    for i in range(expert):
        expert_index = expert_indices[i]
        h = hs[expert_index]
        out = h @ ws[i].t()
        outputs.append(out)
    # outputs = torch.cat(outputs, dim=0)
    return outputs


def fp8_gmm(hs, wqs, smooth_scale, w_scale):
    expert, ims, dim = wqs.shape
    outputs = []
    for i in range(expert):
        x_q, x_scale = triton_slide_smooth_quant(hs[i], smooth_scale)
        output = torch._scaled_mm(x_q,
                                  wqs.view(torch.int8)[i].view(
                                      torch.float8_e4m3fn).t(),
                                  scale_a=x_scale.view(-1, 1),
                                  scale_b=w_scale[:, i * ims:(i + 1) * ims],
                                  out_dtype=hs[0].dtype,
                                  use_fast_accum=True)
        outputs.append(output)
    return outputs


def split_moe(hs, logits, wgs, wus, wds, n_act=4):
    # hs: [bs, length, dim]
    # logits: [bs, length, expert]
    # w1s: [expert, im, dim]
    assert logits.dtype == torch.float32
    bs, length, dim = hs.shape
    hs = hs.view(bs * length, dim)
    expert, ims, dim = wgs.shape
    logits = logits.view(bs * length, expert)
    double_logits = logits.to(torch.float64) * (
                1 + torch.arange(expert, device=hs.device,
                                 dtype=torch.float64) * 1e-12)
    double_top_values, top_indices = torch.topk(double_logits, n_act, dim=-1,
                                                largest=True, sorted=True)
    mask_logits = torch.where(double_logits >= double_top_values[:, -1:],
                              logits, -10000.0)
    mask_probs = torch.softmax(mask_logits, -1).to(hs.dtype)
    expert_indices = []
    outputs = []
    for i in range(expert):
        expert_index = torch.where(mask_logits[:, i] > -5000)[0]
        expert_indices.append(expert_index.float() + 0.001 * i)
        expert_prob = mask_probs[expert_index, i:i + 1]
        h = hs[expert_index]
        act = torch.nn.functional.silu(h @ wgs[i].t()) * (h @ wus[i].t())
        out = act @ wds[i].t()
        out *= expert_prob
        outputs.append(out)
    outputs = torch.cat(outputs, dim=0)
    expert_indices = torch.cat(expert_indices, dim=0)
    gather_indices = torch.argsort(expert_indices, dim=-1, descending=False,
                                   stable=False)
    outputs = outputs[gather_indices]
    outputs = outputs.view(bs * length, n_act, dim)
    outputs = outputs.sum(1)
    return outputs.view(bs, length, dim)


def moe_ref(hs, logits, ws, vs, n_act=4):
    # hs: [bs, length, dim]
    # logits: [bs, length, expert]
    # w1s: [expert, im, dim]
    assert logits.dtype == torch.float32
    bs, length, dim = hs.shape
    hs = hs.view(bs * length, dim)
    expert, dim, ims = vs.shape
    logits = logits.view(bs * length, expert)
    double_logits = logits.to(torch.float64) * (
                1 + torch.arange(expert, device=hs.device,
                                 dtype=torch.float64) * 1e-12)
    double_top_values, top_indices = torch.topk(double_logits, n_act, dim=-1,
                                                largest=True, sorted=True)
    mask_logits = torch.where(double_logits >= double_top_values[:, -1:],
                              logits, -10000.0)
    mask_probs = torch.softmax(mask_logits, -1).to(
        hs.dtype)  # [bs*length, expert]
    outputs = []
    for i in range(expert):
        gus = hs @ ws[i].t()
        act = torch.nn.functional.silu(gus[:, :ims]) * gus[:, ims:]
        out = act @ vs[i].t()
        outputs.append(out)
    outputs = torch.stack(outputs, 2)  # [bs*length, dim, expert]
    outputs = torch.einsum("bde,be->bd", outputs, mask_probs)
    return outputs.view(bs, length, dim)


def moe(hs, logits, ws, vs, n_act=4):
    # hs: [bs, length, dim]
    # logits: [bs, length, expert]
    # w1s: [expert, im, dim]
    assert logits.dtype == torch.float32
    bs, length, dim = hs.shape
    hs = hs.view(bs * length, dim)
    expert, dim, ims = vs.shape
    logits = logits.view(bs * length, expert)
    double_logits = logits.to(torch.float64) * (
                1 + torch.arange(expert, device=hs.device,
                                 dtype=torch.float64) * 1e-12)
    double_top_values, top_indices = torch.topk(double_logits, n_act, dim=-1,
                                                largest=True, sorted=True)
    mask_logits = torch.where(double_logits >= double_top_values[:, -1:],
                              logits, -10000.0)
    mask_probs = torch.softmax(mask_logits, -1).to(hs.dtype)
    expert_float_indices = []
    outputs = []
    for i in range(expert):
        expert_index, = torch.where(mask_logits[:, i] > -5000)
        expert_float_indices.append(expert_index.float() + 0.001 * i)
        expert_prob = mask_probs[expert_index, i:i + 1]
        h = hs[expert_index]
        gus = h @ ws[i].t()
        act = torch.nn.functional.silu(gus[:, :ims]) * gus[:, ims:]
        out = act @ vs[i].t()
        out *= expert_prob
        outputs.append(out)
    outputs = torch.cat(outputs, dim=0)
    expert_float_indices = torch.cat(expert_float_indices, dim=0)
    gather_indices = torch.argsort(expert_float_indices, dim=-1,
                                   descending=False, stable=False)
    outputs = outputs[gather_indices]
    outputs = outputs.view(bs * length, n_act, dim)
    outputs = outputs.sum(1)
    return outputs.view(bs, length, dim)


def naive_fp8_moe(hs, logits, wfns, vfns, n_act=4):
    # hs: [bs, length, dim]
    # logits: [bs, length, expert]
    # w1s: [expert, im, dim]
    assert logits.dtype == torch.float32
    bs, length, dim = hs.shape
    hs = hs.view(bs * length, dim)
    expert = len(wfns)
    dim, ims = vfns[0].weight.shape
    logits = logits.view(bs * length, expert)
    double_logits = logits.to(torch.float64) * (
                1 + torch.arange(expert, device=hs.device,
                                 dtype=torch.float64) * 1e-12)
    double_top_values, top_indices = torch.topk(double_logits, n_act, dim=-1,
                                                largest=True, sorted=True)
    mask_logits = torch.where(double_logits >= double_top_values[:, -1:],
                              logits, -10000.0)
    mask_probs = torch.softmax(mask_logits, -1).to(hs.dtype)
    expert_float_indices = []
    outputs = []
    for i in range(expert):
        expert_index, = torch.where(mask_logits[:, i] > -5000)
        expert_float_indices.append(expert_index.float() + 0.001 * i)
        n_token = expert_index.size(0)
        if n_token % 128 != 0:
            pad_token = 128 - n_token % 128
            expert_index = torch.nn.functional.pad(expert_index, (0, pad_token),
                                                   "constant", 0)
        expert_prob = mask_probs[expert_index, i:i + 1]
        h = hs[expert_index]
        gus = wfns[i](h)
        act = torch.nn.functional.silu(gus[:, :ims]) * gus[:, ims:]
        out = vfns[i](act)
        out = out * expert_prob
        if n_token % 128 != 0:
            out = out[:n_token]
        outputs.append(out)
    outputs = torch.cat(outputs, dim=0)
    expert_float_indices = torch.cat(expert_float_indices, dim=0)
    gather_indices = torch.argsort(expert_float_indices, dim=-1,
                                   descending=False, stable=False)
    outputs = outputs[gather_indices]
    outputs = outputs.view(bs * length, n_act, dim)
    outputs = outputs.sum(1)
    return outputs.view(bs, length, dim)


def fused_fp8_moe_forward(hs, logits, ws, vs, n_act=4):
    # hs: [bs, length, dim]
    # logits: [bs, length, expert]
    # w1s: [expert, im, dim]
    assert logits.dtype == torch.float32
    bs, length, dim = hs.shape
    hs = hs.view(bs * length, dim)
    expert, ims2, dim = ws.shape
    ims = ims2 // 2
    logits = logits.view(bs * length, expert)
    double_logits = logits.to(torch.float64) * (
                1 + torch.arange(expert, device=hs.device,
                                 dtype=torch.float64) * 1e-12)
    double_top_values, top_indices = torch.topk(double_logits, n_act, dim=-1,
                                                largest=True, sorted=True)
    mask_logits = torch.where(double_logits >= double_top_values[:, -1:],
                              logits, -10000.0)
    mask_probs = torch.softmax(mask_logits, -1).to(hs.dtype)
    expert_indices = []
    float_expert_indices = []
    expert_probs = []
    outputs = []
    n_tokens = []
    for i in range(expert):
        expert_index, = torch.where(mask_logits[:, i] > -5000)
        expert_indices.append(expert_index)
        float_expert_indices.append(expert_index.float() + 0.001 * i)
        n_token = expert_index.size(0)
        n_tokens.append(n_token)
        if n_token % 128 != 0:
            pad_token = 128 - n_token % 128
            expert_index = torch.nn.functional.pad(expert_index, (0, pad_token),
                                                   "constant", 0)
        expert_probs.append(mask_probs[expert_index, i:i + 1])

    x_q, w_q, x_scale, w_scale, smooth_scale = triton_smooth_quant_nt(hs,
                                                                      ws.view(
                                                                          expert * ims * 2,
                                                                          dim))
    w_q = w_q.view(expert, ims * 2, dim)
    w_scale = w_scale.view(expert, ims * 2)
    pad_xs = triton_batch_slice_and_pad(x_q, expert_indices, expert=expert,
                                        block=128)
    pad_x_scales = triton_batch_slice_and_pad(x_scale, expert_indices,
                                              expert=expert, block=128)
    for i in range(expert):
        h = pad_xs[i]
        gus = torch._scaled_mm(h,
                               w_q.view(torch.int8)[i].view(
                                   torch.float8_e4m3fn).t(),
                               scale_a=pad_x_scales[i],
                               scale_b=w_scale[i:i + 1],
                               out_dtype=hs.dtype,
                               use_fast_accum=True)

        act = torch.nn.functional.silu(gus[:, :ims]) * gus[:, ims:]
        out = smooth_quant_forward(act, vs[i])[0]
        out *= expert_probs[i]
        outputs.append(out[:n_tokens[i]])
    outputs = torch.cat(outputs, dim=0)
    float_expert_indices = torch.cat(float_expert_indices, dim=0)
    gather_indices = torch.argsort(float_expert_indices, dim=-1,
                                   descending=False, stable=False)
    outputs = outputs[gather_indices]
    outputs = outputs.view(bs * length, n_act, dim)
    outputs = outputs.sum(1)
    return outputs.view(bs, length, dim)


def slide_fp8_moe_forward(hs, logits, ws, w_scales, vs, v_scales,
                          w_smooth_scale, v_smooth_scale, n_act=4):
    # hs: [bs, length, dim]
    # logits: [bs, length, expert]
    # w1s: [expert, im, dim]
    assert logits.dtype == torch.float32
    bs, length, dim = hs.shape
    hs = hs.view(bs * length, dim)
    expert, ims2, dim = ws.shape
    ims = ims2 // 2
    logits = logits.view(bs * length, expert)
    double_logits = logits.to(torch.float64) * (
                1 + torch.arange(expert, device=hs.device,
                                 dtype=torch.float64) * 1e-12)
    double_top_values, top_indices = torch.topk(double_logits, n_act, dim=-1,
                                                largest=True, sorted=True)
    mask_logits = torch.where(double_logits >= double_top_values[:, -1:],
                              logits, -10000.0)
    mask_probs = torch.softmax(mask_logits, -1).to(hs.dtype)
    expert_indices = []
    float_expert_indices = []
    expert_probs = []
    outputs = []
    n_tokens = []
    for i in range(expert):
        expert_index, = torch.where(mask_logits[:, i] > -5000)
        expert_indices.append(expert_index)
        float_expert_indices.append(expert_index.float() + 0.001 * i)
        n_token = expert_index.size(0)
        n_tokens.append(n_token)
        if n_token % 128 != 0:
            pad_token = 128 - n_token % 128
            expert_index = torch.nn.functional.pad(expert_index, (0, pad_token),
                                                   "constant", 0)
        expert_probs.append(mask_probs[expert_index, i:i + 1])

    x_q, x_scale = triton_slide_smooth_quant(hs, w_smooth_scale)
    w_scale = w_scales.view(expert, ims * 2)
    v_scale = v_scales.view(expert, dim)
    pad_xs = triton_batch_slice_and_pad(x_q, expert_indices, expert=expert,
                                        block=128)
    pad_x_scales = triton_batch_slice_and_pad(x_scale, expert_indices,
                                              expert=expert, block=128)
    for i in range(expert):
        h = pad_xs[i]
        gus = torch._scaled_mm(h,
                               ws.view(torch.int8)[i].view(
                                   torch.float8_e4m3fn).t(),
                               scale_a=pad_x_scales[i],
                               scale_b=w_scale[i:i + 1],
                               out_dtype=hs.dtype,
                               use_fast_accum=True)

        act = torch.nn.functional.silu(gus[:, :ims]) * gus[:, ims:]
        a_q, a_scale = triton_slide_smooth_quant(act, v_smooth_scale)
        out = torch._scaled_mm(a_q,
                               vs.view(torch.int8)[i].view(
                                   torch.float8_e4m3fn).t(),
                               scale_a=a_scale,
                               scale_b=v_scale[i:i + 1],
                               out_dtype=hs.dtype,
                               use_fast_accum=True)
        out *= expert_probs[i]
        outputs.append(out[:n_tokens[i]])
        outputs.append(out)
    outputs = torch.cat(outputs, dim=0)
    float_expert_indices = torch.cat(float_expert_indices, dim=0)
    gather_indices = torch.argsort(float_expert_indices, dim=-1,
                                   descending=False, stable=False)
    outputs = outputs[gather_indices]
    outputs = outputs.view(bs * length, n_act, dim)
    outputs = outputs.sum(1)
    return outputs.view(bs, length, dim)


class GLMTopNRouter(torch.nn.Module):
    """
    This implementation is equivalent to the standard
    TopN MoE with full capacity without dropp tokens.
    """

    def __init__(self, expert, hidden_size, top_k):
        super().__init__()
        self.num_experts = expert

        self.classifier = torch.nn.Linear(
            hidden_size, self.num_experts, bias=False
        )
        self.top_k = top_k

    def _cast_classifier(self):
        r"""
        `bitsandbytes` `Linear8bitLt` layers does not support manual casting Therefore we need to check if they are an
        instance of the `Linear8bitLt` class by checking special attributes.
        """
        pass

    def _compute_router_probabilities(self,
                                      hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self.classifier(hidden_states)

        return router_logits

    def _route_tokens(self, router_logits: torch.Tensor):
        router_probs = torch.nn.functional.softmax(router_logits, dim=-1,
                                                   dtype=torch.float)

        topk_weight, topk_experts_index = torch.topk(router_probs, self.top_k,
                                                     dim=-1)
        topk_weight /= topk_weight.sum(dim=-1, keepdim=True)
        return topk_weight, topk_experts_index

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        router_logits = self._compute_router_probabilities(hidden_states)

        router_probs, topk_experts_index = self._route_tokens(router_logits)

        return router_probs, router_logits, topk_experts_index


if __name__ == '__main__':

    # https://yuque.antfin.com/graph/fvptqx/qv4331ph6we9gv04#iKkO
    # bs, length, dim, ims, expert, n_act = 8, 4096, 2048, 1408, 64, 6  # lite
    # bs, length, dim, ims, expert, n_act = 8, 4096, 5376, 3072, 64, 4  # plus
    bs, length, dim, ims, expert, n_act = 4, 4096, 7168, 5376, 64, 2  # max
    dtype = torch.bfloat16
    device = 'cuda:0'
    hidden_states = 0.1 * torch.randn((bs, length, dim), dtype=dtype,
                                      device=device)
    gate_weights = 0.1 * torch.randn((expert, ims, dim), dtype=dtype,
                                     device=device)
    up_weights = 0.1 * torch.randn((expert, ims, dim), dtype=dtype,
                                   device=device)
    down_weights = 0.1 * torch.randn((expert, dim, ims), dtype=dtype,
                                     device=device)
    gate_up_weights = torch.cat([gate_weights, up_weights], dim=1)
    logits = torch.randn((bs, length, expert), dtype=torch.float32,
                         device=device)
    expert_indices = get_expert_indices(logits, n_act=n_act)
    split_hidden_states = [hidden_states.view(bs * length, dim)[x] for x in
                           expert_indices]
    gate_up_scales = torch.rand((1, expert * ims * 2), dtype=torch.float32,
                                device=device)
    down_scales = torch.rand((1, expert * dim), dtype=torch.float32,
                             device=device)
    gate_up_smooth_scale = torch.rand((dim,), dtype=torch.float32,
                                      device=device)
    down_smooth_scale = torch.rand((ims,), dtype=torch.float32, device=device)

    wfns = []
    vfns = []
    for i in range(expert):
        wfns.append(
            QuantLinear(in_features=dim, out_features=2 * ims, dtype=dtype,
                        device=device))
        vfns.append(QuantLinear(in_features=ims, out_features=dim, dtype=dtype,
                                device=device))

    if False:
        # test can run in `h800` aistudio exp
        from atorch.modules.moe.grouped_gemm_moe import Grouped_GEMM_MoE

        atorch_moe = Grouped_GEMM_MoE(
            hidden_size=dim,
            expert_intermediate_size=ims,
            output_dropout_prob=False,
            num_experts=expert,
            topk=n_act,
            use_swiglu=True,
            use_bias=False,
            initializer_range=0.01,
            use_expert_parallelism=False,
            expert_parallel_group=None,
            transpose_w1=True,
            merge_w1_v1=True,
            implementation_type="MegaBlocks",
            token_dispatcher_type="AllToAll",
            is_scale_gradient=True,
        ).to(device).to(torch.bfloat16)
        router = GLMTopNRouter(expert, dim, n_act).to(device).to(torch.bfloat16)
        router_probs, router_logits, topk_experts_index = router(hidden_states)
        o = atorch_moe(hidden_states, router_probs, topk_experts_index)

    gmm_ref_flops = bs * length * dim * ims * n_act * 2 * 2
    ref_flops = bs * length * dim * ims * n_act * 3 * 2

    org_output = moe_ref(hidden_states, logits, gate_up_weights, down_weights,
                         n_act=n_act)
    opt_output = moe(hidden_states, logits, gate_up_weights, down_weights,
                     n_act=n_act)
    # torch.testing.assert_close(org_output,opt_output,rtol=0.05,atol=0.01)

    # naive_fp8_moe(hidden_states, logits, wfns, vfns, n_act=n_act)
    # fused_fp8_moe_forward(hidden_states, logits, gate_up_weights, down_weights, n_act=n_act)
    # slide_fp8_moe_forward(hidden_states, logits, gate_up_weights.to(torch.float8_e4m3fn), gate_up_scales, down_weights.to(torch.float8_e4m3fn), down_scales, gate_up_smooth_scale, down_smooth_scale, n_act=4)

    # print(expert*ims*2)

    n_repeat = 100
    # print(f'\n{bs=} {length=} {dim=} {ims=} {expert=} {n_act=}')
    # benchmark_func(gmm, split_hidden_states, gate_up_weights, ref_flops=gmm_ref_flops, n_repeat=n_repeat)
    # benchmark_func(slice_and_gmm, hidden_states, expert_indices, gate_up_weights, ref_flops=gmm_ref_flops, n_repeat=n_repeat)
    # benchmark_func(fp8_gmm, split_hidden_states, gate_up_weights.to(torch.float8_e4m3fn), gate_up_smooth_scale,  gate_up_scales, ref_flops=gmm_ref_flops, n_repeat=n_repeat)

    benchmark_func(split_moe, hidden_states, logits, gate_weights, up_weights,
                   down_weights, n_act=n_act, ref_flops=ref_flops,
                   n_repeat=n_repeat)
    ref_time = benchmark_func(moe, hidden_states, logits, gate_up_weights,
                              down_weights, n_act=n_act, ref_flops=ref_flops,
                              n_repeat=n_repeat)
    # benchmark_func(naive_fp8_moe, hidden_states, logits, wfns, vfns, n_act=n_act, ref_flops=ref_flops, ref_time=ref_time, n_repeat=n_repeat)
    # benchmark_func(fused_fp8_moe_forward, hidden_states, logits, gate_up_weights, down_weights, n_act=n_act, ref_flops=ref_flops, ref_time=ref_time,n_repeat=n_repeat)
    # benchmark_func(slide_fp8_moe_forward, hidden_states, logits,  gate_up_weights.to(torch.float8_e4m3fn), gate_up_scales, down_weights.to(torch.float8_e4m3fn), down_scales, gate_up_smooth_scale, down_smooth_scale, n_act=n_act, ref_flops=ref_flops, ref_time=ref_time,n_repeat=n_repeat)
