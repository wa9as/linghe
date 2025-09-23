import torch
import triton
import triton.language as tl





@triton.jit
def silu_and_block_quant_forward_kernel(x_ptr, 
                                        out_ptr, scale_ptr,
                                        transpose_output_ptr, 
                                        transpose_scale_ptr,
                                        M, 
                                        n: tl.constexpr, 
                                        ROUND: tl.constexpr,
                                        OUTPUT_MODE: tl.constexpr):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)

    offs = rid * 128 * n * 2 + cid * 128 + tl.arange(0, 128)[:, None] * n * 2 + tl.arange(0, 128)[None, :]
    indices = rid * 128 + tl.arange(0, 128)
    mask = indices[:, None] < M


    x1 = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    x2 = tl.load(x_ptr + n + offs, mask=mask).to(tl.float32)
    x = x1 * tl.sigmoid(x1) * x2
    # x1 = tl.load(x_ptr + offs, mask=mask)
    # x2 = tl.load(x_ptr + n + offs, mask=mask)
    # x = tl.sigmoid(x1.to(tl.float32)) * x1 * x2

    if OUTPUT_MODE % 2 == 0:
        scale = tl.maximum(tl.max(x.abs(), 1) / 448, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        
        tl.store(scale_ptr + rid * 128 + cid * M + tl.arange(0, 128), scale, mask=indices < M)
        xq = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + rid * 128 * n + cid * 128 + tl.arange(0, 128)[:, None] * n + tl.arange(0, 128)[None, :], xq, mask=mask)

    if OUTPUT_MODE > 0:
        scale = tl.maximum(tl.max(x.abs(), 0) / 448, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(transpose_scale_ptr + rid * n + cid * 128 + tl.arange(0, 128), scale)
        xq = (x / scale).to(out_ptr.dtype.element_ty)
        tl.store(transpose_output_ptr + rid * 128 + cid * 128 * M + tl.arange(0, 128)[:, None] * M + tl.arange(0, 128)[
                                                            None, :], tl.trans(xq), mask=indices[None, :] < M)




# used in shared expert
def triton_silu_and_block_quant_forward(x, out=None, scale=None,
                                  round_scale=False,
                                  output_mode=2):
    # row-wise read, row-wise write
    M, N = x.shape
    n = N // 2
    device = x.device
    if out is None:
        out = torch.empty((M, N // 2), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.empty((N // 2 // 128, M), device=device, dtype=torch.float32)

    transpose_output = torch.empty((N // 2, M), device=device, dtype=torch.float8_e4m3fn) 
    transpose_scale = torch.empty((triton.cdiv(M, 128), N // 2), device=device, dtype=torch.float32)

    grid = (triton.cdiv(M, 128), n // 128)
    silu_and_block_quant_forward_kernel[grid](
        x,
        out,
        scale,
        transpose_output, 
        transpose_scale,
        M,
        n,
        round_scale,
        output_mode,
        num_stages=2,
        num_warps=16
    )

    return out, scale, transpose_output, transpose_scale



@triton.jit
def silu_and_block_quant_backward_kernel(g_ptr, x_ptr,  
                                        dx_ptr,
                                        dx_scale_ptr, 
                                        transpose_dx_ptr,
                                        transpose_dx_scale_ptr,
                                        M, 
                                        n: tl.constexpr,
                                        ROUND: tl.constexpr):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)
    nb = n // 128
    offs = rid * 128 * n * 2 + cid * 128 + tl.arange(0, 128)[:, None] * n * 2 + tl.arange(0, 128)[None, :]
    toffs = rid * 128 + cid * M * 128 + tl.arange(0, 128)[:, None] * M + tl.arange(0, 128)[
                                                            None, :]
    idx = rid * 128 + tl.arange(0, 128)
    mask = idx[:, None] < M
    x1 = tl.load(x_ptr + offs, mask=mask)#.to(tl.float32)
    x2 = tl.load(x_ptr + n + offs, mask=mask)#.to(tl.float32)
    g = tl.load(g_ptr + rid * 128 * n + cid * 128 + 
                    tl.arange(0, 128)[:, None] * n + 
                    tl.arange(0, 128)[None, :], mask=mask)#.to(tl.float32)
    sigmoid = tl.sigmoid(x1.to(tl.float32))
    dx1 = sigmoid * g * x2 * (1 + x1 * (1 - sigmoid))  # change order to trigger autocast
    scale1 = tl.maximum(
        tl.max(dx1.abs(), 1) / 448, 1e-30)
    if ROUND:
        scale1 = tl.exp2(tl.ceil(tl.log2(scale1)))
    tl.store(dx_scale_ptr + cid * M + rid * 128 + tl.arange(0, 128), scale1, mask=idx < M)

    qdx1 = (dx1 / scale1[:, None]).to(dx_ptr.dtype.element_ty)
    tl.store(dx_ptr + offs, qdx1, mask=mask)

    scale1 = tl.maximum(
        tl.max(dx1.abs(), 0) / 448, 1e-30)
    if ROUND:
        scale1 = tl.exp2(tl.ceil(tl.log2(scale1)))
    tl.store(transpose_dx_scale_ptr + rid * n * 2 + cid * 128 + tl.arange(0, 128), scale1)

    qdx1 = (dx1 / scale1[None, :]).to(dx_ptr.dtype.element_ty)
    tl.store(transpose_dx_ptr + toffs, tl.trans(qdx1), mask=idx[None, :] < M)

    dx2 = sigmoid * g * x1
    scale2 = tl.maximum(
        tl.max(dx2.abs(), 1) / 448, 1e-30)
    if ROUND:
        scale2 = tl.exp2(tl.ceil(tl.log2(scale2)))
    tl.store(dx_scale_ptr + cid * M + rid * 128 + M * nb + tl.arange(0, 128), scale2, mask=idx < M)

    qdx2 = (dx2 / scale2[:, None]).to(dx_ptr.dtype.element_ty)
    tl.store(dx_ptr + offs + n, qdx2, mask=idx[:, None] < M)

    scale2 = tl.maximum(
        tl.max(dx2.abs(), 0) / 448, 1e-30)
    if ROUND:
        scale2 = tl.exp2(tl.ceil(tl.log2(scale2)))
    tl.store(transpose_dx_scale_ptr + rid * n * 2 + n + cid * 128 + tl.arange(0, 128), scale2)

    qdx2 = (dx2 / scale2[None, :]).to(dx_ptr.dtype.element_ty)
    tl.store(transpose_dx_ptr + M * n + toffs, tl.trans(qdx2), mask=idx[None,:] < M)


# used in shared expert
def triton_silu_and_block_quant_backward(g, x, 
                                   round_scale=False):
    # row-wise read, row-wise write
    M, N = x.shape
    n = N // 2
    device = x.device
    dx = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)

    dx_scale = torch.empty((N//128, M), device=device, dtype=torch.float32)
    scale_shape = (triton.cdiv(M, 128), N)
    transpose_dx = torch.empty((N, M), device=device, dtype=torch.float8_e4m3fn)
    transpose_dx_scale = torch.empty(scale_shape, device=device, dtype=torch.float32)


    assert M % 128 == 0
    grid = (M//128, N // 256)
    silu_and_block_quant_backward_kernel[grid](
        g,
        x,
        dx,
        dx_scale,
        transpose_dx,
        transpose_dx_scale,
        M,
        n,
        round_scale,
        num_stages=2,
        num_warps=8
    )
    return dx, dx_scale, transpose_dx, transpose_dx_scale


@triton.jit
def batch_weighted_silu_and_block_quant_forward_kernel(x_ptr, weight_ptr,
                                                 out_ptr,
                                                 scale_ptr, 
                                                 transpose_output_ptr, 
                                                 transpose_scale_ptr,
                                                 count_ptr,
                                                 accum_ptr, 
                                                 n: tl.constexpr,
                                                 E: tl.constexpr,
                                                 ROUND: tl.constexpr,
                                                 OUTPUT_MODE: tl.constexpr):
    eid = tl.program_id(axis=0)
    rid = tl.program_id(axis=1)
    cid = tl.program_id(axis=2)

    count = tl.load(count_ptr + eid)
    ei = tl.load(accum_ptr + eid)
    si = ei - count
    c = tl.cdiv(count, 128)

    if rid >= c:
        return

    nb = n // 128

    counts = tl.load(count_ptr + tl.arange(0, E))
    n_blocks = tl.cdiv(counts, 128)
    transpose_scale_off = tl.sum(tl.where(tl.arange(0, E)< eid, n_blocks, 0))

    offs = si * n * 2 + rid * 128 * n * 2 + cid * 128 + tl.arange(0, 128)[:, None] * n * 2 + tl.arange(0, 128)[None, :]
    hoffs = si * n + rid * 128 * n + cid * 128 + tl.arange(0, 128)[:, None] * n + tl.arange(0, 128)[None, :]
    toffs = si * n + rid * 128 + cid * count * 128 + tl.arange(0, 128)[:, None] * count + tl.arange(0, 128)[
                                                            None, :]
    indices = rid * 128 + tl.arange(0, 128)
    mask = indices[:, None] < count
    w = tl.load(weight_ptr + si + indices, mask=indices < count).to(
            tl.float32)
    x1 = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    x2 = tl.load(x_ptr + n + offs, mask=mask).to(
        tl.float32)

    x = x1 * tl.sigmoid(x1) * x2 * w[:, None]

    if OUTPUT_MODE % 2 == 0:
        scale = tl.maximum(tl.max(tl.abs(x), 1) / 448, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(scale_ptr + si * nb + cid * count + rid * 128 + tl.arange(0, 128), scale, mask=indices < count)

        xq = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + hoffs, xq, mask=mask)

    if OUTPUT_MODE > 0:
        scale = tl.maximum(tl.max(tl.abs(x), 0) / 448, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(transpose_scale_ptr + transpose_scale_off * n + rid * n + cid * 128 + tl.arange(0, 128), scale)

        xq = tl.trans((x / scale).to(out_ptr.dtype.element_ty))
        tl.store(transpose_output_ptr + toffs, xq, mask=indices[None, :] < count)



# used in routed experts
def triton_batch_weighted_silu_and_block_quant_forward(x, 
                                                 weight, 
                                                 counts, 
                                                 splits=None,
                                                 out=None, 
                                                 scale=None,
                                                 round_scale=False,
                                                 output_mode=2):
    # row-wise read, row-wise write
    M, N = x.shape
    n = N // 2
    n_experts = counts.shape[0]
    assert N <= 8192
    device = x.device
    if out is None:
        out = torch.empty((M, n), device=device, dtype=torch.float8_e4m3fn)

    assert splits is not None, 'batch mode need splits to launch kernels'
    blocks = sum([(x+127)//128 for x in splits])
    transpose_output = torch.empty((M * n), device=device, dtype=torch.float8_e4m3fn)
    transpose_scale = torch.empty((blocks * n), device=device, dtype=torch.float32)
    # intra layout and inner layput are not consist,
    # tensors will be viewed after splitting
    scale = torch.empty((M * n // 128,), device=device, dtype=torch.float32)

    if M == 0:
        return out, scale, transpose_output, transpose_scale

    accums = torch.cumsum(counts, 0)
    
    grid = (n_experts, triton.cdiv(max(splits), 128), n//128)
    batch_weighted_silu_and_block_quant_forward_kernel[grid](
        x,
        weight,
        out,
        scale,
        transpose_output,
        transpose_scale,
        counts,
        accums,
        n,
        len(splits),
        round_scale,
        output_mode,
        num_stages=2,
        num_warps=8
    )


    return out, scale, transpose_output, transpose_scale




@triton.jit
def batch_weighted_silu_and_block_quant_backward_kernel(g_ptr, x_ptr, weight_ptr,
                                                  count_ptr,
                                                  accum_ptr, 
                                                  dx_ptr,
                                                  dx_scale_ptr, 
                                                  transpose_dx_ptr, 
                                                  transpose_dx_scale_ptr,
                                                  dw_ptr, 
                                                  n: tl.constexpr,
                                                  E: tl.constexpr,
                                                  ROUND: tl.constexpr):
    eid = tl.program_id(axis=0)
    rid = tl.program_id(axis=1)
    cid = tl.program_id(axis=2)

    count = tl.load(count_ptr + eid)
    si = tl.load(accum_ptr + eid) - count

    if rid >= tl.cdiv(count, 128):
        return 
    
    nb = n // 128
    transpose_off = tl.sum(tl.where(tl.arange(0, E) < eid, tl.cdiv(tl.load(count_ptr + tl.arange(0, E)), 128), 0))

    offs = si * n * 2 + rid * 128 * n * 2 + cid * 128 + tl.arange(0, 128)[:, None] * n * 2 + tl.arange(0, 128)[None, :]
    # hoffs = si * n + tid * 128 * n + tl.arange(0, 128)[:, None] * n + tl.arange(0, 128)[None, :]
    # toffs = si * n * 2 + tid * 128 + tl.arange(0, 128)[:, None] * count + tl.arange(0, 128)[None, :]
    idx = rid * 128 + tl.arange(0, 128)
    w = tl.load(weight_ptr + si + idx, mask=idx < count).to(tl.float32)[:, None]

    x1 = tl.load(x_ptr + offs, mask=idx[:, None] < count) #.to(tl.float32)
    x2 = tl.load(x_ptr + n + offs, mask=idx[:, None] < count) #.to(tl.float32)
    g = tl.load(g_ptr +  si * n + rid * 128 * n + 128 * cid + 
                tl.arange(0, 128)[:, None] * n + 
                tl.arange(0, 128)[None, :], 
                mask=idx[:, None] < count) #.to(tl.float32)
    sigmoid = tl.sigmoid(x1.to(tl.float32))

    dw = tl.sum(sigmoid * x1 * x2 * g, 1)
    tl.store(dw_ptr + si * nb + cid + idx * nb, dw, mask=idx < count)

    dx = sigmoid * g * x2 * w * (1 + x1 * (1 - sigmoid))
    scale = tl.maximum(
        tl.max(dx.abs(), 1) / 448, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    tl.store(dx_scale_ptr + si * nb * 2 + cid * count + rid * 128 + tl.arange(0, 128), scale, mask=idx < count)

    tl.store(dx_ptr + offs, dx / scale[:, None], mask=idx[:, None] < count)

    scale = tl.maximum(
        tl.max(dx.abs(), 0) / 448, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    tl.store(transpose_dx_scale_ptr + transpose_off * n * 2 + rid * n * 2 + cid * 128 + tl.arange(0, 128), scale)

    qdx = tl.trans((dx / scale[None, :]).to(dx_ptr.dtype.element_ty))
    # tl.store(transpose_dx_ptr + toffs, qdx, mask=idx[None, :] < count)
    tl.store(transpose_dx_ptr + si * n * 2 + rid * 128 + cid * 128 * count + 
             tl.arange(0, 128)[:, None] * count + 
             tl.arange(0, 128)[None, :], 
             qdx, 
             mask=idx[None, :] < count)

    dx = sigmoid * g * x1 * w
    scale = tl.maximum(
        tl.max(dx.abs(), 1) / 448, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    tl.store(dx_scale_ptr + si * nb * 2 + cid * count + rid * 128 + count * nb + tl.arange(0, 128), scale, mask=idx < count)
    tl.store(dx_ptr + n + offs, dx / scale[:, None], mask=idx[:, None] < count)

    scale = tl.maximum(
        tl.max(dx.abs(), 0) / 448, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    qdx = tl.trans((dx / scale[None, :]).to(dx_ptr.dtype.element_ty))
    tl.store(transpose_dx_scale_ptr + transpose_off * n * 2 + rid * n * 2 + n + cid * 128 + tl.arange(0, 128), scale)
    tl.store(transpose_dx_ptr + count * n + si * n * 2 + rid * 128 + cid * 128 * count + tl.arange(0, 128)[:, None] * count + tl.arange(0, 128)[None, :], qdx, mask=idx[None, :] < count)





# used in routed experts
def triton_batch_weighted_silu_and_block_quant_backward(g, x, weight, 
                                                  counts,
                                                  splits=None,
                                                  round_scale=False):
    # row-wise read, row-wise write
    M, N = x.shape
    n = N // 2
    n_expert = counts.shape[0]
    assert N <= 8192 and 8192 % N == 0
    assert splits is not None, 'batch mode need splits to launch kernels'

    device = x.device

    accums = torch.cumsum(counts, 0)

    dx = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn) 

    # intra layout and inner layput are not consist,
    # tensors will be viewed after splitting
    dx_scale = torch.empty((N // 128 * M), device=device, dtype=torch.float32)

    s = sum([(x+127)//128 for x in splits])
    transpose_dx = torch.empty((N * M), device=device, dtype=torch.float8_e4m3fn)  
    transpose_dx_scale = torch.empty((s * N), device=device, dtype=torch.float32) 
    if s == 0:
        dw = torch.empty_like(weight)
        return dx, dx_scale, dw, transpose_dx, transpose_dx_scale

    # grid = (n_expert, triton.cdiv(max(splits), 128))
    grid = (n_expert, triton.cdiv(max(splits), 128), N//256)
    dws = torch.empty((M, N//256), device=device, dtype=torch.float32)
    batch_weighted_silu_and_block_quant_backward_kernel[grid](
        g,
        x,
        weight,
        counts,
        accums,
        dx,
        dx_scale,
        transpose_dx,
        transpose_dx_scale,
        dws,
        n,
        n_expert,
        round_scale,
        num_stages=3,
        num_warps=16
    )
    dw = dws.sum(1, keepdim=True).to(weight.dtype)
    return dx, dx_scale, dw, transpose_dx, transpose_dx_scale
