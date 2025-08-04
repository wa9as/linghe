import torch



def test_backward_with_stride(M=4096, N=4096):
    device = 'cuda:0'
    dtype = torch.bfloat16
    x = torch.randn(M, N, dtype=dtype, device=device, requires_grad=True)
    y = x[:16,:16]
    grad = torch.randn(16,16,dtype=dtype, device=device)
    y.backward(grad)
    print(x.grad.shape)




if __name__ == '__main__':
    test_backward_with_stride(M=4096, N=4096)