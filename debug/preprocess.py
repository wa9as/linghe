
import numpy as np
import torch
import matplotlib.pyplot as plt


src = 'safetensor'
if src == 'pkl':
    file_name = '/agent/liangchen/fp8/step_003900_pid_1320_ppid_1189.pkl'

    d = torch.load(file_name, weights_only=True)
    # print(d.keys())
    idx = 25
    x = d[f'glm.transformer.layers.{idx}.mlp.down_proj.fwd_x']['tensor']
    w = d[f'glm.transformer.layers.{idx}.mlp.down_proj.fwd_w']['tensor']
    y = d[f'glm.transformer.layers.{idx}.mlp.down_proj.fwd_y']['tensor']

    torch.save({'x':x, 'w':w, 'y':y},f'dump_down_{idx}.pkl')
elif src == 'pth':
    idx = 27
    # name = 'mlp.down_proj'
    name = 'mlp.up_proj'
    d = torch.load('/agent/zixuan/forward_hooks.pth', weights_only=True)
    x = d[f'model.layers.{idx}.{name}']['input']
    w = d[f'model.layers.{idx}.{name}']['weight']
    db = torch.load('/agent/zixuan/backward_hooks.pth', weights_only=True)
    y = db[f'model.layers.{idx}.{name}']['grad_output']
    torch.save({'x':x, 'w':w, 'y':y},f'{name}.{idx}.pkl')
elif src == 'safetensor':

    from safetensors import safe_open
    # model_path = '/mntnlp/common_base_model/Meta-Llama-3-8B-Instruct'
    model_path = '/mntnlp/common_base_model/Llama-3.1-8B-Instruct'
    for i in range(1,5):
        with safe_open(f"{model_path}/model-0000{i}-of-00004.safetensors", framework="pt", device=0) as f:
            for k in f.keys():
                if 'norm' in k:
                    v = f.get_tensor(k)
                    print(k,v.abs().max().item(),v.abs().mean().item())

else:
    # file_name = 'down.pkl'
    # d = torch.load(file_name, weights_only=True)
    # x = d['x']['tensor'][0].float()
    # w = d['w']['tensor'].t().contiguous().float()
    # y = d['y']['tensor'][0].float()

    file_name = 'down_fb_26.pkl'
    d = torch.load(file_name, weights_only=True)
    x = d['x'][0].float()
    w = d['w'].t().contiguous().float()
    y = d['y'][0].float()

    x = x.abs().cpu().numpy()
    w = w.abs().cpu().numpy()
    y = y.abs().cpu().numpy()

    # M:22 N:4096 K:13440
    M, K = x.shape
    K, N = w.shape

    xp = []
    for i in range(16):
        xp.append(x[:,i*K//16:(i+1)*K//16])
    xp = np.concatenate(xp, 0)

    yp = []
    for i in range(8):
        yp.append(y[:,i*N//8:(i+1)*N//8])
    yp = np.concatenate(yp, 0)

    # xc = np.minimum(xp, 10)
    # yc = np.minimum(xp, 10)

    #
    # s = 21
    # sigma = 5
    # hs = (s-1)//2
    # kernel = np.zeros((s, s))
    # for i in range(s):
    #     for j in range(s):
    #         kernel[i,j] = np.exp((-((i-hs)**2+(j-hs)**2)/(sigma*sigma)))
    # kernel = kernel/np.sum(kernel)
    #
    # xpm, xpn = xp.shape
    # xb = np.zeros((xpm+2*hs, xpn+2*hs))
    # for i in range(s):
    #     for j in range(s):
    #         scale = kernel[i,j]
    #         xb[i:xpm+i,j:xpn+j] += scale*xp
    #
    #
    # ypm, ypn = yp.shape
    # yb = np.zeros((ypm+2*hs, ypn+2*hs))
    # for i in range(s):
    #     for j in range(s):
    #         scale = kernel[i,j]
    #         yb[i:ypm+i,j:ypn+j] += scale*yp


    # s = 11
    # vw = s
    # hs = (s-1)//2
    # sigma = 4
    # kernel = np.zeros((vw, s))
    # for i in range(vw):
    #     for j in range(s):
    #         kernel[i,j] = np.exp((-((i-hs)**2+(j-hs)**2)/(sigma*sigma)))
    # kernel = kernel/np.sum(kernel)
    # # print(kernel)
    # wpm, wpn = w.shape
    # wb = np.zeros((wpm+2*hs, wpn+2*hs))
    # for i in range(vw):
    #     for j in range(s):
    #         scale = kernel[i,j]
    #         wb[i:wpm+i,j:wpn+j] += scale*w

    xb = torch.nn.functional.max_pool2d(torch.from_numpy(xp)[None],11).numpy()[0]
    yb = torch.nn.functional.max_pool2d(torch.from_numpy(yp)[None],11).numpy()[0]
    wb = torch.nn.functional.max_pool2d(torch.from_numpy(w)[None],21).numpy()[0]


    fmt = 'png'
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(xb, cmap='gray')
    # plt.show()
    plt.axis('off')
    plt.savefig(f"figures/x.{fmt}", bbox_inches='tight',dpi=600)
    plt.close('all')


    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(yb, cmap='gray')
    # plt.show()
    plt.axis('off')
    plt.savefig(f"figures/y.{fmt}", bbox_inches='tight', dpi=600)
    plt.close('all')

    #
    fig, ax = plt.subplots(figsize=(64, 18))
    ax.imshow(wb, cmap='gray', vmax=wb.max())
    # plt.show()
    plt.axis('off')
    plt.savefig(f"figures/w.{fmt}", bbox_inches='tight', dpi=600)
    plt.close('all')

    #
    # block_size = 512
    # for i in range(K//block_size):
    #     for j in range(N//block_size):
    #         if i==0 or j == 0:
    #             fig, ax = plt.subplots(figsize=(12, 12))
    #             ax.imshow(w[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size], cmap='gray')
    #             # plt.show()
    #             plt.axis('off')
    #             plt.savefig(f"figures/w_{i}_{j}.{fmt}", bbox_inches='tight', dpi=600)
    #     plt.close('all')

