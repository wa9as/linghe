
import torch
import torch.nn.functional as F
import triton

import time
import os
import random


def benchmark_func(fn, *args, n_warmup=100, n_repeat=1000, ref_flops=None, ref_bytes=None, ref_time=None, name='', **kwargs):
    func_name = fn.__name__

    for i in range(n_warmup):
        fn(*args,**kwargs)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    
    ts = time.time()
    for i in range(n_repeat):
        start_events[i].record()
        fn(*args,**kwargs)
        end_events[i].record()
    
    torch.cuda.synchronize() 
    
    # times = sum([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
    # average_event_time = times * 1000 / n_repeat

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times = sorted(times)
    clip = max(1,n_repeat//100)
    times = sum(times[clip:-clip])
    
    average_event_time = times * 1000 / (n_repeat - 2*clip)
    
    fs = ''
    if ref_flops is not None:
        flops = ref_flops/1e12/(average_event_time/1e6)
        fs = f'FLOPS:{flops:.2f}T'
    bs = ''
    if ref_bytes is not None:
        bs = f'bandwidth:{ref_bytes/average_event_time/1e3:.3f}G/S'
    ss = ''
    if ref_time is not None:
        ss = f'speedup:{ref_time/average_event_time:.3f}'

    print(f'{func_name:<30} {name} time:{average_event_time:.1f} us {fs} {bs} {ss}')
    return average_event_time

