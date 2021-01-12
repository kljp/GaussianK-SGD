# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import time
import utils
from compression import compressors

compstr = 'topk'
#compstr = 'gaussian'
#compstr = 'dgcsampling'
compressor = compressors[compstr]

def bench_single_topk(n, ratio, num_iters):
    tensor = torch.rand(n).float().cuda()
    k = int(ratio * n)
    values, indexes = torch.topk(torch.abs(tensor.data), k=k, sorted=False)
    torch.cuda.synchronize()
    stime = time.time()
    for i in range(num_iters):
        values, indexes = torch.topk(torch.abs(tensor.data), k=k, sorted=False)
    torch.cuda.synchronize()
    etime = time.time()
    time_used = (etime-stime)/num_iters
    return time_used


def bench_compressor(n, ratio, num_iters, compressor):
    tensor = torch.rand(n).float().cuda()
    #tensor = torch.normal(0.0, 0.5, size=n).float().cuda()
    name = 'name:%d%f' % (n, ratio)
    k = int(ratio * n)
    sigma_scale = 3
    if compstr == 'sigmathres':
        sigma_scale = utils.get_approximate_sigma_scale(ratio)
    tr, indexes, values = compressor.compress(tensor, name, sigma_scale=sigma_scale, ratio=ratio)
    torch.cuda.synchronize()
    stime = time.time()
    for i in range(num_iters):
        tr, indexes, values = compressor.compress(tensor, name, sigma_scale=sigma_scale, ratio=ratio)
    torch.cuda.synchronize()
    etime = time.time()
    time_used = (etime-stime)/num_iters
    compressor.clear()
    return time_used


def bench():
    #ns = range(2**10, 2**20, 1024) 
    #ns = ns+range(2**20, 2**29, 2**20) 
    ns = range(2**20, 2**29, 2**20) 
    ratio = 0.001
    for n in ns:
        num_iters = 50
        if n > 2**19:
            num_iters = 10
        #t = bench_single_topk(n, ratio, num_iters)
        t = bench_compressor(n, ratio, num_iters, compressor)
        print('%d,%f'%(n,t))



if __name__ == '__main__':
    bench()
