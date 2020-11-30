# -*- coding: utf-8 -*-
from __future__ import print_function

import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

compressor = 'topk'
dnn='resnet20'
bs=32
lr=0.1
LOGHOME='./gradients/allreduce-comp-%s-gwarmup-convergence-thres-512000kbytes/%s-n4-bs%d-lr%.4f-ns1-ds0.001' % (compressor, dnn, bs, lr);
iterations=range(0, 1000) 

#fig, ax= plt.subplots(1, 1,figsize=(5.5,4.0))
fig, ax= plt.subplots()

ks = np.arange(0.00, 1.01, 0.01)
#print(ks)
for i in iterations:
    if i % 100 != 0 or i == 0:
        continue
    fn = '%s/r0_gradients_iter_%d.npy' % (LOGHOME, i)
    grad = np.load(fn).flatten()
    d = len(grad)
    abs_grad = np.abs(grad)
    sorted_grad = np.sort(abs_grad)[::-1]
    #ax.plot(np.log(np.arange(1, d+1)),np.log(sorted_grad), label='iter-%d'%i)
    #ax.plot(np.log10(np.arange(1, d+1)),np.log10(sorted_grad), label='iter-%d'%i)
    ax.plot(np.arange(1, d+1),sorted_grad, label='iter-%d'%i)

    #xnorm = np.linalg.norm(sorted_grad)
    #reals = []
    #ours = []
    #previous = []
    #for r in ks: 
    #    k = int(r*d)
    #    topk = sorted_grad[0:k]
    #    topknorm = np.linalg.norm(topk)
    #    #if k <= d:
    #    #    print(topknorm, xnorm)
    #    realbound = (xnorm**2 - 2*topknorm*xnorm + topknorm**2)/xnorm**2
    #    reals.append(realbound)
    #    ourbound = (1-k*1.0/d)**2
    #    ours.append(ourbound)
    #    previousbound = (1-k*1.0/d)
    #    previous.append(previousbound)
    #x = 1- np.array(ks)
    #ax.plot(x, reals, label='alpha^2, iter=%d'%i)
#ax.plot(x, previous, label='1-k/d')

#ax.plot(np.log10(np.arange(1, d+1)),np.log10(sorted_grad), label='iter-%d'%i)
ax.legend()
#ax.set_xlabel('1-k/d')
ax.set_xlabel('i')
ax.set_ylabel('x(i)')
#ax.set_xlabel('log(i)')
#ax.set_ylabel('log(|x|(pi(i)))')
#ax.set_xlabel('log10(i)')
#ax.set_ylabel('log10(|x|(pi(i)))')
#ax.set_xlabel('i')
#ax.set_ylabel('|x|(pi(i))')
plt.show()
