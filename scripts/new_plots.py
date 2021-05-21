# -*- coding: utf-8 -*-
from __future__ import print_function

import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import invweibull
from sympy import harmonic
from gmpy2 import mpz, mpq, fac

compressor = 'topk'
dnn='resnet20'
bs=32
lr=0.1
LOGHOME='./gradients/allreduce-comp-%s-gwarmup-convergence-thres-512000kbytes/%s-n4-bs%d-lr%.4f-ns1-ds0.001' % (compressor, dnn, bs, lr);

def raw_plots():
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


def _harmonic4(a, b, m):
    if b-a == 1:
        return mpq(1,a)**m
    middle = (a+b)//2
    return _harmonic4(a,middle,m) + _harmonic4(middle,b,m)

def harmonic4(n, m):
    return _harmonic4(1,n+1,m)

def harmonic_number(n, m):
    """
    sum_{k=1}^n 1/k**m
    """
    #return harmonic(n, m).n(10)
    return harmonic4(n,m)

def compute_alpha(k, d, m, c1, c0):
    numerator = k * c0**2 + 2*c0*c1*harmonic_number(k, m) + c1**2 * harmonic_number(k, 2*m)
    dominator = d * c0**2 + 2*c0*c1*harmonic_number(d, m) + c1**2 * harmonic_number(d, 2*m)
    return 1. - numerator/dominator

def generate_random(d, t='gaussian'):
    if t == 'gaussian':
        data = np.random.standard_normal(size=d)
    elif t == 'gumbel':
        data = np.random.gumbel(size=d)
    elif t == 'frechet':
        c = 1.2
        mean, var, skew, kurt = invweibull.stats(c, moments='mvsk')
        data = np.linspace(invweibull.ppf(0.01, c), invweibull.ppf(0.99, c), d)
    elif t == 'weibull':
        data = np.random.weibull(1, size=d)
    else:
        raise 'Unsupported distribution: %s' % t
    return data


def func_powerlaw(x, m, b, a):
    return b + x**m * a 
    #return b * (x**m * a)

def fit_powerlaw_with_distribution():
    fig, ax= plt.subplots()
    d = 100000
    ratio = 0.01
    k = int(ratio * d)
    distribution = 'gaussian'
    #distribution = 'gumbel'
    #distribution = 'weibull'
    #distribution = 'frechet'
    grad = generate_random(d, t=distribution)
    abs_grad = np.abs(grad)
    sorted_grad = np.sort(abs_grad)[::-1]
    x = np.arange(1, d+1)
    y = np.array(sorted_grad) 
    ax.plot(np.arange(1, d+1),sorted_grad, label=distribution)

    #sol1, _ = curve_fit(func_powerlaw, x, y, p0 = np.asarray([1,10**6,0]))
    sol1, _ = curve_fit(func_powerlaw, x, y, maxfev = 8000)
    m, c1, c0 = sol1
    ax.plot(x, func_powerlaw(x, *sol1), '--', label='%s-fit'%distribution)
    print('[m, b, a]: ', sol1) 

    plt.legend()
    plt.show()

def fit_powerlaw():
    fig, ax= plt.subplots()
    iterations=range(0, 1000) 
    ratio = 0.001
    for i in iterations:
        if i % 200 != 0 or i == 0:
            continue
        fn = '%s/r0_gradients_iter_%d.npy' % (LOGHOME, i)
        grad = np.load(fn).flatten()
        d = len(grad)
        k = int(ratio * d)
        #distribution = 'gaussian'
        #grad = generate_random(d, t=distribution)
        abs_grad = np.abs(grad)
        sorted_grad = np.sort(abs_grad)[::-1]
        #sorted_grad = np.power(sorted_grad, 2)
        #sorted_grad /= np.linalg.norm(sorted_grad)
        x = np.arange(1, d+1)
        y = np.array(sorted_grad) 
        sol1, _ = curve_fit(func_powerlaw, x, y, p0 = np.asarray([1,10**5,0]))
        m, c1, c0 = sol1
        #ax.plot(np.log10(np.arange(1, d+1)),np.log10(sorted_grad), label='iter-%d'%i)
        ax.plot(np.arange(1, d+1),sorted_grad, label='iter-%d'%i)
        ax.plot(x, func_powerlaw(x, *sol1), '--', label='iter-%d-fit'%i)
        print('iter-%d, [m, b, a]: ' % i, sol1) #, 'ratio: ', ratio(k, d, *sol1))
        topk_alpha = compute_alpha(k, d, -m, c1, c0)
        randk_alpha = 1-ratio
        retangle_alpha = (1-ratio)**2
        print('iter-%d topk_alpha: ' % i, topk_alpha, ', randk_alpha: ', randk_alpha, ', retangle_alpha: ', retangle_alpha)
    plt.xlabel('i')
    plt.ylabel(r'$\pi(i)$')
    plt.legend()
    plt.show()

def compared_bounds():
    d = int(25e6)
    ratio = 0.001
    k = int(ratio*d)
    bound1 = 1-ratio
    bound2 = (1-ratio)**2
    #bound3 = 


if __name__ == '__main__':
    #raw_plots()
    fit_powerlaw()
    #fit_powerlaw_with_distribution()
