#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 12:25:02 2021

@author: marcos
"""

import numpy as np
from scipy.special import gamma, kv
import matplotlib.pyplot as plt

def mean_func(x):
    return np.zeros(len(x))
    
def get_r(x1, x2):
    return np.subtract.outer(x1, x2)

def matern_kernel(r, l = 1.0, v = 1.5):
    r = np.abs(r)
    r[r == 0] = 1e-8
    part1 = 2 ** (1 - v) / gamma(v)
    #part2 = (np.sqrt(2 * v) * r / l) ** v
    #part3 = kv(v, np.sqrt(2 * v) * r / l)
    part2 = ( r / l) ** v
    part3 = kv(v, r / l)       
    return part1 * part2 * part3

def sample_prior(
        x, mean_func, cov_func, cov_args = {}, 
        random_seed = -1, n_samples = 5):
    x_mean = mean_func(x)
    x_cov = cov_func(get_r(x, x), **cov_args)
    random_seed = int(random_seed)
    if random_seed < 0:
        prng = np.random
    else:
        prng = np.random.RandomState(random_seed)
    out = prng.multivariate_normal(x_mean, x_cov, n_samples)
    return out,x_cov

def spectrum(s, l = 1.0, v = 0.5):
    D = 1
    part1 = 2**D*np.pi**(D/2)*gamma( v + (D/2))*(2 * v) ** v
    part2 = gamma(v) * l ** (2 * v)
    part3 = ((2* v)/ l + 4 * np.pi ** 2 * s ** 2) ** -(v + D/2)
    return (part1/part2)*part3


eles = [0.05, 0.1, 0.2]
plt.figure(figsize = (10, 3))
x = np.linspace(0, 1, 200)
for l_i, l in enumerate(eles):
    y,x_cov = sample_prior(x, mean_func, matern_kernel, 
                     {"l": l}, n_samples = 50000)
    plt.subplot(131 + l_i)
    pvar = np.sqrt(np.diag(np.dot(x_cov,x_cov.T)))
    plt.plot(x,pvar,'r-')
    for i in range(50):
        plt.plot(x, y[i, :], "k-", alpha = 0.3)        
    plt.title("$\\ell$ = " + str(l))
    
plt.figure(figsize = (10,3))
r = np.linspace(0, 1, 200)
for l_i, l in enumerate(eles):
    rr = matern_kernel(r, l=l)
    plt.subplot(131 + l_i)
    plt.title("$\\ell$ = " + str(l))
    plt.plot(r,rr)
    
plt.figure(figsize = (10,3))
s = np.linspace(0, 1, 200)
for l_i, l in enumerate(eles):
    sp = spectrum(s, l=l)
    plt.subplot(131 + l_i)
    plt.title("$\\ell$ = " + str(l))
    plt.semilogy(s,sp)