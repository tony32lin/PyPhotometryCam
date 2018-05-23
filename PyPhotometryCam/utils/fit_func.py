import numpy as np 

def gaussian_with_baseline(x,norm,mu,sigma,bs):
    return norm*np.exp(-1*(x-mu)**2/sigma**2)+bs 

def linfunc(x,offset):
    return x+offset

def ze_lin_func(x,tau,offset):
    return tau*(x-1)+offset
