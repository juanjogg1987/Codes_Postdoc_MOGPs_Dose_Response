import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from importlib import reload
import TransferLearning_Kernels
reload(TransferLearning_Kernels)
from TransferLearning_Kernels import TL_Kernel_var, Kernel_CrossDomains, TLRelatedness,NNetwork_kern,Kernel_Sig2Constrained

import torch

def sigmoid_4_param(x, x0, L, k, d):
    """ Comparing with Dennis Wang's sigmoid:
    x0 -  p - position, correlation with IC50 or EC50
    L = 1 in Dennis Wang's sigmoid, protect from devision by zero if x is too small
    k = -1/s (s -shape parameter)
    d - determines the vertical position of the sigmoid - shift on y axis - better fitting then Dennis Wang's sigmoid

    """
    return ( 1/ (L + np.exp(-k*(x-x0))) + d)

x = torch.linspace(-3,3,100)
y_sig = sigmoid_4_param(x,-1,1,-10.5,0)
y_sig2 = sigmoid_4_param(x,-1,1,-10.5,0)

plt.close('all')

plt.figure(1)
plt.plot(x,y_sig)

def covar(x1,length=1):
    x1 = x1.detach().numpy()
    x1 =x1
    x2 = x1.copy()
    x1_m,x2_m = np.meshgrid(x1,x2)
    sig = 0.005  #Play with this value to see how the covariance increases or decreases its values
    kxx = sig*np.exp(-0.5/(length**2)*(x1_m-x2_m)**2)
    return kxx

mycovar =Kernel_Sig2Constrained()

#cov = covar(x)
#cov2 = covar(x,0.1)

mycovar.length = 0.5
cov = mycovar(x).evaluate()

cov2 = covar(x,0.5)

plt.figure(2,figsize=(10, 5))
plt.figure(3,figsize=(10, 5))
Nsamples = 100
for i in range(Nsamples):
    # Uncomment the first line below to use only diag of covariance
    # or Uncomment second line for use complete covariance

    # ysample = np.random.multivariate_normal(y.flatten(), np.diag(np.diag(cov)))
    ysample = np.random.multivariate_normal(y_sig.flatten(), cov.detach().numpy())
    ysample2 = np.random.multivariate_normal(y_sig2.flatten(), cov2)

    plt.figure(2)
    plt.plot(x, ysample2, '-', alpha=0.7)
    plt.figure(3)
    plt.plot(x, ysample, '-', alpha=0.7)

# plt.ylim([1.5,2.2])
# plt.ylim([-2.2,2.2])
#plt.xlabel(r'$x$', fontsize=17)
#plt.ylabel(r'$y \sim \mathcal{N}(f(x),\Sigma)$', fontsize=17)