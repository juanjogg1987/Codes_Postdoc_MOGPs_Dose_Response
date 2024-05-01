from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim
import gpytorch

#import matplotlib as mpl
#mpl.use("TkAgg")  # or can use 'TkAgg', whatever you have/prefer
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from importlib import reload
import TransferLearning_Kernels
reload(TransferLearning_Kernels)
from TransferLearning_Kernels import TL_Kernel_var, Kernel_CrossDomains, TLRelatedness,NNetwork_kern

from numpy import linalg as la
import numpy as np
import os

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
def isPD_torch(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = torch.linalg.cholesky(B)
        return True
    except torch.linalg.LinAlgError:
        return False
def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    if isPD(A):
        return A

    print("Correcting Matrix to be PSD!!")
    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

class LogMarginalLikelihood(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,mu,L,y):
        y = y.T.reshape(-1, 1)
        assert mu.shape == y.shape
        alpha = torch.linalg.solve(L.t(), torch.linalg.solve(L, y-mu))
        N = y.shape[0]
        return -0.5*torch.matmul(y.t()-mu.t(),alpha)-torch.diag(L).log().sum()-N*0.5*torch.log(torch.tensor([2*torch.pi]))
class TLMOGaussianProcess(nn.Module):
    'This model expects the data from source and target domains with:'
    'idxS as a list of labels with integer values between 0 to NDomains-2'
    'by deafult the idxT for the target domain will be labeled as NDomains-1'
    'i.e., if NDomains = 3, the idxS is a list with values between 0 and 1, by default idxT has all values equal to 2'
    'The model expects the input and output of the target domain: xT (shape: [N,P]), yT (shape: [N,D])'
    'with P as the number of input features and D as the number of outputs'
    'The input and output of the source domains: xS, yS with a list idxS with their labels as per their domains'
    'NOTE: The model would expect the data to be sorted according to the idxS'
    'For instance, idxS = [0,0,1,2,2,2] implies NDomains = 4 with N = 2 in first domain,'
    'N = 1 in second domain, and N = 3 in third domain, if N = 2 in fourth (target) domain, then idxT = [3,3]'
    'The model also would expect a list of the Drug Concentrations used, it should have the same size as D output'
    def __init__(self,xT,yT,xS,yS,idxS,DrugC_T,DrugC_S,NDomains):
        super().__init__()
        self.Douts_T = yT.shape[1]
        self.Douts_S = yS.shape[1]
        #self.DrugC_T = torch.Tensor(DrugC_T)[:,None] #We treat the Drug Concentrations as a float tensor vector [N,1]
        #self.DrugC_S = torch.Tensor(DrugC_S)[:, None]  # We treat the Drug Concentrations as a float tensor vector [N,1]
        self.NDomains = NDomains

        assert DrugC_T.shape == yT.shape  # DrugC_T should same shape as yT
        assert DrugC_S.shape == yS.shape  # DrugC_S should same shape as yS
        self.Pfeat = xT.shape[1]
        #torch.kron(mat1, mat2.reshape(-1)).reshape(-1, 429)
        #self.xT = torch.kron(torch.ones(self.Douts, 1),xT)  #This is to replicate xT as per the D number of outputs
        self.xT = torch.kron(torch.ones(self.Douts_T, 1), xT.reshape(-1)).reshape(-1, self.Pfeat)  # This is to replicate xT as per the D number of outputs
        #self.xS = torch.kron(torch.ones(self.Douts, 1), xS)  # This is to replicate xS as per the D number of outputs
        self.xS = torch.kron(torch.ones(self.Douts_S, 1),xS.reshape(-1)).reshape(-1, self.Pfeat)  #This is to replicate xS as per the D number of outputs
        self.yT = yT.T.reshape(-1, 1)  # This is to vectorise by staking the columns (vect(yT))
        self.yS = yS.T.reshape(-1,1) #This is to vectorise by staking the columns (vect(yS))

        self.mS = torch.zeros_like(self.yS)
        self.mT = torch.zeros_like(self.yT)
        self.all_m = torch.cat([self.mS, self.mT])

        #self.DrugC_xT = torch.kron(self.DrugC_T,torch.ones(xT.shape[0], 1)) #Rep. Concentr. similar to coreginalisation
        #self.DrugC_xS = torch.kron(self.DrugC_S,torch.ones(xS.shape[0], 1)) #Rep. Concentr. similar to coreginalisation

        self.DrugC_xT = DrugC_T.T.reshape(-1, 1)
        self.DrugC_xS = DrugC_S.T.reshape(-1, 1)

        self.idxS = idxS * self.Douts_S #Replicate the Source domain index as per the number of outputs
        self.idxT = [NDomains - 1] * xT.shape[0] * self.Douts_T #Replicate the target domain index as per the number of outputs
        self.all_y = torch.cat([self.yS, self.yT])

        self.TLKern1 = Kernel_CrossDomains(NDomains=NDomains)
        self.TLKern2 = Kernel_CrossDomains(NDomains=NDomains)
        self.TLKern3 = Kernel_CrossDomains(NDomains=NDomains)
        self.TLKern4 = Kernel_CrossDomains(NDomains=NDomains)

        self.TLCovariance = [self.TLKern1, self.TLKern2, self.TLKern3, self.TLKern4]
        #self.TLCovariance = [Kernel_CrossDomains(NDomains=NDomains),Kernel_CrossDomains(NDomains=NDomains),Kernel_CrossDomains(NDomains=NDomains)] #gpytorch.kernels.RBFKernel()

        self.LambdaDiDj = TLRelatedness(NDomains=NDomains)

        self.LMCkern1 = NNetwork_kern() #gpytorch.kernels.RBFKernel() #gpytorch.kernels.MaternKernel(1.5)
        self.LMCkern2 = NNetwork_kern() #gpytorch.kernels.RBFKernel() #gpytorch.kernels.MaternKernel(1.5)
        self.LMCkern3 = NNetwork_kern() #gpytorch.kernels.RBFKernel() #gpytorch.kernels.MaternKernel(2.5)
        self.LMCkern4 = NNetwork_kern()
        self.CoregCovariance = [self.LMCkern1, self.LMCkern2, self.LMCkern3, self.LMCkern4]
        #self.CoregCovariance = [gpytorch.kernels.MaternKernel(1.5),gpytorch.kernels.MaternKernel(2.5),gpytorch.kernels.MaternKernel(2.5)]

        self.Train_mode = True
        #self.lik_std_noise = torch.nn.Parameter(1.0*torch.ones(NDomains)) #0.3*torch.rand(NDomains)+0.01
        self.coef1 = torch.nn.Parameter(1.0 * torch.randn(NDomains))
        self.coef2 = torch.nn.Parameter(1.0 * torch.randn(NDomains))
        #self.lik_std_noise = 0.1 * torch.ones(NDomains)
        self.mu_star = torch.zeros_like(self.yT) #mu has the shape of the new replicated along the outputs yT
        self.L = torch.eye(self.yT.shape[0])
        self.all_L = torch.eye(self.all_y.shape[0])
        "TODO: I might need the self.LSS also here in order to be able to predict without optimising"

    def noise_func(self,DrugC_x,idx):
        std_lik = self.coef1[idx]*torch.log2(-np.log2(0.1)+DrugC_x[:,0]+1)+self.coef2[idx]
        return std_lik

    def forward(self,xT, DrugC_new = None,noiseless = True):
        if self.Train_mode:
            #xT = torch.kron(torch.ones(self.Douts, 1), xT)
            xT = torch.kron(torch.ones(self.Douts_T, 1), xT.reshape(-1)).reshape(-1, self.Pfeat)
            assert (xT == self.xT).sum() == (xT.shape[0]*xT.shape[1]) #This is just to check if the xT to init the model is the same
            assert xT.shape[1] == self.xT.shape[1]  #Thiis is to check if the xT to init the model has same Pfeatures

            "Below the TLCovariance and LambdaDiDj have to coincide in the same indexes idx1 and idx2"
            # Here we compute the Covariance matrices between source-target, source-source and target domains
            KTS = self.LambdaDiDj(xT,self.xS,idx1=self.idxT,idx2=self.idxS).evaluate()*(self.CoregCovariance[0](self.DrugC_xT,self.DrugC_xS).evaluate()*self.TLCovariance[0](xT,self.xS,idx1=self.idxT,idx2=self.idxS).evaluate() + \
                                                                                        self.CoregCovariance[1](self.DrugC_xT,self.DrugC_xS).evaluate()*self.TLCovariance[1](xT, self.xS, idx1=self.idxT, idx2=self.idxS).evaluate()+\
                                                                                        self.CoregCovariance[2](self.DrugC_xT,self.DrugC_xS).evaluate()*self.TLCovariance[2](xT, self.xS, idx1=self.idxT, idx2=self.idxS).evaluate()+\
                                                                                        self.CoregCovariance[3](self.DrugC_xT,self.DrugC_xS).evaluate()*self.TLCovariance[3](xT, self.xS, idx1=self.idxT, idx2=self.idxS).evaluate())

            KSS = self.LambdaDiDj(self.xS,idx1=self.idxS).evaluate()*(self.CoregCovariance[0](self.DrugC_xS,self.DrugC_xS).evaluate()*self.TLCovariance[0](self.xS,idx1=self.idxS).evaluate()+ \
                                                                      self.CoregCovariance[1](self.DrugC_xS,self.DrugC_xS).evaluate()*self.TLCovariance[1](self.xS, idx1=self.idxS).evaluate()+\
                                                                      self.CoregCovariance[2](self.DrugC_xS,self.DrugC_xS).evaluate()*self.TLCovariance[2](self.xS, idx1=self.idxS).evaluate()+\
                                                                      self.CoregCovariance[3](self.DrugC_xS,self.DrugC_xS).evaluate()*self.TLCovariance[3](self.xS, idx1=self.idxS).evaluate())

            KTT = self.LambdaDiDj(self.xT,idx1=self.idxT).evaluate()*(self.CoregCovariance[0](self.DrugC_xT,self.DrugC_xT).evaluate()*self.TLCovariance[0](self.xT,idx1=self.idxT).evaluate()+ \
                                                                      self.CoregCovariance[1](self.DrugC_xT,self.DrugC_xT).evaluate() * self.TLCovariance[1](self.xT, idx1=self.idxT).evaluate()+\
                                                                      self.CoregCovariance[2](self.DrugC_xT,self.DrugC_xT).evaluate() * self.TLCovariance[2](self.xT, idx1=self.idxT).evaluate()+\
                                                                      self.CoregCovariance[3](self.DrugC_xT,self.DrugC_xT).evaluate() * self.TLCovariance[3](self.xT, idx1=self.idxT).evaluate())

            #print(f"LambdaDiDj xT xS: {self.LambdaDiDj(xT,self.xS,idx1=self.idxT,idx2=self.idxS).evaluate()}")
            #print(f"LambdaDiDj xS xS: {self.LambdaDiDj(self.xS,idx1=self.idxS).evaluate()}")
            #print(f"LambdaDiDj xT xT: {self.LambdaDiDj(self.xT,idx1=self.idxT).evaluate()}")
            #print(f"KTS: {KTS}")
            #print(f"KSS: {KSS}")
            #print(f"KTT: {KTT}")

            #print(f"KTS CoregCov[0]: {self.CoregCovariance[0](self.DrugC_xT, self.DrugC_xS).evaluate()}")
            #print(f"KTS TLCov: {self.TLCovariance[0](xT, self.xS, idx1=self.idxT, idx2=self.idxS).evaluate()}")
            #print(f"KTS CoregCov[0]*TLCov: {self.CoregCovariance[0](self.DrugC_xT,self.DrugC_xS).evaluate()*self.TLCovariance[0](xT,self.xS,idx1=self.idxT,idx2=self.idxS).evaluate()}")

            # Here we include the respective noise terms associated to each domain
            lik_std_noise_xS = self.noise_func(self.DrugC_xS, self.idxS)
            lik_std_noise_xS = lik_std_noise_xS + torch.sign(lik_std_noise_xS)*1.0e-4
            lik_std_noise_xT = self.noise_func(self.DrugC_xT, self.idxT)
            lik_std_noise_xT = lik_std_noise_xT + torch.sign(lik_std_noise_xT)*1.0e-4
            CSS = KSS + torch.diag(lik_std_noise_xS.pow(2))
            CTT = KTT + torch.diag(lik_std_noise_xT.pow(2))

            # The code below aim to correct for numerical instabilities when CSS becomes Non-PSD
            if not isPD_torch(CSS):
                CSS_aux = CSS.clone()
                with torch.no_grad():
                    CSS_aux = torch.from_numpy(nearestPD(CSS_aux.numpy()))
                CSS = 0.0*CSS + CSS_aux  #This operation aims to keep the gradients working over lik_std_noise

            self.LSS = torch.linalg.cholesky(CSS)
            # alphaSS1 = torch.linalg.solve(self.LSS, self.yS)
            alphaSS1 = torch.linalg.solve(self.LSS, self.yS-self.mS)
            alphaSS = torch.linalg.solve(self.LSS.t(), alphaSS1)

            # Compute the mean of the conditional distribution p(yT|XT,XS,yS)
            self.mu_star = torch.matmul(KTS,alphaSS) + self.mT
            # Compute the Covariance of the conditional distribution p(yT|XT,XS,yS)
            vTT = torch.linalg.solve(self.LSS, KTS.t())
            C_star = CTT - torch.matmul(vTT.t(),vTT) #+ 1e-4*torch.eye(xT.shape[0])  #jitter?
            # The code below aim to correct for numerical instabilities when CSS becomes Non-PSD
            if not isPD_torch(C_star):
                C_star_aux = C_star.clone()
                with torch.no_grad():
                    C_star_aux = torch.from_numpy(nearestPD(C_star_aux.numpy()))
                C_star = 0.0 * C_star + C_star_aux  # This operation aims to keep the gradients working over lik_std_noise
            self.L = torch.linalg.cholesky(C_star)

            return self.mu_star, self.L  # here we return the mean and covariance
        else:
            "We replicate the target domain index as per the number of drug concentration we want to test"
            "notice that it is not limited to D number of outputs, but actually the number of concentrations"
            # Here we compute the full covariance of xS and xT together
            xST = torch.cat([self.xS, self.xT])
            idxST = self.idxS + self.idxT
            self.DrugC_xSxT = torch.cat([self.DrugC_xS, self.DrugC_xT])
            all_K_xST = self.LambdaDiDj(xST, idx1=idxST).evaluate()*(self.CoregCovariance[0](self.DrugC_xSxT).evaluate() * self.TLCovariance[0](xST, idx1=idxST).evaluate()+ \
                                                                     self.CoregCovariance[1](self.DrugC_xSxT).evaluate() * self.TLCovariance[1](xST, idx1=idxST).evaluate()+\
                                                                     self.CoregCovariance[2](self.DrugC_xSxT).evaluate() * self.TLCovariance[2](xST, idx1=idxST).evaluate()+\
                                                                     self.CoregCovariance[3](self.DrugC_xSxT).evaluate() * self.TLCovariance[3](xST, idx1=idxST).evaluate())

            lik_std_noise_xST = self.noise_func(self.DrugC_xSxT, idxST)
            lik_std_noise_xST = lik_std_noise_xST + torch.sign(lik_std_noise_xST)*1.0e-4
            all_K_xST_noise = all_K_xST + torch.diag(lik_std_noise_xST.pow(2))  # + 0.1*torch.eye(xST.shape[0]) #Jitter?

            if not isPD_torch(all_K_xST_noise):
                with torch.no_grad():
                    all_K_xST_noise = torch.from_numpy(nearestPD(all_K_xST_noise.numpy()))
            self.all_L = torch.linalg.cholesky(all_K_xST_noise)  # + 1e-4*torch.eye(xST.shape[0])

            # Here we receive a list of possible drug concentrations to predict
            if DrugC_new is None:
                #TODO
                DrugC_new = self.DrugC_xT
                assert 1==0
                #DrugC_new = torch.Tensor(DrugC_new)[:, None]
                #DrugC_xT = torch.kron(DrugC_new, torch.ones(xT.shape[0], 1))

            else:
                assert DrugC_new.shape[0] == xT.shape[0]
                DrugC_xT = DrugC_new.T.reshape(-1, 1)
            #Below we replicate the target domain index as per the number of drug concentration we want to test
            NewDouts = DrugC_new.shape[1]

            idxT = [self.NDomains - 1] * xT.shape[0] * NewDouts

            "Be careful with operation using xT.shape, from here it changes the original shape"
            #xT = torch.kron(torch.ones(NewDouts, 1), xT)
            xT = torch.kron(torch.ones(NewDouts, 1), xT.reshape(-1)).reshape(-1, self.Pfeat)
            alpha1 = torch.linalg.solve(self.all_L, self.all_y-self.all_m)
            alpha = torch.linalg.solve(self.all_L.t(), alpha1)
            KTT_xnew_xnew = self.LambdaDiDj(xT, idx1=idxT).evaluate()*(self.CoregCovariance[0](DrugC_xT).evaluate() * self.TLCovariance[0](xT, idx1=idxT).evaluate()+ \
                                                                       self.CoregCovariance[1](DrugC_xT).evaluate() * self.TLCovariance[1](xT,idx1=idxT).evaluate()+\
                                                                       self.CoregCovariance[2](DrugC_xT).evaluate() * self.TLCovariance[2](xT,idx1=idxT).evaluate()+\
                                                                       self.CoregCovariance[3](DrugC_xT).evaluate() * self.TLCovariance[3](xT,idx1=idxT).evaluate())
            xST = torch.cat([self.xS, self.xT])
            idxST = self.idxS+self.idxT

            # Rep. Concentr. similar to coreginalisation
            K_xnew_xST = self.LambdaDiDj(xT,xST, idx1=idxT,idx2=idxST).evaluate()*(self.CoregCovariance[0](DrugC_xT,self.DrugC_xSxT).evaluate() * self.TLCovariance[0](xT,xST, idx1=idxT,idx2=idxST).evaluate()+ \
                                                                                   self.CoregCovariance[1](DrugC_xT,self.DrugC_xSxT).evaluate() * self.TLCovariance[1](xT, xST, idx1=idxT, idx2=idxST).evaluate()+\
                                                                                   self.CoregCovariance[2](DrugC_xT,self.DrugC_xSxT).evaluate() * self.TLCovariance[2](xT, xST, idx1=idxT, idx2=idxST).evaluate()+\
                                                                                   self.CoregCovariance[3](DrugC_xT,self.DrugC_xSxT).evaluate() * self.TLCovariance[3](xT, xST, idx1=idxT, idx2=idxST).evaluate())

            f_mu = torch.matmul(K_xnew_xST, alpha)
            mT_pred = torch.zeros_like(f_mu)
            f_mu = torch.matmul(K_xnew_xST, alpha) + mT_pred
            v = torch.linalg.solve(self.all_L, K_xnew_xST.t())

            if noiseless:
                f_Cov = KTT_xnew_xnew - torch.matmul(v.t(),v) #+ 1e-2*torch.eye(xT.shape[0])  #I had to add this Jitter
                f_Cov = torch.from_numpy(nearestPD(f_Cov.numpy()))
            else:
                lik_std_noise_xT = self.noise_func(DrugC_xT, idxT)
                lik_std_noise_xT = lik_std_noise_xT + torch.sign(lik_std_noise_xT)*1.0e-4
                f_Cov = KTT_xnew_xnew - torch.matmul(v.t(),v) + torch.diag(lik_std_noise_xT.pow(2)) + 1e-5*torch.eye(xT.shape[0])
                f_Cov = torch.from_numpy(nearestPD(f_Cov.numpy()))
            return f_mu, f_Cov

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Nseed = 2
torch.manual_seed(Nseed)
import math
import pandas as pd
import random
import warnings
warnings.filterwarnings("ignore")
##from sklearn.preprocessing import MinMaxScaler
random.seed(Nseed)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Here we preprocess and prepare our data"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
_FOLDER = "/home/juanjo/Work_Postdoc/my_codes_postdoc/Dataset_5Cancers/GDSC2_dataset_ForSarcoma/"
#_FOLDER = "/rds/general/user/jgiraldo/home/Dataset_5Cancers/GDSC2_EGFR_PI3K_MAPK_Top5cancers/" #HPC path
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def sigmoid_4_param(x, x0, L, k, d):
    """ Comparing with Dennis Wang's sigmoid:
    x0 -  p - position, correlation with IC50 or EC50
    L = 1 in Dennis Wang's sigmoid, protect from devision by zero if x is too small
    k = -1/s (s -shape parameter)
    d - determines the vertical position of the sigmoid - shift on y axis - better fitting then Dennis Wang's sigmoid

    """
    return ( 1/ (L + np.exp(-k*(x-x0))) + d)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import getopt
import sys

warnings.filterwarnings("ignore")
#os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

class commandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'n:r:p:w:s:t:i:d:')
        # opts = dict(opts)
        # print(opts)
        self.N_iter = 2    #number of iterations
        self.which_seed = 20 #108 #103 #29 #35 #123  #change seed to initialise the hyper-parameters
        self.weight = 1.0  #use weights 0.3, 0.5, 1.0 and 2.0
        self.bash = "None"
        self.sel_cancer_Source = 3
        self.sel_cancer_Target = 5
        self.idx_CID_Target = 0  #This is just an integer from 0 to max number of CosmicIDs in Target cancer.
        self.which_drug = 1022   #1062(22) 1057(19) 2096(17) #This is the drug we will select as test for the target domain.

        for op, arg in opts:
            # print(op,arg)
            if op == '-n':
                self.N_iter = arg
            if op == '-r':  # (r)and seed
                self.which_seed = arg
            if op == '-p':  # (p)ython bash
                self.bash = arg
            if op == '-w':
                self.weight = arg
            if op == '-s':
                self.sel_cancer_Source = arg
            if op == '-t':
                self.sel_cancer_Target = arg
            if op == '-i':
                self.idx_CID_Target = arg
            if op == '-d':
                self.which_drug = arg


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
config = commandLine()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
dict_cancers={0:'BRCA_GDSC2_drugresponse_source_uM.csv',1:'COREAD_GDSC2_drugresponse_source_uM.csv',
              2:'LUAD_GDSC2_drugresponse_source_uM.csv',3:'SKCM_GDSC2_drugresponse_source_uM.csv',
              4:'SCLC_GDSC2_drugresponse_source_uM.csv',5:'MB_GDSC2_drugresponse_fullcurve_uM.csv'}

indx_cancer_train = np.array([0,1,2,3,4])
#indx_cancer_train = np.array([int(config.sel_cancer_Source)])

#name_file_cancer = dict_cancers[indx_cancer_train[0]]
name_file_cancer_target = dict_cancers[int(config.sel_cancer_Target)]
#print("Source Cancer:",name_file_cancer)
print("Target Cancer:",name_file_cancer_target)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
df_to_read_target = pd.read_csv(_FOLDER + name_file_cancer_target)#.sample(n=N_CellLines,random_state = rand_state_N)

"Split of data into Training and Testing for the Source and Target domains"
for k,idx_cancer in enumerate(indx_cancer_train):
    name_file_cancer = dict_cancers[idx_cancer]
    print("Source Cancer:", name_file_cancer)
    df_to_read = pd.read_csv(_FOLDER + name_file_cancer)#.sample(n=N_CellLines,random_state = rand_state_N)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # "Split of data into Training and Testing for the Source and Target domains"
    # Index_sel = (df_to_read["DRUG_ID"] == 1036) | (df_to_read["DRUG_ID"] == 1061)| (df_to_read["DRUG_ID"] == 1373) \
    #             | (df_to_read["DRUG_ID"] == 1039) | (df_to_read["DRUG_ID"] == 1560) | (df_to_read["DRUG_ID"] == 1057) \
    #             | (df_to_read["DRUG_ID"] == 1059)| (df_to_read["DRUG_ID"] == 1062) | (df_to_read["DRUG_ID"] == 2096) \
    #             | (df_to_read["DRUG_ID"] == 2045)

    if k ==0:
        #df_SourceCancer_all = df_to_read[Index_sel]
        df_SourceCancer_all = df_to_read  # df_to_read[Index_sel]
    else:
        #df_SourceCancer_all = pd.concat([df_SourceCancer_all, df_to_read[Index_sel]])  #df_to_read[Index_sel]
        df_SourceCancer_all = pd.concat([df_SourceCancer_all, df_to_read])

df_all = df_SourceCancer_all.reset_index().drop(columns=['index'])
df_source = df_all.dropna().sample(n=200,random_state = 2)


# Index_sel_target = (df_to_read_target["DRUG_ID"] == 1036) | (df_to_read_target["DRUG_ID"] == 1061)| (df_to_read_target["DRUG_ID"] == 1373) \
#             | (df_to_read_target["DRUG_ID"] == 1039) | (df_to_read_target["DRUG_ID"] == 1560) | (df_to_read_target["DRUG_ID"] == 1057) \
#             | (df_to_read_target["DRUG_ID"] == 1059)| (df_to_read_target["DRUG_ID"] == 1062) | (df_to_read_target["DRUG_ID"] == 2096) \
#             | (df_to_read_target["DRUG_ID"] == 2045)

#df_TargetCancer_all = df_to_read_target[Index_sel_target]
df_TargetCancer_all = df_to_read_target
df_all_target = df_TargetCancer_all.reset_index().drop(columns=['index'])
df_all_target = df_all_target.dropna()

myset_target = set(df_all_target['COSMIC_ID'].values)
myLabels = np.arange(0,myset_target.__len__())
CosmicIDs_All_Target = list(myset_target)
"Here we order the list of target COSMIC_IDs from smallest CosmicID to biggest"
CosmicIDs_All_Target.sort()
CellLine_pos = int(config.idx_CID_Target) #37
print(f"The CosmicID of the selected Target Cell-line: {CosmicIDs_All_Target[CellLine_pos]}")
CosmicID_target = CosmicIDs_All_Target[CellLine_pos]
df_target = df_all_target[df_all_target['COSMIC_ID']==CosmicID_target].reset_index().drop(columns=['index'])
#df_target = df_all_target.iloc[0:200]

"Here we select the drug we will use as testing"
which_drug = int(config.which_drug) #1057
idx_test = np.where(df_target['DRUG_ID']==which_drug)[0]
assert idx_test.shape[0]>0 #The drug selected was not tested in the cell-line
idx_train = np.delete(np.arange(0,df_target.shape[0]),idx_test)

df_target_test = df_target.iloc[idx_test]
df_target_train = df_target.iloc[idx_train]

Name_DrugID_train = df_target_train['DRUG_ID'].values
Name_DrugID_test = df_target_test['DRUG_ID'].values

df_source_and_target = pd.concat([df_source,df_target_train])

## "Here we just check that from the column index 25 the input features start"

start_pos_features = 25
print("first feat Source:",df_all.columns[start_pos_features])
print("first feat Target:",df_all_target.columns[start_pos_features])
assert df_all.columns[start_pos_features] == df_all_target.columns[start_pos_features]

"Below we do until -2 'cause the last two columns of the dataframe are just informative names"
df_feat = df_source_and_target[df_source_and_target.columns[start_pos_features:-2]]
Names_All_features = df_source_and_target.columns[start_pos_features:-2]
Idx_Non_ZeroStd = np.where(df_feat.std()!=0.0)
Names_features_NonZeroStd = Names_All_features[Idx_Non_ZeroStd]

"This is a scaler to scale features in similar ranges"
scaler = MinMaxScaler().fit(df_source_and_target[Names_features_NonZeroStd])

xT_train = scaler.transform(df_target_train[Names_features_NonZeroStd])
xT_test = scaler.transform(df_target_test[Names_features_NonZeroStd])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Here we extract from the dataframe the D outputs related to each dose concentration"
"Below we select 7 concentration since GDSC2 has that number for the Drugs"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Dnorm_cell = 8  #For GDSC2 this is the number of dose concentrations, we include zero concentration

"Here we extract the target domain outputs yT"
x_dose_T = df_target["fd_num_" + str(0)].values[:, None]
yT = np.clip(df_target["norm_cells_" + str(0)].values[:, None], -1.0e-1, 1.2)
print(yT.shape)
for i in range(1, Dnorm_cell):
    yT = np.concatenate((yT, np.clip(df_target["norm_cells_" + str(i)].values[:, None], -1.0e-1, 1.2)), 1)
    x_dose_T = np.concatenate((x_dose_T, df_target["fd_num_" + str(i)].values[:, None]), 1)

print("yT size: ", yT.shape)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Since we fitted the dose response curves with a Sigmoid4_parameters function"
"We extract the optimal coefficients in order to reproduce such a Sigmoid4_parameters fitting"
"Here we extract the target domain parameters"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
params_4_sig_target = df_target["param_" + str(1)].values[:, None]
for i in range(2, 5):  #here there are four params for sigmoid4
    params_4_sig_target = np.concatenate((params_4_sig_target, df_target["param_" + str(i)].values[:, None]), 1)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"In this sections we will extract the summary metrics AUC, Emax and IC50 from the Sigmoid4_parameters functions"
"These metrics are used as the references to compute the error metrics"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from sklearn import metrics
from importlib import reload
import Utils_TransferLearning
import Utils_TransferLearning as MyUtils
reload(MyUtils)

"The function Extract_IC50_AUC_Emax is implemented in Utils_TransferLearning.py to extract the summary metrics"
#print(f"No Log: {x_dose_T}")

add_to_log = 0.1
assert add_to_log==0.1  #if it is not 0.1 then we should change the log2(0.1) in model.noise_func or receive as value
x_dose_T = np.log2(x_dose_T + add_to_log)

# "THIS IS JUST A CHECK BELOW"
# x_dose_T = np.repeat(np.array(np.linspace(0.0,1.0,8))[None,:], x_dose_T.shape[0], axis=0)

#print(f"With Log: {x_dose_T}")
#x_lin = np.log2( np.array([np.linspace(xrange[0],xrange[-1],1000) for xrange in x_dose_T]) + add_to_log)
#x_real_dose = np.linspace(0.142857, 1, Dnorm_cell)  #Here is Dnorm_cell due to using GDSC2 that has 7 doses

#TODO the params_4_sig are not working with the current dose concentrations so we cannot extract summary metrics
#TODO I have to ask Evelyn how are the params_4_sig computed

# Ydose50,Ydose_res,IC50,AUC,Emax = MyUtils.Extract_IC50_AUC_Emax(params_4_sig_target,x_lin)
# AUC = np.array(AUC)[:, None]
# IC50 = np.array(IC50)[:, None]
# Emax = np.array(Emax)[:, None]

# def my_plot(posy,fig_num,Ydose50,Ydose_res,IC50,AUC,Emax,x_lin,x_real_dose,y_train_drug):
#     plt.figure(fig_num)
#     #plt.plot(x_lin, Ydose_res[posy])
#     plt.plot(x_real_dose, y_train_drug[posy, :], '.')
#     #plt.plot(IC50[posy], Ydose50[posy], 'rx')
#     #plt.plot(x_lin, np.ones_like(x_lin)*Emax[posy], 'r') #Plot a horizontal line as Emax
#     #plt.title(f"AUC = {AUC[posy]}")
#     #plt.legend(['Sigmoid4','Observations','IC50','Emax'])
#
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# "Here we can visualise the values of the GDSC2 dataset with the fitting of Sigmoid4_parameters function"
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# posy = 0   #select the location you want to plot, do not exceed the Ytrain length
# my_plot(posy,1,Ydose50,Ydose_res,IC50,AUC,Emax,x_lin[posy,:],x_dose_T[posy,:],yT)

yT_train = yT[idx_train]
yT_test = yT[idx_test]

"Here we define the set of all the possible cell-lines (Cosmic_ID) that belong to Melanoma cancer"
"NOTE: Not all cell-lines have been tested with all ten drugs, so some cell-line might have more"
"dose response curves than others"

myset_source = set(df_source['COSMIC_ID'].values)
myLabels = np.arange(0,myset_source.__len__())
CosmicID_labels = list(myset_source)
"Here we order the list from smallest CosmicID to biggest"
CosmicID_labels.sort()
dict_CosmicID2Label = dict(zip(CosmicID_labels,myLabels))
df_source_sort = df_source.sort_values(by='COSMIC_ID')
CosmicID_source = df_source_sort['COSMIC_ID'].values
idx_S = [dict_CosmicID2Label[CosID] for CosID in CosmicID_source]

"Here we extract the source domain inputs xS so that they coincide with yS sorted by COSMIC_ID"
xS_train = scaler.transform(df_source_sort[Names_features_NonZeroStd])

"Here we extract the source domain outputs yS"

x_dose_S = df_source_sort["fd_num_" + str(0)].values[:, None]
yS_train = np.clip(df_source_sort["norm_cells_" + str(0)].values[:, None], -1.0e-1, 1.2)
print(yS_train.shape)
for i in range(1, Dnorm_cell):
    yS_train = np.concatenate((yS_train, np.clip(df_source_sort["norm_cells_" + str(i)].values[:, None], -1.0e-1, 1.2)), 1)
    x_dose_S = np.concatenate((x_dose_S, df_source_sort["fd_num_" + str(i)].values[:, None]),1)

print("yS size: ", yS_train.shape)

x_dose_S = np.log2(x_dose_S + add_to_log)

# "THIS IS JUST A CHECK BELOW"
# x_dose_S = np.repeat(np.array(np.linspace(0.0,1.0,8))[None,:], x_dose_S.shape[0], axis=0)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Make all variable passed to the model tensor to operate in pytorch"
indT_concentr = np.array([0,5,6,7])
xT_all_train = xT_train.copy()
yT_all_train = yT_train[:,indT_concentr].copy()
xT_train = torch.from_numpy(xT_train)
xS_train = torch.from_numpy(xS_train)
yT_train_AllConc = torch.from_numpy(yT_train)
yT_train = torch.from_numpy(yT_train[:,indT_concentr])
yS_train = torch.from_numpy(yS_train)
xT_test = torch.from_numpy(xT_test)
yT_test_AllConc = torch.from_numpy(yT_test)
yT_test = torch.from_numpy(yT_test[:,indT_concentr])

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print(f"new shape for yS_train: {yS_train.shape}")
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"This is the number of source domains plus target domain."
"In this cancer application we are refering to each domain as each cell-line with the Cosmic_ID"
"There are CosmicID_labels.__len__() cell-lines or CosmicIDs in the Source (Melanoma) + 1 cell-line of the Target"
"Then the total NDomains = CosmicID_labels.__len__() + 1"
NDomains = CosmicID_labels.__len__() + 1 #CosmicID_labels.__len__() contains the CosmicIDs of Source Domains
print(f"Number Nsource Domains: {CosmicID_labels.__len__()}, so total NDomains Source + Target: {NDomains}")

"Drug_S concentrations for training"
# Drug_S_range = list(np.linspace(0.0,1.0,8))
# DrugC_S = torch.Tensor(Drug_S_range) #0:NSource_concentr
# DrugC_S = DrugC_S.repeat(yS_train.shape[0],1)

DrugC_S = torch.from_numpy(x_dose_S)

"Drug_T concentrations for training and testing"
DrugC_T_AllConc = torch.from_numpy(x_dose_T[idx_train])
DrugC_T = DrugC_T_AllConc[:,indT_concentr]
DrugC_T_test_AllConc = torch.from_numpy(x_dose_T[idx_test])
DrugC_T_test = DrugC_T_test_AllConc[:,indT_concentr]

model = TLMOGaussianProcess(xT_train,yT_train,xS_train,yS_train,idxS=idx_S,DrugC_T=DrugC_T,DrugC_S=DrugC_S,NDomains=NDomains)
## model.covariance.length=0.05
myseed = int(config.which_seed)
torch.manual_seed(myseed)   #Ex1: 15 (run 100 iter)  #Exp2 (906826): 35  (run 100 iter)
with torch.no_grad():
    #model.lik_std_noise= torch.nn.Parameter(2.0*torch.ones(NDomains)) #torch.nn.Parameter(0.5*torch.randn(NDomains))

    model.TLCovariance[0].length = float(config.weight)*1.0*np.sqrt(xT_train.shape[1])*torch.rand(NDomains)[:,None] #2
    model.TLCovariance[1].length = float(config.weight)*15.0*np.sqrt(xT_train.shape[1]) * torch.rand(NDomains)[:, None] #6
    model.TLCovariance[2].length = float(config.weight)*30.0*np.sqrt(xT_train.shape[1]) * torch.rand(NDomains)[:, None]#10
    model.TLCovariance[3].length = float(config.weight)*40.0*np.sqrt(xT_train.shape[1]) * torch.rand(NDomains)[:,None]
    #model.CoregCovariance[0].lengthscale = 500*8*torch.rand(1) #8*
    #model.CoregCovariance[1].lengthscale = 500*10*torch.rand(1)  #10*
    #model.CoregCovariance[2].lengthscale = 500*20*torch.rand(1)  #20*

    model.CoregCovariance[0].sig = 0.1 * torch.rand(1)  # 8*
    model.CoregCovariance[1].sig = 5 * torch.rand(1)  # 10*
    model.CoregCovariance[2].sig = 10 * torch.rand(1)  # 20*
    model.CoregCovariance[3].sig = 20 * torch.rand(1)

    model.CoregCovariance[0].sig0 = 0.1 * torch.rand(1)  # 8*
    model.CoregCovariance[1].sig0 = 5 * torch.rand(1)  # 10*
    model.CoregCovariance[2].sig0 = 10 * torch.rand(1)  # 20*
    model.CoregCovariance[3].sig0 = 20 * torch.rand(1)

    model.LambdaDiDj.muDi = 3*torch.rand(NDomains)[:, None]
    model.LambdaDiDj.bDi = 3*torch.rand(NDomains)[:, None]
    #print(model.LambdaDiDj.muDi)
#print(f"Noises std: {model.lik_std_noise}")

"Training process below"
def myTrain(model,xT_train,yT_train,myLr = 1e-2,Niter = 1):
    optimizer = optim.Adam(model.parameters(),lr=myLr)
    loss_fn = LogMarginalLikelihood()
    flag = 1
    for iter in range(Niter):
        # Forward pass
        mu, L = model(xT_train)

        # Backprop
        loss = -loss_fn(mu,L,yT_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss.item() < 0 and flag ==1:
        #if iter==100:  #70
            flag = 0
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr'] * 0.01

        print(f"i: {iter+1}, Loss: {loss.item()}")
        # print(f"TLlength1 {model.TLCovariance[2].length}")
        # print(f"CoregCov1 {model.CoregCovariance[0].lengthscale}")
        # print(f"CoregCov2 {model.CoregCovariance[1].lengthscale}")
        # print(f"CoregCov3 {model.CoregCovariance[2].lengthscale}")
        # print(f"muDi: {model.LambdaDiDj.bDi}")

"Train the model with all yT training data"
myTrain(model,xT_train,yT_train,myLr = 5e-2,Niter = int(config.N_iter))
def bypass_params(model_trained,model_cv):
    #model_cv.lik_std_noise = model_trained.lik_std_noise
    model_cv.coef1 = model_trained.coef1
    model_cv.coef2 = model_trained.coef2

    "Bypass lengthscales"
    for i in range(model_trained.TLCovariance.__len__()):
        model_cv.TLCovariance[i].length = model_trained.TLCovariance[i].length.clone()
        "Bypass variances"
        model_cv.TLCovariance[i].variance = model_trained.TLCovariance[i].variance.clone()

        # model_cv.CoregCovariance[0].lengthscale = model_trained.CoregCovariance[0].lengthscale.clone()
        # model_cv.CoregCovariance[1].lengthscale = model_trained.CoregCovariance[1].lengthscale.clone()
        # model_cv.CoregCovariance[2].lengthscale = model_trained.CoregCovariance[2].lengthscale.clone()

        model_cv.CoregCovariance[i].sig = model_trained.CoregCovariance[i].sig.clone()

        model_cv.CoregCovariance[i].sig0 = model_trained.CoregCovariance[i].sig0.clone()

    model_cv.LambdaDiDj.muDi = model_trained.LambdaDiDj.muDi.clone()
    model_cv.LambdaDiDj.bDi = model_trained.LambdaDiDj.bDi.clone()

# model_check = TLMOGaussianProcess(xT_train,yT_train,xS_train,yS_train,idxS=idx_S,DrugC_T=DrugC_T,DrugC_S=DrugC_S,NDomains=NDomains)
# print(f"Model check train BEFORE bypass:")
# myTrain(model_check, xT_train, yT_train, myLr=1e-20, Niter=1)
# bypass_params(model,model_check)
#
# print(f"Model check train AFTER bypass:")
# myTrain(model_check, xT_train, yT_train, myLr=1e-20, Niter=1)
#
# print(f"Model ORIGINAL train check small stepsize:")
# myTrain(model, xT_train, yT_train, myLr=1e-20, Niter=1)

"Leave one out cross-validation"
Val_LML = LogMarginalLikelihood()
TestLogLoss_All = []
#for i in range(2):
for i in range(yT_all_train.shape[0]):
    model_cv = []
    yT_train_cv = np.delete(yT_all_train,i,axis=0)
    DrugC_T_train_cv = np.delete(DrugC_T.numpy(),i,axis=0)
    DrugC_T_val_cv = DrugC_T.numpy()[i:i+1,:]
    yT_val_cv = yT_all_train[i:i+1,:]
    xT_train_cv = np.delete(xT_all_train, i, axis=0)
    xT_val_cv = xT_all_train[i:i + 1, :]
    print(f"shape cv train:{xT_train_cv.shape}")
    #print(f"cv train:{xT_train_cv}")
    print(f"shape cv val:{xT_val_cv.shape}")
    #print(f"cv val:{xT_val_cv}")
    xT_train_cv = torch.from_numpy(xT_train_cv)
    yT_train_cv = torch.from_numpy(yT_train_cv)
    xT_val_cv = torch.from_numpy(xT_val_cv)
    yT_val_cv = torch.from_numpy(yT_val_cv)
    DrugC_T_train_cv = torch.from_numpy(DrugC_T_train_cv)
    DrugC_T_val_cv = torch.from_numpy(DrugC_T_val_cv)

    "model fit with Cross-val"
    model_cv = TLMOGaussianProcess(xT_train_cv, yT_train_cv, xS_train, yS_train, idxS=idx_S, DrugC_T=DrugC_T_train_cv,DrugC_S=DrugC_S, NDomains=NDomains)
    bypass_params(model, model_cv)  #Here we bypass the fitted parameters from the MOGP trained over all data
    myTrain(model_cv, xT_train_cv, yT_train_cv, myLr=1e-15, Niter=1) #Here we could refine hyper-params a bit if wished
    "Here we put the model in prediciton mode"
    model_cv.eval()
    model_cv.Train_mode = False
    #DrugCtoPred_cv = list(np.linspace(0.142857, 1, 7))
    with torch.no_grad():
        "NOTE: It is important to validate the model with the noisy prediction"
        "I noticed it is necessary to include the outputs' uncertainty when validating"
        "that is the why noiseless=False, this guarantees including the noise uncertainty of the outputs in preditions"
        "validating with noiseless=True can lead to selecting an underfitted model"
        mpred_cv, Cpred_cv = model_cv(xT_val_cv, DrugC_new=DrugC_T_val_cv, noiseless=False)
        Lpred_cv = torch.linalg.cholesky(Cpred_cv)  #Here we compute Cholesky since Val_LML gets L Cholesky of Cpred
        Val_loss = -Val_LML(mpred_cv, Lpred_cv, yT_val_cv)
        print(f"Val Loss: {Val_loss.item()}")
    TestLogLoss_All.append(Val_loss.numpy())

print(f"Mean cv ValLogLoss: {np.mean(TestLogLoss_All)}")

path_home = '/home/juanjo/Work_Postdoc/my_codes_postdoc/'
#path_home = '/rds/general/user/jgiraldo/home/TransferLearning_Results/'
path_val = path_home+'Jobs_TLMOGP_OneCell_OneDrug_Testing/TargetCancer'+str(config.sel_cancer_Target)+'/Drug_'+str(config.which_drug)+'/CellLine'+str(config.idx_CID_Target)+'_CID'+str(CosmicID_target)+'/'

# check whether directory already exists
if not os.path.exists(path_val):
  #os.mkdir(path_val)   #Use this for a single dir
  os.makedirs(path_val) #Use this for a multiple sub dirs

"Here we save the Validation Log Loss in path_val in order to have a list of different bashes to select the best model"
f = open(path_val+'Validation.txt', "a")
f.write(f"\nbash{str(config.bash)}, ValLogLoss:{np.mean(TestLogLoss_All)}, CrossVal_N:{yT_train.shape[0]}")
f.close()

"Here we have to assign the flag to change from self.Train_mode = True to False"
print("check difference between model.eval and model.train")
model.eval()
model.Train_mode = False
plot_test = False
if plot_test:
    x_test = xT_test.clone()
    y_test = yT_test.clone()
    DrugCtoPred_exact = DrugC_T_test.clone()
    Name_DrugID_plot = Name_DrugID_test
    plotname = 'Test'
else:
    x_test = xT_train.clone()
    y_test = yT_train.clone()
    DrugCtoPred_exact = DrugC_T.clone()
    Name_DrugID_plot = Name_DrugID_train
    plotname = 'Train'

"The Oversample_N below is to generate equaly spaced drug concentrations between the original 7 drug concentrations"
"i.e., if Oversample_N = 2: means that each 2 positions we'd have the original drug concentration tested in cell-line"
"we'd have DrugCtoPred = [0.1428, 0.2142, 0.2857, 0.3571,0.4285,0.4999,0.5714,0.6428,0.7142,0.7857,0.8571,0.9285,1.0]"

Oversample_N = 35
#DrugCtoPred_range = list(np.linspace(0.5,1,NTarget_concentr+(NTarget_concentr-1)*(Oversample_N-1)))
DrugCtoPred_range = np.array(np.linspace(-3.5,4.0,7+6*(Oversample_N-1)))
#DrugCtoPred_range = list(np.linspace(0.142857,1,7+6*(Oversample_N-1)))
DrugCtoPred = torch.from_numpy(DrugCtoPred_range)
DrugCtoPred = DrugCtoPred.repeat(x_test.shape[0],1)

"Below we refer as exact in the sense that those are the exact location of drug concent. for which we have exact data"

#ind_NTarget_concentr = np.array([0,4,5,6,7])
#DrugCtoPred_exact_range = np.array(np.linspace(0.0,1.0,8))[ind_NTarget_concentr]
#DrugCtoPred_exact = torch.Tensor(DrugCtoPred_exact_range)
#DrugCtoPred_exact = DrugCtoPred_exact.repeat(x_test.shape[0],1)

with torch.no_grad():
    mpred, Cpred = model(x_test,DrugC_new = DrugCtoPred,noiseless=False)
    "To assess the TestLogLoss we have to also include the noise uncertainty, so we use noiseless=False"
    mpred_exact, Cpred_exact = model(x_test, DrugC_new=DrugCtoPred_exact, noiseless=False)
    Lpred_exact = torch.linalg.cholesky(Cpred_exact)  # Here we compute Cholesky since Val_LML gets L Cholesky of Cpred
    Test_loss = -Val_LML(mpred_exact, Lpred_exact, y_test)
    print(f"Test Loss: {Test_loss.item()}")

yT_pred = mpred.reshape(DrugCtoPred.shape[1],x_test.shape[0]).T

"Plot the prediction for the test yT"
from torch.distributions.multivariate_normal import MultivariateNormal
plt.close('all')
#plt.switch_backend('agg')
for i in range(x_test.shape[0]):
    plt.figure(i+1)
    plt.ylim([-0.02,1.1])
    #plt.plot(x,mpred1,'.')
    plt.plot(DrugCtoPred[i,:],yT_pred[i,:],'blue')
    #plt.plot(IC50_pred[i],0.5,'x')   # Plot an x in predicted IC50 location
    #plt.plot(x_lin, np.ones_like(x_lin) * Emax_pred[i], 'r')  # Plot a horizontal line as Emax
    if plot_test:
        #plt.plot(DrugC_T_test[i,:], yT_test[i,:], 'ro')
        plt.plot(DrugC_T_test_AllConc[i,:], yT_test_AllConc[i, :], 'b*')
        plt.plot(DrugC_T_test_AllConc[i, indT_concentr], yT_test_AllConc[i, indT_concentr], 'ro')
        xlim1 = DrugC_T_test_AllConc[i,0]; xlim2 = DrugC_T_test_AllConc[i,-1]
        plt.xlim([xlim1-.1*np.abs(xlim1),xlim2+.1*np.abs(xlim2)])
    else:
        #plt.plot(DrugC_T[i, :], yT_train[i, :], 'ro')
        plt.plot(DrugC_T_AllConc[i,:], yT_train_AllConc[i, :], 'b*')
        plt.plot(DrugC_T_AllConc[i, indT_concentr], yT_train_AllConc[i, indT_concentr], 'ro')
        xlim1 = DrugC_T_AllConc[i,0]; xlim2 = DrugC_T_AllConc[i,-1]
        plt.xlim([xlim1-.1*np.abs(xlim1),xlim2+.1*np.abs(xlim2)])
    plt.title(f"CosmicID: {CosmicID_target}, {plotname} DrugID: {Name_DrugID_plot[i]}",fontsize=14)
    plt.xlabel('Dose concentration',fontsize=14)
    plt.ylabel('Cell viability',fontsize=14)
    plt.grid(True)

    # for j in range(30):
    #     i_sample = MultivariateNormal(loc=mpred[:, 0], covariance_matrix=Cpred)
    #     yT_pred_sample = i_sample.sample().reshape(DrugCtoPred.__len__(), x_test.shape[0]).T
    #     plt.plot(DrugCtoPred, yT_pred_sample[i, :], alpha=0.1)

    std_pred = torch.sqrt(torch.diag(Cpred)).reshape(DrugCtoPred.shape[1], x_test.shape[0]).T
    plt.plot(DrugCtoPred[i,:], yT_pred[i, :] + 2.0 * std_pred[i,:], '--b')
    plt.plot(DrugCtoPred[i,:], yT_pred[i, :] - 2.0 * std_pred[i,:], '--b')

    # # check whether directory already exists
    # path_plot = path_val + 'Test_plot/'
    # if not os.path.exists(path_plot):
    #     # os.mkdir(path_val)   #Use this for a single dir
    #     os.makedirs(path_plot)  # Use this for a multiple sub dirs
    # plt.savefig(path_plot+'plotbash'+str(config.bash)+'.pdf')

#plt.savefig('./Plots_April16_2024/CID'+str(CosmicID_target)+'_Train'+str(i)+'.pdf')