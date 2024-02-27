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
from TransferLearning_Kernels import TL_Kernel_var, Kernel_CrossDomains, TLRelatedness, Kernel_ICM,Kernel_ICM_LessParams

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
    def __init__(self,xT,yT,xS,yS,idxS,DrugC,NDomains):
        super().__init__()
        self.Douts = yT.shape[1]
        self.DrugC = torch.Tensor(DrugC)[:,None] #We treat the Drug Concentrations as a float tensor vector [N,1]
        self.NDomains = NDomains
        #self.Q = 2    #Number of Kernel_CrossDomains to linearly combine
        assert self.DrugC.shape[0] == yT.shape[1]  #DrugC length should be equal to D number of outputs
        self.Pfeat = xT.shape[1]
        #torch.kron(mat1, mat2.reshape(-1)).reshape(-1, 429)
        #self.xT = torch.kron(torch.ones(self.Douts, 1),xT)  #This is to replicate xT as per the D number of outputs
        self.xT = torch.kron(torch.ones(self.Douts, 1), xT.reshape(-1)).reshape(-1, self.Pfeat)  # This is to replicate xT as per the D number of outputs
        #self.xS = torch.kron(torch.ones(self.Douts, 1), xS)  # This is to replicate xS as per the D number of outputs
        self.xS = torch.kron(torch.ones(self.Douts, 1),xS.reshape(-1)).reshape(-1, self.Pfeat)  #This is to replicate xS as per the D number of outputs
        self.yT = yT.T.reshape(-1, 1)  # This is to vectorise by staking the columns (vect(yT))
        self.yS = yS.T.reshape(-1,1) #This is to vectorise by staking the columns (vect(yS))
        self.DrugC_xT = torch.kron(self.DrugC,torch.ones(xT.shape[0], 1)) #Rep. Concentr. similar to coreginalisation
        self.DrugC_xS = torch.kron(self.DrugC,torch.ones(xS.shape[0], 1)) #Rep. Concentr. similar to coreginalisation
        self.idxS = idxS * self.Douts #Replicate the Source domain index as per the number of outputs
        self.idxT = [NDomains - 1] * xT.shape[0] * self.Douts #Replicate the target domain index as per the number of outputs
        self.all_y = torch.cat([self.yS, self.yT])
        self.TLCovariance = [Kernel_CrossDomains(NDomains=NDomains),Kernel_CrossDomains(NDomains=NDomains),Kernel_CrossDomains(NDomains=NDomains)] #gpytorch.kernels.RBFKernel()
        self.LambdaDiDj = TLRelatedness(NDomains=NDomains)
        #self.CoregCovariance = gpytorch.kernels.RBFKernel()
        self.CoregCovariance = [gpytorch.kernels.MaternKernel(1.5),gpytorch.kernels.MaternKernel(2.5),gpytorch.kernels.MaternKernel(2.5)]
        self.Train_mode = True
        self.lik_std_noise = torch.nn.Parameter(1.0*torch.ones(NDomains)) #0.3*torch.rand(NDomains)+0.01
        #self.lik_std_noise = 0.05 * torch.ones(NDomains)
        self.mu_star = torch.zeros(self.yT.shape) #mu has the shape of the new replicated along the outputs yT
        self.L = torch.eye(self.yT.shape[0])
        self.all_L = torch.eye(self.all_y.shape[0])
        "TODO: I might need the self.LSS also here in order to be able to predict without optimising"

    def forward(self,xT, DrugC_new = None,noiseless = True):
        if self.Train_mode:
            #xT = torch.kron(torch.ones(self.Douts, 1), xT)
            xT = torch.kron(torch.ones(self.Douts, 1), xT.reshape(-1)).reshape(-1, self.Pfeat)
            assert (xT == self.xT).sum() == (xT.shape[0]*xT.shape[1]) #This is just to check if the xT to init the model is the same
            assert xT.shape[1] == self.xT.shape[1]  #Thiis is to check if the xT to init the model has same Pfeatures

            "Below the TLCovariance and LambdaDiDj have to coincide in the same indexes idx1 and idx2"
            # Here we compute the Covariance matrices between source-target, source-source and target domains
            KTS = self.LambdaDiDj(xT,self.xS,idx1=self.idxT,idx2=self.idxS).evaluate()*(self.CoregCovariance[0](self.DrugC_xT,self.DrugC_xS).evaluate()*self.TLCovariance[0](xT,self.xS,idx1=self.idxT,idx2=self.idxS).evaluate() + \
                                                                                        self.CoregCovariance[1](self.DrugC_xT,self.DrugC_xS).evaluate()*self.TLCovariance[1](xT, self.xS, idx1=self.idxT, idx2=self.idxS).evaluate()+\
                                                                                        self.CoregCovariance[2](self.DrugC_xT,self.DrugC_xS).evaluate()*self.TLCovariance[2](xT, self.xS, idx1=self.idxT, idx2=self.idxS).evaluate())
            KSS = self.LambdaDiDj(self.xS,idx1=self.idxS).evaluate()*(self.CoregCovariance[0](self.DrugC_xS,self.DrugC_xS).evaluate()*self.TLCovariance[0](self.xS,idx1=self.idxS).evaluate()+ \
                                                                      self.CoregCovariance[1](self.DrugC_xS,self.DrugC_xS).evaluate()*self.TLCovariance[1](self.xS, idx1=self.idxS).evaluate()+\
                                                                      self.CoregCovariance[2](self.DrugC_xS,self.DrugC_xS).evaluate()*self.TLCovariance[2](self.xS, idx1=self.idxS).evaluate())
            KTT = self.LambdaDiDj(self.xT,idx1=self.idxT).evaluate()*(self.CoregCovariance[0](self.DrugC_xT,self.DrugC_xT).evaluate()*self.TLCovariance[0](self.xT,idx1=self.idxT).evaluate()+ \
                                                                      self.CoregCovariance[1](self.DrugC_xT,self.DrugC_xT).evaluate() * self.TLCovariance[1](self.xT, idx1=self.idxT).evaluate()+\
                                                                      self.CoregCovariance[2](self.DrugC_xT,self.DrugC_xT).evaluate() * self.TLCovariance[2](self.xT, idx1=self.idxT).evaluate())

            # Here we include the respective noise terms associated to each domain
            CSS = KSS + torch.diag(self.lik_std_noise[self.idxS].pow(2))
            CTT = KTT + torch.diag(self.lik_std_noise[self.idxT].pow(2))

            # The code below aim to correct for numerical instabilities when CSS becomes Non-PSD
            if not isPD_torch(CSS):
                CSS_aux = CSS.clone()
                with torch.no_grad():
                    CSS_aux = torch.from_numpy(nearestPD(CSS_aux.numpy()))
                CSS = 0.0*CSS + CSS_aux  #This operation aims to keep the gradients working over lik_std_noise

            self.LSS = torch.linalg.cholesky(CSS)
            alphaSS1 = torch.linalg.solve(self.LSS, self.yS)
            alphaSS = torch.linalg.solve(self.LSS.t(), alphaSS1)

            # Compute the mean of the conditional distribution p(yT|XT,XS,yS)
            self.mu_star = torch.matmul(KTS,alphaSS)
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
                                                                     self.CoregCovariance[2](self.DrugC_xSxT).evaluate() * self.TLCovariance[2](xST, idx1=idxST).evaluate())
            all_K_xST_noise = all_K_xST + torch.diag(self.lik_std_noise[idxST].pow(2))  # + 0.1*torch.eye(xST.shape[0]) #Jitter?
            if not isPD_torch(all_K_xST_noise):
                with torch.no_grad():
                    all_K_xST_noise = torch.from_numpy(nearestPD(all_K_xST_noise.numpy()))
            self.all_L = torch.linalg.cholesky(all_K_xST_noise)  # + 1e-4*torch.eye(xST.shape[0])

            # Here we receive a list of possible drug concentrations to predict
            if DrugC_new is None:
                DrugC_new = self.DrugC
            else:
                DrugC_new = torch.Tensor(DrugC_new)[:, None]
            #Below we replicate the target domain index as per the number of drug concentration we want to test
            NewDouts = DrugC_new.shape[0]
            idxT = [self.NDomains - 1] * xT.shape[0] * NewDouts
            DrugC_xT = torch.kron(DrugC_new, torch.ones(xT.shape[0], 1))
            "Be careful with operation using xT.shape, from here it changes the original shape"
            #xT = torch.kron(torch.ones(NewDouts, 1), xT)
            xT = torch.kron(torch.ones(NewDouts, 1), xT.reshape(-1)).reshape(-1, self.Pfeat)
            alpha1 = torch.linalg.solve(self.all_L, self.all_y)
            alpha = torch.linalg.solve(self.all_L.t(), alpha1)
            KTT_xnew_xnew = self.LambdaDiDj(xT, idx1=idxT).evaluate()*(self.CoregCovariance[0](DrugC_xT).evaluate() * self.TLCovariance[0](xT, idx1=idxT).evaluate()+ \
                                                                       self.CoregCovariance[1](DrugC_xT).evaluate() * self.TLCovariance[1](xT,idx1=idxT).evaluate()+\
                                                                       self.CoregCovariance[2](DrugC_xT).evaluate() * self.TLCovariance[2](xT,idx1=idxT).evaluate())
            xST = torch.cat([self.xS, self.xT])
            idxST = self.idxS+self.idxT

            # Rep. Concentr. similar to coreginalisation
            K_xnew_xST = self.LambdaDiDj(xT,xST, idx1=idxT,idx2=idxST).evaluate()*(self.CoregCovariance[0](DrugC_xT,self.DrugC_xSxT).evaluate() * self.TLCovariance[0](xT,xST, idx1=idxT,idx2=idxST).evaluate()+ \
                                                                                   self.CoregCovariance[1](DrugC_xT,self.DrugC_xSxT).evaluate() * self.TLCovariance[1](xT, xST, idx1=idxT, idx2=idxST).evaluate()+\
                                                                                   self.CoregCovariance[2](DrugC_xT,self.DrugC_xSxT).evaluate() * self.TLCovariance[2](xT, xST, idx1=idxT, idx2=idxST).evaluate())

            f_mu = torch.matmul(K_xnew_xST, alpha)
            v = torch.linalg.solve(self.all_L, K_xnew_xST.t())

            if noiseless:
                f_Cov = KTT_xnew_xnew - torch.matmul(v.t(),v) #+ 1e-2*torch.eye(xT.shape[0])  #I had to add this Jitter
                f_Cov = torch.from_numpy(nearestPD(f_Cov.numpy()))
            else:
                f_Cov = KTT_xnew_xnew - torch.matmul(v.t(),v) + torch.diag(self.lik_std_noise[idxT].pow(2)) + 1e-5*torch.eye(xT.shape[0])
                f_Cov = torch.from_numpy(nearestPD(f_Cov.numpy()))
            return f_mu, f_Cov

class BMKMOGaussianProcess(nn.Module):
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
    def __init__(self,xS,yS,idxS,DrugC,NDomains):
        super().__init__()
        self.Douts = yS.shape[1]
        self.DrugC = torch.Tensor(DrugC)[:,None] #We treat the Drug Concentrations as a float tensor vector [N,1]
        assert self.DrugC.shape[0] == yS.shape[1]  #DrugC length should be equal to D number of outputs
        self.Pfeat = xS.shape[1]
        self.xS = torch.kron(torch.ones(self.Douts, 1), xS.reshape(-1)).reshape(-1, self.Pfeat)
        self.yS = yS.T.reshape(-1,1) #This is to vectorise by staking the columns (vect(yS))
        self.DrugC_xS = torch.kron(self.DrugC,torch.ones(xS.shape[0], 1)) #Rep. Concentr. similar to coreginalisation
        self.idxS = idxS * self.Douts #Replicate the Source domain index as per the number of outputs
        #self.idxT = [NDomains - 1] * xT.shape[0] * self.Douts #Replicate the target domain index as per the number of outputs
        #self.all_y = torch.cat([self.yS, self.yT])
        assert NDomains == (max(idxS)+1) #This is to assert that the Domains meant by user coincide with max label
        #self.TLCovariance = Kernel_ICM(NDomains=NDomains)+Kernel_ICM(NDomains=NDomains)+Kernel_ICM(NDomains=NDomains)#+Kernel_ICM(NDomains=NDomains)
        self.TLCovariance = Kernel_ICM(NDomains=NDomains)
        for i in range(NDomains-1):
            self.TLCovariance += Kernel_ICM(NDomains=NDomains)
        #self.CoregCovariance = gpytorch.kernels.RBFKernel()
        self.CoregCovariance = [gpytorch.kernels.RBFKernel(),gpytorch.kernels.RBFKernel(),gpytorch.kernels.RBFKernel()]
        self.Train_mode = True
        self.lik_std_noise = torch.nn.Parameter(1.0*torch.ones(NDomains)) #torch.tensor([0.07])
        #self.lik_std_noise = 0.05 * torch.ones(NDomains)
        #self.mu_star = torch.zeros(self.yT.shape) #mu has the shape of the new replicated along the outputs yT
        self.LSS = torch.eye(self.yS.shape[0])
        #self.all_L = torch.eye(self.all_y.shape[0])
        "TODO: I need to erase all related to xT since now we only have all data inside xS"

    def forward(self,xS, DrugC_new = None,NDomain_sel = None,noiseless = True):
        if self.Train_mode:
            #xS = torch.kron(torch.ones(self.Douts, 1), xS)
            xS = torch.kron(torch.ones(self.Douts, 1), xS.reshape(-1)).reshape(-1, self.Pfeat)
            assert (xS == self.xS).sum() == (xS.shape[0] * xS.shape[1])  # This is just to check if the xT to init the model is the same
            assert xS.shape[1] == self.xS.shape[1]  # Thiis is to check if the xT to init the model has same Pfeatures
            # Here we compute the Covariance matrices between source-source domains
            #KSS = self.CoregCovariance(self.DrugC_xS,self.DrugC_xS).evaluate()*self.TLCovariance(xS,idx1=self.idxS).evaluate()
            KSS = (self.CoregCovariance[0](self.DrugC_xS, self.DrugC_xS).evaluate()+
                   self.CoregCovariance[1](self.DrugC_xS, self.DrugC_xS).evaluate()+
                   self.CoregCovariance[2](self.DrugC_xS, self.DrugC_xS).evaluate())* self.TLCovariance(xS,idx1=self.idxS).evaluate()

            # Here we include the respective noise terms associated to each domain
            CSS = KSS + torch.diag(self.lik_std_noise[self.idxS].pow(2))

            # The code below aim to correct for numerical instabilities when CSS becomes Non-PSD
            if not isPD_torch(CSS):
                CSS_aux = CSS.clone()
                with torch.no_grad():
                    CSS_aux = torch.from_numpy(nearestPD(CSS_aux.numpy()))
                CSS = 0.0*CSS + CSS_aux  #This operation aims to keep the gradients working over lik_std_noise

            self.LSS = torch.linalg.cholesky(CSS)
            #TODO I should use a prior with mean in 1.0 so that the model predicts values in 1 when is uncertain!!
            self.mu_star = torch.zeros_like(self.yS)
            return self.mu_star, self.LSS  # here we return the mean and covariance
        else:
            "We replicate the target domain index as per the number of drug concentration we want to test"
            "notice that it is not limited to D number of outputs, but actually the number of concentrations"
            # Here we receive a list of possible drug concentrations to predict
            if DrugC_new is None:
                DrugC_new = self.DrugC
            else:
                DrugC_new = torch.Tensor(DrugC_new)[:, None]
            #Below we replicate the target domain index as per the number of drug concentration we want to test
            NewDouts = DrugC_new.shape[0]
            "Below NDomain_sel has to be the domain (i.e., D-th output we want to predict)"
            if NDomain_sel is None:
                NDomain_sel = 0
                print("NDomain_sel = 0 by default, this would predict for domain 0. Set NDomain_sel as per the domain to predict")
            idxS_new = [NDomain_sel] * xS.shape[0] * NewDouts  #Here xS is the new input value for which we want to pred
            DrugC_xS = torch.kron(DrugC_new, torch.ones(xS.shape[0], 1))
            "Be careful with operation using xT.shape, from here it changes the original shape"
            xS = torch.kron(torch.ones(NewDouts, 1), xS.reshape(-1)).reshape(-1, self.Pfeat)
            alpha1 = torch.linalg.solve(self.LSS, self.yS)  #Here LSS contains info of all source domains (all outputs)
            alpha = torch.linalg.solve(self.LSS.t(), alpha1)
            KSS_xnew_xnew = (self.CoregCovariance[0](DrugC_xS).evaluate()+
                             self.CoregCovariance[1](DrugC_xS).evaluate()+
                             self.CoregCovariance[2](DrugC_xS).evaluate())* self.TLCovariance(xS, idx1=idxS_new).evaluate()

            # Rep. Concentr. similar to coreginalisation
            K_xnew_xS = (self.CoregCovariance[0](DrugC_xS,self.DrugC_xS).evaluate()+
                         self.CoregCovariance[1](DrugC_xS,self.DrugC_xS).evaluate()+
                         self.CoregCovariance[2](DrugC_xS,self.DrugC_xS).evaluate())* self.TLCovariance(xS,self.xS, idx1=idxS_new,idx2=self.idxS).evaluate()

            f_mu = torch.matmul(K_xnew_xS, alpha)
            v = torch.linalg.solve(self.LSS, K_xnew_xS.t())

            if noiseless:
                f_Cov = KSS_xnew_xnew - torch.matmul(v.t(),v) #+ 1e-2*torch.eye(xT.shape[0])  #I had to add this Jitter
                f_Cov = torch.from_numpy(nearestPD(f_Cov.numpy()))
            else:
                f_Cov = KSS_xnew_xnew - torch.matmul(v.t(),v) + torch.diag(self.lik_std_noise[idxS_new].pow(2)) + 1e-5*torch.eye(xS.shape[0])
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
_FOLDER = "/home/juanjo/Work_Postdoc/my_codes_postdoc/Dataset_5Cancers/GDSC2_EGFR_PI3K_MAPK_Top5cancers/"
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
        self.N_iter = 40    #number of iterations
        self.which_seed = 35    #change seed to initialise the hyper-parameters
        self.weight = 1.0  #use weights 0.3, 0.5, 1.0 and 2.0
        self.bash = "None"
        self.sel_cancer_Source = 3
        self.sel_cancer_Target = 0
        self.idx_CID_Target = 17#0  #This is just an integer from 0 to max number of CosmicIDs in Target cancer.
        self.which_drug = 2096#1560   #This is the drug we will select as test for the target domain.

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
dict_cancers={0:'GDSC2_EGFR_PI3K_MAPK_Breast_1000FR.csv',1:'GDSC2_EGFR_PI3K_MAPK_COAD_1000FR.csv',
              2:'GDSC2_EGFR_PI3K_MAPK_LUAD.csv',3:'GDSC2_EGFR_PI3K_MAPK_melanoma.csv',4:'GDSC2_EGFR_PI3K_MAPK_SCLC.csv'}

#indx_cancer = np.array([0,1,2,3,4])
indx_cancer_train = np.array([int(config.sel_cancer_Source)])

name_file_cancer = dict_cancers[indx_cancer_train[0]]
name_file_cancer_target = dict_cancers[int(config.sel_cancer_Target)]
print("Source Cancer:",name_file_cancer)
print("Target Cancer:",name_file_cancer_target)

df_to_read = pd.read_csv(_FOLDER + name_file_cancer)#.sample(n=N_CellLines,random_state = rand_state_N)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
df_to_read_target = pd.read_csv(_FOLDER + name_file_cancer_target)#.sample(n=N_CellLines,random_state = rand_state_N)

"Split of data into Training and Testing for the Source and Target domains"
Index_sel = (df_to_read["DRUG_ID"] == 1036) | (df_to_read["DRUG_ID"] == 1061)| (df_to_read["DRUG_ID"] == 1373) \
            | (df_to_read["DRUG_ID"] == 1039) | (df_to_read["DRUG_ID"] == 1560) | (df_to_read["DRUG_ID"] == 1057) \
            | (df_to_read["DRUG_ID"] == 1059)| (df_to_read["DRUG_ID"] == 1062) | (df_to_read["DRUG_ID"] == 2096) \
            | (df_to_read["DRUG_ID"] == 2045)

df_SourceCancer_all = df_to_read[Index_sel]
df_all = df_SourceCancer_all.reset_index().drop(columns=['index'])
df_source = df_all.dropna()

Index_sel_target = (df_to_read_target["DRUG_ID"] == 1036) | (df_to_read_target["DRUG_ID"] == 1061)| (df_to_read_target["DRUG_ID"] == 1373) \
            | (df_to_read_target["DRUG_ID"] == 1039) | (df_to_read_target["DRUG_ID"] == 1560) | (df_to_read_target["DRUG_ID"] == 1057) \
            | (df_to_read_target["DRUG_ID"] == 1059)| (df_to_read_target["DRUG_ID"] == 1062) | (df_to_read_target["DRUG_ID"] == 2096) \
            | (df_to_read_target["DRUG_ID"] == 2045)

df_TargetCancer_all = df_to_read_target[Index_sel_target]
df_all_target = df_TargetCancer_all.reset_index().drop(columns=['index'])
df_all_target = df_all_target.dropna()

"COSMIC_IDs for Breast:"
"The ones below have 9 drugs tested"
"The cell line 946359 (all nores); 749709 (1 resp 1 parcial)  has been tested in 9 of the 10 drugs"
"907046 (2 resp 1 parcial); 1290798 (3 parcial); 905946 (1 resp 1 parcial)"
"908121 (2 resp 1 parcial); 1240172 (non resp); 906826 (2 resp should try this one)"

"The ones below have 8 drugs tested"
"1298157 (1 resp 1 parcial); 910927 (2 resp 1 parcial)"
"910948 (Exp:0,2,6 seed 35)"


"COSMIC_IDs for COAD: 909748, 905961, 905937, 1240123, 1659928 and 909755 with (Exp:1,3,5 seed 35), "

"COSMIC_IDs for LUAD: 906805, 1298537, 724878, 687777, 908475 (Exp:0,3,6 seed 35), "

"COSMIC_IDs for SCLC: 713885, 687985, 906808, 1299062, 910692, 687997, 688015, 1240189, 1322212, 688027  (Exp:2,4,5 seed 35), "

myset_target = set(df_all_target['COSMIC_ID'].values)
myLabels = np.arange(0,myset_target.__len__())
CosmicIDs_All_Target = list(myset_target)
"Here we order the list of target COSMIC_IDs from smallest CosmicID to biggest"
CosmicIDs_All_Target.sort()
CellLine_pos = int(config.idx_CID_Target) #37
print(f"The CosmicID of the selected Target Cell-line: {CosmicIDs_All_Target[CellLine_pos]}")
CosmicID_target = CosmicIDs_All_Target[CellLine_pos] #906826 #910927 #1298157 #906826 #1240172 #908121 #905946 #1290798 #907046 #749709 #946359
df_target = df_all_target[df_all_target['COSMIC_ID']==CosmicID_target].reset_index().drop(columns=['index'])

#idx_train = np.array([0,1,3,4,5,6,7])  #Exp1:3,4,8 ,Exp2 (906826):0,2,6  Exp3 (749709):1,6,8
#idx_test = np.delete(np.arange(0,df_target.shape[0]),idx_train)


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

# Here we just check that from the column index 25 the input features start
start_pos_features = 25
print("first feat Source:",df_all.columns[start_pos_features])
print("first feat Target:",df_all_target.columns[start_pos_features])
assert df_all.columns[start_pos_features] == df_all_target.columns[start_pos_features]

df_feat = df_source_and_target[df_source_and_target.columns[start_pos_features:]]
Names_All_features = df_source_and_target.columns[start_pos_features:]
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
Dnorm_cell = 7  #For GDSC2 this is the number of dose concentrations

"Here we extract the target domain outputs yT"

yT = np.clip(df_target["norm_cells_" + str(1)].values[:, None], 1.0e-9, np.inf)
print(yT.shape)
for i in range(2, Dnorm_cell+1):
    yT = np.concatenate((yT, np.clip(df_target["norm_cells_" + str(i)].values[:, None], 1.0e-9, np.inf)), 1)

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

"Be careful that x starts from 0.111111 for 9 or 5 drug concentrations in GDSC1 dataset"
"but x starts from 0.142857143 for the case of 7 drug concentrations in GDSC2 dataset"
"The function Get_IC50_AUC_Emax is implemented in Utils_SummaryMetrics_KLRelevance.py to extract the summary metrics"
x_lin = np.linspace(0.142857, 1, 1000)
x_real_dose = np.linspace(0.142857, 1, Dnorm_cell)  #Here is Dnorm_cell due to using GDSC2 that has 7 doses
Ydose50,Ydose_res,IC50,AUC,Emax = MyUtils.Get_IC50_AUC_Emax(params_4_sig_target,x_lin,x_real_dose)
AUC = np.array(AUC)[:, None]
IC50 = np.array(IC50)[:, None]
Emax = np.array(Emax)[:, None]

def my_plot(posy,fig_num,Ydose50,Ydose_res,IC50,AUC,Emax,x_lin,x_real_dose,y_train_drug):
    plt.figure(fig_num)
    plt.plot(x_lin, Ydose_res[posy])
    plt.plot(x_real_dose, y_train_drug[posy, :], '.')
    plt.plot(IC50[posy], Ydose50[posy], 'rx')
    plt.plot(x_lin, np.ones_like(x_lin)*Emax[posy], 'r') #Plot a horizontal line as Emax
    plt.title(f"AUC = {AUC[posy]}")
    plt.legend(['Sigmoid4','Observations','IC50','Emax'])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Here we can visualise the values of the GDSC2 dataset with the fitting of Sigmoid4_parameters function"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
posy = 0   #select the location you want to plot, do not exceed the Ytrain length
my_plot(posy,0,Ydose50,Ydose_res,IC50,AUC,Emax,x_lin,x_real_dose,yT)

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

yS_train = np.clip(df_source_sort["norm_cells_" + str(1)].values[:, None], 1.0e-9, np.inf)
print(yS_train.shape)
for i in range(2, Dnorm_cell+1):
    yS_train = np.concatenate((yS_train, np.clip(df_source_sort["norm_cells_" + str(i)].values[:, None], 1.0e-9, np.inf)), 1)

print("yS size: ", yS_train.shape)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Make all variable passed to the model tensor to operate in pytorch"
xT_all_train = xT_train.copy()
yT_all_train = yT_train.copy()
xT_train = torch.from_numpy(xT_train)
yT_train = torch.from_numpy(yT_train)
xS_train = torch.from_numpy(xS_train)
yS_train = torch.from_numpy(yS_train)
xT_test = torch.from_numpy(xT_test)
yT_test = torch.from_numpy(yT_test)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"This is the number of source domains plus target domain."
"In this cancer application we are refering to each domain as each cell-line with the Cosmic_ID"
"There are CosmicID_labels.__len__() cell-lines or CosmicIDs in the Source (Melanoma) + 1 cell-line of the Target"
"Then the total NDomains = CosmicID_labels.__len__() + 1"
NDomains = CosmicID_labels.__len__() + 1 #CosmicID_labels.__len__() contains the CosmicIDs of Source Domains
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
idx_T = [NDomains - 1] * xT_train.shape[0]   #Here we generate the label for the target domain
yST_train = torch.cat([yS_train,yT_train])
xST_train = torch.cat([xS_train,xT_train])
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print(f"Number Nsource Domains: {CosmicID_labels.__len__()}, so total NDomains Source + Target: {NDomains}")
DrugC = list(np.linspace(0.142857,1.0,7))
assert DrugC.__len__() == yS_train.shape[1] and DrugC.__len__() == yT_train.shape[1]
model = BMKMOGaussianProcess(xST_train,yST_train,idxS=idx_S+idx_T,DrugC=DrugC,NDomains=NDomains)
#TODO I need to work on a general way to create a linear combination of Q kernels_ICM

myseed = int(config.which_seed)
torch.manual_seed(myseed)   #Ex1: 15 (run 100 iter)  #Exp2 (906826): 35  (run 100 iter)
with torch.no_grad():
    model.lik_std_noise = torch.nn.Parameter(1.0*torch.randn(NDomains))
    for i_kern in range(model.TLCovariance.kernels.__len__()):
        #model.TLCovariance.kernels[i_kern].adq = 0.2*torch.randn(NDomains)[:,None]
        model.TLCovariance.kernels[i_kern].length = 2.*np.sqrt(xS_train.shape[1])*torch.rand(1)

    model.CoregCovariance[0].lengthscale = 0.1 *torch.rand(1) #1*
    model.CoregCovariance[1].lengthscale = 0.1*torch.rand(1)  #1*
    model.CoregCovariance[2].lengthscale = 0.1*torch.rand(1)  #1*
    #print(model.LambdaDiDj.muDi)
#print(f"Noises std: {model.lik_std_noise}")

"Training process below"
def myTrain(model,x_train,y_train,myLr = 1e-2,Niter = 1):
    optimizer = optim.Adam(model.parameters(),lr=myLr)
    loss_fn = LogMarginalLikelihood()

    for iter in range(Niter):
        # Forward pass
        mu, L = model(x_train)

        # Backprop
        loss = -loss_fn(mu,L,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter==100:  #70
            optimizer.param_groups[0]['lr']=3e-3
        #print(model.TLCovariance.length)
        print(f"i: {iter+1}, Loss: {loss.item()}")

"Train the model with all yT training data"
myTrain(model,xST_train,yST_train,myLr = 3e-2,Niter = int(config.N_iter))
def bypass_params(model_trained,model_cv):
    model_cv.lik_std_noise = model_trained.lik_std_noise
    model_cv.CoregCovariance[0].lengthscale = model_trained.CoregCovariance[0].lengthscale.clone()
    model_cv.CoregCovariance[1].lengthscale = model_trained.CoregCovariance[1].lengthscale.clone()
    model_cv.CoregCovariance[2].lengthscale = model_trained.CoregCovariance[2].lengthscale.clone()

    for i_kern in range(model_trained.TLCovariance.kernels.__len__()):
        model_cv.TLCovariance.kernels[i_kern].length = model_cv.TLCovariance.kernels[i_kern].length.clone()
        model_cv.TLCovariance.kernels[i_kern].adq = model_cv.TLCovariance.kernels[i_kern].adq.clone()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Create a K-fold for cross-validation"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from sklearn.model_selection import KFold, cross_val_score
Ndata = yST_train.shape[0]
Xind = np.arange(Ndata)
nsplits = 5 #Ndata
k_fold = KFold(n_splits=nsplits,random_state=1,shuffle=True)
list_folds = list(k_fold.split(yST_train))
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"Leave one out cross-validation"
Val_LML = LogMarginalLikelihood()
TestLogLoss_All = []
for Nfold in range(0,nsplits):
    model_cv = []
    train_ind, val_ind = list_folds[Nfold]
    idx_ST_numpy = np.array(idx_S + idx_T)
    idx_ST = list(idx_ST_numpy[train_ind])
    yST_train_cv = yST_train[train_ind,:].clone()
    yST_val_cv = yST_train[val_ind, :].clone()
    xST_train_cv = xST_train[train_ind,:].clone()
    xST_val_cv = xST_train[val_ind, :].clone()
    #xT_val_cv = xT_all_train[i:i + 1, :]
    #print(f"shape cv yST train:{yST_train_cv.shape}")
    #print(f"shape cv xST train:{xST_train_cv.shape}")
    print(f"shape cv yST val:{yST_val_cv.shape}")
    print(f"shape cv xST val:{xST_val_cv.shape}")

    "model fit with Cross-val"
    #model_cv = TLMOGaussianProcess(xT_train_cv, yT_train_cv, xS_train, yS_train, idxS=idx_S, DrugC=DrugC, NDomains=NDomains)

    model_cv = BMKMOGaussianProcess(xST_train_cv, yST_train_cv, idxS=idx_ST, DrugC=DrugC, NDomains=NDomains)

    bypass_params(model, model_cv)  #Here we bypass the fitted parameters from the MOGP trained over all data
    myTrain(model_cv, xST_train_cv, yST_train_cv, myLr=1e-5, Niter=1) #Here we could refine hyper-params a bit if wished
    "Here we put the model in prediciton mode"
    model_cv.eval()
    model_cv.Train_mode = False
    DrugCtoPred_cv = list(np.linspace(0.142857, 1, 7))
    with torch.no_grad():
        "NOTE: It is important to validate the model with the noisy prediction"
        "I noticed it is necessary to include the outputs' uncertainty when validating"
        "that is the why noiseless=False, this guarantees including the noise uncertainty of the outputs in preditions"
        "validating with noiseless=True can lead to selecting an underfitted model"
        mpred_cv, Cpred_cv = model_cv(xST_val_cv, DrugC_new=DrugCtoPred_cv, noiseless=False)
        Lpred_cv = torch.linalg.cholesky(Cpred_cv)  #Here we compute Cholesky since Val_LML gets L Cholesky of Cpred
        Val_loss = -Val_LML(mpred_cv, Lpred_cv, yST_val_cv)
        print(f"Val Loss: {Val_loss.item()}")
    TestLogLoss_All.append(Val_loss.numpy())

print(f"Mean cv ValLogLoss: {np.mean(TestLogLoss_All)}")

#TODO: allow the model to receive a variable idxS_new to indicate what are the domains (outputs) we want to predict

#
# #path_home = '/home/juanjo/Work_Postdoc/my_codes_postdoc/'
# path_home = '/rds/general/user/jgiraldo/home/TransferLearning_Results/'
# path_val = path_home+'Jobs_TLMOGP_OneCell_OneDrug_Testing/TargetCancer'+str(config.sel_cancer_Target)+'/Drug_'+str(config.which_drug)+'/CellLine'+str(config.idx_CID_Target)+'_CID'+str(CosmicID_target)+'/'
#
# # check whether directory already exists
# if not os.path.exists(path_val):
#   #os.mkdir(path_val)   #Use this for a single dir
#   os.makedirs(path_val) #Use this for a multiple sub dirs
#
# "Here we save the Validation Log Loss in path_val in order to have a list of different bashes to select the best model"
# f = open(path_val+'Validation.txt', "a")
# f.write(f"\nbash{str(config.bash)}, ValLogLoss:{np.mean(TestLogLoss_All)}, CrossVal_N:{yT_train.shape[0]}")
# f.close()

# "Here we have to assign the flag to change from self.Train_mode = True to False"
# print("check difference between model.eval and model.train")
# model.eval()
# model.Train_mode = False
# plot_test = True
# if plot_test:
#     x_test = xT_test.clone()
#     y_test = yT_test.clone()
#     Name_DrugID_plot = Name_DrugID_test
#     plotname = 'Test'
# else:
#     x_test = xT_train.clone()
#     y_test = yT_train.clone()
#     Name_DrugID_plot = Name_DrugID_train
#     plotname = 'Train'
#
# "The Oversample_N below is to generate equaly spaced drug concentrations between the original 7 drug concentrations"
# "i.e., if Oversample_N = 2: means that each 2 positions we'd have the original drug concentration tested in cell-line"
# "we'd have DrugCtoPred = [0.1428, 0.2142, 0.2857, 0.3571,0.4285,0.4999,0.5714,0.6428,0.7142,0.7857,0.8571,0.9285,1.0]"
# Oversample_N = 15
# DrugCtoPred = list(np.linspace(0.142857,1,7+6*(Oversample_N-1)))
# "Below we refer as exact in the sense that those are the exact location of drug concent. for which we have exact data"
# DrugCtoPred_exact = list(np.linspace(0.142857, 1, 7))
# #DrugCtoPred = list(np.linspace(0.142857,1,28))
# sel_concentr = 0
# with torch.no_grad():
#     #model(x_test, DrugC_new=[DrugC[sel_concentr]], NDomain_sel=2, noiseless=True)
#     mpred, Cpred = model(x_test,DrugC_new = DrugCtoPred, NDomain_sel=54,noiseless=False)
#     "To assess the TestLogLoss we have to also include the noise uncertainty, so we use noiseless=False"
#     mpred_exact, Cpred_exact = model(x_test, DrugC_new=DrugCtoPred_exact, NDomain_sel=54, noiseless=False)
#     Lpred_exact = torch.linalg.cholesky(Cpred_exact)  # Here we compute Cholesky since Val_LML gets L Cholesky of Cpred
#     #Test_loss = -Val_LML(mpred_exact, Lpred_exact, y_test)
#     #print(f"Test Loss: {Test_loss.item()}")
#
# yT_pred = mpred.reshape(DrugCtoPred.__len__(),x_test.shape[0]).T
#
# "Compute AUC of prediction"
# AUC_pred = []
# Ncurves = yT_pred.shape[0]
# for i in range(Ncurves):
#     AUC_pred.append(metrics.auc(DrugCtoPred, yT_pred[i,:]))
# AUC_pred = np.array(AUC_pred)[:,None]
#
# "Compute IC50 of prediction"
# x_dose_new = np.array(DrugCtoPred)
# Ydose50_pred = []
# IC50_pred = []
# Emax_pred = []
# for i in range(Ncurves):
#     y_resp_interp = yT_pred[i,:].clone() #pchip_interpolate(x_dose, y_resp, x_dose_new)
#
#     Emax_pred.append(y_resp_interp[-1])
#
#     res1 = y_resp_interp < 0.507
#     res2 = y_resp_interp > 0.493
#     res_aux = np.where(res1 & res2)[0]
#     if (res1 & res2).sum() > 0:
#         res_IC50 = np.arange(res_aux[0], res_aux[0] + res_aux.shape[0]) == res_aux
#         res_aux = res_aux[res_IC50].copy()
#     else:
#         res_aux = res1 & res2
#
#     if (res1 & res2).sum() > 0:
#         Ydose50_pred.append(y_resp_interp[res_aux].mean())
#         IC50_pred.append(x_dose_new[res_aux].mean())
#     elif y_resp_interp[-1] < 0.5:
#         for dose_j in range(x_dose_new.shape[0]):
#             if (y_resp_interp[dose_j] < 0.5):
#                 break
#         Ydose50_pred.append(y_resp_interp[dose_j])
#         aux_IC50 = x_dose_new[dose_j]  # it has to be a float not an array to avoid bug
#         IC50_pred.append(aux_IC50)
#     else:
#         Ydose50_pred.append(0.5)
#         IC50_pred.append(1.5)
#
# IC50_pred = np.array(IC50_pred)[:,None]
# Emax_pred = np.array(Emax_pred)[:,None]
#
# "Below we use ::Oversample_N in order to obtain the exact locations of drug concentration for which we have data"
# "in order to compute the error between the yT_test (true observed values) and the yT_pred (predicted)"
# if plot_test:
#     Test_MSE = torch.mean((yT_test-yT_pred[:,::Oversample_N])**2)
#     print(f"Test_MSE: {Test_MSE}")
#     AUC_test = AUC[idx_test]
#     print(f"Test AUC_MAE:{np.mean(np.abs(AUC_pred - AUC_test))}")
#     IC50_test = IC50[idx_test]
#     print(f"Test IC50_MSE:{np.mean((IC50_pred - IC50_test)**2)}")
#     Emax_test = Emax[idx_test]
#     print(f"Test Emax_MAE:{np.mean(np.abs(Emax_pred - Emax_test))}")
# else:
#     "Here we just allow the errors of training when we wish to explore their specific values"
#     Train_MSE = torch.mean((yT_train - yT_pred[:, ::Oversample_N]) ** 2)
#     print(f"Train_MSE: {Train_MSE}")
#     AUC_test = AUC[idx_train]
#     print(f"Train AUC_MAE:{np.mean(np.abs(AUC_pred - AUC_test))}")
#     IC50_test = IC50[idx_train]
#     print(f"Train IC50_MSE:{np.mean((IC50_pred - IC50_test) ** 2)}")
#     Emax_test = Emax[idx_train]
#     print(f"Train Emax_MAE:{np.mean(np.abs(Emax_pred - Emax_test))}")
#
# # "Here we save the Test Log Loss metric in the same folder path_val where we had also saved the Validation Log Loss"
# # f = open(path_val + 'Test.txt', "a")
# # f.write(f"\nbash{str(config.bash)}, TestLogLoss:{Test_loss.item()}, IC50_MSE:{np.mean((IC50_pred - IC50_test) ** 2)}, AUC_MAE:{np.mean(np.abs(AUC_pred - AUC_test))}, Emax_MAE:{np.mean(np.abs(Emax_pred - Emax_test))}, CrossVal_N:{yT_train.shape[0]}")
# # f.close()
#
# "Plot the prediction for the test yT"
# from torch.distributions.multivariate_normal import MultivariateNormal
# plt.close('all')
# #plt.switch_backend('agg')
# for i in range(x_test.shape[0]):
#     plt.figure(i+1)
#     plt.ylim([-0.02,1.1])
#     #plt.plot(x,mpred1,'.')
#     plt.plot(DrugCtoPred,yT_pred[i,:],'blue')
#     #plt.plot(IC50_pred[i],0.5,'x')   # Plot an x in predicted IC50 location
#     #plt.plot(x_lin, np.ones_like(x_lin) * Emax_pred[i], 'r')  # Plot a horizontal line as Emax
#     if plot_test:
#         plt.plot(DrugC, yT_test[i,:], 'ro')
#     else:
#         plt.plot(DrugC, yT_train[i, :], 'ro')
#
#     plt.title(f"CosmicID: {CosmicID_target}, {plotname} DrugID: {Name_DrugID_plot[i]}",fontsize=14)
#     plt.xlabel('Dose concentration',fontsize=14)
#     plt.ylabel('Cell viability',fontsize=14)
#     plt.grid(True)
#
#     # for j in range(30):
#     #     i_sample = MultivariateNormal(loc=mpred[:, 0], covariance_matrix=Cpred)
#     #     yT_pred_sample = i_sample.sample().reshape(DrugCtoPred.__len__(), x_test.shape[0]).T
#     #     plt.plot(DrugCtoPred, yT_pred_sample[i, :], alpha=0.1)
#
#     std_pred = torch.sqrt(torch.diag(Cpred)).reshape(DrugCtoPred.__len__(), x_test.shape[0]).T
#     plt.plot(DrugCtoPred, yT_pred[i, :]+2.0*std_pred[i,:], '--b')
#     plt.plot(DrugCtoPred, yT_pred[i, :] - 2.0 * std_pred[i, :], '--b')
#
#     # # check whether directory already exists
#     # path_plot = path_val + 'Test_plot/'
#     # if not os.path.exists(path_plot):
#     #     # os.mkdir(path_val)   #Use this for a single dir
#     #     os.makedirs(path_plot)  # Use this for a multiple sub dirs
#     # plt.savefig(path_plot+'plotbash'+str(config.bash)+'.pdf')
#
