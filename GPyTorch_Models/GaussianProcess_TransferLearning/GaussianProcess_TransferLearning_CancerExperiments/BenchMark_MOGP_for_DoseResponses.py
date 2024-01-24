import torch
from torch import nn, optim
import gpytorch
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from importlib import reload
import TransferLearning_Kernels
reload(TransferLearning_Kernels)
from TransferLearning_Kernels import TL_Kernel_var

from numpy import linalg as la
import numpy as np

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
    def __init__(self,xS,yS,idxS,DrugC,NDomains):
        super().__init__()
        self.Douts = yS.shape[1]
        self.DrugC = torch.Tensor(DrugC)[:,None] #We treat the Drug Concentrations as a float tensor vector [N,1]
        assert self.DrugC.shape[0] == yS.shape[1]  #DrugC length should be equal to D number of outputs
        #self.xT = torch.kron(torch.ones(self.Douts, 1),xT)  #This is to replicate xT as per the D number of outputs
        self.xS = torch.kron(torch.ones(self.Douts, 1),xS)  #This is to replicate xS as per the D number of outputs
        #self.yT = yT.T.reshape(-1, 1)  # This is to vectorise by staking the columns (vect(yT))
        self.yS = yS.T.reshape(-1,1) #This is to vectorise by staking the columns (vect(yS))
        #self.DrugC_xT = torch.kron(self.DrugC,torch.ones(xT.shape[0], 1)) #Rep. Concentr. similar to coreginalisation
        self.DrugC_xS = torch.kron(self.DrugC,torch.ones(xS.shape[0], 1)) #Rep. Concentr. similar to coreginalisation
        self.idxS = idxS * self.Douts #Replicate the Source domain index as per the number of outputs
        #self.idxT = [NDomains - 1] * xT.shape[0] * self.Douts #Replicate the target domain index as per the number of outputs
        #self.all_y = torch.cat([self.yS, self.yT])
        assert NDomains == (max(idxS)+1) #This is to assert that the Domains meant by user coincide with max label
        self.TLCovariance = TL_Kernel_var(NDomains=NDomains) #gpytorch.kernels.RBFKernel()
        self.CoregCovariance = gpytorch.kernels.RBFKernel()
        self.Train_mode = True
        self.lik_std_noise = torch.nn.Parameter(1.0*torch.ones(NDomains)) #torch.tensor([0.07])
        #self.lik_std_noise = 0.05 * torch.ones(NDomains)
        #self.mu_star = torch.zeros(self.yT.shape) #mu has the shape of the new replicated along the outputs yT
        self.LSS = torch.eye(self.yS.shape[0])
        #self.all_L = torch.eye(self.all_y.shape[0])
        "TODO: I need to erase all related to xT since now we only have all data inside xS"

    def forward(self,xS, DrugC_new = None,noiseless = True):
        if self.Train_mode:
            xS = torch.kron(torch.ones(self.Douts, 1), xS)
            assert (xS == self.xS).sum() == xS.shape[0] #This is just to check if the xS to init the model is the same
            # Here we compute the Covariance matrices between source-target, source-source and target domains
            #KTS = self.CoregCovariance(self.DrugC_xT,self.DrugC_xS).evaluate()*self.TLCovariance(xT,self.xS,idx1=self.idxT,idx2=self.idxS).evaluate()
            KSS = self.CoregCovariance(self.DrugC_xS,self.DrugC_xS).evaluate()*self.TLCovariance(xS,idx1=self.idxS).evaluate()
            #KTT = self.CoregCovariance(self.DrugC_xT,self.DrugC_xT).evaluate()*self.TLCovariance(self.xT,idx1=self.idxT).evaluate()

            # Here we include the respective noise terms associated to each domain
            CSS = KSS + torch.diag(self.lik_std_noise[self.idxS].pow(2))
            #CTT = KTT + torch.diag(self.lik_std_noise[self.idxT].pow(2))

            # The code below aim to correct for numerical instabilities when CSS becomes Non-PSD
            if not isPD_torch(CSS):
                CSS_aux = CSS.clone()
                with torch.no_grad():
                    CSS_aux = torch.from_numpy(nearestPD(CSS_aux.numpy()))
                CSS = 0.0*CSS + CSS_aux  #This operation aims to keep the gradients working over lik_std_noise

            self.LSS = torch.linalg.cholesky(CSS)
            #alphaSS1 = torch.linalg.solve(self.LSS, self.yS)
            #alphaSS = torch.linalg.solve(self.LSS.t(), alphaSS1)

            # Compute the mean of the conditional distribution p(yT|XT,XS,yS)
            #self.mu_star = torch.matmul(KTS,alphaSS)
            # Compute the Covariance of the conditional distribution p(yT|XT,XS,yS)
            #vTT = torch.linalg.solve(self.LSS, KTS.t())
            #C_star = CTT - torch.matmul(vTT.t(),vTT) #+ 1e-4*torch.eye(xT.shape[0])  #jitter?
            #self.L = torch.linalg.cholesky(C_star)
            #TODO I should use a prior with mean in 1.0 so that the model predicts values in 1 when is uncertain!!
            self.mu_star = torch.zeros_like(self.yS)
            return self.mu_star, self.LSS  # here we return the mean and covariance
        else:
            "We replicate the target domain index as per the number of drug concentration we want to test"
            "notice that it is not limited to D number of outputs, but actually the number of concentrations"
            # Here we compute the full covariance of xS and xT together
            xST = torch.cat([self.xS, self.xT])
            idxST = self.idxS + self.idxT
            self.DrugC_xSxT = torch.cat([self.DrugC_xS, self.DrugC_xT])
            all_K_xST = self.CoregCovariance(self.DrugC_xSxT).evaluate() * self.TLCovariance(xST, idx1=idxST).evaluate()
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
            idxT = [self.TLCovariance.NDomains - 1] * xT.shape[0] * NewDouts
            DrugC_xT = torch.kron(DrugC_new, torch.ones(xT.shape[0], 1))
            "Be careful with operation using xT.shape, from here it changes the original shape"
            xT = torch.kron(torch.ones(NewDouts, 1), xT)
            alpha1 = torch.linalg.solve(self.all_L, self.all_y)
            alpha = torch.linalg.solve(self.all_L.t(), alpha1)
            KTT_xnew_xnew = self.CoregCovariance(DrugC_xT).evaluate() * self.TLCovariance(xT, idx1=idxT).evaluate()
            xST = torch.cat([self.xS, self.xT])
            idxST = self.idxS+self.idxT

            # Rep. Concentr. similar to coreginalisation
            K_xnew_xST = self.CoregCovariance(DrugC_xT,self.DrugC_xSxT).evaluate() * self.TLCovariance(xT,xST, idx1=idxT,idx2=idxST).evaluate()

            f_mu = torch.matmul(K_xnew_xST, alpha)
            v = torch.linalg.solve(self.all_L, K_xnew_xST.t())

            if noiseless:
                f_Cov = KTT_xnew_xnew - torch.matmul(v.t(),v) #+ 1e-2*torch.eye(xT.shape[0])  #I had to add this Jitter
                f_Cov = torch.from_numpy(nearestPD(f_Cov.numpy()))
            else:
                f_Cov = KTT_xnew_xnew - torch.matmul(v.t(),v) + torch.diag(self.lik_std_noise[idxT].pow(2)) + 1e-5*torch.eye(xT.shape[0])
                f_Cov = torch.from_numpy(nearestPD(f_Cov.numpy()))
            return f_mu, f_Cov

# Nseed = 3
# torch.manual_seed(Nseed)
# import random
# random.seed(Nseed)
# x1 = torch.rand(200,1)
# x2 = x1[0:30]
# y1 = torch.exp(1*x1)*torch.sin(10*x1)*torch.cos(3*x1) + 0.2*torch.rand(*x1.shape)
# y2 = torch.exp(1.5*x2)*torch.sin(8*x2)*torch.cos(2.7*x2) + 0.3*torch.rand(*x2.shape)
# idx1 = [0]*y1.shape[0]
#
# "Here (x2,y2) is Target domain and (x1,y1) is source domain"
# model = TLGaussianProcess(x2,y2,x1,y1,idx1,NDomains=2)
# #model.covariance.length=0.1
# print(model(x2))

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Nsize = 150
Nseed = 2
torch.manual_seed(Nseed)
import math
import random
random.seed(Nseed)
#train_x1 = torch.linspace(0,1,Nsize)
x1 = torch.rand(Nsize)[:,None]
torch.manual_seed(Nseed)
random.seed(Nseed)

indx = torch.arange(50,90)
indx0 = torch.arange(0,100)
#print(indx)
x0 = x1[indx0]
x2 = x1[indx]
#x1 = x0

y2 = x2*torch.sin(-8*x2 * (2 * math.pi))*torch.cos(0.3+2*x2 * (2 * math.pi)) + torch.randn(x2.size()) * 0.05 #+ x2
Many_x2 = torch.rand(500)[:,None]

y0 = torch.cos(7*x0 * (2 * math.pi)) + torch.randn(x0.size()) * 0.2
y1 = torch.sin(4*x1 * (2 * math.pi))*torch.sin(3*x1 * (2 * math.pi)) + torch.randn(x1.size()) * 0.1
#train_y2 = -torch.cos(train_x1*train_x2 * (2 * math.pi)) + torch.randn(train_x2.size()) * 0.2
Many_y2 = Many_x2*torch.sin(-8*Many_x2 * (2 * math.pi))*torch.cos(0.3+2*Many_x2 * (2 * math.pi)) + torch.randn(Many_x2.size()) * 0.05 #+ Many_x2
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

y0 = torch.cat([y0,0.2*y0],1)
y1 = torch.cat([y1,0.2*y1+0.3*(y1**2)],1)
y2 = torch.cat([y2,-0.4*y2],1)
Many_y2 = torch.cat([Many_y2,-0.4*Many_y2],1)
train_xS = torch.cat([x0,x1])
train_yS = torch.cat([y0,y1])
idx0 = [0]*y0.shape[0]
idx1 = [1]*y1.shape[0]
NDomains = 2 #3
DrugC = [0.1,0.5]
assert DrugC.__len__() == y0.shape[1] and DrugC.__len__() == y1.shape[1] and DrugC.__len__() == y2.shape[1]
model = TLMOGaussianProcess(train_xS,train_yS,idxS=idx0+idx1,DrugC=DrugC,NDomains=NDomains)
plt.plot(Many_x2,Many_y2[:,0],'y.',alpha=0.5)
plt.plot(Many_x2,Many_y2[:,1],'m.',alpha=0.5)
# model.covariance.length=0.05
torch.manual_seed(5)
with torch.no_grad():
    model.lik_std_noise=torch.nn.Parameter(torch.randn(NDomains))
    model.TLCovariance.length = 0.1*torch.rand(NDomains)[:,None]
    print(model.TLCovariance.length)
print(f"Noises std: {model.lik_std_noise}")

"Training process below"
myLr = 5e-3
Niter = 200
optimizer = optim.Adam(model.parameters(),lr=myLr)
loss_fn = LogMarginalLikelihood()

for iter in range(Niter):
    # Forward pass
    mu, L = model(train_xS)

    # Backprop
    loss = -loss_fn(mu,L,train_yS)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(model.TLCovariance.length)
    print(f"Loss: {loss.item()}")


# "Here we have to assign the flag to change from self.Train_mode = True to False"
# print("check difference between model.eval and model.train")
# model.eval()
# model.Train_mode = False
# x_test = torch.linspace(0, 1, 200)[:,None]
# sel_concentr = 0
# with torch.no_grad():
#     #mpred1,Cpred1 = model(x)
#     mpred, Cpred = model(x_test,DrugC_new = [DrugC[sel_concentr]],noiseless=True)
#
# plt.figure(1)
# #plt.plot(x,mpred1,'.')
# plt.plot(x_test,mpred,'blue')
# plt.plot(x2,y2,'k.')
# plt.plot(x2,y2[:,sel_concentr],'r.')
# from torch.distributions.multivariate_normal import MultivariateNormal
# for i in range(50):
#     i_sample = MultivariateNormal(loc=mpred[:, 0], covariance_matrix=Cpred)
#     plt.plot(x_test,i_sample.sample(),alpha = 0.1)
# #plt.plot(x_test, mpred+2*torch.sqrt(torch.diag(Cpred)[:,None]),'c--')
# #plt.plot(x_test, mpred-2*torch.sqrt(torch.diag(Cpred)[:,None]),'c--')
#

"TODO: add restrictions to the likelihood noise, and set to optimise the variance instead of std!!!"
