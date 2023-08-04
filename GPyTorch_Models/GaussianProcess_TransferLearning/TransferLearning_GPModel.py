import torch
from torch import nn, optim
import gpytorch
import matplotlib.pyplot as plt
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
        alpha = torch.linalg.solve(L.t(), torch.linalg.solve(L, y-mu))
        N = y.shape[0]
        return -0.5*torch.matmul(y.t()-mu.t(),alpha)-torch.diag(L).log().sum()-N*0.5*torch.log(torch.tensor([2*torch.pi]))
class TLGaussianProcess(nn.Module):
    'This model expects the data from source and target domains with:'
    'idxS as a list of labels with integer values between 0 to NDomains-2'
    'by deafult the idxT for the target domain will be labeled as NDomains-1'
    'i.e., if NDomains = 3, the idxS is a list with values between 0 and 1, by default idxT has all values equal to 2'
    'The model expects the input and output of the target domain: xT, yT'
    'The input and output of the source domains: xS, yS with a list idxS with their labels as per their domains'
    'NOTE: The model would expect the data to be sorted according to the idxS'
    'For instance, idxS = [0,0,1,2,2,2] implies NDomains = 4 with N = 2 in first domain,'
    'N = 1 in second domain, and N = 3 in third domain, if N = 2 in fourth (target) domain, then idxT = [3,3]'
    def __init__(self,xT,yT,xS,yS,idxS,NDomains):
        super().__init__()
        self.xT = xT
        self.yT = yT
        self.xS = xS
        self.yS = yS
        self.idxS = idxS
        self.idxT = [NDomains - 1] * xT.shape[0]
        self.all_y = torch.cat([self.yS, self.yT])
        self.covariance = TL_Kernel_var(NDomains=NDomains) #gpytorch.kernels.RBFKernel()
        self.Train_mode = True
        self.lik_std_noise = torch.nn.Parameter(1*torch.ones(NDomains)) #torch.tensor([0.07])
        #self.lik_std_noise = 0.05 * torch.ones(NDomains)
        self.mu_star = torch.zeros(yT.shape)
        self.L = torch.eye(yT.shape[0])
        self.all_L = torch.eye(self.all_y.shape[0])
        "TODO: I might need the self.LSS also here in order to be able to predict without optimising"


    def forward(self,xT, noiseless = True):
        if self.Train_mode:
            # Here we compute the Covariance matrices between source-target, source-source and target domains
            KTS = self.covariance(xT,self.xS,idx1=self.idxT,idx2=self.idxS).evaluate()
            KSS = self.covariance(self.xS,idx1=self.idxS).evaluate()
            KTT = self.covariance(self.xT,idx1=self.idxT).evaluate()

            # Here we include the respective noise terms associated to each domain
            CSS = KSS + torch.diag(self.lik_std_noise[self.idxS].pow(2))
            CTT = KTT + torch.diag(self.lik_std_noise[self.idxT].pow(2))

            #Knn_noise = Knn + self.lik_std_noise.pow(2) * torch.eye(Knn.shape[0])
            with torch.no_grad():
                CSS = torch.from_numpy(nearestPD(CSS.numpy()))
            self.LSS = torch.linalg.cholesky(CSS)
            alphaSS1 = torch.linalg.solve(self.LSS, self.yS)
            alphaSS = torch.linalg.solve(self.LSS.t(), alphaSS1)

            # Compute the mean of the conditional distribution p(yT|XT,XS,yS)
            self.mu_star = torch.matmul(KTS,alphaSS)
            # Compute the Covariance of the conditional distribution p(yT|XT,XS,yS)
            vTT = torch.linalg.solve(self.LSS, KTS.t())
            C_star = CTT - torch.matmul(vTT.t(),vTT) #+ 1e-4*torch.eye(xT.shape[0])  #jitter?
            self.L = torch.linalg.cholesky(C_star)

            # Here we compute the full covariance of xS and xT together
            xST = torch.cat([self.xS, self.xT])
            idxST= self.idxS+self.idxT
            all_K_xST = self.covariance(xST, idx1=idxST).evaluate()
            all_K_xST_noise = all_K_xST + torch.diag(self.lik_std_noise[idxST].pow(2)) #+ 0.1*torch.eye(xST.shape[0]) #Jitter?
            with torch.no_grad():
                all_K_xST_noise = torch.from_numpy(nearestPD(all_K_xST_noise.numpy()))
            self.all_L = torch.linalg.cholesky(all_K_xST_noise) #+ 1e-4*torch.eye(xST.shape[0])
            return self.mu_star, self.L  # here we return the mean and covariance
        else:
            "TODO all this part of the prediction of the model"
            idxT = [self.covariance.NDomains - 1] * xT.shape[0]
            alpha1 = torch.linalg.solve(self.all_L, self.all_y)
            alpha = torch.linalg.solve(self.all_L.t(), alpha1)
            KTT_xnew_xnew = self.covariance(xT, idx1=idxT).evaluate()
            xST = torch.cat([self.xS, self.xT])
            idxST = self.idxS+self.idxT
            K_xnew_xST = self.covariance(xT,xST, idx1=idxT,idx2=idxST).evaluate()

            f_mu = torch.matmul(K_xnew_xST, alpha)
            v = torch.linalg.solve(self.all_L, K_xnew_xST.t())

            if noiseless:
                f_Cov = KTT_xnew_xnew - torch.matmul(v.t(),v) #+ 1e-2*torch.eye(xT.shape[0])  #I had to add this Jitter
                f_Cov = torch.from_numpy(nearestPD(f_Cov.numpy()))
            else:
                f_Cov = KTT_xnew_xnew - torch.matmul(v.t(),v) + 10*torch.diag(self.lik_std_noise[idxT].pow(2)) + 1e-5*torch.eye(xT.shape[0])
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
Nsize = 100
Nseed = 2
torch.manual_seed(Nseed)
import math
import random
random.seed(Nseed)
#train_x1 = torch.linspace(0,1,Nsize)
x1 = torch.rand(Nsize)[:,None]
torch.manual_seed(Nseed)
random.seed(Nseed)

indx = torch.arange(0,30)
indx0 = torch.arange(0,50)
print(indx)
x0 = x1[indx0]
x2 = x1[indx]

y2 = x2*torch.sin(-8*x2 * (2 * math.pi))*torch.cos(0.3+2*x2 * (2 * math.pi)) + torch.randn(x2.size()) * 0.2 + x2
Many_x2 = torch.rand(500)

y0 = torch.cos(7*x0 * (2 * math.pi)) + torch.randn(x0.size()) * 0.1
y1 = torch.sin(4*x1 * (2 * math.pi))*torch.sin(3*x1 * (2 * math.pi)) + torch.randn(x1.size()) * 0.1
#train_y2 = -torch.cos(train_x1*train_x2 * (2 * math.pi)) + torch.randn(train_x2.size()) * 0.2
Many_y2 = Many_x2*torch.sin(-8*Many_x2 * (2 * math.pi))*torch.cos(0.3+2*Many_x2 * (2 * math.pi)) + torch.randn(Many_x2.size()) * 0.1 + Many_x2
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

train_xS = torch.cat([x0,x1])
train_yS = torch.cat([y0,y1])
idx0 = [0]*y0.shape[0]
idx1 = [1]*y1.shape[0]
model = TLGaussianProcess(x2,y2,train_xS,train_yS,idxS=idx0+idx1,NDomains=3)
plt.plot(Many_x2,Many_y2,'m.')
model.covariance.length=0.01
#model.lik_std_noise=0.1
"Training process below"
myLr = 1e-2
Niter = 1000
optimizer = optim.Adam(model.parameters(),lr=myLr)
loss_fn = LogMarginalLikelihood()

for iter in range(Niter):
    # Forward pass
    mu, L = model(x2)

    # Backprop
    loss = -loss_fn(mu,L,y2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item()}")


"Here we have to assign the flag to change from self.Train_mode = True to False"
print("check difference between model.eval and model.train")
model.eval()
model.Train_mode = False
x_test = torch.linspace(0, 1, 500)[:,None]
with torch.no_grad():
    #mpred1,Cpred1 = model(x)
    mpred, Cpred = model(x_test,noiseless=False)

plt.figure(1)
#plt.plot(x,mpred1,'.')
plt.plot(x_test,mpred,'blue')
plt.plot(x2,y2,'k.')
from torch.distributions.multivariate_normal import MultivariateNormal
# for i in range(50):
#     i_sample = MultivariateNormal(loc=mpred[:, 0], covariance_matrix=Cpred)
#     plt.plot(x_test,i_sample.sample(),alpha = 0.2)
plt.plot(x_test, mpred+2*torch.diag(Cpred)[:,None],'c--')
plt.plot(x_test, mpred-2*torch.diag(Cpred)[:,None],'c--')


"TODO: add restrictions to the likelihood noise, and set to optimise the variance instead of std!!!"
