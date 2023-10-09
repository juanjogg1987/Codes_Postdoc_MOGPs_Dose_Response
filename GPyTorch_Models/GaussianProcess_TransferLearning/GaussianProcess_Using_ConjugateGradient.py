import torch
from torch import nn, optim
import gpytorch
import matplotlib.pyplot as plt
from gpytorch.utils.cholesky import psd_safe_cholesky
from importlib import reload

import Utils_ToScale_MOGPs as MyUtils
reload(MyUtils)


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
    def forward(self,Knn_noise,y):
        #alpha = torch.linalg.solve(L.t(), torch.linalg.solve(L, y))
        N = y.shape[0]
        Knn_i_y, log_det_Knn = MyUtils.CG_Lanczos(Knn_noise,y,t = 100,p_iter = 30)
        #return -0.5*torch.matmul(y.t(),alpha)-torch.diag(L).log().sum()-N*0.5*torch.log(torch.tensor([2*torch.pi]))
        return -0.5*torch.matmul(y.t(),Knn_i_y)-0.5*log_det_Knn-N*0.5*torch.log(torch.tensor([2*torch.pi]))
class GaussianProcess(nn.Module):
    def __init__(self,x,y):
        super().__init__()
        self.x = x
        self.y = y
        #self.covariance = gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.GreaterThan(1e-2))
        self.covariance = gpytorch.kernels.RBFKernel()
        self.Train_mode = True
        #self.lik_std_noise = torch.tensor([0.3])#torch.nn.Parameter(torch.tensor([1.0])) #torch.tensor([0.07])
        self.lik_std_noise = torch.nn.Parameter(torch.tensor([0.3])) #torch.tensor([0.07])
        self.L = torch.eye(y.shape[0])
        self.Knn_noise = torch.eye(y.shape[0])
    def forward(self,x, noiseless = True):
        if self.Train_mode:
            print(f"lik_std:{self.lik_std_noise}")
            print(f"length:{self.covariance.lengthscale}")
            Knn = self.covariance(x).evaluate() #+ 1e-5*torch.eye(x.shape[0])
            #N = x.shape[0]
            #Knn = torch.zeros(N, N)
            # dK_dtheta = torch.zeros(N, N)
            # for i in range(N):
            #     for j in range(N):
            #         element = self.covariance(x[i],x[j]).evaluate()
            #         Knn[i, j] = element
            #         element.backward(retain_graph=True)
            #         #print(self.covariance.raw_lengthscale.grad)
            #         dK_dtheta[i, j] = self.covariance.raw_lengthscale.grad
            # print(dK_dtheta)
            self.Knn_noise = Knn + self.lik_std_noise.pow(2)*torch.eye(Knn.shape[0])
            #self.L = torch.linalg.cholesky(self.Knn_noise)
            #self.L = psd_safe_cholesky(Knn_noise,jitter=1e-5)
            return self.Knn_noise  #here we might return the mean and covariance (or just covar if mean is zeros)
        else:
            #self.L = torch.linalg.cholesky(self.Knn_noise)
            #alpha1 = torch.linalg.solve(self.L, self.y)
            #alpha = torch.linalg.solve(self.L.t(), alpha1)
            K_xnew_x = self.covariance(x,self.x).evaluate()
            K_xnew_xnew = self.covariance(x).evaluate()
            #f_mu = torch.matmul(K_xnew_x, alpha)
            Kxx_i_y, _ = MyUtils.CG_Lanczos(self.Knn_noise, y, t=100, p_iter=100,tolerance=1.0e-1)
            f_mu = torch.matmul(K_xnew_x, Kxx_i_y)
            #v = torch.linalg.solve(self.L,K_xnew_x.t())
            #Kxx_i_Kx_xnew = torch.linalg.solve(self.Knn_noise,K_xnew_x.t())
            Kxx_i_Kx_xnew,_ = MyUtils.CG_Lanczos(self.Knn_noise, K_xnew_x.t(), t=100, p_iter=100,tolerance=1.0e-1)
            if noiseless:
                #f_Cov = K_xnew_xnew - torch.matmul(v.t(),v) #+ 1e-5*torch.eye(x.shape[0])  #I had to add this Jitter
                f_Cov = K_xnew_xnew - torch.matmul(K_xnew_x, Kxx_i_Kx_xnew)  # + 1e-5*torch.eye(x.shape[0])  #I had to add this Jitter
                f_Cov = torch.from_numpy(nearestPD(f_Cov.numpy()))
            else:
                f_Cov = K_xnew_xnew - torch.matmul(K_xnew_x, Kxx_i_Kx_xnew)  #+ 1e-5*torch.eye(x.shape[0])
                f_Cov = torch.from_numpy(nearestPD(f_Cov.numpy())) + self.lik_std_noise.pow(2) * torch.eye(x.shape[0])
                #f_Cov = torch.from_numpy(nearestPD(f_Cov.numpy()))
            return f_mu, f_Cov

Nseed = 5
torch.manual_seed(Nseed)
import random
random.seed(Nseed)
x = torch.rand(60,1)
y = torch.exp(1*x)*torch.sin(30*x)*torch.cos(3*x) + 0.1*torch.randn(*x.shape)

model = GaussianProcess(x,y)
#model.lik_std_noise = torch.nn.Parameter(torch.tensor([0.7]))
model.covariance.lengthscale=0.1
print(model(x))

"Training process below"
myLr = 1.0e-3
Niter = 1000
optimizer = optim.Adam(model.parameters(),lr=myLr)
loss_fn = LogMarginalLikelihood()

for iter in range(Niter):


    # Forward pass
    L = model(x)

    # Backprop
    loss = -loss_fn(L,y)
    optimizer.zero_grad()
    loss.backward()
    #print(f"lenth before:{model.covariance.lengthscale}")
    try:
        #print(f"grad:{model.covariance.raw_lengthscale.grad}")
        assert model.covariance.raw_lengthscale.grad.isnan() != True
        optimizer.step()
        #assert model.covariance.lengthscale.isnan() != True
        print(f"Loss: {loss.item()}")
    except:
        print("Except IN")
        print(f"lenth:{model.covariance.lengthscale}")
        break


"Here we have to assign the flag to change from self.Train_mode = True to False"
print("check difference between model.eval and model.train")
model.eval()
model.Train_mode = False
x_test = torch.linspace(0, 1, 200)[:,None]
with torch.no_grad():
    #mpred1,Cpred1 = model(x)
    mpred, Cpred = model(x_test,noiseless=False)

plt.figure(1)
#plt.plot(x,mpred1,'.')

from torch.distributions.multivariate_normal import MultivariateNormal
for i in range(50):
    i_sample = MultivariateNormal(loc=mpred[:,0], covariance_matrix=Cpred)
    plt.plot(x_test,i_sample.sample(),alpha = 0.2)

plt.plot(x,y,'r.',markersize=10)
plt.plot(x_test,mpred[:,0],'b--',alpha = 0.8)
plt.plot(x_test,mpred[:,0]+2.0*torch.sqrt(torch.diag(Cpred)),'b--',alpha = 0.8)  #This is to plot the standard dev
plt.plot(x_test,mpred[:,0]-2.0*torch.sqrt(torch.diag(Cpred)),'b--',alpha = 0.8)  #This is to plot the standard dev