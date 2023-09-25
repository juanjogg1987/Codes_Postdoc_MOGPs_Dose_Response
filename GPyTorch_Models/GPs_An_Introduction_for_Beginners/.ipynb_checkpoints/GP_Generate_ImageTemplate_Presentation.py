import torch
from torch import nn, optim
import gpytorch
import matplotlib.pyplot as plt
from gpytorch.utils.cholesky import psd_safe_cholesky

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
    def forward(self,L,y):
        alpha = torch.linalg.solve(L.t(), torch.linalg.solve(L, y))
        N = y.shape[0]
        return -0.5*torch.matmul(y.t(),alpha)-torch.diag(L).log().sum()-N*0.5*torch.log(torch.tensor([2*torch.pi]))
class GaussianProcess(nn.Module):
    def __init__(self,x,y):
        super().__init__()
        self.x = x
        self.y = y
        self.covariance = gpytorch.kernels.RBFKernel()
        self.Train_mode = True
        self.lik_std_noise = torch.nn.Parameter(torch.tensor([1.0])) #torch.tensor([0.07])
        self.L = torch.eye(y.shape[0])
    def forward(self,x, noiseless = True):
        if self.Train_mode:
            Knn = self.covariance(x).evaluate() #+ 1e-3*torch.eye(x.shape[0])
            Knn_noise = Knn + self.lik_std_noise.pow(2)*torch.eye(Knn.shape[0])
            self.L = torch.linalg.cholesky(Knn_noise)
            #self.L = psd_safe_cholesky(Knn_noise,jitter=1e-5)
            return self.L  #here we might return the mean and covariance (or just covar if mean is zeros)
        else:
            alpha1 = torch.linalg.solve(self.L, self.y)
            alpha = torch.linalg.solve(self.L.t(), alpha1)
            K_xnew_x = self.covariance(x,self.x).evaluate()
            K_xnew_xnew = self.covariance(x).evaluate()
            f_mu = torch.matmul(K_xnew_x,alpha)
            v = torch.linalg.solve(self.L,K_xnew_x.t())
            if noiseless:
                f_Cov = K_xnew_xnew - torch.matmul(v.t(),v) #+ 1e-5*torch.eye(x.shape[0])  #I had to add this Jitter
                f_Cov = torch.from_numpy(nearestPD(f_Cov.numpy()))
            else:
                f_Cov = K_xnew_xnew - torch.matmul(v.t(), v) + self.lik_std_noise.pow(2) * torch.eye(x.shape[0]) #+ 1e-5*torch.eye(x.shape[0])
                f_Cov = torch.from_numpy(nearestPD(f_Cov.numpy()))
            return f_mu, f_Cov

Nseed = 3
torch.manual_seed(Nseed)
import random
random.seed(Nseed)
x = torch.rand(50,1)
y = torch.exp(1*x)*torch.sin(10*x)*torch.cos(3*x) + 0.3*torch.rand(*x.shape)

model = GaussianProcess(x,y)
#model.covariance.lengthscale=0.1
print(model(x))

"Training process below"
myLr = 1e-2
Niter = 500
optimizer = optim.Adam(model.parameters(),lr=myLr)
loss_fn = LogMarginalLikelihood()

for iter in range(Niter):
    # Forward pass
    L = model(x)

    # Backprop
    loss = -loss_fn(L,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item()}")


"Here we have to assign the flag to change from self.Train_mode = True to False"
print("check difference between model.eval and model.train")
model.eval()
model.Train_mode = False
x_test = torch.linspace(0, 1, 100)[:,None]
with torch.no_grad():
    #mpred1,Cpred1 = model(x)
    mpred, Cpred = model(x_test,noiseless=False)

plt.figure(1)
#plt.plot(x,mpred1,'.')

from torch.distributions.multivariate_normal import MultivariateNormal
for i in range(50):
    i_sample = MultivariateNormal(loc=mpred[:,0], covariance_matrix=Cpred)
    plt.plot(x_test,i_sample.sample(),alpha = 0.2)

plt.plot(x,y,'k.')
