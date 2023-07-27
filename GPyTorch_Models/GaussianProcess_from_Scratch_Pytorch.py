import torch
from torch import nn, optim
import gpytorch
import matplotlib.pyplot as plt

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
            Knn = self.covariance(x).evaluate()
            Knn_noise = Knn + self.lik_std_noise.pow(2)*torch.eye(Knn.shape[0])
            self.L = torch.linalg.cholesky(Knn_noise)
            return self.L  #here we might return the mean and covariance (or just covar if mean is zeros)
        else:
            alpha1 = torch.linalg.solve(self.L, self.y)
            alpha = torch.linalg.solve(self.L.t(), alpha1)
            K_xnew_x = self.covariance(x,self.x).evaluate()
            K_xnew_xnew = self.covariance(x).evaluate()
            f_mu = torch.matmul(K_xnew_x,alpha)
            v = torch.linalg.solve(self.L,K_xnew_x.t())
            if noiseless:
                f_Cov = K_xnew_xnew - torch.matmul(v.t(),v) + 1e-5*torch.eye(x.shape[0])  #I had to add this Jitter
            else:
                f_Cov = K_xnew_xnew - torch.matmul(v.t(), v) + self.lik_std_noise.pow(2) * torch.eye(x.shape[0])
            return f_mu, f_Cov

Nseed = 3
torch.manual_seed(Nseed)
import random
random.seed(Nseed)
x = torch.rand(20,1)
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
    mpred, Cpred = model(x_test,noiseless=True)

plt.figure(1)
#plt.plot(x,mpred1,'.')

from torch.distributions.multivariate_normal import MultivariateNormal
for i in range(50):
    i_sample = MultivariateNormal(loc=mpred[:,0], covariance_matrix=Cpred)
    plt.plot(x_test,i_sample.sample(),alpha = 0.2)

plt.plot(x,y,'k.')
