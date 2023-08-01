import torch
from torch import nn, optim
import gpytorch
import matplotlib.pyplot as plt
from TransferLearning_Kernels import TL_Kernel_var

class LogMarginalLikelihood(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,mu,L,y):
        alpha = torch.linalg.solve(L.t(), torch.linalg.solve(L, y-mu))
        N = y.shape[0]
        return -0.5*torch.matmul(y.t()-mu.t(),alpha)-torch.diag(L).log().sum()-N*0.5*torch.log(torch.tensor([2*torch.pi]))
class GaussianProcess(nn.Module):
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
        self.covariance = TL_Kernel_var(NDomains=NDomains) #gpytorch.kernels.RBFKernel()
        self.Train_mode = True
        self.lik_std_noise = torch.nn.Parameter(torch.ones(NDomains)) #torch.tensor([0.07])
        self.mu_star = torch.zeros(y.shape)
        self.L = torch.eye(y.shape[0])
    def forward(self,xT, noiseless = True):
        if self.Train_mode:
            # Here we compute the Covariance matrices between source-target, source-source and target domains
            idxT = [self.covariance.NDomains-1]*xT.shape[0]
            KTS = self.covariance(xT,self.xS,idx1=idxT,idx2=self.idxS).evaluate()
            KSS = self.covariance(self.xS,idx1=self.idxS).evaluate()
            KTT = self.covariance(self.xT,idx1=idxT).evaluate()

            # Here we include the respective noise terms associated to each domain
            CSS = KSS + torch.diag(self.lik_std_noise[self.idxS].pow(2))
            CTT = KTT + torch.diag(self.lik_std_noise[idxT].pow(2))

            #Knn_noise = Knn + self.lik_std_noise.pow(2) * torch.eye(Knn.shape[0])
            self.LSS = torch.linalg.cholesky(CSS)
            alphaSS1 = torch.linalg.solve(self.LSS, self.yS)
            alphaSS = torch.linalg.solve(self.LSS.t(), alphaSS1)

            # Compute the mean of the conditional distribution p(yT|XT,XS,yS)
            self.mu_star = torch.matmul(KTS,alphaSS)
            # Compute the Covariance of the conditional distribution p(yT|XT,XS,yS)
            vTT = torch.linalg.solve(self.LSS, KTS.t())
            C_star = CTT - torch.matmul(vTT.t(),vTT)
            self.L = torch.linalg.cholesky(C_star)
            return self.mu_star, self.L  # here we return the mean and covariance
        else:
            "TODO all this part of the prediction of the model"
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
