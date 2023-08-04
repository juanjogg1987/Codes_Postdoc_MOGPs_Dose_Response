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
        self.covariance = TL_Kernel_var(NDomains=NDomains) #gpytorch.kernels.RBFKernel()
        self.Train_mode = True
        self.lik_std_noise = torch.nn.Parameter(0.1*torch.ones(NDomains)) #torch.tensor([0.07])
        self.mu_star = torch.zeros(yT.shape)
        self.L = torch.eye(yT.shape[0])
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
            alpha1 = torch.linalg.solve(self.L, self.yT-self.mu_star)
            alpha = torch.linalg.solve(self.L.t(), alpha1)

            idxT = [self.covariance.NDomains - 1] * xT.shape[0]
            KTS_xnew_xS = self.covariance(xT, self.xS, idx1=idxT, idx2=self.idxS).evaluate()
            KTS = self.covariance(self.xT, self.xS, idx1=self.idxT, idx2=self.idxS).evaluate()


            KTT_xnew_xT = self.covariance(xT,self.xT, idx1=idxT,idx2=self.idxT).evaluate()

            # We need to build f_Cov = Kstar(xnew,xnew) - Kstar(xnew,xT)*(C^-1)*(Kstar(xnew,xT))^T
            # where Kstar(.,.) = KTT(.,.) - KTS(.,xS)*(CSS)^-1*KTS(.,xS)^T
            KTT_xnew_xnew = self.covariance(xT,idx1=idxT).evaluate()

            vTT_xS_xnew = torch.linalg.solve(self.LSS, KTS_xnew_xS.t())
            Kstar_xnew_xnew = KTT_xnew_xnew-torch.matmul(vTT_xS_xnew.t(),vTT_xS_xnew)

            vTT_xS_xT = torch.linalg.solve(self.LSS, KTS.t())
            Kstar_xnew_xT = KTT_xnew_xT - torch.matmul(vTT_xS_xnew.t(),vTT_xS_xT)

            vstar_xS_xnew = torch.linalg.solve(self.L, Kstar_xnew_xT.t())

            # We need to build f_mu = mu_star(xnew) - Kstar(xnew,xT)*(C^-1)*(yT-mu_star(xT))
            # here mu_star(xT) is simply self.mu_star

            alphaSS1 = torch.linalg.solve(self.LSS, self.yS)
            alphaSS = torch.linalg.solve(self.LSS.t(), alphaSS1)
            mu_star_new = torch.matmul(KTS_xnew_xS, alphaSS)

            f_mu = mu_star_new - torch.matmul(Kstar_xnew_xT, alpha)

            if noiseless:
                #f_Cov = K_xnew_xnew - torch.matmul(v.t(),v) + 1e-5*torch.eye(x.shape[0])  #I had to add this Jitter
                f_Cov = Kstar_xnew_xnew - torch.matmul(vstar_xS_xnew.t(), vstar_xS_xnew) + 1e-4*torch.eye(xT.shape[0])
            else:
                #f_Cov = K_xnew_xnew - torch.matmul(v.t(), v) + self.lik_std_noise.pow(2) * torch.eye(x.shape[0])
                f_Cov = Kstar_xnew_xnew - torch.matmul(vstar_xS_xnew.t(), vstar_xS_xnew) + torch.diag(self.lik_std_noise[idxT].pow(2))
            return f_mu, f_Cov

Nseed = 3
torch.manual_seed(Nseed)
import random
random.seed(Nseed)
x1 = torch.rand(100,1)
x2 = x1
y1 = torch.exp(1*x1)*torch.sin(10*x1)*torch.cos(3*x1) + 0.3*torch.rand(*x1.shape)
y2 = torch.exp(1.5*x2)*torch.sin(8*x2)*torch.cos(2.7*x2) + 0.1*torch.rand(*x2.shape)
idx1 = [0]*y1.shape[0]
"Here (x2,y2) is Target domain and (x1,y1) is source domain"
model = TLGaussianProcess(x2,y2,x1,y1,idx1,NDomains=2)
#model.covariance.lengthscale=0.1
print(model(x2))

"Training process below"
myLr = 1e-2
Niter = 500
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
x_test = torch.linspace(0, 1, 100)[:,None]
with torch.no_grad():
    #mpred1,Cpred1 = model(x)
    mpred, Cpred = model(x_test,noiseless=False)

plt.figure(1)
#plt.plot(x,mpred1,'.')
plt.plot(x_test,mpred,'blue')
plt.plot(x2,y2,'k.')
from torch.distributions.multivariate_normal import MultivariateNormal
# for i in range(50):
#     i_sample = MultivariateNormal(loc=model.mu_star[:,0], covariance_matrix=torch.matmul(model.L,model.L.t()))
#     #i_sample = MultivariateNormal(loc=mpred[:, 0], covariance_matrix=Cpred)
#     plt.plot(x2,i_sample.sample(),'.',alpha = 0.2)
plt.plot(x2.numpy(),model.mu_star.detach().numpy(),'.',alpha = 0.2)


