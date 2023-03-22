import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch

plt.close('all')
typeinput = "unif"   #select unif or lin
which_model = "IndMOGP"
generative = "ICM_3"
np.random.seed(16)
Nall = 200  #200
# x1 = np.linspace(0,1,Nall)
# #x1 = np.linspace(0,2,Nall)**(1.0/5.0)
# #x2 = np.linspace(-0.5,0.8,Nall)**2
# x2 = np.exp(np.linspace(-0.2,0.8,Nall))
# x3 = np.linspace(-1,1,Nall)**3
# #x4 = np.linspace(0,4,Nall)**(1.0/5.0)
#
# x4 = np.log(np.linspace(0.01,3,Nall))
# # x1 = np.linspace(0,1,Nall)
# # x2 = np.linspace(0.2,0.8,Nall)**2
# # x3 = np.linspace(-1,1,Nall)**3
# # x4 = np.linspace(0,4,Nall)**(1.0/5.0)

x5 = 0.1*np.random.randn(Nall)

if typeinput == "unif":
    x1= np.random.uniform(-1,1,Nall) #np.linspace(0,1,Nall)
    x2= np.random.uniform(-1,1,Nall)
    x3= np.random.uniform(-1,1,Nall)
    x4= np.random.uniform(-1,1,Nall)
elif typeinput == "lin":
    x1 = np.linspace(0,1,Nall)
    x2 = np.exp(np.linspace(-0.2,0.8,Nall))
    x3 = np.linspace(-1,1,Nall)**3
    x4 = np.log(np.linspace(0.01,3,Nall))

#x1.sort();x2.sort();x3.sort();x4.sort()

X = np.concatenate((x1[:,None],x2[:,None],x3[:,None],x4[:,None]),1)
#X = x1.copy()

#Dim = 5
Q = 4

k5 = gpytorch.kernels.RBFKernel(ard_num_dims=4)  #if ARD include ard_num_dims=5
k6 = gpytorch.kernels.RBFKernel(ard_num_dims=4)
k7 = gpytorch.kernels.RBFKernel(ard_num_dims=4)
k8 = gpytorch.kernels.RBFKernel(ard_num_dims=4)

# k1.lengthscale = [0.0701,2.0,3.0,1.0]#2.0
# k2.lengthscale = [2.0,0.0603,1.0,3.0]#1.3
# k3.lengthscale = [3.0,1.0,0.0702,2.00]#0.8
# k4.lengthscale = [1.0,3.0,2.00,0.0704]#0.1

if generative =="ICM_1":
    Dtasks = 3
    print("Generative ICM")
    k1 = gpytorch.kernels.RBFKernel()  # if ARD include ard_num_dims=5
    k2 = gpytorch.kernels.RBFKernel()
    k3 = gpytorch.kernels.RBFKernel()
    k4 = gpytorch.kernels.RBFKernel()
    k1.lengthscale = 0.2 #1.2#2.0
    k2.lengthscale = 0.2 #0.03#1.3
    k3.lengthscale = 0.2 #0.5#0.8
    k4.lengthscale = 0.2 #0.07#0.1
elif generative =="ICM_2":
    Dtasks = 3
    print("Generative ICM 2")
    k1 = gpytorch.kernels.RBFKernel(ard_num_dims=4)  # if ARD include ard_num_dims=5
    k2 = gpytorch.kernels.RBFKernel(ard_num_dims=4)
    k3 = gpytorch.kernels.RBFKernel(ard_num_dims=4)
    k4 = gpytorch.kernels.RBFKernel(ard_num_dims=4)
    k1.lengthscale = [1.2, 10.0, 10.0, 10.0]  # 2.0
    k2.lengthscale = [10.0, 1.0, 10.0, 10.0]  # 1.3
    k3.lengthscale = [10.0, 10.0, 0.5, 10.0]  # 0.8
    k4.lengthscale = [10.0, 10.0, 10.0, 0.2]  # 0.1
elif generative =="ICM_3":
    Dtasks = 2
    print("Generative ICM 3")
    k1 = gpytorch.kernels.RBFKernel(ard_num_dims=4)  # if ARD include ard_num_dims=5
    k2 = gpytorch.kernels.RBFKernel(ard_num_dims=4)
    k3 = gpytorch.kernels.RBFKernel(ard_num_dims=4)
    k4 = gpytorch.kernels.RBFKernel(ard_num_dims=4)
    k1.lengthscale = [1.2, 10.0, 10.0, 10.0]  # 2.0
    k2.lengthscale = [10.0, 0.8, 10.0, 10.0]  # 1.3
    k3.lengthscale = [10.0, 10.0, 0.5, 10.0]  # 0.8
    k4.lengthscale = [10.0, 10.0, 10.0, 0.2]  # 0.1
elif generative =="ICM_4":
    Dtasks = 2
    print("Generative ICM 4")
    k1 = gpytorch.kernels.RBFKernel(ard_num_dims=4)  # if ARD include ard_num_dims=5
    k2 = gpytorch.kernels.RBFKernel(ard_num_dims=4)
    k3 = gpytorch.kernels.RBFKernel(ard_num_dims=4)
    k4 = gpytorch.kernels.RBFKernel(ard_num_dims=4)
    k1.lengthscale = [1.2, 10.0, 10.0, 10.0]  # 2.0
    k2.lengthscale = [10.0, 0.9, 10.0, 10.0]  # 1.3
    k3.lengthscale = [10.0, 10.0, 0.5, 10.0]  # 0.8
    k4.lengthscale = [10.0, 10.0, 10.0, 0.2]  # 0.1
else:
    print("Generative LMC")
    k1 = gpytorch.kernels.RBFKernel(ard_num_dims=4)  # if ARD include ard_num_dims=5
    k2 = gpytorch.kernels.RBFKernel(ard_num_dims=4)
    k3 = gpytorch.kernels.RBFKernel(ard_num_dims=4)
    k4 = gpytorch.kernels.RBFKernel(ard_num_dims=4)
    k1.lengthscale = [1.2,10.0,10.0,10.0]#2.0
    k2.lengthscale = [10.0,0.8,10.0,10.0]#1.3
    k3.lengthscale = [10.0,10.0,0.5,10.0]#0.8
    k4.lengthscale = [10.0,10.0,10.0,0.2]#0.1

    k5.lengthscale = [0.2, 10.0, 10.0, 10.0]  # 2.0
    k6.lengthscale = [10.0, 0.3, 10.0, 10.0]  # 1.3
    k7.lengthscale = [10.0, 10.0, 0.6, 10.0]  # 0.8
    k8.lengthscale = [10.0, 10.0, 10.0, 1.0]  # 0.1

K1 = k1(torch.tensor(X)).numpy()
K2 = k2(torch.tensor(X)).numpy()
K3 = k3(torch.tensor(X)).numpy()
K4 = k4(torch.tensor(X)).numpy()

K5 = k5(torch.tensor(X)).numpy()
K6 = k6(torch.tensor(X)).numpy()
K7 = k7(torch.tensor(X)).numpy()
K8 = k8(torch.tensor(X)).numpy()

Kern_q = [K1,K2,K3,K4]
Kern2_q = [K5,K6,K7,K8]

F = np.zeros((Nall,Dtasks))
u_q = np.zeros((Nall,Q))
#W = np.array([[1.0,-0.5,0.3,-1.5],[0.01,-0.3,1.1,0.7],[-0.6,0.5,-0.6,0.7],[1.1,-1.3,-0.6,0.08]])
if generative =="ICM_3":
    W = np.array([[1.0,1.0,0.0,0.0],[0.0,1.0,1.0,1.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])
elif generative =="ICM_4":
    W = np.array([[1.0,1.0,1.0,1.0],[1.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
else:
    W = np.array([[1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0]])
#W = np.ones((Dtasks,Q))
np.random.seed(15)
for q in range(Q):
    #u_q[:,q] = np.random.multivariate_normal(np.zeros(Nall), Kern_q[q], 1).flatten()
    for d in range(Dtasks):
        if True:#d<2:
            u_q[:, q] = np.random.multivariate_normal(np.zeros(Nall), Kern_q[q], 1).flatten()
        #else:
        #    u_q[:, q] = np.random.multivariate_normal(np.zeros(Nall), Kern2_q[q], 1).flatten()
        print("d,q",W[d,q])
        F[:,d] += W[d,q]*u_q[:,q]

plt.figure(0)
plt.plot(F,'x')
plt.figure(10)
plt.plot(u_q,'x')
#plt.plot(F[:,1],'rx')

#index_N = np.random.permutation(Nall)
index_N = np.arange(0,Nall)

#Xall = np.concatenate((x1[:,None],x2[:,None],x3[:,None],x4[:,None],x5[:,None]),1)
Xall = np.concatenate((x1[:,None],x2[:,None],x3[:,None],x4[:,None]),1)
#Xall = x1[:,None].copy()

standard_X = False
if standard_X:
    Xall = (Xall-np.mean(Xall,0))/np.std(Xall,0)

#Xall = x1[:,None].copy()
#Xall = np.concatenate((x2[:,None],x4[:,None]),1)
#Xall = np.concatenate((x2[:,None],x4[:,None],x3[:,None],x1[:,None]),1)

Yall = F + 0.01*np.random.randn(x1.shape[0],Dtasks)

standard_Y = False
if standard_Y:
    Yall = (Yall-np.mean(Yall,0))/np.std(Yall,0)
    F = (F - np.mean(F, 0)) / np.std(F, 0)

Ntrain = Nall#//3
Xtrain = Xall[index_N[0:Ntrain],:].copy()
Ytrain = Yall[index_N[0:Ntrain],:].copy()
#Yclean = F[index_N[0:Ntrain],:].copy()

#Xtrain = Xall[index_N[Ntrain:2*Ntrain],:].copy()
#Ytrain = Yall[index_N[Ntrain:2*Ntrain],:].copy()
#Yclean = F[index_N[Ntrain:2*Ntrain],:].copy()

Xval = Xall[index_N[0:Ntrain],:].copy()
Yval = Yall[index_N[0:Ntrain],:].copy()
Yclean = F[index_N[0:Ntrain],:].copy()


standard_Y = True
if standard_X:
    Xtrain = (Xtrain-np.mean(Xtrain,0))/np.std(Xtrain,0)


import math
import torch
import gpytorch
from matplotlib import pyplot as plt
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"SEED"
torch.manual_seed(49)  #23 for Ind  #use 49 for MOGP ICM3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
rank = Ytrain.shape[1]   #Rank for the MultitaskKernel, we make rank equal to number of tasks
num_tasks = Ytrain.shape[1]
Dim = Xtrain.shape[1]
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
train_x = torch.from_numpy(Xtrain.astype(np.float32))
train_y = torch.from_numpy(Ytrain.astype(np.float32))
val_x = torch.from_numpy(Xval.astype(np.float32))
val_y = torch.from_numpy(Yval.astype(np.float32))
#val_x = torch.from_numpy(Xtrain.astype(np.float32))
#val_y = torch.from_numpy(Ytrain.astype(np.float32))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

myseed = 15  #int(config.which_seed)
np.random.seed(myseed)

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )

        size_dims = Dim
        #mykern = gpytorch.kernels.MaternKernel(nu=0.5,ard_num_dims=size_dims)
        mykern = gpytorch.kernels.RBFKernel(ard_num_dims=size_dims)
        #mykern = gpytorch.kernels.RBFKernel()

        self.covar_module = gpytorch.kernels.MultitaskKernel(
            mykern, num_tasks=num_tasks, rank=rank
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

num_latents = Q

class MultitaskGPModelLMC(gpytorch.models.ApproximateGP):
    def __init__(self):
        # Let's use a different set of inducing points for each latent function
        #inducing_points = torch.rand(num_latents, 30, Dim)
        inducing_points = torch.from_numpy(np.repeat(Xtrain[None,np.random.permutation(Ntrain)[0:50],:],num_latents,0).astype(np.float32))

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=Dtasks,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        size_dims = Dim
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=size_dims,batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks,noise_constraint=gpytorch.constraints.Interval(1.0e-6,1.0e-1),rank=num_tasks)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if which_model == "IndMOGP":
    print("Running IndMOGP...")
    model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood)
elif which_model=="SpMOGP":
    print("Running SparseMOGP...")
    #model = MultitaskGPModel(train_x, train_y, likelihood)
    model = MultitaskGPModelLMC()
else:
    print("Running MOGP...")
    # model = MultitaskGPModel(train_x, train_y, likelihood)
    model = MultitaskGPModel(train_x, train_y, likelihood)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
num_epochs = 3000 #int(config.N_iter_epoch)

model.train()
likelihood.train()

if which_model=="SpMOGP":
    # Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.005)
else:
    # Our loss object We're using the Exact Marginal Log Likelohood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Ntrain,_ = Ytrain.shape
show_each = 100 #Ntrain//train_loader.batch_size
#refine_lr = [0.01,0.005,0.001,0.0005]
refine_lr = [0.007,0.005,0.001,0.0005]
#refine_lr = [0.005,0.001,0.0005,0.0001]

if which_model=="SpMOGP":
    refine_lr = [0.03, 0.007, 0.003, 0.001]

refine_num_epochs = [num_epochs,int(num_epochs*0.5),int(num_epochs*0.2),int(num_epochs*0.2)]
for Nrefine in range(len(refine_lr)):
    print(f"\nRefine Learning Rate {Nrefine}; lr={refine_lr[Nrefine]}")
    for g in optimizer.param_groups:
        g['lr'] = refine_lr[Nrefine]

    for j in range(refine_num_epochs[Nrefine]):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        if j%(show_each)==0:
            print(f"Iter {j}, Loss {loss}")

        loss.backward()
        optimizer.step()
    #break
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Set into eval mode
model.eval()
likelihood.eval()

# Initialize plots
# fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #test_x = torch.linspace(0, 1, 51)
    predictions = likelihood(model(val_x))
    mean = predictions.mean
    #lower, upper = predictions.confidence_region()
    lower = mean - 2*torch.sqrt(predictions.variance)
    upper = mean + 2*torch.sqrt(predictions.variance)

for task in range(num_tasks):
    # Plot training data as black stars
    plt.figure(task+1)
    plt.plot(val_y[:, task].detach().numpy(), 'k.')
    plt.plot(Yclean[:, task], '-r')
    # Predictive mean as blue line
    plt.plot(mean[:, task].numpy(), '.b')
    # Shade in confidence
    plt.plot(lower[:, task].numpy(), 'c--')
    plt.plot(upper[:, task].numpy(), 'c--')
    #ax.fill_between(lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
    #plt.ylim([-0.1, 1.3])
    plt.legend(['Observed Data', 'y', 'mean_pred','2std'])
    plt.title(f'Task {task + 1} {which_model}')


print("MSE:",np.mean((mean.numpy()-val_y.numpy())**2))
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"KL Relevance for MOGP"
from scipy.linalg import cholesky,cho_solve
from GPy.util import linalg

#Kullback-Leibler Divergence between Gaussian distributions
def KLD_Gaussian(m1,V1,m2,V2,use_diag=False):
    Dim = m1.shape[0]
    # Cholesky decomposition of the covariance matrix
    if use_diag:
        L1 = np.diag(np.sqrt(np.diag(V1)))
        L2 = np.diag(np.sqrt(np.diag(V2)))
        V2_inv = np.diag(1.0/np.diag(V2))
    else:
        L1 = cholesky(V1, lower=True)
        L2 = cholesky(V2, lower=True)
        V2_inv, _  = linalg.dpotri(np.asfortranarray(L2))
        #V2_inv = np.linalg.inv(V2)
    #print(V2_inv)

    KL = 0.5 * np.sum(V2_inv * V1) + 0.5 * np.dot((m1-m2).T, np.dot(V2_inv, (m1-m2))) \
         - 0.5 * Dim + 0.5 * 2. * np.sum(np.log(np.abs(np.diag(L2)))) - 0.5 * 2. * np.sum(np.log(np.abs(np.diag(L1))))
    return np.abs(KL)  #This is to avoid any negative due to numerical instability

def KLD_Gaussian_NoChol(m1,V1,m2,V2,use_diag=False):
    Dim = m1.shape[0]
    #print("shape m1", m1.shape)
    # Cholesky decomposition of the covariance matrix
    if use_diag:
        V1 = np.diag(np.sqrt(np.diag(V1)))
        V2 = np.diag(np.sqrt(np.diag(V2)))
        V2_inv = np.diag(1.0/np.diag(V2))
    else:
        #V2_inv, _  = linalg.dpotri(np.asfortranarray(L2))
        V2_inv = np.linalg.inv(V2)
    #print(V2_inv)

    KL = 0.5 * np.sum(V2_inv * V1) + 0.5 * np.dot((m1-m2).T, np.dot(V2_inv, (m1-m2))) \
         - 0.5 * Dim + 0.5 * np.log(np.linalg.det(V2)) - 0.5 * np.log(np.linalg.det(V1))
    return KL  #This is to avoid any negative due to numerical instability

N,P = train_x.shape   #Do we need to merge the train with val in here?

relevance = np.zeros((N,P))
delta = 1.0e-4
jitter = 1.0e-15
x_plus = np.zeros((1,P))

#which_p = int(config.feature)
print(f"Analysing {P} Features...")
for p in range(P):
    for n in range(N):
        #x_plus = X[n,:].copy()
        x_plus = train_x[n:n+1, :].clone()
        x_minus = train_x[n:n + 1, :].clone()
        x_plus[0,p] = x_plus[0,p]+delta
        x_minus[0, p] = x_minus[0, p] - delta

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # test_x = torch.linspace(0, 1, 51)
            predict_xn = likelihood(model(train_x[n:n+1,:]))
            predict_xn_delta = likelihood(model(x_plus))
            predict_xn_delta_min = likelihood(model(x_minus))

            m1 = predict_xn.mean        #np.ones((D, 1))  # The dimension of this is related to the number of Outputs D
            m2 = predict_xn_delta.mean  #np.ones((D, 1))  # The dimension of this is related to the number of Outputs D
            m2_minus = predict_xn_delta_min.mean  # np.ones((D, 1))  # The dimension of this is related to the number of Outputs D
            #np.random.seed(1)
            use_diag = False

            if use_diag:
                V1 = np.diag(predict_xn.variance.numpy()[0,:])  # np.dot(A.T, A)  # The dimension of this is related to the number of Outputs D
                V2 = np.diag(predict_xn_delta.variance.numpy()[0,:])  # The dimension of this is related to the number of Outputs D
                V2_minus = np.diag(predict_xn_delta_min.variance.numpy()[0,:])  # The dimension of this is related to the number of Outputs D
            else:
                V1 = predict_xn.covariance_matrix.numpy()  #np.dot(A.T, A)  # The dimension of this is related to the number of Outputs D
                V2 = predict_xn_delta.covariance_matrix.numpy()  # The dimension of this is related to the number of Outputs D
                V2_minus = predict_xn_delta_min.covariance_matrix.numpy()  # The dimension of this is related to the number of Outputs D


            #Below I got rid of the 2 inside sqrt(2*KL), the author's code does not use the 2.0 inside.
            if False:#which_model=='IndMOGP':
                outp = 1
                KL_plus = np.sqrt(KLD_Gaussian(m1[:,outp:outp+1].numpy().T, V1[outp:outp+1,outp:outp+1], m2[:,outp:outp+1].numpy().T, V2[outp:outp+1,outp:outp+1],use_diag=use_diag) + jitter)  # In code the authors don't use the Mult. by 2
                KL_minus = np.sqrt(KLD_Gaussian(m1[:,outp:outp+1].numpy().T, V1[outp:outp+1,outp:outp+1], m2_minus[:,outp:outp+1].numpy().T, V2_minus[outp:outp+1,outp:outp+1],use_diag=use_diag) + jitter)  # In code the authors don't use the Mult. by 2
                relevance[n, p] = 0.5 * (KL_plus + KL_minus) / delta
            else:
                KL_plus = np.sqrt(KLD_Gaussian(m1.numpy().T,V1,m2.numpy().T,V2,use_diag=use_diag)+jitter) #In code the authors don't use the Mult. by 2
                KL_minus = np.sqrt(KLD_Gaussian(m1.numpy().T, V1, m2_minus.numpy().T,V2_minus,use_diag=use_diag)+jitter)  # In code the authors don't use the Mult. by 2
                relevance[n, p] = 0.5*(KL_plus+KL_minus)/delta
                #relevance[n, p] = KL_plus  / delta
#print(relevance)
mean_KLRel = np.mean(relevance,0)
print(f"Relevance of Features Normalised:\n {mean_KLRel/max(mean_KLRel)}")