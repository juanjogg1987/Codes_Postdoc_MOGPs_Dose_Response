import numpy as np
import matplotlib.pyplot as plt
Nall = 20  #400

which_toy = "toy3"

if which_toy =="toy1":
    x2_all = np.linspace(0,1,Nall)
    x1_all = np.linspace(0.2,0.8,Nall)**2
    x3_all = np.linspace(-1,1,Nall)**3

    np.random.seed(15)
    index_N = np.random.permutation(Nall)

    x2 = x2_all[index_N[0:Nall//2]]
    x1 = x1_all[index_N[0:Nall//2]]
    x3 = x3_all[index_N[0:Nall//2]]

    x2_val = x2_all[index_N[Nall//2:]]
    x1_val = x1_all[index_N[Nall//2:]]
    x3_val = x3_all[index_N[Nall//2:]]

    #x1.sort();x2.sort();x3.sort()

    np.random.seed(15)
    x4 = 0.5*np.random.randn(x1.shape[0])
    x4_val = 0.5*np.random.randn(x1_val.shape[0])

    """"""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""
    Xtrain = np.concatenate((x1[:, None], x2[:, None], x3[:, None], x4[:, None]), 1)
    Xval = np.concatenate((x1_val[:, None], x2_val[:, None], x3_val[:, None], x4_val[:, None]), 1)
    # Xtrain = np.concatenate((x1[:,None],x2[:,None],x3[:,None]),1)
    # Xval = np.concatenate((x1_val[:,None],x2_val[:,None],x3_val[:,None]),1)
    """"""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""
    y1_clean = 1.5*np.sin(5*x1)*np.cos(3*x2) + 0.8*np.sin(10*x2) + 0.1*np.sin(15*x3)
    y2_clean = 1.5*np.sin(5*x1) + 0.8*np.sin(10*x2)*np.cos(3*x2)
    y3_clean = 1.5*np.sin(5*x3)

    y1_clean_val = 1.5*np.sin(5*x1_val)*np.cos(3*x2_val) + 0.8*np.sin(10*x2_val) + 0.1*np.sin(15*x3_val)
    y2_clean_val = 1.5*np.sin(5*x1_val) + 0.8*np.sin(10*x2_val)*np.cos(3*x2_val)
    y3_clean_val = 1.5*np.sin(5*x3_val)
elif which_toy=="toy2":
    # x2_all = np.linspace(0, 1, Nall)
    # x1_all = np.linspace(0.2, 0.8, Nall) ** 2
    # x3_all = np.linspace(-1, 1, Nall) ** 3
    # x4_all = np.exp(np.linspace(-1, 1, Nall))

    x2_all = np.random.uniform(0,1,Nall) #np.linspace(0, 1, Nall)
    x1_all = np.random.uniform(0.2,0.8,Nall)**2 #np.linspace(0.2, 0.8, Nall) ** 2
    x3_all = np.random.uniform(-1,1,Nall)**3 #np.linspace(-1, 1, Nall) ** 3
    x4_all = np.exp(np.random.uniform(-1,1,Nall)) #np.exp(np.linspace(-1, 1, Nall))

    np.random.seed(15)
    index_N = np.random.permutation(Nall)

    x2 = x2_all[index_N[0:Nall // 2]]
    x1 = x1_all[index_N[0:Nall // 2]]
    x3 = x3_all[index_N[0:Nall // 2]]
    x4 = x4_all[index_N[0:Nall // 2]]

    x2_val = x2_all[index_N[Nall // 2:]]
    x1_val = x1_all[index_N[Nall // 2:]]
    x3_val = x3_all[index_N[Nall // 2:]]
    x4_val = x4_all[index_N[Nall // 2:]]

    # x1.sort();x2.sort();x3.sort()

    np.random.seed(15)
    x5 = 0.5 * np.random.randn(x1.shape[0])
    x5_val = 0.5 * np.random.randn(x1_val.shape[0])

    """"""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""
    Xtrain = np.concatenate((x1[:, None], x2[:, None], x3[:, None], x4[:, None], x5[:, None]), 1)
    Xval = np.concatenate((x1_val[:, None], x2_val[:, None], x3_val[:, None], x4_val[:, None], x5_val[:, None]), 1)
    #Xtrain = np.concatenate((x1[:,None],x3[:,None],x2[:,None],x4[:,None]),1)
    #Xval = np.concatenate((x1_val[:,None],x3_val[:,None],x2_val[:,None],x4_val[:,None]),1)
    #Xtrain = x1[:,None].copy()
    #Xval = x1_val[:, None].copy()
    """"""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""
    y1_clean = 1.5 * np.sin(5 * x1) * np.cos(3 * x2) + 0.8 * np.sin(10 * x2) + 0.1 * np.sin(15 * x3)
    y2_clean = 1.5 * np.sin(5 * x1) + 0.8 * np.sin(10 * x2) * np.cos(3 * x2) - 0.5 *np.cos(8*x4)
    y3_clean = 1.5 * np.sin(5 * x3) - 0.5*np.cos(8*x4)

    y1_clean_val = 1.5 * np.sin(5 * x1_val) * np.cos(3 * x2_val) + 0.8 * np.sin(10 * x2_val) + 0.1 * np.sin(15 * x3_val)
    y2_clean_val = 1.5 * np.sin(5 * x1_val) + 0.8 * np.sin(10 * x2_val) * np.cos(3 * x2_val)- 0.5 *np.cos(8*x4_val)
    y3_clean_val = 1.5 * np.sin(5 * x3_val)- 0.5*np.cos(8*x4_val)
elif which_toy=="toy3":
    # x2_all = np.linspace(0, 1, Nall)
    # x1_all = np.linspace(0.2, 0.8, Nall) ** 2
    # x3_all = np.linspace(-1, 1, Nall) ** 3
    # x4_all = np.exp(np.linspace(-1, 1, Nall))

    x2_all = np.random.uniform(0,1,Nall) #np.linspace(0, 1, Nall)
    x1_all = np.random.uniform(0.2,0.8,Nall)**2 #np.linspace(0.2, 0.8, Nall) ** 2
    x3_all = np.random.uniform(-1,1,Nall)**3 #np.linspace(-1, 1, Nall) ** 3
    x4_all = np.exp(np.random.uniform(-1,1,Nall)) #np.exp(np.linspace(-1, 1, Nall))

    np.random.seed(15)
    index_N = np.random.permutation(Nall)

    x2 = x2_all[index_N[0:Nall // 2]]
    x1 = x1_all[index_N[0:Nall // 2]]
    x3 = x3_all[index_N[0:Nall // 2]]
    x4 = x4_all[index_N[0:Nall // 2]]

    x2_val = x2_all[index_N[Nall // 2:]]
    x1_val = x1_all[index_N[Nall // 2:]]
    x3_val = x3_all[index_N[Nall // 2:]]
    x4_val = x4_all[index_N[Nall // 2:]]

    # x1.sort();x2.sort();x3.sort()

    np.random.seed(15)
    x5 = 0.05 * np.random.randn(x1.shape[0])
    x5_val = 0.05 * np.random.randn(x1_val.shape[0])

    """"""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""
    Xtrain = np.concatenate((x1[:, None], x2[:, None], x3[:, None], x4[:, None], x5[:, None]), 1)
    Xval = np.concatenate((x1_val[:, None], x2_val[:, None], x3_val[:, None], x4_val[:, None], x5_val[:, None]), 1)
    #Xtrain = np.concatenate((x2[:,None],x3[:,None]),1)
    #Xval = np.concatenate((x2_val[:,None],x3_val[:,None]),1)
    #Xtrain = x1[:,None].copy()
    #Xval = x1_val[:, None].copy()
    """"""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""
    y1_clean = 1.5 * np.sin(5 * x1) * np.cos(3 * x2) + 0.8 * np.sin(10 * x2) + 0.1 * np.sin(15 * x3)
    y2_clean = 1.5 * np.sin(5 * x1) + 0.8 * np.sin(10 * x2) * np.cos(3 * x2) - 0.5 *np.cos(8*x4)
    y3_clean = 1.5 * np.sin(5 * x3) - 0.5*np.cos(8*x4)

    y1_clean_val = 1.5 * np.sin(5 * x1_val) * np.cos(3 * x2_val) + 0.8 * np.sin(10 * x2_val) + 0.1 * np.sin(15 * x3_val)
    y2_clean_val = 1.5 * np.sin(5 * x1_val) + 0.8 * np.sin(10 * x2_val) * np.cos(3 * x2_val)- 0.5 *np.cos(8*x4_val)
    y3_clean_val = 1.5 * np.sin(5 * x3_val)- 0.5*np.cos(8*x4_val)

y1 = y1_clean + 0.05*np.random.randn(x1.shape[0])
y2 = y2_clean + 0.05*np.random.randn(x1.shape[0])
y3 = y3_clean + 0.05*np.random.randn(x1.shape[0])

y1_val = y1_clean_val + 0.05*np.random.randn(x1_val.shape[0])
y2_val = y2_clean_val + 0.05*np.random.randn(x1_val.shape[0])
y3_val = y3_clean_val + 0.05*np.random.randn(x1_val.shape[0])

Ytrain = np.concatenate((y1[:,None],y2[:,None],y3[:,None]),1)
Yclean = np.concatenate((y1_clean[:,None],y2_clean[:,None],y3_clean[:,None]),1)
Yclean_val = np.concatenate((y1_clean_val[:,None],y2_clean_val[:,None],y3_clean_val[:,None]),1)

Yval = np.concatenate((y1_val[:,None],y2_val[:,None],y3_val[:,None]),1)

standard_X = False
if standard_X:
    Xtrain = (Xtrain-np.mean(Xtrain,0))/np.std(Xtrain,0)


import math
import torch
import gpytorch
from matplotlib import pyplot as plt
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"SEED"
torch.manual_seed(12)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
num_tasks = Ytrain.shape[1]
rank = num_tasks   # Rank for the MultitaskKernel
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



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Note: When using ExactMOGP you have to use rank in likelihood also to avoid problem with the std in predictions"
#likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks,noise_constraint=gpytorch.constraints.Interval(1.0e-5,1.0e-1),rank=num_tasks)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model = MultitaskGPModel(train_x, train_y, likelihood)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
num_epochs = 2000 #int(config.N_iter_epoch)

model.train()
likelihood.train()

# Our loss object We're using the Exact Marginal Log Likelohood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
Ntrain,_ = Ytrain.shape
show_each = 100 #Ntrain//train_loader.batch_size
refine_lr = [0.01,0.005,0.001,0.0005]
#refine_lr = [0.03,0.008,0.004,0.0002]
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
    lower, upper = predictions.confidence_region()

plt.close('all')
for task in range(num_tasks):
    # Plot training data as black stars
    plt.figure(task)
    plt.plot(val_y[:, task].detach().numpy(), 'k.')
    plt.plot(Yclean_val[:, task], '-r')
    # Predictive mean as blue line
    plt.plot(mean[:, task].numpy(), '.b')
    # Shade in confidence
    plt.plot(lower[:, task].numpy(), 'c--')
    plt.plot(upper[:, task].numpy(), 'c--')
    #ax.fill_between(lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
    #plt.ylim([-0.1, 1.3])
    plt.legend(['Observed Data', 'y', 'mean_pred','2std'])
    plt.title(f'Task {task + 1} MOGP')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
my_MSE = np.mean((mean.numpy()-val_y.numpy())**2)
print("MSE val:",my_MSE)
print("Neg Marginal Log Likelihood:",-mll(model(val_x), val_y))
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
            KL_plus = np.sqrt(KLD_Gaussian(m1.numpy().T,V1,m2.numpy().T,V2,use_diag=use_diag)) #In code the authors don't use the Mult. by 2
            KL_minus = np.sqrt(KLD_Gaussian(m1.numpy().T, V1, m2_minus.numpy().T,V2_minus,use_diag=use_diag))  # In code the authors don't use the Mult. by 2
            relevance[n, p] = 0.5*(KL_plus+KL_minus)/delta
#print(relevance)
print(f"Relevance of Features:\n {np.mean(relevance,0)}")