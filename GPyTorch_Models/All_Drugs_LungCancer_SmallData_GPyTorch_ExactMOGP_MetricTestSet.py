import pandas as pd
import numpy as np
import time
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import scipy.optimize as opt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import gc

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import os

import scipy as sp

# _FOLDER = "/home/ac1jjgg/MOGP_GPyTorch/GDSC1_Small_Dataset/"
_FOLDER = "/home/juanjo/Work_Postdoc/my_codes_postdoc/GPyTorch_Models/GDSC1_Small_Dataset/"

df_train = pd.read_csv(_FOLDER+"PI3k_train.csv")  #Contain dataset prepared by Subhashini
df_test = pd.read_csv(_FOLDER+"PI3K_test.csv")  #Contain dataset prepared by Subhashini
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# we realised that the column "molecular_formula" is a string like and also
#We get rid of the two columns with std == 0
df_train_clean = df_train.drop(columns=['molecular_formula','covalent_unit_count','N'])
df_test_clean = df_test.drop(columns=['molecular_formula','covalent_unit_count','N'])
print("std for features train:", (df_train_clean[df_train_clean.columns[30:]].std(0)==0.0).sum() )
print("std for features test:", (df_test_clean[df_test_clean.columns[30:]].std(0)==0.0).sum() )
print("It might be ok to have more than zero in the test set.")
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
df_train_values = df_train_clean[df_train_clean.columns[30:]].values
df_test_values = df_test_clean[df_test_clean.columns[30:]].values
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#df_test_clean.columns[30]
X_train_features = df_train_clean[df_train_clean.columns[30:]].values
X_test_features = df_test_clean[df_test_clean.columns[30:]].values
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
y_train_drug = np.clip(df_train_clean["norm_cells_"+str(1)].values[:,None],1.0e-9,np.inf)
y_test_drug =  np.clip(df_test_clean["norm_cells_"+str(1)].values[:,None],1.0e-9,np.inf)
print(y_train_drug.shape)
for i in range(2,10):
    y_train_drug = np.concatenate((y_train_drug,np.clip(df_train_clean["norm_cells_"+str(i)].values[:,None],1.0e-9,np.inf)),1)
    y_test_drug = np.concatenate((y_test_drug,np.clip(df_test_clean["norm_cells_"+str(i)].values[:,None],1.0e-9,np.inf)),1)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
params_4_sig_train = df_train_clean["param_"+str(1)].values[:,None]
params_4_sig_test = df_test_clean["param_"+str(1)].values[:,None]
for i in range(2,5):
    params_4_sig_train = np.concatenate((params_4_sig_train,df_train_clean["param_"+str(i)].values[:,None]),1)
    params_4_sig_test = np.concatenate((params_4_sig_test,df_test_clean["param_"+str(i)].values[:,None]),1)
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
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import matplotlib.pyplot as plt
from sklearn import metrics
x_lin = np.linspace(0,1,1000)
x_real_dose = np.linspace(0.111111,1,9)
x_lin_tile = np.tile(x_lin,(params_4_sig_train.shape[0],1))
#(x_lin,params_4_sig_train.shape[0],1).shape
Ydose_res = []
AUC = []
for i in range(params_4_sig_train.shape[0]):
    Ydose_res.append(sigmoid_4_param(x_lin_tile[i,:],*params_4_sig_train[i,:]))
    AUC.append(metrics.auc(x_lin_tile[i,:],Ydose_res[i]))

posy = 12
#plt.plot(x_lin,Ydose_res[posy])
#plt.plot(x_real_dose,y_train_drug[posy,:],'.')
#plt.title(f"AUC = {AUC[posy]}")
#print(AUC[posy])
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import matplotlib.pyplot as plt
from sklearn import metrics
x_lin = np.linspace(0,1,1000)
x_real_dose = np.linspace(0.111111,1,9)
x_lin_tile = np.tile(x_lin,(params_4_sig_test.shape[0],1))
#(x_lin,params_4_sig_train.shape[0],1).shape
Ydose_res_test = []
AUC_test = []
for i in range(params_4_sig_test.shape[0]):
    Ydose_res_test.append(sigmoid_4_param(x_lin_tile[i,:],*params_4_sig_test[i,:]))
    AUC_test.append(metrics.auc(x_lin_tile[i,:],Ydose_res_test[i]))

posy = 12
#plt.plot(x_lin,Ydose_res_test[posy])
#plt.plot(x_real_dose,y_test_drug[posy,:],'.')
#plt.title(f"AUC = {AUC_test[posy]}")
#print(AUC_test[posy])
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
AUC = np.array(AUC)
AUC_test = np.array(AUC_test)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Xall = X_train_features.copy()
Xtest = X_test_features.copy()

Yall = y_train_drug.copy()
Ytest =  y_test_drug.copy()

AUC_all =  AUC[:,None].copy()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import getopt
import sys

warnings.filterwarnings("ignore")
#os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

class commandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'i:s:k:r:d:p:')
        # opts = dict(opts)
        # print(opts)
        self.N_iter_epoch = 3    #number of iterations
        self.which_seed = 1010    #change seed to initialise the hyper-parameters
        self.rank = 1
        self.scale = 1.0
        self.split_dim = 2
       # self.Nfold = 0
        self.bash = "None"

        for op, arg in opts:
            # print(op,arg)
            if op == '-i':
                self.N_iter_epoch = arg
            if op == '-r':  # (r)and seed
                self.which_seed = arg
            if op == '-k':  # ran(k)
                self.rank = arg
            if op == '-s':  # length(s)cale width
                self.scale = arg
            if op == '-p':  # (p)ython bash
                self.bash = arg
            if op == '-d':  # split of the additive kernel
                self.split_dim = arg
            # if op == '-f':  # Nfold to select for Cross-Validation
            #     self.Nfold = arg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
config = commandLine()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Create a K-fold for cross-validation"
from sklearn.model_selection import KFold, cross_val_score
N_per_out = Xall.shape[0]
Xind = np.arange(N_per_out)
k_fold = KFold(n_splits=5,shuffle=True,random_state=0)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Nfold = 0
select_fold =  3  #int(config.Nfold)
MSEMean_Folds = []
MSEMed_Folds = []
All_Models = []
for train_id, val_id in k_fold.split(Xind):
    if Nfold == select_fold:
        print(f"Using Fold {Nfold}")
        Xval = Xall[val_id, :].copy()
        Xtrain = Xall[train_id, :].copy()
        Yval = Yall[val_id, :].copy()
        Ytrain = Yall[train_id, :].copy()
        break
    Nfold +=1

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"SEED"
torch.manual_seed(0)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
rank = int(config.rank)   # Rank for the MultitaskKernel
num_tasks = Ytrain.shape[1]
split_dim = int(config.split_dim)
Dim = Xall.shape[1]
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
train_x = torch.from_numpy(Xtrain.astype(np.float32))
train_y = torch.from_numpy(Ytrain.astype(np.float32))
val_x = torch.from_numpy(Xval.astype(np.float32))
val_y = torch.from_numpy(Yval.astype(np.float32))

test_x = torch.from_numpy(Xtest.astype(np.float32))
test_y = torch.from_numpy(Ytest.astype(np.float32))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

myseed = int(config.which_seed)
np.random.seed(myseed)

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )

        size_dims = (Dim) // split_dim

        # mykern = gpytorch.kernels.RBFKernel(active_dims=torch.tensor(list(np.arange(0,size_dims))),batch_shape=torch.Size([num_latents]))
        mykern = gpytorch.kernels.LinearKernel(active_dims=torch.tensor(list(np.arange(0, size_dims))),
                                               ard_num_dims=size_dims) + gpytorch.kernels.MaternKernel(nu=0.5,
                                                                                                       active_dims=torch.tensor(
                                                                                                           list(
                                                                                                               np.arange(
                                                                                                                   0,
                                                                                                                   size_dims))),
                                                                                                       ard_num_dims=size_dims)
        for i in range(1, split_dim):
            if i != (split_dim - 1):
                mykern = mykern + gpytorch.kernels.LinearKernel(
                    active_dims=torch.tensor(list(np.arange(size_dims * i, size_dims * i + size_dims))), ard_num_dims=size_dims) + gpytorch.kernels.MaternKernel(
                    nu=0.5, active_dims=torch.tensor(list(np.arange(size_dims * i, size_dims * i + size_dims))), ard_num_dims=size_dims)
                # print(torch.tensor(list(np.arange(size_dims*i,size_dims*i+size_dims))))
            else:
                last_dims = Dim - size_dims * i
                mykern = mykern + gpytorch.kernels.LinearKernel(
                    active_dims=torch.tensor(list(np.arange(size_dims * i, Dim))), ard_num_dims=last_dims) + gpytorch.kernels.MaternKernel(
                    nu=0.5, active_dims=torch.tensor(list(np.arange(size_dims * i, Dim))), ard_num_dims=last_dims)
                # print(torch.tensor(list(np.arange(size_dims*i,Dim))))

        self.covar_module = gpytorch.kernels.MultitaskKernel(
            mykern, num_tasks=num_tasks, rank=rank
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks,
                                                              noise_constraint=gpytorch.constraints.Interval(1.0e-9,
                                                                                                             0.5e-2))
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model = MultitaskGPModel(train_x, train_y, likelihood)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
for i in range(split_dim):
    d1,d2 = model.covar_module.data_covar_module.kernels[2 * i + 1].lengthscale.size()
    mylengthscale = float(config.scale) * np.sqrt(Dim) * np.random.rand(d1,d2)
    model.covar_module.data_covar_module.kernels[2*i+1].lengthscale = torch.tensor(mylengthscale)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
m_trained = "12"
print("loading model ",m_trained)
state_dict = torch.load('m_'+m_trained+'.pth')
model.load_state_dict(state_dict)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# this is for running the notebook in our testing framework
import os
num_epochs = int(config.N_iter_epoch)

model.train()
likelihood.train()

# Our loss object We're using the Exact Marginal Log Likelohood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
Ntrain,_ = Ytrain.shape
show_each = 100 #Ntrain//train_loader.batch_size
refine_lr = [0.1,0.05,0.01,0.005]
refine_num_epochs = [num_epochs,int(num_epochs*0.5),int(num_epochs*0.2),int(num_epochs*0.2)]
for Nrefine in range(len(refine_lr)):
    print(f"\nRefine Learning Rate {Nrefine}; lr={refine_lr[Nrefine]}")
    for g in optimizer.param_groups:
        g['lr'] = refine_lr[Nrefine]

    for j in range(num_epochs):
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

# plt.close('all')
# for task in range(num_tasks):
#     # Plot training data as black stars
#     plt.figure(task)
#     plt.plot(val_y[:, task].detach().numpy(), 'k.')
#     # Predictive mean as blue line
#     plt.plot(mean[:, task].numpy(), '.b')
#     # Shade in confidence
#     #ax.fill_between(test_x.numpy(), lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
#     plt.ylim([-0.1, 1.3])
#     plt.legend(['Observed Data', 'Mean', 'Confidence'])
#     plt.title(f'Task {task + 1}')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
r2_scr = r2_score(val_y.T, mean.T, multioutput='raw_values')[:,None]
myMSE = np.mean((val_y.numpy()-mean.numpy())**2,1)
myEmax_Err = np.abs(np.mean(val_y.numpy()[:,-2:],1)-np.mean(mean.numpy()[:,-2:],1))
myAUC_Err = np.zeros(val_y.numpy().shape[0])
from sklearn import metrics
for i in range(val_y.numpy().shape[0]):
    AUC_val = metrics.auc(np.linspace(0.11,1,9),val_y.numpy()[i,:])
    AUC_pred = metrics.auc(np.linspace(0.11, 1, 9), mean.numpy()[i, :])
    myAUC_Err[i] = np.abs(AUC_val-AUC_pred)

print("Median R2: ",np.median(r2_scr))
print("Mean R2: ",np.mean(r2_scr))
print("Median MSE: ",np.median(myMSE))
print("Mean MSE: ",np.mean(myMSE))
print("Median Emax_abs:",np.median(myEmax_Err))
print("Mean Emax_abs:",np.mean(myEmax_Err))
print("Median AUC_abs:",np.median(myAUC_Err))
print("Mean AUC_abs:",np.mean(myAUC_Err))

MSEMean_Folds.append(np.mean(myMSE))
MSEMed_Folds.append(np.median(myMSE))
All_Models.append(model)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Nrow = 77//4
Ncolumn = 4
fig, axs = plt.subplots(Nrow, Ncolumn, figsize=(50,110*(Nrow//10)))

#test_y, mean
posy = 0
for i in range(Nrow):
    for j in range(Ncolumn):
        #posy = np.random.randint(val_y.shape[0])
        axs[i, j].plot(np.linspace(0.11,1,9), val_y[posy,:],'xb')
        axs[i, j].plot(np.linspace(0.11,1,9), val_y[posy,:],'.b')
        axs[i, j].plot(np.linspace(0.11,1,9), mean[posy,:],'-r')
        axs[i, j].plot(np.linspace(0.11,1,9), lower[posy,:],'--c')
        axs[i, j].plot(np.linspace(0.11,1,9), upper[posy,:],'--c')
        axs[i, j].set_title("r2 = "+str(r2_scr[posy,:]))
        axs[i, j].set_ylim([-0.05, 1.05])
        posy = posy + 1
        if posy == 855:
            break

print(posy)
fig.savefig('WinnerModel_SmallDataset_Val.pdf',dpi=300)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## Make predictions over Test Set
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #test_x = torch.linspace(0, 1, 51)
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()
plt.close('all')
# for task in range(num_tasks):
#     # Plot training data as black stars
#     plt.figure(task)
#     plt.plot(test_y[:, task].detach().numpy(), 'k.')
#     # Predictive mean as blue line
#     plt.plot(mean[:, task].numpy(), '.b')
#     # Shade in confidence
#     #ax.fill_between(test_x.numpy(), lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
#     plt.ylim([-0.1, 1.3])
#     plt.legend(['Observed Data', 'Mean', 'Confidence'])
#     plt.title(f'Task {task + 1}')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
r2_scr = r2_score(test_y.T, mean.T, multioutput='raw_values')[:,None]
myMSE = np.mean((test_y.numpy()-mean.numpy())**2,1)
myEmax_Err = np.abs(np.mean(test_y.numpy()[:,-2:],1)-np.mean(mean.numpy()[:,-2:],1))
myAUC_Err = np.zeros(test_y.numpy().shape[0])
from sklearn import metrics
for i in range(test_y.numpy().shape[0]):
    AUC_val = metrics.auc(np.linspace(0.11,1,9),test_y.numpy()[i,:])
    AUC_pred = metrics.auc(np.linspace(0.11, 1, 9), mean.numpy()[i, :])
    myAUC_Err[i] = np.abs(AUC_val-AUC_pred)

print("Median R2 Test: ",np.median(r2_scr))
print("Mean R2 Test: ",np.mean(r2_scr))
print("Median MSE Test: ",np.median(myMSE))
print("Mean MSE Test: ",np.mean(myMSE))
print("Median Emax_abs Test:",np.median(myEmax_Err))
print("Mean Emax_abs Test:",np.mean(myEmax_Err))
print("Median AUC_abs Test:",np.median(myAUC_Err))
print("Mean AUC_abs Test:",np.mean(myAUC_Err))

Nrow = 96//4
Ncolumn = 4
fig, axs = plt.subplots(Nrow, Ncolumn, figsize=(50,110*(Nrow//10)))

#test_y, mean
posy = 0
for i in range(Nrow):
    for j in range(Ncolumn):
        #posy = np.random.randint(val_y.shape[0])
        axs[i, j].plot(np.linspace(0.11,1,9), test_y[posy,:],'xb', markersize=14)
        axs[i, j].plot(np.linspace(0.11,1,9), test_y[posy,:],'.b', markersize=12)
        axs[i, j].plot(np.linspace(0.11,1,9), mean[posy,:],'-r',linewidth=3, markersize=12)
        axs[i, j].plot(np.linspace(0.11,1,9), lower[posy,:],'--c',linewidth=2, markersize=12)
        axs[i, j].plot(np.linspace(0.11,1,9), upper[posy,:],'--c',linewidth=2, markersize=12)
        axs[i, j].set_title(f" Test{posy}, MSE = {myMSE[posy]:.5f}",fontsize=30)
        axs[i, j].set_ylim([-0.05, 1.05])
        posy = posy + 1
        if posy == 855:
            break

print(posy)
#fig.savefig('WinnerModelMOGP_SmallDataset_Test.pdf',dpi=300)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Plot of the data from Functional Random Forest model"
df_pred_FRF = pd.read_csv("./FRF_pred.txt",header=None)
mean_FRF = df_pred_FRF.values
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
r2_scr = r2_score(test_y.T, mean_FRF.T, multioutput='raw_values')[:,None]
myMSE_FRF = np.mean((test_y.numpy()-mean_FRF)**2,1)
myEmax_Err = np.abs(np.mean(test_y.numpy()[:,-2:],1)-np.mean(mean_FRF[:,-2:],1))
myAUC_Err = np.zeros(test_y.numpy().shape[0])
from sklearn import metrics
for i in range(test_y.numpy().shape[0]):
    AUC_val = metrics.auc(np.linspace(0.11,1,9),test_y.numpy()[i,:])
    AUC_pred = metrics.auc(np.linspace(0.11, 1, 9), mean_FRF[i, :])
    myAUC_Err[i] = np.abs(AUC_val-AUC_pred)

print("Median R2 Test FRFmodel: ",np.median(r2_scr))
print("Mean R2 Test FRFmodel: ",np.mean(r2_scr))
print("Median MSE Test FRFmodel: ",np.median(myMSE_FRF))
print("Mean MSE Test FRFmodel: ",np.mean(myMSE_FRF))
print("Median Emax_abs Test FRFmodel:",np.median(myEmax_Err))
print("Mean Emax_abs Test FRFmodel:",np.mean(myEmax_Err))
print("Median AUC_abs Test FRFmodel:",np.median(myAUC_Err))
print("Mean AUC_abs Test FRFmodel:",np.mean(myAUC_Err))
Nrow = 96//4
Ncolumn = 4
#fig, axs = plt.subplots(Nrow, Ncolumn, figsize=(50,110*(Nrow//10)))

#test_y, mean
posy = 0
for i in range(Nrow):
    for j in range(Ncolumn):
        #posy = np.random.randint(val_y.shape[0])
        #axs[i, j].plot(np.linspace(0.11,1,9), test_y[posy,:],'xb', markersize=14)
        #axs[i, j].plot(np.linspace(0.11,1,9), test_y[posy,:],'.b', markersize=12)
        axs[i, j].plot(np.linspace(0.11,1,9), mean_FRF[posy,:],'-g',linewidth=3, markersize=12)
        axs[i, j].set_title(f"{posy}, MSE_GP = {myMSE[posy]:.5f}, MSE_FRF = {myMSE_FRF[posy]:.5f}", fontsize=30)
        #axs[i, j].set_ylim([-0.05, 1.05])
        posy = posy + 1
        if posy == 855:
            break

print(posy)
#fig.savefig('WinnerModelFRF_SmallDataset_Test.pdf',dpi=300)
fig.savefig('WinnerModels_FRF_vs_MOGP_SmallDataset_Test.pdf',dpi=300)


















# BestFold_posy = np.where(np.min(MSEMed_Folds)==MSEMed_Folds)[0][0]
# f= open("Metrics.txt","a+")
# f.write("bash"+str(config.bash)+f" MSEMed_Aver={np.mean(MSEMed_Folds):0.4f} MSEMean_aver={np.mean(MSEMean_Folds):0.4f} BestFold_Pos ={BestFold_posy}\n")
# f.close()
#
# #final_path = '/data/ac1jjgg/Data_Marina/GPyTorch_results/LungCancer_SmallDataset_ExactMOGP/'
# final_path ='model_LungCancer_SmallData/'
# if not os.path.exists(final_path):
#     os.makedirs(final_path)
# torch.save(All_Models[BestFold_posy].state_dict(), final_path+'m_'+str(config.bash)+'.pth')