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

#_FOLDER = "/data/ac1jjgg/Data_Marina/results_with_NonAffecting_Drugs/"  #HPC folder
_FOLDER = "/home/juanjo/Work_Postdoc/my_codes_postdoc/DrugProfiles-master/results_with_NonAffecting_Drugs/"

### Coding Part

with open(_FOLDER + "drug_ids_50.txt", 'r') as f:
    drug_ids_50 = [np.int32(line.rstrip('\n')) for line in f]

with open(_FOLDER + "X_features_cancer_cell_lines.txt", 'r') as f:
    X_cancer_cell_lines = [line.rstrip('\n') for line in f]
# *****************************************

with open(_FOLDER + "X_PubChem_properties.txt", 'r') as f:
    X_PubChem_properties = [line.rstrip('\n') for line in f]
# *****************************************

with open(_FOLDER + "X_features_Targets.txt", 'r') as f:
    X_targets = [line.rstrip('\n') for line in f]
# *****************************************

with open(_FOLDER + "X_features_Target_Pathway.txt", 'r') as f:
    X_target_pathway = [line.rstrip('\n') for line in f]
# *****************************************
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


GDSC_Info = pd.read_csv(_FOLDER+"Cell_list_GDSC.csv")  #Contains info of cancer types for both GDSC1 and GDSC2
df_GDSC1 = GDSC_Info[GDSC_Info["Dataset"]=="GDSC1"]
df_OneCancer = df_GDSC1[df_GDSC1["Tissue"]=="lung"].reset_index()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
all_columns = X_cancer_cell_lines + X_PubChem_properties + X_targets + X_target_pathway + ["MAX_CONC"]

train_df = pd.read_csv(_FOLDER + "train08_merged_fitted_sigmoid4_123_with_drugs_properties_min10.csv").drop(
    ["Unnamed: 0", "Unnamed: 0.1"], axis=1)
test_df = pd.read_csv(_FOLDER + "test02_merged_fitted_sigmoid4_123_with_drugs_properties_min10.csv").drop(
    ["Unnamed: 0", "Unnamed: 0.1"], axis=1)

train_df_50 = train_df.set_index("DRUG_ID").loc[drug_ids_50, :].copy()
test_df_50 = test_df.set_index("DRUG_ID").loc[drug_ids_50, :].copy()

datasets = ["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4"]

X_feat_dict = {"Dataset 1": X_cancer_cell_lines,
               "Dataset 2": ["MAX_CONC"] + X_targets + X_target_pathway + X_cancer_cell_lines,
               "Dataset 3": ["MAX_CONC"] + X_PubChem_properties + X_cancer_cell_lines,
               "Dataset 4": ["MAX_CONC"] + X_PubChem_properties + X_targets + X_target_pathway + X_cancer_cell_lines}

### Coefficient_1

train_drug = train_df_50.copy()
test_drug = test_df_50.copy()

data_set = "Dataset 4"
X_columns = X_feat_dict[data_set]
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
df_train_drug_new = train_drug[train_drug["COSMIC_ID"] == df_OneCancer["COSMICID"][0]]
for i in range(1, df_OneCancer.shape[0]):
    df_aux = train_drug[train_drug["COSMIC_ID"] == df_OneCancer["COSMICID"][i]]
    df_train_drug_new = pd.concat([df_train_drug_new, df_aux])

df_test_drug_new = test_drug[test_drug["COSMIC_ID"] == df_OneCancer["COSMICID"][0]]
for i in range(1, df_OneCancer.shape[0]):
    df_aux = test_drug[test_drug["COSMIC_ID"] == df_OneCancer["COSMICID"][i]]
    df_test_drug_new = pd.concat([df_test_drug_new, df_aux])

df_train_drug_new = df_train_drug_new.reset_index()
df_test_drug_new = df_test_drug_new.reset_index()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
scaler = MinMaxScaler().fit(df_train_drug_new[X_columns])
Xtrain_drug = scaler.transform(df_train_drug_new[X_columns])
Xtest_drug = scaler.transform(df_test_drug_new[X_columns])

y_train_drug = np.clip(df_train_drug_new["norm_cells_"+str(1)].values[:,None],1.0e-9,np.inf)
y_test_drug =  np.clip(df_test_drug_new["norm_cells_"+str(1)].values[:,None],1.0e-9,np.inf)
print(y_train_drug.shape)
for i in range(2,10):
    y_train_drug = np.concatenate((y_train_drug,np.clip(df_train_drug_new["norm_cells_"+str(i)].values[:,None],1.0e-9,np.inf)),1)
    y_test_drug = np.concatenate((y_test_drug,np.clip(df_test_drug_new["norm_cells_"+str(i)].values[:,None],1.0e-9,np.inf)),1)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
params_4_sig_train = df_train_drug_new["param_"+str(1)].values[:,None]
params_4_sig_test = df_test_drug_new["param_"+str(1)].values[:,None]
for i in range(2,5):
    params_4_sig_train = np.concatenate((params_4_sig_train,df_train_drug_new["param_"+str(i)].values[:,None]),1)
    params_4_sig_test = np.concatenate((params_4_sig_test,df_test_drug_new["param_"+str(i)].values[:,None]),1)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

### Training data for the GP ###
output_dim = y_train_drug.shape[1]
N_per_out = Xtrain_drug.shape[0]
#aux_label_outputs = np.tile(np.arange(0,output_dim),(N_per_out,1)).T.reshape(-1) #Create labels for X
Xall = Xtrain_drug.copy()  #Here replicate (10,1) due to having ten outputs.
#Xtrain = np.concatenate((Xtrain,aux_label_outputs[:,None]),axis=1)
Yall = y_train_drug.copy()

### Testing data for the GP ###
N_per_out_test = Xtest_drug.shape[0]
#aux_label_outputs_test = np.tile(np.arange(0,output_dim),(N_per_out_test,1)).T.reshape(-1) #Create labels for X
Xtest = Xtest_drug.copy()  #Here replicate (10,1) due to having ten outputs.
#Xtest= np.concatenate((Xtest,aux_label_outputs_test[:,None]),axis=1)
Ytest = y_test_drug.copy()
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
x_lin = np.linspace(0,1,1000)
x_lin_tile = np.tile(x_lin,(params_4_sig_train.shape[0],1))
#(x_lin,params_4_sig_train.shape[0],1).shape
Ydose_res = []
IC50_all = []
Ydose50_all = []
for i in range(params_4_sig_train.shape[0]):
    Ydose_res.append(sigmoid_4_param(x_lin_tile[i,:],*params_4_sig_train[i,:]))
    res1=(Ydose_res[i]<0.507)
    res2=(Ydose_res[i]>0.493)
    if (res1 & res2).sum()>0:
        Ydose50_all.append(Ydose_res[i][res1 & res2].mean())
        IC50_all.append(x_lin[res1 & res2].mean())
    else:
        Ydose50_all.append(x_lin[-1])
        IC50_all.append(x_lin[-1])

Ydose50_all = np.array(Ydose50_all)
IC50_all = np.array(IC50_all)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import getopt
import sys

warnings.filterwarnings("ignore")
#os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

class commandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'l:b:m:i:w:s:r:d:p:f:')
        # opts = dict(opts)
        # print(opts)
        self.minibatch = 100   # mini-batch size for stochastic inference
        self.inducing = 200    #number of inducing points
        self.N_iter_epoch = 3    #number of iterations
        self.num_latentGPs = 9  #number of latent GPs
        #self.which_model = 'VIK'   #LMC  #CPM
        self.which_seed = 1010    #change seed to initialise the hyper-parameters
        self.weight = 0.01
        self.scale = 1.0
        self.split_dim = 2
        self.Nfold = 0
        self.bash = "None"

        for op, arg in opts:
            # print(op,arg)
            if op == '-l':
                self.num_latentGPs = arg
            if op == '-b':
                self.minibatch = arg
            if op == '-m':
                self.inducing = arg
            if op == '-i':
                self.N_iter_epoch = arg
            if op == '-r':  # (r)and seed
                self.which_seed = arg
            if op == '-w':  # (r)and seed
                self.weight = arg
            if op == '-s':  # (r)and seed
                self.scale = arg
            if op == '-p':  # (p)ython bash
                self.bash = arg
            if op == '-d':  # split of the additive kernel
                self.split_dim = arg
            if op == '-f':  # Nfold to select for Cross-Validation
                self.Nfold = arg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
config = commandLine()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Create a K-fold for cross-validation"
from sklearn.model_selection import KFold, cross_val_score
Xind = np.arange(N_per_out)
k_fold = KFold(n_splits=5,shuffle=True,random_state=0)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Nfold = 0
select_fold = int(config.Nfold)
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
minibatch = int(config.minibatch)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
train_x = torch.from_numpy(Xtrain.astype(np.float32))
train_y = torch.from_numpy(Ytrain.astype(np.float32))
val_x = torch.from_numpy(Xval.astype(np.float32))
val_y = torch.from_numpy(Yval.astype(np.float32))

from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=minibatch, shuffle=True)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
num_latents = int(config.num_latentGPs) #9
num_tasks = Ytrain.shape[1]
split_dim = int(config.split_dim)
num_inducing = int(config.inducing)

myseed = int(config.which_seed)

np.random.seed(myseed)
minis = Xall.min(0)
maxis = Xall.max(0)
Dim = Xall.shape[1]
Z = np.linspace(minis[0], maxis[0], num_inducing).reshape(1, -1)
for i in range(Dim - 1):
    Zaux = np.linspace(minis[i + 1], maxis[i + 1], num_inducing)
    Z = np.concatenate((Z, Zaux[np.random.permutation(num_inducing)].reshape(1, -1)), axis=0)
Z = 1.0 * Z.T

class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self):
        # Let's use a different set of inducing points for each latent function
        #inducing_points = torch.rand(num_latents, num_inducing, Dim)
        inducing_points = torch.tensor(np.repeat(Z[None,:,:],num_latents,axis=0).astype(np.float32))

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
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))

        size_dims = (Dim) // split_dim

        # mykern = gpytorch.kernels.RBFKernel(active_dims=torch.tensor(list(np.arange(0,size_dims))),batch_shape=torch.Size([num_latents]))
        mykern = gpytorch.kernels.LinearKernel(active_dims=torch.tensor(list(np.arange(0, size_dims))),
                                               batch_shape=torch.Size([num_latents]),
                                               ard_num_dims=size_dims) + gpytorch.kernels.MaternKernel(nu=0.5,
                                                                                                       active_dims=torch.tensor(
                                                                                                           list(
                                                                                                               np.arange(
                                                                                                                   0,
                                                                                                                   size_dims))),
                                                                                                       batch_shape=torch.Size(
                                                                                                           [
                                                                                                               num_latents]),
                                                                                                       ard_num_dims=size_dims)
        for i in range(1, split_dim):
            if i != (split_dim - 1):
                mykern = mykern + gpytorch.kernels.LinearKernel(
                    active_dims=torch.tensor(list(np.arange(size_dims * i, size_dims * i + size_dims))),
                    batch_shape=torch.Size([num_latents]), ard_num_dims=size_dims) + gpytorch.kernels.MaternKernel(
                    nu=0.5, active_dims=torch.tensor(list(np.arange(size_dims * i, size_dims * i + size_dims))),
                    batch_shape=torch.Size([num_latents]), ard_num_dims=size_dims)
                # print(torch.tensor(list(np.arange(size_dims*i,size_dims*i+size_dims))))
            else:
                last_dims = Dim - size_dims * i
                mykern = mykern + gpytorch.kernels.LinearKernel(
                    active_dims=torch.tensor(list(np.arange(size_dims * i, Dim))),
                    batch_shape=torch.Size([num_latents]), ard_num_dims=last_dims) + gpytorch.kernels.MaternKernel(
                    nu=0.5, active_dims=torch.tensor(list(np.arange(size_dims * i, Dim))),
                    batch_shape=torch.Size([num_latents]), ard_num_dims=last_dims)
                # print(torch.tensor(list(np.arange(size_dims*i,Dim))))

        self.covar_module = gpytorch.kernels.ScaleKernel(mykern,
                                                         batch_shape=torch.Size([num_latents])
                                                         )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model = MultitaskGPModel()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
np.random.seed(myseed)

#myweights = 0.2 * np.ones((num_latents,num_tasks))
myweights = float(config.weight) * np.random.rand(num_latents,num_tasks)
model.variational_strategy.lmc_coefficients = torch.nn.Parameter(torch.tensor(myweights.astype(np.float32)))

for i in range(split_dim):
    d1,d2,d3 = model.covar_module.base_kernel.kernels[2 * i + 1].lengthscale.size()
    mylengthscale = float(config.scale) * np.sqrt(Dim) * np.random.rand(d1,d2,d3)
    #mylengthscale = 0.2*np.ones((d1,d2,d3))
    model.covar_module.base_kernel.kernels[2*i+1].lengthscale = torch.tensor(mylengthscale)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks,
                                                              noise_constraint=gpytorch.constraints.Interval(1.0e-5,
                                                                                                             1.0e-3))
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
m_trained = config.bash
print("loading model ",m_trained)
state_dict = torch.load('./model_LungCancer_HighAdditiveSplitDim/m_'+m_trained+'.pth')
model.load_state_dict(state_dict)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# # this is for running the notebook in our testing framework
# import os
# #smoke_test = ('CI' in os.environ)
# #num_epochs = 1 if smoke_test else 2
# num_epochs = int(config.N_iter_epoch)
#
# model.train()
# likelihood.train()
#
# optimizer = torch.optim.Adam([
#     {'params': model.parameters()},
#     {'params': likelihood.parameters()},
# ], lr=0.005)
#
# # Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
# mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#
# Ntrain,_ = Ytrain.shape
# show_each = Ntrain//train_loader.batch_size
# refine_lr = [0.005,0.001,0.0005,0.0001]
# refine_num_epochs = [num_epochs,int(num_epochs*0.5),int(num_epochs*0.2),int(num_epochs*0.2)]
# for Nrefine in range(len(refine_lr)):
#     print(f"\nRefine Learning Rate {Nrefine}; lr={refine_lr[Nrefine]}")
#     for g in optimizer.param_groups:
#         g['lr'] = refine_lr[Nrefine]
#
#     for i in range(refine_num_epochs[Nrefine]):
#         print(f"Epoch {i}")
#         # Within each iteration, we will go over each minibatch of data
#         #minibatch_iter = tqdm(train_loader, desc = 'Minibatch')
#         for j, (x_batch, y_batch) in enumerate(train_loader):
#             optimizer.zero_grad()
#             output = model(x_batch)
#             loss = -mll(output, y_batch)
#             if j%(show_each//2)==0:
#                 print(f"Minbatch {j}, Loss {loss}")
#
#             loss.backward()
#             optimizer.step()
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# # Set into eval mode
# model.eval()
# likelihood.eval()

# Initialize plots
#fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))

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
    # Predictive mean as blue line
    plt.plot(mean[:, task].numpy(), '.b')
    # Shade in confidence
    #ax.fill_between(test_x.numpy(), lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
    plt.ylim([-0.1, 1.3])
    plt.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.title(f'Task {task + 1}')

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
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from scipy.interpolate import interp1d
x_dose = np.linspace(0.111111,1.0,9)
x_dose_new = np.linspace(0.111111,1.0,1000)
Ydose50_val = []
IC50_val = []
for i in range(val_y.numpy().shape[0]):
    y_resp = mean.numpy()[i,:].copy()
    f = interp1d(x_dose, y_resp)
    f2 = interp1d(x_dose, y_resp, kind='quadratic')

    y_resp_interp = f2(x_dose_new)
    res1 = y_resp_interp<0.507
    res2 = y_resp_interp>0.493
    if (res1 & res2).sum()>0:
        Ydose50_val.append( y_resp_interp[res1 & res2].mean())
        IC50_val.append( x_dose_new[res1 & res2].mean())
    else:
        Ydose50_val.append( x_dose_new[-1])
        IC50_val.append( x_dose_new[-1])

Ydose50_val = np.array(Ydose50_val)
IC50_val = np.array(IC50_val)

print("IC50 MSE:",np.mean((IC50_all[val_id]-IC50_val)**2))

IC50_Y_Val = IC50_all[val_id]
pos_NonAffect = IC50_Y_Val==1.0
pos_Affect = pos_NonAffect==False
print("IC50 MSE val NonAffect:",np.mean((IC50_Y_Val[pos_NonAffect]-IC50_val[pos_NonAffect])**2))
print("IC50 MSE val Affect:",np.mean((IC50_Y_Val[pos_Affect]-IC50_val[pos_Affect])**2))

from scipy.stats import spearmanr
spear_corr, p_value = spearmanr(IC50_Y_Val, IC50_val)
print ("Spearman Corr: ",spear_corr)
print("Spearman p-value: ", p_value)

# import matplotlib.pyplot as plt
# plt.figure(12)
# plt.plot(x_dose, y_resp, 'o', x_dose_new, f(x_dose_new), '-', x_dose_new, y_resp_interp, '--')
# plt.legend(['data', 'linear', 'cubic'], loc='best')
# plt.plot(x_dose,val_y[posy_check,:],'xr')
# plt.plot(IC50,Ydose50,'xb')
# plt.show()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# f= open("Metrics.txt","a+")
# f.write("bash"+str(config.bash)+f" R2Med={np.median(r2_scr):0.3f} R2Mean={r2_scr.mean():0.3f} MSEMed={np.median(myMSE):0.4f} MSEMean={myMSE.mean():0.4f} EmaxMed={np.median(myEmax_Err):0.4f} EmaxMean={myEmax_Err.mean():0.4f} AUCMed={np.median(myAUC_Err):0.4f} AUCMean={myAUC_Err.mean():0.4f}\n")
# f.close()
#
# f= open("Summary_Metric.txt","a+")
# summary_metric_med = (-1.0*np.median(r2_scr)+np.median(myMSE)+np.median(myEmax_Err)+np.median(myAUC_Err))/4.0
# summary_metric_mean = (-1.0*np.mean(r2_scr)+np.mean(myMSE)+np.mean(myEmax_Err)+np.mean(myAUC_Err))/4.0
# f.write("bash"+str(config.bash)+f" Summary_Med={summary_metric_med:0.5f} Summary_Mean={summary_metric_mean:0.5f}\n")
# f.close()
#
# print("\nMedian Summary:", summary_metric_med)
# print("\nMean Summary:", summary_metric_mean)
#
# #final_path = '/data/ac1jjgg/Data_Marina/GPyTorch_results/LungCancer_GPAdditiveCorrectRealY/'
# final_path ='model_LungCancer/'
# if not os.path.exists(final_path):
#     os.makedirs(final_path)
#
# #torch.save(model.state_dict(), final_path+'m_'+str(config.bash)+'.pth')

Nrow = 107//2
Ncolumn = 4
fig, axs = plt.subplots(Nrow, Ncolumn, figsize=(30,50*(Nrow//10)))

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
#fig.savefig('validation_m_130_0-212.pdf',dpi=300)