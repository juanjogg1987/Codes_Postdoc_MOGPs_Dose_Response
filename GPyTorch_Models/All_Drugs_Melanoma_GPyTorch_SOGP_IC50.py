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

# _FOLDER = "/home/ac1jjgg/MOGP_GPyTorch/FiveCancersDataSet/"
_FOLDER = "/home/juanjo/Work_Postdoc/my_codes_postdoc/GPyTorch_Models/FiveCancersDataSet/"
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

name_for_KLrelevance = 'Melanoma_BRAF_Targeted_Drugs.csv'

df_train_No_MolecForm = pd.read_csv(_FOLDER + name_for_KLrelevance)  # Contain Train dataset prepared by Subhashini-Evelyn
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
try:
    df_train_No_MolecForm = df_train_No_MolecForm.drop(columns='Drug_Name')
except:
    pass

# Here we just check that from the column index 28 the input features start
print(df_train_No_MolecForm.columns[28])

print("Columns with std equal zero:")
print(np.where(df_train_No_MolecForm.std(0) == 0.0))
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
scaler = MinMaxScaler().fit(df_train_No_MolecForm[df_train_No_MolecForm.columns[28:]])
X_train_features = scaler.transform(df_train_No_MolecForm[df_train_No_MolecForm.columns[28:]])

y_train_drug = np.clip(df_train_No_MolecForm["norm_cells_" + str(1)].values[:, None], 1.0e-9, np.inf)
print(y_train_drug.shape)
for i in range(2, 10):
    y_train_drug = np.concatenate(
        (y_train_drug, np.clip(df_train_No_MolecForm["norm_cells_" + str(i)].values[:, None], 1.0e-9, np.inf)), 1)

print("Ytrain size: ", y_train_drug.shape)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
params_4_sig_train = df_train_No_MolecForm["param_" + str(1)].values[:, None]
for i in range(2, 5):
    params_4_sig_train = np.concatenate(
        (params_4_sig_train, df_train_No_MolecForm["param_" + str(i)].values[:, None]), 1)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import matplotlib.pyplot as plt
from sklearn import metrics

plt.close('all')

x_lin = np.linspace(0.111111, 1, 1000)
x_real_dose = np.linspace(0.111111, 1, 9)
x_lin_tile = np.tile(x_lin, (params_4_sig_train.shape[0], 1))
# (x_lin,params_4_sig_train.shape[0],1).shape
Ydose_res = []
AUC = []
IC50 = []
Ydose50 = []
Emax = []
for i in range(params_4_sig_train.shape[0]):
    Ydose_res.append(sigmoid_4_param(x_lin_tile[i, :], *params_4_sig_train[i, :]))
    AUC.append(metrics.auc(x_lin_tile[i, :], Ydose_res[i]))
    Emax.append(Ydose_res[i][-1])
    res1 = (Ydose_res[i] < 0.507)
    res2 = (Ydose_res[i] > 0.493)
    if (res1 & res2).sum() > 0:
        Ydose50.append(Ydose_res[i][res1 & res2].mean())
        IC50.append(x_lin[res1 & res2].mean())
    else:
        Ydose50.append(0.5)
        IC50.append(1.5) #IC50.append(x_lin[-1])

# posy = 65
# plt.figure(Nfold)
# plt.plot(x_lin, Ydose_res[posy])
# plt.plot(x_real_dose, y_train_drug[posy, :], '.')
# plt.plot(IC50[posy], Ydose50[posy], 'rx')
# plt.plot(x_lin, np.ones_like(x_lin)*Emax[posy], 'r') #Plot a horizontal line as Emax
# plt.title(f"AUC = {AUC[posy]}")
# print(AUC[posy])
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Compute Log(AUC)? R/ Not for Functional Random Forest Model
AUC = np.array(AUC)
IC50 = np.array(IC50)
Emax = np.array(Emax)

"Below we select just the columns with std higher than zero"
ind_nonzero_std = np.where(X_train_features.std(0)!=0.0)[0]
AllNames_Features = df_train_No_MolecForm.columns[28:]
Name_Features_Melanoma = AllNames_Features[ind_nonzero_std]

#Xall_aux = X_train_features[:,ind_nonzero_std].copy()
#ind_rel_asper_GP = np.array([93,231,344,259,131,163,284,53,95,105,247,299,196,102,47])   #KL_SOGP_IC50
#ind_rel_asper_GP = np.array([172,108,187,52,238])   #KL_SOGP_IC50 with rbF
#ind_rel_asper_GP = np.array([126,381,378,348,365,366,375,362,379,355])   #KL_MOGP
#ind_rel_asper_GP = np.array([157,3,4,44,26,61,1,27,17,124])   #Lasso

#Xall = Xall_aux[:,ind_rel_asper_GP]
Xall = X_train_features[:,ind_nonzero_std].copy()
Yall = y_train_drug.copy()

AUC_all = AUC[:, None].copy()
IC50_all = IC50[:, None].copy()
Emax_all = Emax[:, None].copy()

print("AUC train size:", AUC_all.shape)
print("IC50 train size:", IC50_all.shape)
print("Emax train size:", Emax_all.shape)
print("X train size:", Xall.shape)
print("Y train size:", Yall.shape)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import getopt
import sys

warnings.filterwarnings("ignore")
#os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

class commandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'i:s:r:d:p:')
        # opts = dict(opts)
        # print(opts)
        self.N_iter_epoch = 1500    #number of iterations
        self.which_seed = 1010    #change seed to initialise the hyper-parameters
        self.scale = 0.1
        self.split_dim = 2
        self.bash = "None"

        for op, arg in opts:
            # print(op,arg)
            if op == '-i':
                self.N_iter_epoch = arg
            if op == '-r':  # (r)and seed
                self.which_seed = arg
            if op == '-s':  # (r)and seed
                self.scale = arg
            if op == '-p':  # (p)ython bash
                self.bash = arg
            if op == '-d':  # split of the additive kernel
                self.split_dim = arg


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
config = commandLine()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
index_Test = np.array([3,22,50])  #index 3: IC50=0.301, index 22: IC50=1.5(nonResponsive) index 50: IC50=0.568
index_Train = np.arange(0,Xall.shape[0])
Xtest_final = Xall[index_Test,:].copy()
Xall = Xall[np.delete(index_Train,index_Test),:].copy()
IC50_test_final = IC50_all[index_Test,:].copy()
IC50_all = IC50_all[np.delete(index_Train,index_Test),:].copy()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Create a K-fold for cross-validation"
from sklearn.model_selection import KFold, cross_val_score
Ndata = Xall.shape[0]
Xind = np.arange(Ndata)
nsplits = 20 #Ndata
k_fold = KFold(n_splits=nsplits)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
NegMLL_AllFolds = []
Emax_abs_AllFolds = []
AUC_abs_AllFolds = []
IC50_MSE_AllFolds = []
Med_MSE_AllFolds = []
AllPred_MSE_AllFolds = []
Mean_MSE_AllFolds = []
Spearman_AllFolds = []
SpearActualIC50_AllFolds = []
All_Models = []

list_folds = list(k_fold.split(Xall))
for Nfold in range(nsplits-1,nsplits+1-1):
    "The first if below is for the cross-val"
    "Then the else is for using all data to save the model trained over all data"
    if Nfold<nsplits:
        train_ind, test_ind = list_folds[Nfold]
        print(f"{test_ind} to Val in IC50")

        Xval = Xall[train_ind].copy()#Xall[test_ind].copy()
        #Xval = Xall[test_ind].copy()
        Xtrain = Xall[train_ind].copy()
        Yval = IC50_all[train_ind].copy()#IC50_all[test_ind].copy()
        #Yval = IC50_all[test_ind].copy()
        Ytrain = IC50_all[train_ind].copy()
    else:
        print(f"Train ovell all Data in IC50")
        Xval = Xtest_final.copy()
        Xtrain = Xall.copy()
        Yval = IC50_test_final.copy()
        Ytrain = IC50_all.copy()
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
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    mean_all = np.zeros_like(Yval)
    models_outs = []
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Dim = Xtrain.shape[1]

    train_x = torch.from_numpy(Xtrain.astype(np.float32))
    train_y = torch.from_numpy(Ytrain[:,0].astype(np.float32))  #Here we train on IC50
    val_x = torch.from_numpy(Xval.astype(np.float32))
    val_y = torch.from_numpy(Yval[:,0].astype(np.float32))      #val for IC50

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #num_latents = int(config.num_latentGPs) #9
    num_tasks = Ytrain.shape[1]
    split_dim = int(config.split_dim)
    #num_inducing = int(config.inducing)

    myseed = int(config.which_seed)
    np.random.seed(myseed)

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()

            size_dims = (Dim) // split_dim

            # mykern = gpytorch.kernels.RBFKernel(active_dims=torch.tensor(list(np.arange(0,size_dims))),batch_shape=torch.Size([num_latents]))
            mykern = gpytorch.kernels.LinearKernel(active_dims=torch.tensor(list(np.arange(0, size_dims))),
                                                   ard_num_dims=size_dims) + gpytorch.kernels.RBFKernel(
                                                                                                           active_dims=torch.tensor(
                                                                                                               list(
                                                                                                                   np.arange(
                                                                                                                       0,
                                                                                                                       size_dims))),
                                                                                                           ard_num_dims=size_dims)
            for i in range(1, split_dim):
                if i != (split_dim - 1):
                    mykern = mykern + gpytorch.kernels.LinearKernel(
                        active_dims=torch.tensor(list(np.arange(size_dims * i, size_dims * i + size_dims))), ard_num_dims=size_dims) + gpytorch.kernels.RBFKernel(active_dims=torch.tensor(list(np.arange(size_dims * i, size_dims * i + size_dims))), ard_num_dims=size_dims)
                    # print(torch.tensor(list(np.arange(size_dims*i,size_dims*i+size_dims))))
                else:
                    last_dims = Dim - size_dims * i
                    mykern = mykern + gpytorch.kernels.LinearKernel(
                        active_dims=torch.tensor(list(np.arange(size_dims * i, Dim))), ard_num_dims=last_dims) + gpytorch.kernels.RBFKernel( active_dims=torch.tensor(list(np.arange(size_dims * i, Dim))), ard_num_dims=last_dims)
                    # print(torch.tensor(list(np.arange(size_dims*i,Dim))))

            #mykern = mykern+gpytorch.kernels.
            self.covar_module = gpytorch.kernels.ScaleKernel(mykern)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
    #likelihood = FixedNoiseGaussianLikelihood(torch.ones(train_x.shape[0]) * 0.01)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1.0e-5, 1.0e-1))
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    model = ExactGPModel(train_x, train_y, likelihood)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    np.random.seed(myseed)

    # myweights = float(config.weight) * np.random.rand(num_latents,num_tasks)
    # model.variational_strategy.lmc_coefficients = torch.nn.Parameter(torch.tensor(myweights.astype(np.float32)))
    #
    for i in range(split_dim):
        d1,d2 = model.covar_module.base_kernel.kernels[2 * i + 1].lengthscale.size()
        mylengthscale = float(config.scale) * np.sqrt(Dim) * np.random.rand(d1,d2)
        #mylengthscale = 0.2*np.ones((d1,d2,d3))
        model.covar_module.base_kernel.kernels[2*i+1].lengthscale = torch.tensor(mylengthscale)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #m_trained = config.bash
    #print("loading model ",m_trained)
    #state_dict = torch.load('./INCLUDEDIR/m_'+m_trained+'.pth')
    #model.load_state_dict(state_dict)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # this is for running the notebook in our testing framework
    import os
    #smoke_test = ('CI' in os.environ)
    #num_epochs = 1 if smoke_test else 2
    num_epochs = int(config.N_iter_epoch)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=0.005)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    Ntrain,_ = Ytrain.shape
    show_each = 10
    refine_lr = [0.005,0.001,0.0005,0.0001]
    #refine_lr = [0.00001, 0.000005, 0.000001, 0.0000005]
    refine_num_epochs = [num_epochs,int(num_epochs*0.5),int(num_epochs*0.2),int(num_epochs*0.2)]
    for Nrefine in range(len(refine_lr)):
        print(f"\nRefine Learning Rate {Nrefine}; lr={refine_lr[Nrefine]}")
        for g in optimizer.param_groups:
            g['lr'] = refine_lr[Nrefine]

        for i in range(refine_num_epochs[Nrefine]):
            # Within each iteration, we will go over each minibatch of data
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            if i%(show_each)==0:
                print(f"Iter {i}, Loss {loss}")

            loss.backward()
            optimizer.step()
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Initialize plots
    #fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #test_x = torch.linspace(0, 1, 51)
        predictions = likelihood(model(val_x))
        mean = predictions.mean
        #lower, upper = predictions.confidence_region()
        pred_var = predictions.variance

    Val_NMLL = -mll(model(val_x), val_y).detach().numpy()
    print("NegMLL Val",Val_NMLL)

    IC50_pred = mean.numpy()[:,None]
    posNoIC50 = IC50_pred > 1.0
    IC50_pred[posNoIC50] = 1.5
    lower = IC50_pred - 2*np.sqrt(pred_var.numpy()[:,None])
    upper = IC50_pred + 2 * np.sqrt(pred_var.numpy()[:,None])

    #plt.close('all')
    # Plot training data as black stars
    plt.figure(Nfold)
    plt.plot(val_y.detach().numpy(), 'kx')
    # Predictive mean as blue line
    plt.plot(IC50_pred, '.b')
    # Shade in confidence
    plt.plot(lower, '--c')
    plt.plot(upper, '--c')

    #ax.fill_between(test_x.numpy(), lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
    plt.ylim([-0.2, 1.6])
    plt.legend(['Observed Data', 'Mean_pred'])
    #plt.title(f'Fold {Nfold}')

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    IC50_MSE = np.mean((Yval - IC50_pred)**2)
    print("IC50 MSE:", IC50_MSE)

    from scipy.stats import spearmanr
    pos_Actual_IC50 = Yval!=1.5
    spear_corr_all, p_value_all = spearmanr(Yval, IC50_pred)
    spear_corr_actualIC50, p_value_actual = spearmanr(Yval[pos_Actual_IC50], IC50_pred[pos_Actual_IC50])
    print ("Spearman_all Corr: ",spear_corr_all)
    print("Spearman p-value: ", p_value_all)
    print("Spearman_actualIC50 Corr: ", spear_corr_actualIC50)
    print("Spearman p-value: ", p_value_actual)

    if Nfold < nsplits:
        NegMLL_AllFolds.append(Val_NMLL.copy())
        IC50_MSE_AllFolds.append(IC50_MSE.copy())
        Spearman_AllFolds.append(spear_corr_all)
        SpearActualIC50_AllFolds.append(spear_corr_actualIC50)
    print("Yval shape",Yval.shape)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
f= open("Metrics.txt","a+")
f.write("bash"+str(config.bash)+f" IC50_MSE={np.mean(IC50_MSE_AllFolds):0.5f}({np.std(IC50_MSE_AllFolds):0.5f}) NegMLL={np.mean(NegMLL_AllFolds):0.5f}({np.std(NegMLL_AllFolds):0.5f}) Spear_ActualIC50={np.mean(SpearActualIC50_AllFolds):0.5f}({np.std(SpearActualIC50_AllFolds):0.5f}) Spear_all={np.mean(Spearman_AllFolds):0.5f}({np.std(Spearman_AllFolds):0.5f}) test_MSE_IC50= {np.mean((Yval-IC50_pred)**2):0.5f} test_NegMLL={Val_NMLL:0.5f} \n")
f.close()

"The last model should have been trained over all dataset without splitting"

# final_path = '/data/ac1jjgg/Data_Marina/GPyTorch_results/Melanoma_SmallDataset_SOGP_IC50/'
# # final_path ='model_Melanoma_SOGP_IC50/'
# if not os.path.exists(final_path):
#    os.makedirs(final_path)
# torch.save(model.state_dict(), final_path+'m_'+str(config.bash)+'.pth')