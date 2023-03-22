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
Xall = X_train_features[:,ind_nonzero_std].copy()
Yall = y_train_drug.copy()

AUC_all = AUC[:, None].copy()
IC50_all = IC50[:, None].copy()
Emax_all = Emax[:, None].copy()

print("AUC train size:", AUC_all.shape)
print("IC50 train size:", IC50_all.shape)
print("Emax train size:", Emax_all.shape)
print("X all data size:", Xall.shape)
print("Y all data size:", Yall.shape)

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
        opts, args = getopt.getopt(sys.argv[1:], 'i:s:k:r:d:p:e:')
        # opts = dict(opts)
        # print(opts)
        self.N_iter_epoch = 100    #number of iterations
        self.which_seed = 1010    #change seed to initialise the hyper-parameters
        self.rank = 1
        self.scale = 0.1
        self.split_dim = 2
        self.feature = 0
        self.bash = "None"

        for op, arg in opts:
            # print(op,arg)
            if op == '-i':
                self.N_iter_epoch = arg
            if op == '-r':  # (r)and seed
                self.which_seed = arg
            if op == '-k':  # ran(k)
                self.rank = arg
            if op == '-s':  # (r)and seed
                self.scale = arg
            if op == '-p':  # (p)ython bash
                self.bash = arg
            if op == '-d':  # split of the additive kernel
                self.split_dim = arg
            if op == '-e':  # Feature to select for the KL_Relevance
                self.feature = arg


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
config = commandLine()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
index_Test = np.array([3,22,50])  #index 3: IC50=0.301, index 22: IC50=1.5(nonResponsive) index 50: IC50=0.568
index_Train = np.arange(0,Xall.shape[0])
Xtest_final = Xall[index_Test,:].copy()
Xall = Xall[np.delete(index_Train,index_Test),:].copy()
Ytest_final = Yall[index_Test,:].copy()
Yall = Yall[np.delete(index_Train,index_Test),:].copy()

IC50_test_final = IC50_all[index_Test,:].copy()
IC50_all = IC50_all[np.delete(index_Train,index_Test),:].copy()
AUC_test_final = AUC_all[index_Test,:].copy()
AUC_all = AUC_all[np.delete(index_Train,index_Test),:].copy()
Emax_test_final = Emax_all[index_Test,:].copy()
Emax_all = Emax_all[np.delete(index_Train,index_Test),:].copy()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Create a K-fold for cross-validation"
from sklearn.model_selection import KFold, cross_val_score
Ndata = Xall.shape[0]
Xind = np.arange(Ndata)
nsplits = 20 #Remember we used 20 for training in HPC
k_fold = KFold(n_splits=nsplits)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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
for Nfold in range(nsplits,nsplits+1):
    "The first if below is for the cross-val"
    "Then the else is for using all data to save the model trained over all data"
    if Nfold<nsplits:
        train_ind, test_ind = list_folds[Nfold]
        print(f"{test_ind} to Val in IC50")

        Xval = Xall[test_ind].copy()
        Xtrain = Xall[train_ind].copy()
        Yval = Yall[test_ind].copy()
        Ytrain = Yall[train_ind].copy()

        Emax_val = Emax_all[test_ind].copy()
        AUC_val = AUC_all[test_ind].copy()
        IC50_val = IC50_all[test_ind].copy()

    else:
        print(f"Train ovell all Data in IC50")
        Xval = Xtest_final.copy()
        Xtrain = Xall.copy()
        Yval = Ytest_final.copy()
        Ytrain = Yall.copy()

        Emax_val = Emax_test_final.copy()
        AUC_val = AUC_test_final.copy()
        IC50_val = IC50_test_final.copy()
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
    rank = int(config.rank)  # Rank for the MultitaskKernel
    Dim = Xtrain.shape[1]

    train_x = torch.from_numpy(Xtrain.astype(np.float32))
    train_y = torch.from_numpy(Ytrain.astype(np.float32))  #Here we train on all dose response curve
    val_x = torch.from_numpy(Xval.astype(np.float32))
    val_y = torch.from_numpy(Yval.astype(np.float32))      #val for all dose response curve

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #num_latents = int(config.num_latentGPs) #9
    num_tasks = Ytrain.shape[1]
    split_dim = int(config.split_dim)
    #num_inducing = int(config.inducing)

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
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks,noise_constraint=gpytorch.constraints.Interval(1.0e-5, 1.0e-3),rank=num_tasks)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    model = MultitaskGPModel(train_x, train_y, likelihood)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    np.random.seed(myseed)

    # myweights = float(config.weight) * np.random.rand(num_latents,num_tasks)
    # model.variational_strategy.lmc_coefficients = torch.nn.Parameter(torch.tensor(myweights.astype(np.float32)))
    #
    for i in range(split_dim):
        d1, d2 = model.covar_module.data_covar_module.kernels[2 * i + 1].lengthscale.size()
        mylengthscale = float(config.scale) * np.sqrt(Dim) * np.random.rand(d1, d2)
        model.covar_module.data_covar_module.kernels[2 * i + 1].lengthscale = torch.tensor(mylengthscale)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #Here we load the model bash*:
    m_trained = config.bash
    state_dict = torch.load(
        '/home/juanjo/Work_Postdoc/my_codes_postdoc/GPyTorch_Models/FiveCancersDataSet/Best_Model_Melanoma/m_' + m_trained + '.pth')
    # state_dict = torch.load("/home/ac1jjgg/MOGP_GPyTorch/FiveCancersDataSet/Best_Model_Melanoma/" +'m_'+ m_trained + '.pth')
    print("loading model ", m_trained)
    model.load_state_dict(state_dict)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Initialize plots
    #fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # test_x = torch.linspace(0, 1, 51)
        predictions = likelihood(model(val_x))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

    plt.close('all')
    for task in range(num_tasks):
        # Plot training data as black stars
        plt.figure(task)
        plt.plot(val_y[:, task].detach().numpy(), 'kx')
        # Predictive mean as blue line
        plt.plot(mean[:, task].numpy(), '.b')
        plt.plot(lower[:, task].numpy(), '--c')
        plt.plot(upper[:, task].numpy(), '--c')
        # Shade in confidence
        #ax.fill_between(test_x.numpy(), lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
        plt.ylim([-0.1, 1.3])
        #plt.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.title(f'Task {task + 1}')

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    from scipy.interpolate import interp1d
    from scipy.interpolate import pchip_interpolate

    x_dose = np.linspace(0.111111, 1.0, 9)
    x_dose_new = np.linspace(0.111111, 1.0, 1000)
    Ydose50_pred = []
    IC50_pred = []
    AUC_pred = []
    Emax_pred = []
    Y_pred_interp_all = []
    for i in range(val_y.numpy().shape[0]):
        y_resp = mean.numpy()[i, :].copy()
        f = interp1d(x_dose, y_resp)
        f2 = interp1d(x_dose, y_resp, kind='cubic')
        y_resp_interp = pchip_interpolate(x_dose, y_resp, x_dose_new)

        #y_resp_interp = f2(x_dose_new)
        Y_pred_interp_all.append(y_resp_interp)
        AUC_pred.append(metrics.auc(x_dose_new, y_resp_interp))
        Emax_pred.append(y_resp_interp[-1])

        res1 = y_resp_interp < 0.507
        res2 = y_resp_interp > 0.493
        if (res1 & res2).sum() > 0:
            Ydose50_pred.append(y_resp_interp[res1 & res2].mean())
            IC50_pred.append(x_dose_new[res1 & res2].mean())
        else:
            Ydose50_pred.append(0.5)
            IC50_pred.append(1.5)

    Ydose50_pred = np.array(Ydose50_pred)
    IC50_pred = np.array(IC50_pred)[:,None]
    AUC_pred = np.array(AUC_pred)[:, None]
    Emax_pred = np.array(Emax_pred)[:, None]

    posy = 0
    plt.figure(Nfold)
    plt.plot(x_dose_new, Y_pred_interp_all[posy])
    plt.plot(x_dose, val_y.numpy()[posy, :], '.')
    plt.plot(IC50_pred[posy], Ydose50_pred[posy], 'rx')
    plt.plot(x_dose_new, np.ones_like(x_dose_new) * Emax_pred[posy], 'r')  # Plot a horizontal line as Emax
    plt.plot(x_dose_new, np.ones_like(x_dose_new) * Emax_val[posy], 'r')  # Plot a horizontal line as Emax
    plt.plot(x_dose,lower.numpy()[posy, :], '--c')
    plt.plot(x_dose,upper.numpy()[posy, :], '--c')
    plt.title(f"AUC = {AUC_pred[posy]}")
    print(AUC_pred[posy])

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    Emax_abs = np.mean(np.abs(Emax_val - Emax_pred))
    AUC_abs = np.mean(np.abs(AUC_val - AUC_pred))
    IC50_MSE = np.mean((IC50_val - IC50_pred) ** 2)
    MSE_curves = np.mean((mean.numpy() - val_y.numpy()) ** 2, 1)
    AllPred_MSE = np.mean((mean.numpy() - val_y.numpy()) ** 2)
    print("IC50 MSE:", IC50_MSE)
    print("AUC MAE:", AUC_abs)
    print("Emax MAE:", Emax_abs)
    Med_MSE = np.median(MSE_curves)
    Mean_MSE = np.mean(MSE_curves)
    print("Med_MSE:", Med_MSE)
    print("Mean_MSE:", Mean_MSE)
    print("All Predictions MSE:", AllPred_MSE)

    from scipy.stats import spearmanr

    pos_Actual_IC50 = IC50_val != 1.5
    spear_corr_all, p_value_all = spearmanr(IC50_val, IC50_pred)
    spear_corr_actualIC50, p_value_actual = spearmanr(IC50_val[pos_Actual_IC50],
                                                      IC50_pred[pos_Actual_IC50])
    print("Spearman_all Corr: ", spear_corr_all)
    print("Spearman p-value: ", p_value_all)
    print("Spearman_actualIC50 Corr: ", spear_corr_actualIC50)
    print("Spearman p-value: ", p_value_actual)

    if Nfold < nsplits:
        Emax_abs_AllFolds.append(Emax_abs.copy())
        AUC_abs_AllFolds.append(AUC_abs.copy())
        IC50_MSE_AllFolds.append(IC50_MSE.copy())
        Med_MSE_AllFolds.append(Med_MSE.copy())
        Mean_MSE_AllFolds.append(Mean_MSE.copy())
        AllPred_MSE_AllFolds.append(AllPred_MSE.copy())
        Spearman_AllFolds.append(spear_corr_all)
        SpearActualIC50_AllFolds.append(spear_corr_actualIC50)

    print("Yval shape",Yval.shape)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

ftest= open("Metrics_Test.txt","a+")
ftest.write("bash"+str(config.bash)+f" Med_MSE={Med_MSE:0.5f} Mean_MSE={Mean_MSE:0.5f} IC50_MSE={IC50_MSE:0.5f} Spear_ActualIC50={spear_corr_actualIC50:0.5f} Spear_all={spear_corr_all:0.5f} AUC_abs={AUC_abs:0.5f} Emax_abs ={Emax_abs:0.5f} \n")
ftest.close()

"The last model should have been trained over all dataset without splitting"

from scipy.linalg import cholesky,cho_solve
from GPy.util import linalg

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Set into eval mode
model.eval()
likelihood.eval()

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
    #print("V2_inv:", V2_inv)
    #print("m2:", m2)

    KL = 0.5 * np.sum(V2_inv * V1) + 0.5 * np.dot((m1-m2).T, np.dot(V2_inv, (m1-m2))) \
         - 0.5 * Dim + 0.5 * 2. * np.sum(np.log(np.abs(np.diag(L2)))) - 0.5 * 2. * np.sum(np.log(np.abs(np.diag(L1))))
    print("abs(KL):",np.abs(KL))
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
    #print("V2_inv:", V2_inv)
    #print("m2:", m2)

    KL = 0.5 * np.sum(V2_inv * V1) + 0.5 * np.dot((m1-m2).T, np.dot(V2_inv, (m1-m2))) \
         - 0.5 * Dim + 0.5 * np.log(np.linalg.det(V2)) - 0.5 * np.log(np.linalg.det(V1))
    return KL  #This is to avoid any negative due to numerical instability

data_x = train_x

N,P = data_x.shape

relevance = np.zeros((N,P))
delta = 1.0e-5
jitter = 1.0e-15
x_plus = np.zeros((1,P))

which_p = int(config.feature)
print(f"Analysing Feature {which_p} of {P}...")
for p in range(which_p,which_p+1):
    for n in range(N):
        #x_plus = X[n,:].copy()
        x_plus = data_x[n:n+1, :].clone()
        x_minus = data_x[n:n + 1, :].clone()
        x_plus[0,p] = x_plus[0,p]+delta
        x_minus[0, p] = x_minus[0, p] - delta

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # test_x = torch.linspace(0, 1, 51)
            predict_xn = likelihood(model(data_x[n:n+1,:]))
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
            KL_plus = np.sqrt(KLD_Gaussian(m1.numpy().T,V1,m2.numpy().T,V2,use_diag=use_diag)+jitter) #In code the authors don't use the Mult. by 2
            KL_minus = np.sqrt(KLD_Gaussian(m1.numpy().T, V1, m2_minus.numpy().T,V2_minus,use_diag=use_diag)+jitter)  # In code the authors don't use the Mult. by 2
            relevance[n, p] = 0.5*(KL_plus+KL_minus)/delta
#print(relevance)
print(f"Relevance of Features:\n {np.mean(relevance,0)}")

f= open("Relevance_Melanoma.txt","a+")
f.write(f"{which_p}")
for n in range(N):
    f.write(",")
    f.write(f"{relevance[n,which_p]:0.5}")
f.write(f"\n")
f.close()