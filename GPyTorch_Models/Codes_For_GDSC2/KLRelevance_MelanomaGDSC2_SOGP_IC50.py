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

# _FOLDER = "/home/ac1jjgg/MOGP_GPyTorch/Codes_for_GDSC2/Dataset_BRAF/GDSC2/"
_FOLDER = "/home/juanjo/Work_Postdoc/my_codes_postdoc/GPyTorch_Models/Dataset_BRAF/GDSC2/"
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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
name_for_KLrelevance = 'GDSC2_melanoma_BRAF.csv'

df_train_No_MolecForm = pd.read_csv(_FOLDER + name_for_KLrelevance)  # Contain Train dataset prepared by Subhashini-Evelyn
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
try:
    df_train_No_MolecForm = df_train_No_MolecForm.drop(columns='Drug_Name')
except:
    pass

# Here we just check that from the column index 25 the input features start
start_pos_features = 25
print(df_train_No_MolecForm.columns[start_pos_features])

#print("Columns with std equal zero:")
#print("Number of columns with zero std:", np.sum(df_train_No_MolecForm.std(0) == 0.0))
#print(np.where(df_train_No_MolecForm.std(0) == 0.0))
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
scaler = MinMaxScaler().fit(df_train_No_MolecForm[df_train_No_MolecForm.columns[start_pos_features:]])
X_train_features = scaler.transform(df_train_No_MolecForm[df_train_No_MolecForm.columns[start_pos_features:]])

"Below we select just 7 concentration since GDSC2 only has such a number"
y_train_drug = np.clip(df_train_No_MolecForm["norm_cells_" + str(1)].values[:, None], 1.0e-9, np.inf)
print(y_train_drug.shape)
for i in range(2, 8):
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
import matplotlib.pyplot as plt
from sklearn import metrics

plt.close('all')

x_lin = np.linspace(0.111111, 1, 1000)
x_real_dose = np.linspace(0.111111, 1, 7)  #Here is 7 due to using GDSC2 that has 7 doses
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
    elif Ydose_res[i][-1]<0.5:
       Ydose50.append(Ydose_res[i].max())
       aux_IC50 = x_lin[np.where(Ydose_res[i].max())[0]][0]  #it has to be a float not an array to avoid bug
       IC50.append(aux_IC50)
    else:
        Ydose50.append(0.5)
        IC50.append(1.5) #IC50.append(x_lin[-1])

# posy = 300
# #plt.figure(Nfold)
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
AllNames_Features = df_train_No_MolecForm.columns[start_pos_features:]  #starts from 25
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
"Best model was: All_Drugs_MelanomaGDSC2_GPyTorch_SOGP_IC50.py -i 1200 -s 0.1000 -r 1015 -d 5 -p 66 -e %d"
class commandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'i:s:r:d:p:e:')
        # opts = dict(opts)
        # print(opts)
        self.N_iter_epoch = 0    #number of iterations
        self.which_seed = 1015    #change seed to initialise the hyper-parameters
        self.scale = 0.1
        self.split_dim = 5
        self.feature = 0
        self.bash = "66"

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
            if op == '-e':  # Feature to select for the KL_Relevance
                self.feature = arg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
config = commandLine()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"The indexes below are 6 with response (98,90,209,139,105,20) and 6 without reponse (303,200,149,199,219,301i.e., IC50=1.5)"
#([ 20,  90,  98, 105, 139, 149, 199, 200, 209, 219, 301, 303])

index_Test = np.array([20,90,98,105,139,149,199,200,209,219,301,303])  #index 3: IC50=0.301, index 22: IC50=1.5(nonResponsive) index 50: IC50=0.568
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
nsplits = 5 #Ndata
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
for Nfold in range(nsplits,nsplits+1):
    "The first if below is for the cross-val"
    "Then the else is for using all data to save the model trained over all data"
    if Nfold<nsplits:
        train_ind, test_ind = list_folds[Nfold]
        print(f"{test_ind} to Val in IC50")

        Xval = Xall[test_ind].copy()
        Xtrain = Xall[train_ind].copy()
        Yval = IC50_all[test_ind].copy()
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
    # Here we load the model bash*:
    m_trained = str(config.bash)
    state_dict = torch.load('/home/juanjo/Work_Postdoc/my_codes_postdoc/GPyTorch_Models/Codes_For_GDSC2/Best_Model_MelanomaGDSC2_SOGP_IC50/m_' + m_trained + '.pth')
    # state_dict = torch.load("/home/ac1jjgg/SOGP_GPyTorch/Codes_for_GDSC2/Best_Model_MelanomaGDSC2_SOGP_IC50/" +'m_'+ m_trained + '.pth')
    print("loading model ", m_trained)
    model.load_state_dict(state_dict)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
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
        # lower, upper = predictions.confidence_region()
        pred_var = predictions.variance

    Val_NMLL = -mll(model(val_x), val_y).detach().numpy()
    print("NegMLL Val", Val_NMLL)

    IC50_pred = mean.numpy()[:,None]
    posNoIC50 = IC50_pred > 1.0
    IC50_pred[posNoIC50] = 1.5
    lower = IC50_pred - 2 * np.sqrt(pred_var.numpy()[:, None])
    upper = IC50_pred + 2 * np.sqrt(pred_var.numpy()[:, None])

    #plt.close('all')
    # Plot training data as black stars
    plt.figure(Nfold)
    plt.plot(val_y.detach().numpy(), 'kx')
    # Predictive mean as blue line
    plt.plot(IC50_pred, '.b')
    # Shade in confidence
    plt.plot(lower, '--c')
    plt.plot(upper, '--c')

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
f= open("Metrics_Test.txt","a+")
f.write("bash"+str(config.bash)+f"  test_MSE_IC50= {np.mean((Yval-IC50_pred)**2):0.5f} test_NegMLL={Val_NMLL:0.5f}\n")
f.close()

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
delta = 1.0e-4
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
            try:
                KL_plus = np.sqrt(KLD_Gaussian(m1.numpy().T,V1,m2.numpy().T,V2,use_diag=use_diag)+jitter) #In code the authors don't use the Mult. by 2
                KL_minus = np.sqrt(KLD_Gaussian(m1.numpy().T, V1, m2_minus.numpy().T, V2_minus, use_diag=use_diag) + jitter)  # In code the authors don't use the Mult. by 2
            except:
                KL_plus = np.sqrt(KLD_Gaussian_NoChol(m1.numpy().T, V1, m2.numpy().T, V2,use_diag=use_diag) + jitter)  # In code the authors don't use the Mult. by 2
                KL_minus = np.sqrt(KLD_Gaussian_NoChol(m1.numpy().T, V1, m2_minus.numpy().T, V2_minus,use_diag=use_diag) + jitter)  # In code the authors don't use the Mult. by 2

            relevance[n, p] = 0.5*(KL_plus+KL_minus)/delta

"Correction of Nan data in the relevance"
NonNan_ind = np.where(~np.isnan(relevance[:,which_p]))
Nan_ind = np.where(np.isnan(relevance[:,which_p]))[0]
relevance[Nan_ind,which_p]= (relevance[NonNan_ind,which_p].max()-relevance[NonNan_ind,which_p].min())/2.0
#print(relevance)
print(f"Relevance of Features:\n {np.mean(relevance,0)}")

f= open("Relevance_MelanomaGDSC2_SOGP_IC50.txt","a+")
f.write(f"{which_p}")
for n in range(N):
    f.write(",")
    f.write(f"{relevance[n,which_p]:0.5}")
f.write(f"\n")
f.close()