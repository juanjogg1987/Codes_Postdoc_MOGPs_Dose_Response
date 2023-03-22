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
Train_names = ['Breast_train_GMLcLa_updated.csv', 'Glioma_train_BMLcLa_updated.csv',
               'LungAdenocarcinoma_train_BGMLc_updated.csv', 'LungCarcinoma_train_BGMLa_updated.csv',
               'Melanoma_train_BGLcLa_updated.csv']
Test_names = ['Breast_test_updated.csv', 'Glioma_test_updated.csv', 'Lung_adenocarcinoma_test_updated.csv',
              'Lung_carcinoma_test_updated.csv', 'Melanoma_test_updated.csv']

Train_5folds_X = []
Val_5folds_X = []
Train_5folds_Y = []
Val_5folds_Y = []
Train_5folds_AUC = []
Val_5folds_AUC = []
Train_5folds_IC50 = []
Val_5folds_IC50 = []
Train_5folds_Emax = []
Val_5folds_Emax = []

for Nfold in range(0, 5):
    print(f"Train Name in Nfold {Nfold}:", Train_names[Nfold])
    print(f"Test Name in Nfold {Nfold}:", Test_names[Nfold])
    df_train_No_MolecForm = pd.read_csv(
        _FOLDER + "Train/" + Train_names[Nfold])  # Contain Train dataset prepared by Subhashini-Evelyn
    df_test_No_MolecForm = pd.read_csv(
        _FOLDER + "/Test/" + Test_names[Nfold])  # Contain Test dataset prepared by Subhashini-Evelyn

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # we realised that the column "molecular_formula" is a string like
    # The updated files by subhashini do not have 'molecular_formula' anymore
    # df_train_No_MolecForm = df_train.drop(columns='molecular_formula')
    # df_test_No_MolecForm = df_test.drop(columns='molecular_formula')
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    try:
        df_train_No_MolecForm = df_train_No_MolecForm.drop(columns='Drug_Name')
    except:
        pass

    try:
        df_test_No_MolecForm = df_test_No_MolecForm.drop(columns='Drug_Name')
    except:
        pass

    ##df_train_No_MolecForm
    ##df_test_No_MolecForm
    ##Checking if both train and test have the same names for their columns
    print((df_train_No_MolecForm.columns == df_test_No_MolecForm.columns).sum())
    # Here we just check that from the column index 28 the input features start
    print(df_train_No_MolecForm.columns[28])

    All_data_together = pd.concat([df_train_No_MolecForm[df_train_No_MolecForm.columns[28:]],
                                   df_test_No_MolecForm[df_test_No_MolecForm.columns[28:]]])
    print("Columns with std equal zero:")
    print(np.where(All_data_together.std(0) == 0.0))

    # df_train_values = df_train_No_MolecForm[df_train_No_MolecForm.columns[29:]].values
    # df_test_values = df_test_No_MolecForm[df_test_No_MolecForm.columns[29:]].values
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # X_train_features = df_train_No_MolecForm[df_train_No_MolecForm.columns[29:]].values
    # X_test_features = df_test_No_MolecForm[df_test_No_MolecForm.columns[29:]].values

    scaler = MinMaxScaler().fit(df_train_No_MolecForm[df_train_No_MolecForm.columns[28:]])
    X_train_features = scaler.transform(df_train_No_MolecForm[df_train_No_MolecForm.columns[28:]])
    X_test_features = scaler.transform(df_test_No_MolecForm[df_test_No_MolecForm.columns[28:]])

    y_train_drug = np.clip(df_train_No_MolecForm["norm_cells_" + str(1)].values[:, None], 1.0e-9, np.inf)
    y_test_drug = np.clip(df_test_No_MolecForm["norm_cells_" + str(1)].values[:, None], 1.0e-9, np.inf)
    print(y_train_drug.shape)
    for i in range(2, 10):
        y_train_drug = np.concatenate(
            (y_train_drug, np.clip(df_train_No_MolecForm["norm_cells_" + str(i)].values[:, None], 1.0e-9, np.inf)), 1)
        y_test_drug = np.concatenate(
            (y_test_drug, np.clip(df_test_No_MolecForm["norm_cells_" + str(i)].values[:, None], 1.0e-9, np.inf)), 1)

    print("Ytrain size: ", y_train_drug.shape)
    print("Ytest size: ", y_test_drug.shape)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    params_4_sig_train = df_train_No_MolecForm["param_" + str(1)].values[:, None]
    params_4_sig_test = df_test_No_MolecForm["param_" + str(1)].values[:, None]
    for i in range(2, 5):
        params_4_sig_train = np.concatenate(
            (params_4_sig_train, df_train_No_MolecForm["param_" + str(i)].values[:, None]), 1)
        params_4_sig_test = np.concatenate((params_4_sig_test, df_test_No_MolecForm["param_" + str(i)].values[:, None]),
                                           1)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    import matplotlib.pyplot as plt
    from sklearn import metrics

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

    # posy = 90
    # plt.figure(Nfold)
    # plt.plot(x_lin, Ydose_res[posy])
    # plt.plot(x_real_dose, y_train_drug[posy, :], '.')
    # plt.plot(IC50[posy], Ydose50[posy], 'rx')
    # plt.plot(x_lin, np.ones_like(x_lin)*Emax[posy], 'r') #Plot a horizontal line as Emax
    # plt.title(f"AUC = {AUC[posy]}")
    # print(AUC[posy])
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    import matplotlib.pyplot as plt
    from sklearn import metrics

    x_lin = np.linspace(0.111111, 1, 1000)
    x_real_dose = np.linspace(0.111111, 1, 9)
    x_lin_tile = np.tile(x_lin, (params_4_sig_test.shape[0], 1))
    # (x_lin,params_4_sig_train.shape[0],1).shape
    Ydose_res_test = []
    AUC_test = []
    IC50_test = []
    Ydose50_test = []
    Emax_test = []
    for i in range(params_4_sig_test.shape[0]):
        Ydose_res_test.append(sigmoid_4_param(x_lin_tile[i, :], *params_4_sig_test[i, :]))
        AUC_test.append(metrics.auc(x_lin_tile[i, :], Ydose_res_test[i]))
        Emax_test.append(Ydose_res_test[i][-1])
        res1 = (Ydose_res_test[i] < 0.507)
        res2 = (Ydose_res_test[i] > 0.493)
        if (res1 & res2).sum() > 0:
            Ydose50_test.append(Ydose_res_test[i][res1 & res2].mean())
            IC50_test.append(x_lin[res1 & res2].mean())
        else:
            Ydose50_test.append(0.5)
            IC50_test.append(1.5) #IC50_test.append(x_lin[-1])

    # posy = 90
    # plt.figure(Nfold)
    # plt.plot(x_lin, Ydose_res_test[posy])
    # plt.plot(x_real_dose, y_test_drug[posy, :], '.')
    # plt.plot(x_lin, np.ones_like(x_lin) * Emax_test[posy], 'r')  #Plot a horizontal line as Emax
    # plt.plot(IC50_test[posy], Ydose50_test[posy], 'rx')
    # plt.title(f"AUC = {AUC_test[posy]}")
    # print(AUC_test[posy])
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Compute Log(AUC)? R/ Not for Functional Random Forest Model
    # AUC = np.log(np.array(AUC))
    AUC = np.array(AUC)
    AUC_test = np.array(AUC_test)[:, None].copy()

    #Ydose50_all = np.array(Ydose50_all)
    IC50 = np.array(IC50)
    IC50_test = np.array(IC50_test)[:,None].copy()

    Emax = np.array(Emax)
    Emax_test = np.array(Emax_test)[:, None].copy()

    Xall = X_train_features.copy()
    Xtest = X_test_features.copy()

    Yall = y_train_drug.copy()
    Ytest = y_test_drug.copy()

    AUC_all = AUC[:, None].copy()
    IC50_all = IC50[:, None].copy()
    Emax_all = Emax[:, None].copy()

    print("AUC train size:", AUC_all.shape)
    print("AUC test size:", AUC_test.shape)
    print("IC50 train size:", IC50_all.shape)
    print("IC50 test size:", IC50_test.shape)
    print("Emax train size:", Emax_all.shape)
    print("Emax test size:", Emax_test.shape)
    print("X train size:", Xall.shape)
    print("X test size:", Xtest.shape)
    print("Y train size:", Yall.shape)
    print("Y test size:", Ytest.shape)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Val_5folds_X.append(Xtest.copy())
    Train_5folds_X.append(Xall.copy())
    Val_5folds_Y.append(Ytest.copy())
    Train_5folds_Y.append(Yall.copy())
    Val_5folds_AUC.append(AUC_test.copy())
    Train_5folds_AUC.append(AUC_all.copy())
    Val_5folds_IC50.append(IC50_test.copy())
    Train_5folds_IC50.append(IC50_all.copy())
    Val_5folds_Emax.append(Emax_test.copy())
    Train_5folds_Emax.append(Emax_all.copy())
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
        self.minibatch = 200   # mini-batch size for stochastic inference
        self.inducing = 1000    #number of inducing points
        self.N_iter_epoch = 3    #number of iterations
        self.num_latentGPs = 4  #number of latent GPs
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
# "Create a K-fold for cross-validation"
# from sklearn.model_selection import KFold, cross_val_score
# Xind = np.arange(N_per_out)
# k_fold = KFold(n_splits=5,shuffle=True,random_state=0)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Emax_abs_5Folds = []
AUC_abs_5Folds = []
IC50_MSE_5Folds = []
Med_MSE_5Folds = []
AllPred_MSE_5Folds = []
Mean_MSE_5Folds = []
Spearman_5Folds = []
SpearActualIC50_5Folds = []
All_Models = []
for Nfold in range(0,5):
    print(f"Using Fold {Nfold}")

    Xval = Val_5folds_X[Nfold].copy()
    Xtrain = Train_5folds_X[Nfold].copy()
    Yval = Val_5folds_Y[Nfold].copy()
    Ytrain = Train_5folds_Y[Nfold].copy()

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
                    self, inducing_points, variational_distribution, learn_inducing_locations=False
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
        {'params': likelihood.parameters()},
    ], lr=0.005)

    # Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    Ntrain,_ = Ytrain.shape
    show_each = Ntrain//train_loader.batch_size
    refine_lr = [0.005,0.001,0.0005,0.0001]
    refine_num_epochs = [num_epochs,int(num_epochs*0.5),int(num_epochs*0.2),int(num_epochs*0.2)]
    for Nrefine in range(len(refine_lr)):
        print(f"\nRefine Learning Rate {Nrefine}; lr={refine_lr[Nrefine]}")
        for g in optimizer.param_groups:
            g['lr'] = refine_lr[Nrefine]

        for i in range(refine_num_epochs[Nrefine]):
            print(f"Epoch {i}")
            # Within each iteration, we will go over each minibatch of data
            #minibatch_iter = tqdm(train_loader, desc = 'Minibatch')
            for j, (x_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                if j%(show_each//2)==0:
                    print(f"Minbatch {j}, Loss {loss}")

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
    from scipy.interpolate import interp1d

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

        y_resp_interp = f2(x_dose_new)
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
    IC50_pred = np.array(IC50_pred)

    # posy = 100
    # plt.figure(Nfold+10)
    # plt.plot(x_dose_new, Y_pred_interp_all[posy])
    # plt.plot(x_dose, val_y.numpy()[posy, :], '.')
    # plt.plot(IC50_pred[posy], Ydose50_pred[posy], 'rx')
    # plt.plot(x_dose_new, np.ones_like(x_dose_new) * Emax_pred[posy], 'r')  # Plot a horizontal line as Emax
    # plt.plot(x_dose_new, np.ones_like(x_dose_new) * Val_5folds_Emax[Nfold][posy], 'r')  # Plot a horizontal line as Emax
    # plt.title(f"AUC = {AUC_pred[posy]}")
    # print(AUC_pred[posy])

    Emax_abs = np.mean(np.abs(Val_5folds_Emax[Nfold]-Emax_pred))
    AUC_abs = np.mean(np.abs(Val_5folds_AUC[Nfold] - AUC_pred))
    IC50_MSE = np.mean((Val_5folds_IC50[Nfold] - IC50_pred)**2)
    MSE_curves = np.mean((mean.numpy()-val_y.numpy())**2,1)
    AllPred_MSE = np.mean((mean.numpy()-val_y.numpy())**2)
    print("IC50 MSE:", IC50_MSE)
    print("AUC MAE:", AUC_abs)
    print("Emax MAE:", Emax_abs)
    Med_MSE = np.median(MSE_curves)
    Mean_MSE = np.mean(MSE_curves)
    print("Med_MSE:", Med_MSE)
    print("Mean_MSE:", Mean_MSE)
    print("All Predictions MSE:", AllPred_MSE)

    from scipy.stats import spearmanr
    pos_Actual_IC50 = Val_5folds_IC50[Nfold]!=1.5
    spear_corr_all, p_value_all = spearmanr(Val_5folds_IC50[Nfold], IC50_pred)
    spear_corr_actualIC50, p_value_actual = spearmanr(Val_5folds_IC50[Nfold][pos_Actual_IC50], IC50_pred[pos_Actual_IC50[:,0]])
    print ("Spearman_all Corr: ",spear_corr_all)
    print("Spearman p-value: ", p_value_all)
    print("Spearman_actualIC50 Corr: ", spear_corr_actualIC50)
    print("Spearman p-value: ", p_value_actual)

    Emax_abs_5Folds.append(Emax_abs.copy())
    AUC_abs_5Folds.append(AUC_abs.copy())
    IC50_MSE_5Folds.append(IC50_MSE.copy())
    Med_MSE_5Folds.append(Med_MSE.copy())
    Mean_MSE_5Folds.append(Mean_MSE.copy())
    AllPred_MSE_5Folds.append(AllPred_MSE.copy())
    Spearman_5Folds.append(spear_corr_all.copy())
    SpearActualIC50_5Folds.append(spear_corr_actualIC50.copy())
    All_Models.append(model)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # Nrow = 107//2
    # Ncolumn = 4
    # fig, axs = plt.subplots(Nrow, Ncolumn, figsize=(30,50*(Nrow//10)))
    #
    # #test_y, mean
    # posy = 0
    # for i in range(Nrow):
    #     for j in range(Ncolumn):
    #         #posy = np.random.randint(val_y.shape[0])
    #         axs[i, j].plot(np.linspace(0.11,1,9), val_y[posy,:],'xb')
    #         axs[i, j].plot(np.linspace(0.11,1,9), val_y[posy,:],'.b')
    #         axs[i, j].plot(np.linspace(0.11,1,9), mean[posy,:],'-r')
    #         axs[i, j].plot(np.linspace(0.11,1,9), lower[posy,:],'--c')
    #         axs[i, j].plot(np.linspace(0.11,1,9), upper[posy,:],'--c')
    #         #axs[i, j].set_title("r2 = "+str(r2_scr[posy,:]))
    #         axs[i, j].set_ylim([-0.05, 1.05])
    #         posy = posy + 1
    #         if posy == 855:
    #             break
    #
    # print(posy)
    # fig.savefig('FiveCancers.pdf',dpi=300)

# f= open("Metrics.txt","a+")
# f.write("bash"+str(config.bash)+f" Med_MSE={np.mean(Med_MSE_5Folds):0.5f}({np.std(Med_MSE_5Folds):0.5f}) Mean_MSE={np.mean(Mean_MSE_5Folds):0.5f}({np.std(Mean_MSE_5Folds):0.5f}) IC50_MSE={np.mean(IC50_MSE_5Folds):0.5f}({np.std(IC50_MSE_5Folds):0.5f}) Spear_ActualIC50={np.mean(SpearActualIC50_5Folds):0.5f}({np.std(SpearActualIC50_5Folds):0.5f}) Spear_all={np.mean(Spearman_5Folds):0.5f}({np.std(Spearman_5Folds):0.5f}) AUC_abs={np.mean(AUC_abs_5Folds):0.5f}({np.std(AUC_abs_5Folds):0.5f}) Emax_abs ={np.mean(Emax_abs_5Folds):0.5f}({np.std(Emax_abs_5Folds):0.5f})\n")
# f.close()
#
# BestFold_posy = np.where(np.min(Mean_MSE_5Folds)==Mean_MSE_5Folds)[0][0]
#
# #final_path = '/data/ac1jjgg/Data_Marina/GPyTorch_results/FiveCancers_SmallDataset_SparseMOGP/'
# final_path ='model_FiveCancers_SmallData/'
# if not os.path.exists(final_path):
#     os.makedirs(final_path)
# torch.save(All_Models[BestFold_posy].state_dict(), final_path+'m_'+str(config.bash)+'.pth')