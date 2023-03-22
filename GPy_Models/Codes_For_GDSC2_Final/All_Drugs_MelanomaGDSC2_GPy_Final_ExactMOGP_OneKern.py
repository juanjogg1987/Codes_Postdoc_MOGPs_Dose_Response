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

_FOLDER = "/home/ac1jjgg/Dataset_BRAF_NoReplica/GDSC2/"
#_FOLDER = "/home/juanjo/Work_Postdoc/my_codes_postdoc/Dataset_BRAF_NoReplica/GDSC2/"
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
name_for_KLrelevance = 'GDSC2_melanoma_BRAF_noreps_v2.csv'

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

posy = 0
#plt.figure(Nfold)
plt.plot(x_lin, Ydose_res[posy],linewidth=2)
plt.plot(x_real_dose, y_train_drug[posy, :],'.',markersize=12)
plt.plot(IC50[posy], Ydose50[posy], 'rx',markersize=12)
plt.plot(x_lin, np.ones_like(x_lin)*Emax[posy], 'r') #Plot a horizontal line as Emax
plt.title(f"AUC = {AUC[posy]}")
print(AUC[posy])
plt.legend(["Sigmoid4","Data","IC50","Emax"])
plt.xlabel(r"Dose concentration ($d_j$)")
plt.ylabel(r"Dose response (living cancer cells)")
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

class commandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'i:s:k:w:r:p:')
        # opts = dict(opts)
        # print(opts)
        self.N_iter_epoch = 1500    #number of iterations
        self.which_seed = 1010    #change seed to initialise the hyper-parameters
        self.rank = 6
        self.scale = 3.0
        self.weight = 0.1
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
            if op == '-w':
                self.weight = arg



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
config = commandLine()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"The indexes below are 6 with response (5,74,101) and 6 without reponse (10, 80, 140, i.e., IC50=1.5)"
##([ 5,  10,  74, 80, 101, 140])

index_Test = np.array([ 5,  10,  74, 80, 101, 140])
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
nsplits = 10 #Ndata
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
Ntasks = 7
list_folds = list(k_fold.split(Xall))
for Nfold in range(0,nsplits+1):
    model = []
    "The first if below is for the cross-val"
    "Then the else is for using all data to save the model trained over all data"
    if Nfold<nsplits:
        train_ind, test_ind = list_folds[Nfold]
        print(f"{test_ind} to Val in IC50")

        Xval_aux = Xall[test_ind].copy()
        Ylabel_val = np.array([i * np.ones(Xval_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]

        Xval = np.concatenate((np.tile(Xval_aux,(Ntasks,1)), Ylabel_val), 1)

        #Xval = Xall[train_ind].copy()
        Xtrain_aux = Xall[train_ind].copy()
        Ylabel_train = np.array([i * np.ones(Xtrain_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]
        Xtrain = np.concatenate((np.tile(Xtrain_aux, (Ntasks, 1)), Ylabel_train), 1)

        Yval = Yall[test_ind].T.flatten().copy()[:,None]
        #Yval = Yall[train_ind].copy()
        Ytrain = Yall[train_ind].T.flatten().copy()[:,None]

        Emax_val = Emax_all[test_ind].copy()
        AUC_val = AUC_all[test_ind].copy()
        IC50_val = IC50_all[test_ind].copy()

        #Emax_val = Emax_all[train_ind].copy()
        #AUC_val = AUC_all[train_ind].copy()
        #IC50_val = IC50_all[train_ind].copy()

    else:
        print(f"Train ovell all Data in IC50")

        Xval_aux = Xtest_final.copy()
        Ylabel_val = np.array([i * np.ones(Xval_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]

        Xval = np.concatenate((np.tile(Xval_aux, (Ntasks, 1)), Ylabel_val), 1)

        Xtrain_aux = Xall.copy()
        Ylabel_train = np.array([i * np.ones(Xtrain_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]
        Xtrain = np.concatenate((np.tile(Xtrain_aux, (Ntasks, 1)), Ylabel_train), 1)

        Yval = Ytest_final.T.flatten().copy()[:,None]
        Ytrain = Yall.T.flatten().copy()[:,None]

        Emax_val = Emax_test_final.copy()
        AUC_val = AUC_test_final.copy()
        IC50_val = IC50_test_final.copy()
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    import GPy
    from matplotlib import pyplot as plt

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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
    "Below we substract one due to being the label associated to the output"
    Dim = Xtrain.shape[1]-1

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #num_latents = int(config.num_latentGPs) #9
    #num_tasks = Ytrain.shape[1]
    #split_dim = int(config.split_dim)
    #num_inducing = int(config.inducing)

    myseed = int(config.which_seed)
    np.random.seed(myseed)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    mykern = GPy.kern.RBF(Dim)
    mykern.lengthscale = float(config.scale)* np.sqrt(Dim) * np.random.rand()
    mykern.variance.fix()
    kern = mykern ** GPy.kern.Coregionalize(1, output_dim=Ntasks,rank=rank)
    model = GPy.models.GPRegression(Xtrain, Ytrain, kern)
    Init_Ws = float(config.weight) * np.random.randn(Ntasks,rank)
    model.kern.coregion.W = Init_Ws
    #model.optimize(optimizer='lbfgsb',messages=True,max_iters=30)
    #model.optimize(max_iters=int(config.N_iter_epoch))
    model.optimize()

    m_pred, v_pred = model.predict(Xval, full_cov=False)
    plt.figure(Nfold+5)
    plt.plot(Yval, 'bx')
    plt.plot(m_pred, 'ro')
    plt.plot(m_pred + 2 * np.sqrt(v_pred), '--m')
    plt.plot(m_pred - 2 * np.sqrt(v_pred), '--m')

    Yval_curve = Yval.reshape(Ntasks, Xval_aux.shape[0]).T.copy()
    m_pred_curve = m_pred.reshape(Ntasks, Xval_aux.shape[0]).T.copy()
    v_pred_curve = v_pred.reshape(Ntasks, Xval_aux.shape[0]).T.copy()

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    "Negative Log Predictive Density (NLPD)"
    Val_NMLL = -np.mean(model.log_predictive_density(Xval,Yval))
    print("NegLPD Val", Val_NMLL)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    from scipy.interpolate import interp1d
    from scipy.interpolate import pchip_interpolate

    x_dose = np.linspace(0.111111, 1.0, 7)
    x_dose_new = np.linspace(0.111111, 1.0, 1000)
    Ydose50_pred = []
    IC50_pred = []
    AUC_pred = []
    Emax_pred = []
    Y_pred_interp_all = []
    std_upper_interp_all = []
    std_lower_interp_all = []
    for i in range(Yval_curve.shape[0]):
        y_resp = m_pred_curve[i, :].copy()
        std_upper = y_resp + 2*np.sqrt(v_pred_curve[i, :])
        std_lower = y_resp - 2 * np.sqrt(v_pred_curve[i, :])
        f = interp1d(x_dose, y_resp)
        #f2 = interp1d(x_dose, y_resp, kind='cubic')
        y_resp_interp = pchip_interpolate(x_dose, y_resp, x_dose_new)
        std_upper_interp = pchip_interpolate(x_dose, std_upper, x_dose_new)
        std_lower_interp = pchip_interpolate(x_dose, std_lower, x_dose_new)

        #y_resp_interp = f2(x_dose_new)
        Y_pred_interp_all.append(y_resp_interp)
        std_upper_interp_all.append(std_upper_interp)
        std_lower_interp_all.append(std_lower_interp)
        AUC_pred.append(metrics.auc(x_dose_new, y_resp_interp))
        Emax_pred.append(y_resp_interp[-1])

        res1 = y_resp_interp < 0.507
        res2 = y_resp_interp > 0.493
        res_aux = np.where(res1 & res2)[0]
        if (res1 & res2).sum()>0:
            res_IC50 = np.arange(res_aux[0],res_aux[0]+ res_aux.shape[0])==res_aux
            res_aux = res_aux[res_IC50].copy()
        else:
            res_aux = res1 & res2

        if (res1 & res2).sum() > 0:
            Ydose50_pred.append(y_resp_interp[res_aux].mean())
            IC50_pred.append(x_dose_new[res_aux].mean())
        elif y_resp_interp[-1] < 0.5:
            Ydose50_pred.append(y_resp_interp[i].max())
            aux_IC50 = x_dose_new[np.where(y_resp_interp[i]==y_resp_interp[i].max())[0]][0] #x_dose_new[np.where(y_resp_interp[i].max())[0]][0]  # it has to be a float not an array to avoid bug
            IC50_pred.append(aux_IC50)
        else:
            Ydose50_pred.append(0.5)
            IC50_pred.append(1.5)

    Ydose50_pred = np.array(Ydose50_pred)
    IC50_pred = np.array(IC50_pred)[:,None]
    AUC_pred = np.array(AUC_pred)[:, None]
    Emax_pred = np.array(Emax_pred)[:, None]

    posy = 5
    plt.figure(Nfold)
    plt.plot(x_dose_new, Y_pred_interp_all[posy])
    plt.plot(x_dose_new, std_upper_interp_all[posy],'b--')
    plt.plot(x_dose_new, std_lower_interp_all[posy], 'b--')
    plt.plot(x_dose, Yval_curve[posy, :], '.')
    plt.plot(IC50_pred[posy], Ydose50_pred[posy], 'rx')
    plt.plot(x_dose_new, np.ones_like(x_dose_new) * Emax_pred[posy], 'r')  # Plot a horizontal line as Emax
    plt.plot(x_dose_new, np.ones_like(x_dose_new) * Emax_val[posy], 'r')  # Plot a horizontal line as Emax
    plt.title(f"AUC = {AUC_pred[posy]}")
    print(AUC_pred[posy])

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    Emax_abs = np.mean(np.abs(Emax_val - Emax_pred))
    AUC_abs = np.mean(np.abs(AUC_val - AUC_pred))
    IC50_MSE = np.mean((IC50_val - IC50_pred) ** 2)
    MSE_curves = np.mean((m_pred_curve - Yval_curve) ** 2, 1)
    AllPred_MSE = np.mean((m_pred_curve - Yval_curve) ** 2)
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
    spear_corr_actualIC50, p_value_actual = spearmanr(IC50_val[pos_Actual_IC50],IC50_pred[pos_Actual_IC50])
    print("Spearman_all Corr: ", spear_corr_all)
    print("Spearman p-value: ", p_value_all)
    print("Spearman_actualIC50 Corr: ", spear_corr_actualIC50)
    print("Spearman p-value: ", p_value_actual)

    if Nfold < nsplits:
        NegMLL_AllFolds.append(Val_NMLL.copy())
        Emax_abs_AllFolds.append(Emax_abs.copy())
        AUC_abs_AllFolds.append(AUC_abs.copy())
        IC50_MSE_AllFolds.append(IC50_MSE.copy())
        Med_MSE_AllFolds.append(Med_MSE.copy())
        Mean_MSE_AllFolds.append(Mean_MSE.copy())
        AllPred_MSE_AllFolds.append(AllPred_MSE.copy())
        Spearman_AllFolds.append(spear_corr_all)
        SpearActualIC50_AllFolds.append(spear_corr_actualIC50)
    #break
    print("Yval shape",Yval.shape)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
f= open("Metrics.txt","a+")
f.write("bash"+str(config.bash)+f" Med_MSE={np.mean(Med_MSE_AllFolds):0.5f}({np.std(Med_MSE_AllFolds):0.5f}) Mean_MSE={np.mean(Mean_MSE_AllFolds):0.5f}({np.std(Mean_MSE_AllFolds):0.5f}) NegLPD={np.mean(NegMLL_AllFolds):0.5f}({np.std(NegMLL_AllFolds):0.5f}) IC50_MSE={np.mean(IC50_MSE_AllFolds):0.5f}({np.std(IC50_MSE_AllFolds):0.5f}) Spear_ActualIC50={np.mean(SpearActualIC50_AllFolds):0.5f}({np.std(SpearActualIC50_AllFolds):0.5f}) Spear_all={np.mean(Spearman_AllFolds):0.5f}({np.std(Spearman_AllFolds):0.5f}) AUC_abs={np.mean(AUC_abs_AllFolds):0.5f}({np.std(AUC_abs_AllFolds):0.5f}) Emax_abs ={np.mean(Emax_abs_AllFolds):0.5f}({np.std(Emax_abs_AllFolds):0.5f})\n")
f.close()

ftest= open("Metrics_Test.txt","a+")
ftest.write("bash"+str(config.bash)+f" Med_MSE={Med_MSE:0.5f} Mean_MSE={Mean_MSE:0.5f} NegLPD={Val_NMLL:0.5f} IC50_MSE={IC50_MSE:0.5f} Spear_ActualIC50={spear_corr_actualIC50:0.5f} Spear_all={spear_corr_all:0.5f} AUC_abs={AUC_abs:0.5f} Emax_abs ={Emax_abs:0.5f} \n")
ftest.close()

"The last model should have been trained over all dataset without splitting"

final_path = '/data/ac1jjgg/Data_Marina/GPy_results/Codes_for_GDSC2_Final/MelanomaGDSC2_SmallDataset_Final_ExactMOGP_OneKern/'
#final_path ='model_MelanomaGDSC2_Final_ExactMOGP_OneKern/'
if not os.path.exists(final_path):
   os.makedirs(final_path)
np.save(final_path+'m_'+str(config.bash)+'.npy', model.param_array)