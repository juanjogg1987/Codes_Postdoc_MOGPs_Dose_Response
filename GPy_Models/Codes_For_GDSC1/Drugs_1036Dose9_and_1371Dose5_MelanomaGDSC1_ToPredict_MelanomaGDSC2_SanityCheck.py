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

_FOLDER = "/home/ac1jjgg/Dataset_BRAF/GDSC1/"
#_FOLDER = "/home/juanjo/Work_Postdoc/my_codes_postdoc/GPyTorch_Models/Dataset_BRAF/GDSC1/"
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
"Read Data for Drug 1036 with 9 Concentrations (or Doses)"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"#This file only contains drugs 1061 and 1036 with 9 concentrations"
name_for_KLrelevance_Dose9 = 'GDSC1_melanoma_BRAF_9conc.csv'

df_train_No_MolecForm_Dose9 = pd.read_csv(_FOLDER + name_for_KLrelevance_Dose9)  # Contain Train dataset prepared by Subhashini-Evelyn
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
df_train_No_MolecForm_Dose9 = df_train_No_MolecForm_Dose9[(df_train_No_MolecForm_Dose9["DRUG_ID"]==1036)]
try:
    df_train_No_MolecForm_Dose9 = df_train_No_MolecForm_Dose9.drop(columns='Drug_Name')
except:
    pass

# Here we just check that from the column index 29 the input features start for Drug1036 dose9
start_pos_features_Dose9 = 29
print(df_train_No_MolecForm_Dose9.columns[start_pos_features_Dose9])

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Read Data for Drug 1371 with 5 Concentrations (or Doses)"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"#This file only contains drugs 1371 and 1373 with 5 concentrations"
name_for_KLrelevance_Dose5 = 'GDSC1_melanoma_BRAF_5conc.csv'

df_train_No_MolecForm_Dose5 = pd.read_csv(_FOLDER + name_for_KLrelevance_Dose5)  # Contain Train dataset prepared by Subhashini-Evelyn
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
df_train_No_MolecForm_Dose5 = df_train_No_MolecForm_Dose5[(df_train_No_MolecForm_Dose5["DRUG_ID"]==1371)]
try:
    df_train_No_MolecForm_Dose5 = df_train_No_MolecForm_Dose5.drop(columns='Drug_Name')
except:
    pass

# Here we just check that from the column index 21 the input features start for Drug1371 dose5
start_pos_features_Dose5 = 21
print(df_train_No_MolecForm_Dose5.columns[start_pos_features_Dose5])

"We concatenate df_dose9 and df_dose5 to scale features together"
df_train_features_concat = pd.concat([df_train_No_MolecForm_Dose9[df_train_No_MolecForm_Dose9.columns[start_pos_features_Dose9:]], df_train_No_MolecForm_Dose5[df_train_No_MolecForm_Dose5.columns[start_pos_features_Dose5:]]])

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Below we scale concatenated features from Dose9 and Dose5"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
scaler = MinMaxScaler().fit(df_train_features_concat)
X_train_features = scaler.transform(df_train_features_concat)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"For Dose 9 responses"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Below we select 9 concentration since GDSC1 has that for Drugs 1036 and 1061"
y_train_drug_Dose9 = np.clip(df_train_No_MolecForm_Dose9["norm_cells_" + str(1)].values[:, None], 1.0e-9, np.inf)
print(y_train_drug_Dose9.shape)
for i in range(2, 10):
    y_train_drug_Dose9 = np.concatenate(
        (y_train_drug_Dose9, np.clip(df_train_No_MolecForm_Dose9["norm_cells_" + str(i)].values[:, None], 1.0e-9, np.inf)), 1)

print("Ytrain Dose 9 size: ", y_train_drug_Dose9.shape)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

params_4_sig_train_Dose9 = df_train_No_MolecForm_Dose9["param_" + str(1)].values[:, None]
for i in range(2, 5):  #here there are four params for sigmoid4
    params_4_sig_train_Dose9 = np.concatenate(
        (params_4_sig_train_Dose9, df_train_No_MolecForm_Dose9["param_" + str(i)].values[:, None]), 1)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"For Dose 9 responses"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import matplotlib.pyplot as plt
from sklearn import metrics

plt.close('all')

x_lin_Dose9 = np.linspace(0.111111, 1, 1000)
x_real_dose_Dose9 = np.linspace(0.111111, 1, 9)  #Here is 9 due to using GDSC1 that has 9 doses for Drugs 1036 and 1061
x_lin_tile_Dose9 = np.tile(x_lin_Dose9, (params_4_sig_train_Dose9.shape[0], 1))
# (x_lin,params_4_sig_train.shape[0],1).shape
Ydose_res_Dose9 = []
AUC_Dose9 = []
IC50_Dose9 = []
Ydose50_Dose9 = []
Emax_Dose9 = []
for i in range(params_4_sig_train_Dose9.shape[0]):
    Ydose_res_Dose9.append(sigmoid_4_param(x_lin_tile_Dose9[i, :], *params_4_sig_train_Dose9[i, :]))
    AUC_Dose9.append(metrics.auc(x_lin_tile_Dose9[i, :], Ydose_res_Dose9[i]))
    Emax_Dose9.append(Ydose_res_Dose9[i][-1])
    res1 = (Ydose_res_Dose9[i] < 0.507)
    res2 = (Ydose_res_Dose9[i] > 0.493)
    if (res1 & res2).sum() > 0:
        Ydose50_Dose9.append(Ydose_res_Dose9[i][res1 & res2].mean())
        IC50_Dose9.append(x_lin_Dose9[res1 & res2].mean())
    elif Ydose_res_Dose9[i][-1]<0.5:
       Ydose50_Dose9.append(Ydose_res_Dose9[i].max())
       aux_IC50_Dose9 = x_lin_Dose9[np.where(Ydose_res_Dose9[i].max())[0]][0]  #it has to be a float not an array to avoid bug
       IC50_Dose9.append(aux_IC50_Dose9)
    else:
        Ydose50_Dose9.append(0.5)
        IC50_Dose9.append(1.5) #IC50.append(x_lin[-1])

posy = 6
plt.figure(0)
plt.plot(x_lin_Dose9, Ydose_res_Dose9[posy])
plt.plot(x_real_dose_Dose9, y_train_drug_Dose9[posy, :], '.')
plt.plot(IC50_Dose9[posy], Ydose50_Dose9[posy], 'rx')
plt.plot(x_lin_Dose9, np.ones_like(x_lin_Dose9)*Emax_Dose9[posy], 'r') #Plot a horizontal line as Emax
plt.title(f"AUC = {AUC_Dose9[posy]}")
print(AUC_Dose9[posy])

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"For Dose 5 responses"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Below we select 9 concentration since GDSC1 has that for Drugs 1036 and 1061"
y_train_drug_Dose5 = np.clip(df_train_No_MolecForm_Dose5["norm_cells_" + str(1)].values[:, None], 1.0e-9, np.inf)
print(y_train_drug_Dose5.shape)
for i in range(2, 6):
    y_train_drug_Dose5 = np.concatenate(
        (y_train_drug_Dose5, np.clip(df_train_No_MolecForm_Dose5["norm_cells_" + str(i)].values[:, None], 1.0e-9, np.inf)), 1)

print("Ytrain Dose 5 size: ", y_train_drug_Dose5.shape)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
params_4_sig_train_Dose5 = df_train_No_MolecForm_Dose5["param_" + str(1)].values[:, None]
for i in range(2, 5):  #here there are four params for sigmoid4
    params_4_sig_train_Dose5 = np.concatenate(
        (params_4_sig_train_Dose5, df_train_No_MolecForm_Dose5["param_" + str(i)].values[:, None]), 1)

x_lin_Dose5 = np.linspace(0.111111, 1, 1000)
x_real_dose_Dose5 = np.linspace(0.111111, 1, 5)  #Here is 5 due to using GDSC1 that has 5 doses for Drugs 1371
x_lin_tile_Dose5 = np.tile(x_lin_Dose5, (params_4_sig_train_Dose5.shape[0], 1))
# (x_lin,params_4_sig_train.shape[0],1).shape
Ydose_res_Dose5 = []
AUC_Dose5 = []
IC50_Dose5 = []
Ydose50_Dose5 = []
Emax_Dose5 = []
for i in range(params_4_sig_train_Dose5.shape[0]):
    Ydose_res_Dose5.append(sigmoid_4_param(x_lin_tile_Dose5[i, :], *params_4_sig_train_Dose5[i, :]))
    AUC_Dose5.append(metrics.auc(x_lin_tile_Dose5[i, :], Ydose_res_Dose5[i]))
    Emax_Dose5.append(Ydose_res_Dose5[i][-1])
    res1 = (Ydose_res_Dose5[i] < 0.507)
    res2 = (Ydose_res_Dose5[i] > 0.493)
    if (res1 & res2).sum() > 0:
        Ydose50_Dose5.append(Ydose_res_Dose5[i][res1 & res2].mean())
        IC50_Dose5.append(x_lin_Dose5[res1 & res2].mean())
    elif Ydose_res_Dose5[i][-1]<0.5:
       Ydose50_Dose5.append(Ydose_res_Dose5[i].max())
       aux_IC50_Dose5 = x_lin_Dose5[np.where(Ydose_res_Dose5[i].max())[0]][0]  #it has to be a float not an array to avoid bug
       IC50_Dose5.append(aux_IC50_Dose5)
    else:
        Ydose50_Dose5.append(0.5)
        IC50_Dose5.append(1.5) #IC50.append(x_lin[-1])

posy = 6
plt.figure(0)
plt.plot(x_lin_Dose5, Ydose_res_Dose5[posy])
plt.plot(x_real_dose_Dose5, y_train_drug_Dose5[posy, :], '.')
plt.plot(IC50_Dose5[posy], Ydose50_Dose5[posy], 'rx')
plt.plot(x_lin_Dose5, np.ones_like(x_lin_Dose5)*Emax_Dose5[posy], 'r') #Plot a horizontal line as Emax
plt.title(f"AUC = {AUC_Dose5[posy]}")
print(AUC_Dose5[posy])
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Compute Log(AUC)? R/ Not for Functional Random Forest Model
AUC_Dose9 = np.array(AUC_Dose9)
IC50_Dose9 = np.array(IC50_Dose9)
Emax_Dose9 = np.array(Emax_Dose9)

AUC_Dose5 = np.array(AUC_Dose5)
IC50_Dose5 = np.array(IC50_Dose5)
Emax_Dose5 = np.array(Emax_Dose5)

"Below we select just the columns with std higher than zero"
ind_nonzero_std = np.where(X_train_features.std(0)!=0.0)[0]
AllNames_Features = df_train_features_concat.columns
Name_Features_Melanoma = AllNames_Features[ind_nonzero_std]
Xall = X_train_features[:,ind_nonzero_std].copy()

Yall_Dose9 = y_train_drug_Dose9.copy()
Yall_Dose5 = y_train_drug_Dose5.copy()

AUC_all_Dose9 = AUC_Dose9[:, None].copy()
IC50_all_Dose9 = IC50_Dose9[:, None].copy()
Emax_all_Dose9 = Emax_Dose9[:, None].copy()

AUC_all_Dose5 = AUC_Dose5[:, None].copy()
IC50_all_Dose5 = IC50_Dose5[:, None].copy()
Emax_all_Dose5 = Emax_Dose5[:, None].copy()

print("AUC Dose9 train size:", AUC_all_Dose9.shape)
print("IC50 Dose9 train size:", IC50_all_Dose9.shape)
print("Emax Dose9 train size:", Emax_all_Dose9.shape)
print("AUC Dose5 train size:", AUC_all_Dose5.shape)
print("IC50 Dose5 train size:", IC50_all_Dose5.shape)
print("Emax Dose5 train size:", Emax_all_Dose5.shape)
print("X all data size:", Xall.shape)
print("Y all Dose9 data size:", Yall_Dose9.shape)
print("Y all Dose5 data size:", Yall_Dose5.shape)

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
        self.rank = 7
        self.scale = 0.1
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
"Drug1036: The indexes below are 1 with response (4) and non-response (44, i.e., IC50 = 1.5)"

"For Drug1036 np.array([4,44])"
"For Drug1371 np.array([10,68,160,171]), values taken from training only Drug1371"
"Notice that below we add 50 (number points for Drug1036) since we concatenated Features of Drug1036 and Drug1371"

index_Test_Dose9 = np.array([4,44])
index_Test_Dose5 = np.array([10,68,160,171])
index_Test = np.concatenate((index_Test_Dose9,index_Test_Dose5+50))  #50 is number of observations for Dose9

index_Train = np.arange(0,Xall.shape[0])
Xtest_final = Xall[index_Test,:].copy()
Xall = Xall[np.delete(index_Train,index_Test),:].copy()

Ytest_final_Dose9 = Yall_Dose9[index_Test_Dose9,:].copy()
Ytest_final_Dose5 = Yall_Dose5[index_Test_Dose5,:].copy()

index_Train_Dose9 = np.arange(0,Yall_Dose9.shape[0])
index_Train_Dose5 = np.arange(0,Yall_Dose5.shape[0])
Yall_Dose9 = Yall_Dose9[np.delete(index_Train_Dose9,index_Test_Dose9),:].copy()
Yall_Dose5 = Yall_Dose5[np.delete(index_Train_Dose5,index_Test_Dose5),:].copy()

IC50_test_final_Dose9 = IC50_all_Dose9[index_Test_Dose9,:].copy()
IC50_all_Dose9 = IC50_all_Dose9[np.delete(index_Train_Dose9,index_Test_Dose9),:].copy()
AUC_test_final_Dose9 = AUC_all_Dose9[index_Test_Dose9,:].copy()
AUC_all_Dose9 = AUC_all_Dose9[np.delete(index_Train_Dose9,index_Test_Dose9),:].copy()
Emax_test_final_Dose9 = Emax_all_Dose9[index_Test_Dose9,:].copy()
Emax_all_Dose9 = Emax_all_Dose9[np.delete(index_Train_Dose9,index_Test_Dose9),:].copy()

IC50_test_final_Dose5 = IC50_all_Dose5[index_Test_Dose5,:].copy()
IC50_all_Dose5 = IC50_all_Dose5[np.delete(index_Train_Dose5,index_Test_Dose5),:].copy()
AUC_test_final_Dose5 = AUC_all_Dose5[index_Test_Dose5,:].copy()
AUC_all_Dose5 = AUC_all_Dose5[np.delete(index_Train_Dose5,index_Test_Dose5),:].copy()
Emax_test_final_Dose5 = Emax_all_Dose5[index_Test_Dose5,:].copy()
Emax_all_Dose5 = Emax_all_Dose5[np.delete(index_Train_Dose5,index_Test_Dose5),:].copy()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Create a K-fold for cross-validation"
from sklearn.model_selection import KFold, cross_val_score
#Ndata = Xall.shape[0]
#Xind = np.arange(Ndata)
nsplits = 10 #
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
Ntasks9 = 9   #Equal to the number of drug concentrations
Ntasks5 = 5   #Equal to the number of drug concentrations
list_folds_Dose9 = list(k_fold.split(Yall_Dose9))
list_folds_Dose5 = list(k_fold.split(Yall_Dose5))
for Nfold in range(0,nsplits+1):
    model = []
    "The first if below is for the cross-val"
    "Then the else is for using all data to save the model trained over all data"
    if Nfold<nsplits:
        train_ind_Dose9, test_ind_Dose9 = list_folds_Dose9[Nfold]
        print(f"{test_ind_Dose9} to Val in IC50 Dose9")

        train_ind_Dose5, test_ind_Dose5 = list_folds_Dose5[Nfold]
        print(f"{test_ind_Dose5} to Val in IC50 Dose5")

        Noffset = Yall_Dose9.shape[0]
        #train_ind = np.concatenate((train_ind_Dose9,train_ind_Dose5+Noffset))
        #test_ind = np.concatenate((test_ind_Dose9, test_ind_Dose5 + Noffset))

        Xval_aux_Dose9 = Xall[test_ind_Dose9].copy()
        Xval_aux_Dose5 = Xall[test_ind_Dose5+Noffset].copy()
        Ylabel_val_Dose9 = np.array([i * np.ones(test_ind_Dose9.shape[0]) for i in range(Ntasks9)]).flatten()[:, None]
        Ylabel_val_Dose5 = np.array([2*i * np.ones(test_ind_Dose5.shape[0]) for i in range(Ntasks5)]).flatten()[:, None]

        #Ylabel_val = np.concatenate((Ylabel_val_Dose9,Ylabel_val_Dose5))

        Xval_Dose9 = np.concatenate((np.tile(Xval_aux_Dose9,(Ntasks9,1)), Ylabel_val_Dose9), 1)
        Xval_Dose5 = np.concatenate((np.tile(Xval_aux_Dose5, (Ntasks5, 1)), Ylabel_val_Dose5), 1)
        Xval = np.concatenate((Xval_Dose9,Xval_Dose5))

        #Xval = Xall[train_ind].copy()
        #Xtrain_aux = Xall[train_ind].copy()
        #Ylabel_train = np.array([i * np.ones(Xtrain_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]
        #Xtrain = np.concatenate((np.tile(Xtrain_aux, (Ntasks, 1)), Ylabel_train), 1)

        Xtrain_aux_Dose9 = Xall[train_ind_Dose9].copy()
        Xtrain_aux_Dose5 = Xall[train_ind_Dose5 + Noffset].copy()
        Ylabel_train_Dose9 = np.array([i * np.ones(train_ind_Dose9.shape[0]) for i in range(Ntasks9)]).flatten()[:, None]
        Ylabel_train_Dose5 = np.array([2 * i * np.ones(train_ind_Dose5.shape[0]) for i in range(Ntasks5)]).flatten()[:,None]

        Xtrain_Dose9 = np.concatenate((np.tile(Xtrain_aux_Dose9, (Ntasks9, 1)), Ylabel_train_Dose9), 1)
        Xtrain_Dose5 = np.concatenate((np.tile(Xtrain_aux_Dose5, (Ntasks5, 1)), Ylabel_train_Dose5), 1)
        Xtrain = np.concatenate((Xtrain_Dose9, Xtrain_Dose5))

        Yval_Dose9 = Yall_Dose9[test_ind_Dose9].T.flatten().copy()[:,None]
        Yval_Dose5 = Yall_Dose5[test_ind_Dose5].T.flatten().copy()[:, None]
        Yval = np.vstack((Yval_Dose9,Yval_Dose5))

        #Ytrain = Yall[train_ind].T.flatten().copy()[:,None]

        Ytrain_Dose9 = Yall_Dose9[train_ind_Dose9].T.flatten().copy()[:,None]
        Ytrain_Dose5 = Yall_Dose5[train_ind_Dose5].T.flatten().copy()[:, None]
        Ytrain = np.vstack((Ytrain_Dose9, Ytrain_Dose5))

        # Emax_val = Emax_all[test_ind].copy()
        # AUC_val = AUC_all[test_ind].copy()
        # IC50_val = IC50_all[test_ind].copy()

        Emax_val_Dose9 = Emax_all_Dose9[test_ind_Dose9].copy()
        AUC_val_Dose9 = AUC_all_Dose9[test_ind_Dose9].copy()
        IC50_val_Dose9  = IC50_all_Dose9[test_ind_Dose9].copy()

        Emax_val_Dose5 = Emax_all_Dose5[test_ind_Dose5].copy()
        AUC_val_Dose5 = AUC_all_Dose5[test_ind_Dose5].copy()
        IC50_val_Dose5 = IC50_all_Dose5[test_ind_Dose5].copy()

        Emax_val = np.vstack((Emax_val_Dose9,Emax_val_Dose5))
        AUC_val = np.vstack((AUC_val_Dose9,AUC_val_Dose5))
        IC50_val = np.vstack((IC50_val_Dose9,IC50_val_Dose5))

    else:
        print(f"Train ovell all Data in IC50")

        Noffset = Yall_Dose9.shape[0]

        #Xval_aux_Dose9 = Xtest_final
        #Xval_aux_Dose5 = Xall[test_ind_Dose5 + Noffset].copy()

        #Xval_aux = Xtest_final.copy()
        Xval_aux_Dose9 = Xtest_final[0:index_Test_Dose9.shape[0]].copy()
        Xval_aux_Dose5 = Xtest_final[index_Test_Dose9.shape[0]:].copy()
        #Ylabel_val = np.array([i * np.ones(Xval_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]
        Ylabel_val_Dose9 = np.array([i * np.ones(index_Test_Dose9.shape[0]) for i in range(Ntasks9)]).flatten()[:, None]
        Ylabel_val_Dose5 = np.array([2 * i * np.ones(index_Test_Dose5.shape[0]) for i in range(Ntasks5)]).flatten()[:,None]

        Xval_Dose9 = np.concatenate((np.tile(Xval_aux_Dose9, (Ntasks9, 1)), Ylabel_val_Dose9), 1)
        Xval_Dose5 = np.concatenate((np.tile(Xval_aux_Dose5, (Ntasks5, 1)), Ylabel_val_Dose5), 1)
        Xval = np.concatenate((Xval_Dose9, Xval_Dose5))

        Xtrain_aux_Dose9 = Xall[0:Noffset].copy()
        Xtrain_aux_Dose5 = Xall[Noffset:].copy()
        Ylabel_train_Dose9 = np.array([i * np.ones(Yall_Dose9.shape[0]) for i in range(Ntasks9)]).flatten()[:,None]
        Ylabel_train_Dose5 = np.array([2 * i * np.ones(Yall_Dose5.shape[0]) for i in range(Ntasks5)]).flatten()[:,None]

        Xtrain_Dose9 = np.concatenate((np.tile(Xtrain_aux_Dose9, (Ntasks9, 1)), Ylabel_train_Dose9), 1)
        Xtrain_Dose5 = np.concatenate((np.tile(Xtrain_aux_Dose5, (Ntasks5, 1)), Ylabel_train_Dose5), 1)
        Xtrain = np.concatenate((Xtrain_Dose9, Xtrain_Dose5))

        Yval_Dose9 = Ytest_final_Dose9.T.flatten().copy()[:, None]
        Yval_Dose5 = Ytest_final_Dose5.T.flatten().copy()[:, None]
        Yval = np.vstack((Yval_Dose9, Yval_Dose5))

        Ytrain_Dose9 = Yall_Dose9.T.flatten().copy()[:, None]
        Ytrain_Dose5 = Yall_Dose5.T.flatten().copy()[:, None]
        Ytrain = np.vstack((Ytrain_Dose9, Ytrain_Dose5))

        Emax_val_Dose9 = Emax_test_final_Dose9.copy()
        AUC_val_Dose9 = AUC_test_final_Dose9.copy()
        IC50_val_Dose9 = IC50_test_final_Dose9.copy()

        Emax_val_Dose5 = Emax_test_final_Dose5.copy()
        AUC_val_Dose5 = AUC_test_final_Dose5.copy()
        IC50_val_Dose5 = IC50_test_final_Dose5.copy()

        Emax_val = np.vstack((Emax_val_Dose9, Emax_val_Dose5))
        AUC_val = np.vstack((AUC_val_Dose9, AUC_val_Dose5))
        IC50_val = np.vstack((IC50_val_Dose9, IC50_val_Dose5))

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

    kern = GPy.kern.RBF(Dim, active_dims=list(np.arange(0, Dim))) ** GPy.kern.Coregionalize(1, output_dim=Ntasks9,rank=rank)
    kern.rbf.lengthscale = float(config.scale) * np.sqrt(Dim) * np.random.rand()
    kern.rbf.variance.fix()   #Fix variance in 1.0
    model = GPy.models.GPRegression(Xtrain, Ytrain, kern)
    Init_Ws = float(config.weight) * np.random.randn(Ntasks9,rank)
    model.kern.coregion.W = Init_Ws
    #model.optimize(optimizer='lbfgsb',messages=True,max_iters=30)
    #model.optimize(max_iters=int(config.N_iter_epoch))
    model.optimize()

    m_pred, v_pred = model.predict(Xval, full_cov=False)
    plt.figure(30)
    plt.plot(Yval, 'bx')
    plt.plot(m_pred, 'ro')
    plt.plot(m_pred + 2 * np.sqrt(v_pred), '--m')
    plt.plot(m_pred - 2 * np.sqrt(v_pred), '--m')

    Nval_Dose9 = Xval_aux_Dose9.shape[0]

    #Yval_curve = Yval.reshape(Ntasks, Xval_aux.shape[0]).T.copy()
    #m_pred_curve = m_pred.reshape(Ntasks, Xval_aux.shape[0]).T.copy()
    #v_pred_curve = v_pred.reshape(Ntasks, Xval_aux.shape[0]).T.copy()

    Yval_curve_Dose9 = Yval_Dose9.reshape(Ntasks9, Nval_Dose9).T.copy()
    m_pred_Dose9 = m_pred[0:Yval_Dose9.shape[0]].copy()
    v_pred_Dose9 = v_pred[0:Yval_Dose9.shape[0]].copy()
    m_pred_curve_Dose9 = m_pred_Dose9.reshape(Ntasks9, Nval_Dose9).T.copy()
    v_pred_curve_Dose9 = v_pred_Dose9.reshape(Ntasks9, Nval_Dose9).T.copy()
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Nval_Dose5 = Xval_aux_Dose5.shape[0]

    Yval_curve_Dose5 = Yval_Dose5.reshape(Ntasks5, Nval_Dose5).T.copy()
    m_pred_Dose5 = m_pred[Yval_Dose9.shape[0]:].copy()    #From the size of the data for Dose9 until the end
    v_pred_Dose5 = v_pred[Yval_Dose9.shape[0]:].copy()    #From the size of the data for Dose9 until the end
    m_pred_curve_Dose5 = m_pred_Dose5.reshape(Ntasks5, Nval_Dose5).T.copy()
    v_pred_curve_Dose5 = v_pred_Dose5.reshape(Ntasks5, Nval_Dose5).T.copy()

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    "Negative Log Predictive Density (NLPD)"
    Val_NMLL = -np.mean(model.log_predictive_density(Xval,Yval))
    print("NegLPD Val", Val_NMLL)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "Interpolation of predictions for Dose9"
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    from scipy.interpolate import interp1d
    from scipy.interpolate import pchip_interpolate

    x_dose_Dose9 = np.linspace(0.111111, 1.0, 9)  #Here 9 due to having 9 dose responses
    x_dose_new_Dose9 = np.linspace(0.111111, 1.0, 1000)
    Ydose50_pred_Dose9 = []
    IC50_pred_Dose9 = []
    AUC_pred_Dose9 = []
    Emax_pred_Dose9 = []
    Y_pred_interp_all_Dose9 = []
    std_upper_interp_all_Dose9 = []
    std_lower_interp_all_Dose9 = []
    for i in range(Yval_curve_Dose9.shape[0]):
        y_resp_Dose9 = m_pred_curve_Dose9[i, :].copy()
        std_upper = y_resp_Dose9 + 2*np.sqrt(v_pred_curve_Dose9[i, :])
        std_lower = y_resp_Dose9 - 2 * np.sqrt(v_pred_curve_Dose9[i, :])
        f = interp1d(x_dose_Dose9, y_resp_Dose9)
        #f2 = interp1d(x_dose, y_resp, kind='cubic')
        y_resp_interp_Dose9 = pchip_interpolate(x_dose_Dose9, y_resp_Dose9, x_dose_new_Dose9)
        std_upper_interp = pchip_interpolate(x_dose_Dose9, std_upper, x_dose_new_Dose9)
        std_lower_interp = pchip_interpolate(x_dose_Dose9, std_lower, x_dose_new_Dose9)

        #y_resp_interp = f2(x_dose_new)
        Y_pred_interp_all_Dose9.append(y_resp_interp_Dose9)
        std_upper_interp_all_Dose9.append(std_upper_interp)
        std_lower_interp_all_Dose9.append(std_lower_interp)
        AUC_pred_Dose9.append(metrics.auc(x_dose_new_Dose9, y_resp_interp_Dose9))
        Emax_pred_Dose9.append(y_resp_interp_Dose9[-1])

        res1 = y_resp_interp_Dose9 < 0.507
        res2 = y_resp_interp_Dose9 > 0.493
        res_aux = np.where(res1 & res2)[0]
        if (res1 & res2).sum()>0:
            res_IC50 = np.arange(res_aux[0],res_aux[0]+ res_aux.shape[0])==res_aux
            res_aux = res_aux[res_IC50].copy()
        else:
            res_aux = res1 & res2

        if (res1 & res2).sum() > 0:
            Ydose50_pred_Dose9.append(y_resp_interp_Dose9[res_aux].mean())
            IC50_pred_Dose9.append(x_dose_new_Dose9[res_aux].mean())
        elif y_resp_interp_Dose9[-1] < 0.5:
            Ydose50_pred_Dose9.append(y_resp_interp_Dose9[i].max())
            aux_IC50_Dose9 = x_dose_new_Dose9[np.where(y_resp_interp_Dose9[i]==y_resp_interp_Dose9[i].max())[0]][0]  # it has to be a float not an array to avoid bug
            IC50_pred_Dose9.append(aux_IC50_Dose9)
        else:
            Ydose50_pred_Dose9.append(0.5)
            IC50_pred_Dose9.append(1.5)

    Ydose50_pred_Dose9 = np.array(Ydose50_pred_Dose9)
    IC50_pred_Dose9 = np.array(IC50_pred_Dose9)[:,None]
    AUC_pred_Dose9 = np.array(AUC_pred_Dose9)[:, None]
    Emax_pred_Dose9 = np.array(Emax_pred_Dose9)[:, None]

    posy = 0
    plt.figure(Nfold+30)
    plt.plot(x_dose_new_Dose9, Y_pred_interp_all_Dose9[posy])
    plt.plot(x_dose_new_Dose9, std_upper_interp_all_Dose9[posy],'b--')
    plt.plot(x_dose_new_Dose9, std_lower_interp_all_Dose9[posy], 'b--')
    plt.plot(x_dose_Dose9, Yval_curve_Dose9[posy, :], '.')
    plt.plot(IC50_pred_Dose9[posy], Ydose50_pred_Dose9[posy], 'rx')
    plt.plot(x_dose_new_Dose9, np.ones_like(x_dose_new_Dose9) * Emax_pred_Dose9[posy], 'r')  # Plot a horizontal line as Emax
    plt.plot(x_dose_new_Dose9, np.ones_like(x_dose_new_Dose9) * Emax_val_Dose9[posy], 'r')  # Plot a horizontal line as Emax
    plt.title(f"AUC Dose9 = {AUC_pred_Dose9[posy]}")
    print(AUC_pred_Dose9[posy])

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "Interpolation of predictions for Dose5"
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    x_dose_Dose5 = np.linspace(0.111111, 1.0, 5)  # Here 5 due to having 5 dose responses
    x_dose_new_Dose5 = np.linspace(0.111111, 1.0, 1000)
    Ydose50_pred_Dose5 = []
    IC50_pred_Dose5 = []
    AUC_pred_Dose5 = []
    Emax_pred_Dose5 = []
    Y_pred_interp_all_Dose5 = []
    std_upper_interp_all_Dose5 = []
    std_lower_interp_all_Dose5 = []
    for i in range(Yval_curve_Dose5.shape[0]):
        y_resp_Dose5 = m_pred_curve_Dose5[i, :].copy()
        std_upper = y_resp_Dose5 + 2 * np.sqrt(v_pred_curve_Dose5[i, :])
        std_lower = y_resp_Dose5 - 2 * np.sqrt(v_pred_curve_Dose5[i, :])
        f = interp1d(x_dose_Dose5, y_resp_Dose5)
        # f2 = interp1d(x_dose, y_resp, kind='cubic')
        y_resp_interp_Dose5 = pchip_interpolate(x_dose_Dose5, y_resp_Dose5, x_dose_new_Dose5)
        std_upper_interp = pchip_interpolate(x_dose_Dose5, std_upper, x_dose_new_Dose5)
        std_lower_interp = pchip_interpolate(x_dose_Dose5, std_lower, x_dose_new_Dose5)

        # y_resp_interp = f2(x_dose_new)
        Y_pred_interp_all_Dose5.append(y_resp_interp_Dose5)
        std_upper_interp_all_Dose5.append(std_upper_interp)
        std_lower_interp_all_Dose5.append(std_lower_interp)
        AUC_pred_Dose5.append(metrics.auc(x_dose_new_Dose5, y_resp_interp_Dose5))
        Emax_pred_Dose5.append(y_resp_interp_Dose5[-1])

        res1 = y_resp_interp_Dose5 < 0.507
        res2 = y_resp_interp_Dose5 > 0.493
        res_aux = np.where(res1 & res2)[0]
        if (res1 & res2).sum() > 0:
            res_IC50 = np.arange(res_aux[0], res_aux[0] + res_aux.shape[0]) == res_aux
            res_aux = res_aux[res_IC50].copy()
        else:
            res_aux = res1 & res2

        if (res1 & res2).sum() > 0:
            Ydose50_pred_Dose5.append(y_resp_interp_Dose5[res_aux].mean())
            IC50_pred_Dose5.append(x_dose_new_Dose5[res_aux].mean())
        elif y_resp_interp_Dose5[-1] < 0.5:
            Ydose50_pred_Dose5.append(y_resp_interp_Dose5[i].max())
            aux_IC50_Dose5 = x_dose_new_Dose5[np.where(y_resp_interp_Dose5[i] == y_resp_interp_Dose5[i].max())[0]][0]  # it has to be a float not an array to avoid bug
            IC50_pred_Dose5.append(aux_IC50_Dose5)
        else:
            Ydose50_pred_Dose5.append(0.5)
            IC50_pred_Dose5.append(1.5)

    Ydose50_pred_Dose5 = np.array(Ydose50_pred_Dose5)
    IC50_pred_Dose5 = np.array(IC50_pred_Dose5)[:, None]
    AUC_pred_Dose5 = np.array(AUC_pred_Dose5)[:, None]
    Emax_pred_Dose5 = np.array(Emax_pred_Dose5)[:, None]

    posy = 0
    plt.figure(Nfold+nsplits)
    plt.plot(x_dose_new_Dose5, Y_pred_interp_all_Dose5[posy])
    plt.plot(x_dose_new_Dose5, std_upper_interp_all_Dose5[posy], 'b--')
    plt.plot(x_dose_new_Dose5, std_lower_interp_all_Dose5[posy], 'b--')
    plt.plot(x_dose_Dose5, Yval_curve_Dose5[posy, :], '.')
    plt.plot(IC50_pred_Dose5[posy], Ydose50_pred_Dose5[posy], 'rx')
    plt.plot(x_dose_new_Dose5, np.ones_like(x_dose_new_Dose5) * Emax_pred_Dose5[posy],'r')  # Plot a horizontal line as Emax
    plt.plot(x_dose_new_Dose5, np.ones_like(x_dose_new_Dose5) * Emax_val_Dose5[posy],'r')  # Plot a horizontal line as Emax
    plt.title(f"AUC Dose5 = {AUC_pred_Dose5[posy]}")
    print(AUC_pred_Dose5[posy])

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    Emax_val = np.vstack((Emax_val_Dose9,Emax_val_Dose5))
    Emax_pred = np.vstack((Emax_pred_Dose9,Emax_pred_Dose5))

    AUC_val = np.vstack((AUC_val_Dose9,AUC_val_Dose5))
    AUC_pred = np.vstack((AUC_pred_Dose9,AUC_pred_Dose5))

    IC50_val = np.vstack((IC50_val_Dose9,IC50_val_Dose5))
    IC50_pred = np.vstack((IC50_pred_Dose9,IC50_pred_Dose5))


    Emax_abs = np.mean(np.abs(Emax_val - Emax_pred))
    AUC_abs = np.mean(np.abs(AUC_val - AUC_pred))
    IC50_MSE = np.mean((IC50_val - IC50_pred) ** 2)
    MSE_curves_Dose9 = np.mean((m_pred_curve_Dose9 - Yval_curve_Dose9) ** 2, 1)
    MSE_curves_Dose5 = np.mean((m_pred_curve_Dose5 - Yval_curve_Dose5) ** 2, 1)
    AllPred_MSE_Dose9 = np.mean((m_pred_curve_Dose9 - Yval_curve_Dose9) ** 2)
    AllPred_MSE_Dose5 = np.mean((m_pred_curve_Dose5 - Yval_curve_Dose5) ** 2)
    AllPred_MSE = np.mean([AllPred_MSE_Dose9,AllPred_MSE_Dose5])
    print("IC50 MSE:", IC50_MSE)
    print("AUC MAE:", AUC_abs)
    print("Emax MAE:", Emax_abs)
    Med_MSE = np.mean([np.median(MSE_curves_Dose9),np.median(MSE_curves_Dose5)])
    Mean_MSE = np.mean([np.mean(MSE_curves_Dose9),np.mean(MSE_curves_Dose5)])
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
        #print("Yval shape",Yval.shape)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
f= open("Metrics.txt","a+")
f.write("bash"+str(config.bash)+f" Med_MSE={np.mean(Med_MSE_AllFolds):0.5f}({np.std(Med_MSE_AllFolds):0.5f}) Mean_MSE={np.mean(Mean_MSE_AllFolds):0.5f}({np.std(Mean_MSE_AllFolds):0.5f}) NegLPD={np.mean(NegMLL_AllFolds):0.5f}({np.std(NegMLL_AllFolds):0.5f}) IC50_MSE={np.mean(IC50_MSE_AllFolds):0.5f}({np.std(IC50_MSE_AllFolds):0.5f}) Spear_ActualIC50={np.mean(SpearActualIC50_AllFolds):0.5f}({np.std(SpearActualIC50_AllFolds):0.5f}) Spear_all={np.mean(Spearman_AllFolds):0.5f}({np.std(Spearman_AllFolds):0.5f}) AUC_abs={np.mean(AUC_abs_AllFolds):0.5f}({np.std(AUC_abs_AllFolds):0.5f}) Emax_abs ={np.mean(Emax_abs_AllFolds):0.5f}({np.std(Emax_abs_AllFolds):0.5f})\n")
f.close()

"The last model should have been trained over all dataset without splitting"

final_path = '/data/ac1jjgg/Data_Marina/GPy_results/Codes_for_GDSC1/Drugs_1036_and_1371_MelanomaGDSC1_SmallDataset_ExactMOGP_ToPredict_MelanomaGDSC2_SanityCheck/'
#final_path ='model_Drugs_1036_and_1371_MelanomaGDSC1_SmallDataset_ExactMOGP_ToPredict_MelanomaGDSC2_SanityCheck/'
if not os.path.exists(final_path):
   os.makedirs(final_path)
np.save(final_path+'m_'+str(config.bash)+'.npy', model.param_array)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Below we prepare the data for Test over Drug1036 from GDSC2 with 7 Doses"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

_FOLDER_GDSC2 = "/home/ac1jjgg/MOGP_GPyTorch/Codes_for_GDSC2/Dataset_BRAF/GDSC2/"
#_FOLDER_GDSC2 = "/home/juanjo/Work_Postdoc/my_codes_postdoc/GPyTorch_Models/Dataset_BRAF/GDSC2/"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
name_for_KLrelevance_GDSC2 = 'GDSC2_melanoma_BRAF.csv'

df_train_No_MolecForm_GDSC2 = pd.read_csv(_FOLDER_GDSC2 + name_for_KLrelevance_GDSC2)  # Contain Train dataset prepared by Subhashini-Evelyn
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
df_train_GDSC2 = df_train_No_MolecForm_GDSC2[(df_train_No_MolecForm_GDSC2["DRUG_ID"]==1036)]
try:
    df_train_GDSC2 = df_train_GDSC2.drop(columns='Drug_Name')
except:
    pass

# Here we just check that from the column index 25 the input features start
start_pos_features_GDSC2 = 25
print(df_train_GDSC2.columns[start_pos_features_GDSC2])

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Below we use exactly the same features used for the training on GDSC1 their names are in Name_Features_Melanoma"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
X_train_features_GDSC2_NonScaled = df_train_GDSC2[Name_Features_Melanoma].copy()
"Instead of using (MaxMin sclaer) scaler.transform we just extract the min and max and make the transformation"
"This is to directly use the features used for training GDSC1 Dose9 and Dose5"
scaler_sel_max = scaler.data_max_[ind_nonzero_std]
scaler_sel_min = scaler.data_min_[ind_nonzero_std]
Xall_GDSC2 = (X_train_features_GDSC2_NonScaled.values-scaler_sel_min)/(scaler_sel_max-scaler_sel_min)

"Below we select just 7 concentration since GDSC2 only has such a number"
y_train_drug_Dose7 = np.clip(df_train_GDSC2["norm_cells_" + str(1)].values[:, None], 1.0e-9, np.inf)
print(y_train_drug_Dose7.shape)
for i in range(2, 8):
    y_train_drug_Dose7 = np.concatenate(
        (y_train_drug_Dose7, np.clip(df_train_GDSC2["norm_cells_" + str(i)].values[:, None], 1.0e-9, np.inf)), 1)

print("Ytrain size: ", y_train_drug_Dose7.shape)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

params_4_sig_train_Dose7 = df_train_GDSC2["param_" + str(1)].values[:, None]
for i in range(2, 5):
    params_4_sig_train_Dose7 = np.concatenate(
        (params_4_sig_train_Dose7, df_train_GDSC2["param_" + str(i)].values[:, None]), 1)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import matplotlib.pyplot as plt
from sklearn import metrics

#plt.close('all')
Kprop = (20.5-3.95)*(1-0.111111)/20.5
xini_adjusted = 0.111111 + ((1-0.111111)-Kprop)
ToMul = (1-0.111111)/Kprop
weighted = (1-0.111111)*ToMul
difference = weighted- (1-0.111111)
x_lin_Dose7 = np.linspace(0.111111, 1, 1000)
x_lin_Dose7_Adjust = np.linspace(xini_adjusted, 1, 1000)
x_real_dose_Dose7 = np.linspace(0.111111-difference, 1, 7)  #Here is 7 due to using GDSC2 that has 7 doses
x_lin_tile_Dose7 = np.tile(x_lin_Dose7, (params_4_sig_train_Dose7.shape[0], 1))
x_lin_tile_Dose7_Adjust = np.tile(x_lin_Dose7_Adjust, (params_4_sig_train_Dose7.shape[0], 1))
# (x_lin,params_4_sig_train.shape[0],1).shape
Ydose_res_Dose7 = []
AUC_Dose7 = []
IC50_Dose7 = []
Ydose50_Dose7 = []
Emax_Dose7 = []
for i in range(params_4_sig_train_Dose7.shape[0]):
    Ydose_res_Dose7.append(sigmoid_4_param(x_lin_tile_Dose7_Adjust[i, :], *params_4_sig_train_Dose7[i, :]))
    AUC_Dose7.append(metrics.auc(x_lin_tile_Dose7[i, :], Ydose_res_Dose7[i]))
    Emax_Dose7.append(Ydose_res_Dose7[i][-1])
    res1 = (Ydose_res_Dose7[i] < 0.507)
    res2 = (Ydose_res_Dose7[i] > 0.493)
    if (res1 & res2).sum() > 0:
        Ydose50_Dose7.append(Ydose_res_Dose7[i][res1 & res2].mean())
        IC50_Dose7.append(x_lin_Dose7[res1 & res2].mean())
    elif Ydose_res_Dose7[i][-1]<0.5:
       Ydose50_Dose7.append(Ydose_res_Dose7[i].max())
       aux_IC50_Dose7 = x_lin_Dose7[np.where(Ydose_res_Dose7[i].max())[0]][0]  #it has to be a float not an array to avoid bug
       IC50_Dose7.append(aux_IC50_Dose7)
    else:
        Ydose50_Dose7.append(0.5)
        IC50_Dose7.append(1.5) #IC50.append(x_lin[-1])

posy = 50
plt.figure(10)
plt.plot(x_lin_Dose7, Ydose_res_Dose7[posy])
plt.plot(x_real_dose_Dose7, y_train_drug_Dose7[posy, :], '.')
plt.plot(IC50_Dose7[posy], Ydose50_Dose7[posy], 'rx')
plt.plot(x_lin_Dose7, np.ones_like(x_lin_Dose7)*Emax_Dose7[posy], 'r') #Plot a horizontal line as Emax
plt.title(f"AUC Dose7 = {AUC_Dose7[posy]}")
print(AUC_Dose7[posy])

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
AUC_Dose7 = np.array(AUC_Dose7)[:,None]
IC50_Dose7 = np.array(IC50_Dose7)[:,None]
Emax_Dose7 = np.array(Emax_Dose7)[:,None]
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Below we are going to test over the GDSC2 data, but only over IC50, AUC and Emax"
"We cannot test over the specific prediction values since the GDSC1 (9 Doses) and GDSC2 (7 Doses)"
"Do not match the doses between each other."
"The prediction over the data from GDSC2 will contain 9 Doses predictions due to using the GDSC1 model"
Xtest_aux_GDSC2 = Xall_GDSC2.copy()
Ylabel_test_GDSC2 = np.array([i * np.ones(Xtest_aux_GDSC2.shape[0]) for i in range(Ntasks9)]).flatten()[:, None]
Xtest_GDSC2 = np.concatenate((np.tile(Xtest_aux_GDSC2, (Ntasks9, 1)), Ylabel_test_GDSC2), 1)
"We cannot compute the NLPD since the doses from GDSC1 and GDSC2 do not match"
#"Negative Log Predictive Density (NLPD)"
#print("NegLPD Test GDSC2 Dose7: ", Val_NLPD_Dose7)

m_pred_GDSC2, v_pred_GDSC2 = model.predict(Xtest_GDSC2, full_cov=False)

Nval_GDSC2 = Xtest_aux_GDSC2.shape[0]

m_pred_curve_GDSC2 = m_pred_GDSC2.reshape(Ntasks9, Nval_GDSC2).T.copy()
v_pred_curve_GDSC2 = v_pred_GDSC2.reshape(Ntasks9, Nval_GDSC2).T.copy()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Interpolation of predictions for GDSC2"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

x_dose_GDSC2 = np.linspace(0.111111, 1.0, 9)  # Here 5 due to having 5 dose responses
x_dose_new_GDSC2 = np.linspace(0.111111, 1.0, 1000)
Ydose50_pred_GDSC2 = []
IC50_pred_GDSC2 = []
AUC_pred_GDSC2 = []
Emax_pred_GDSC2 = []
Y_pred_interp_all_GDSC2 = []
std_upper_interp_all_GDSC2 = []
std_lower_interp_all_GDSC2 = []
for i in range(m_pred_curve_GDSC2.shape[0]):
    y_resp_GDSC2 = m_pred_curve_GDSC2[i, :].copy()
    std_upper = y_resp_GDSC2 + 2 * np.sqrt(v_pred_curve_GDSC2[i, :])
    std_lower = y_resp_GDSC2 - 2 * np.sqrt(v_pred_curve_GDSC2[i, :])
    f = interp1d(x_dose_GDSC2, y_resp_GDSC2)
    #f2 = interp1d(x_dose_GDSC2, y_resp_GDSC2, kind='quadratic')
    #f2_upper = interp1d(x_dose_GDSC2, std_upper, kind='quadratic')
    #f2_lower = interp1d(x_dose_GDSC2, std_lower, kind='quadratic')
    y_resp_interp_GDSC2 = pchip_interpolate(x_dose_GDSC2, y_resp_GDSC2, x_dose_new_GDSC2)
    std_upper_interp = pchip_interpolate(x_dose_GDSC2, std_upper, x_dose_new_GDSC2)
    std_lower_interp = pchip_interpolate(x_dose_GDSC2, std_lower, x_dose_new_GDSC2)

    #y_resp_interp_GDSC2 = f2(x_dose_new_GDSC2)
    #std_upper_interp = f2_upper(x_dose_new_GDSC2)
    #std_lower_interp = f2_lower(x_dose_new_GDSC2)

    Y_pred_interp_all_GDSC2.append(y_resp_interp_GDSC2)
    std_upper_interp_all_GDSC2.append(std_upper_interp)
    std_lower_interp_all_GDSC2.append(std_lower_interp)
    AUC_pred_GDSC2.append(metrics.auc(x_dose_new_GDSC2, y_resp_interp_GDSC2))
    Emax_pred_GDSC2.append(y_resp_interp_GDSC2[-1])

    res1 = y_resp_interp_GDSC2 < 0.507
    res2 = y_resp_interp_GDSC2 > 0.493
    res_aux = np.where(res1 & res2)[0]
    if (res1 & res2).sum() > 0:
        res_IC50 = np.arange(res_aux[0], res_aux[0] + res_aux.shape[0]) == res_aux
        res_aux = res_aux[res_IC50].copy()
    else:
        res_aux = res1 & res2

    if (res1 & res2).sum() > 0:
        Ydose50_pred_GDSC2.append(y_resp_interp_GDSC2[res_aux].mean())
        IC50_pred_GDSC2.append(x_dose_new_GDSC2[res_aux].mean())
    elif y_resp_interp_GDSC2[-1] < 0.5:
        Ydose50_pred_GDSC2.append(y_resp_interp_GDSC2[i].max())
        aux_IC50_GDSC2 = x_dose_new_GDSC2[np.where(y_resp_interp_GDSC2[i] == y_resp_interp_GDSC2[i].max())[0]][0]  # it has to be a float not an array to avoid bug
        IC50_pred_GDSC2.append(aux_IC50_GDSC2)
    else:
        Ydose50_pred_GDSC2.append(0.5)
        IC50_pred_GDSC2.append(1.5)

Ydose50_pred_GDSC2 = np.array(Ydose50_pred_GDSC2)
IC50_pred_GDSC2 = np.array(IC50_pred_GDSC2)[:, None]
AUC_pred_GDSC2 = np.array(AUC_pred_GDSC2)[:, None]
Emax_pred_GDSC2 = np.array(Emax_pred_GDSC2)[:, None]

#posy = 11
plt.figure(60)
plt.plot(x_dose_new_GDSC2, Y_pred_interp_all_GDSC2[posy])
plt.plot(x_dose_new_GDSC2, std_upper_interp_all_GDSC2[posy], 'b--')
plt.plot(x_dose_new_GDSC2, std_lower_interp_all_GDSC2[posy], 'b--')
#plt.plot(x_dose_Dose7, Yval_curve_Dose7[posy, :], '.')
plt.plot(x_real_dose_Dose7, y_train_drug_Dose7[posy, :], '.')
plt.plot(IC50_pred_GDSC2[posy], Ydose50_pred_GDSC2[posy], 'rx')
plt.plot(x_dose_new_GDSC2, np.ones_like(x_dose_new_GDSC2) * Emax_pred_GDSC2[posy],'b--')  # Plot a horizontal line as Emax
plt.plot(x_dose_new_GDSC2, np.ones_like(x_dose_new_GDSC2) * Emax_Dose7[posy],'r--')  # Plot a horizontal line as Emax
plt.title(f"AUC Dose7 = {AUC_pred_GDSC2[posy]}")
print(AUC_pred_GDSC2[posy])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Emax_abs_Dose7 = np.mean(np.abs(Emax_Dose7 - Emax_pred_GDSC2))
AUC_abs_Dose7 = np.mean(np.abs(AUC_Dose7 - AUC_pred_GDSC2))
IC50_MSE_Dose7 = np.mean((IC50_Dose7 - IC50_pred_GDSC2) ** 2)
#MSE_curves_Dose7 = np.mean((m_pred_curve_Dose7 - Yval_curve_Dose7) ** 2, 1)
#AllPred_MSE_Dose7 = np.mean((m_pred_curve_Dose7 - Yval_curve_Dose7) ** 2)
print("GDSC2 IC50 MSE:", IC50_MSE_Dose7)
print("GDSC2 AUC MAE:", AUC_abs_Dose7)
print("GDSC2 Emax MAE:", Emax_abs_Dose7)
#Med_MSE = np.median(MSE_curves_Dose7)
#Mean_MSE = np.mean(MSE_curves_Dose7)
#print("Med_MSE:", Med_MSE)
#print("Mean_MSE:", Mean_MSE)
#print("All Predictions MSE:", AllPred_MSE)

from scipy.stats import spearmanr

pos_Actual_IC50 = IC50_Dose7 != 1.5
spear_corr_all, p_value_all = spearmanr(IC50_Dose7, IC50_pred_GDSC2)
spear_corr_actualIC50, p_value_actual = spearmanr(IC50_Dose7[pos_Actual_IC50],IC50_pred_GDSC2[pos_Actual_IC50])
print("Spearman_all Corr: ", spear_corr_all)
print("Spearman p-value: ", p_value_all)
print("Spearman_actualIC50 Corr: ", spear_corr_actualIC50)
print("Spearman p-value: ", p_value_actual)

ftest= open("Metrics_Test_GDSC2.txt","a+")
ftest.write("bash"+str(config.bash)+f" IC50_MSE={IC50_MSE_Dose7:0.5f} Spear_ActualIC50={spear_corr_actualIC50:0.5f} Spear_all={spear_corr_all:0.5f} AUC_abs={AUC_abs_Dose7:0.5f} Emax_abs ={Emax_abs_Dose7:0.5f} \n")
ftest.close()
