import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
import os

#_FOLDER = "/home/ac1jjgg/Dataset_5Cancers/GDSC2_EGFR_PI3K_MAPK_Top5cancers/"
_FOLDER = "/home/juanjo/Work_Postdoc/my_codes_postdoc/Dataset_5Cancers/GDSC2_EGFR_PI3K_MAPK_Top5cancers/"
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
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import getopt
import sys

warnings.filterwarnings("ignore")
#os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

class commandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'i:s:k:w:r:p:c:a:n:t:')
        # opts = dict(opts)
        # print(opts)
        self.N_iter_epoch = 1200    #number of iterations
        self.which_seed = 1011    #change seed to initialise the hyper-parameters
        self.rank = 7
        self.scale = 1
        self.weight = 1
        self.bash = "1"
        self.N_CellLines = 144   #Try to put this values as multiple of Num_drugs
        self.sel_cancer = 3
        self.seed_for_N = 3
        self.N_5thCancer_ToBe_Included = 9 #Try to put this values as multiple of Num_drugs

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
            if op == '-c':
                self.N_CellLines = arg
            if op == '-a':
                self.sel_cancer = arg
            if op == '-n':
                self.seed_for_N = arg
            if op == '-t':
                self.N_5thCancer_ToBe_Included = arg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
config = commandLine()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
dict_cancers={0:'GDSC2_EGFR_PI3K_MAPK_Breast_1000FR.csv',1:'GDSC2_EGFR_PI3K_MAPK_COAD_1000FR.csv',
              2:'GDSC2_EGFR_PI3K_MAPK_LUAD.csv',3:'GDSC2_EGFR_PI3K_MAPK_melanoma.csv',4:'GDSC2_EGFR_PI3K_MAPK_SCLC.csv'}

indx_cancer = np.array([0,1,2,3,4])
indx_cancer_test = np.array([int(config.sel_cancer)])
indx_cancer_train = np.delete(indx_cancer,indx_cancer_test)

name_feat_file = "GDSC2_EGFR_PI3K_MAPK_allfeatures.csv"
name_feat_file_nozero = "GDSC2_EGFR_PI3K_MAPK_features_NonZero_Drugs1036_1061_1373.csv" #"GDSC2_EGFR_PI3K_MAPK_features_NonZero.csv"
Num_drugs = 3
N_CellLines_perDrug = int(config.N_CellLines)//Num_drugs
N_CellLines = int(config.N_CellLines)
rand_state_N = int(config.seed_for_N)
for i in range(0,4):
    name_for_KLrelevance = dict_cancers[indx_cancer_train[i]]
    print(name_for_KLrelevance)
    if i==0:
        df_to_read = pd.read_csv(_FOLDER + name_for_KLrelevance)#.sample(n=N_CellLines,random_state = rand_state_N)
        df_4Cancers_train_d1 = df_to_read[df_to_read["DRUG_ID"]==1036].dropna()#.sample(n=N_CellLines,random_state = rand_state_N)
        df_4Cancers_train_d2 = df_to_read[df_to_read["DRUG_ID"]==1061].dropna()#.sample(n=N_CellLines,random_state = rand_state_N)
        df_4Cancers_train_d3 = df_to_read[df_to_read["DRUG_ID"] == 1373].dropna()#.sample(n=N_CellLines,random_state=rand_state_N)
        df_4Cancers_train_Only_d1 = df_4Cancers_train_d1.copy()
        df_4Cancers_train_Only_d2 = df_4Cancers_train_d2.copy()
        df_4Cancers_train_Only_d3 = df_4Cancers_train_d3.copy()
    else:
        df_to_read = pd.read_csv(_FOLDER + name_for_KLrelevance)#.sample(n=N_CellLines,random_state = rand_state_N)
        df_4Cancers_train_d1 = df_to_read[df_to_read["DRUG_ID"]==1036].dropna()#.sample(n=N_CellLines,random_state = rand_state_N)
        df_4Cancers_train_d2 = df_to_read[df_to_read["DRUG_ID"] == 1061].dropna()#.sample(n=N_CellLines,random_state=rand_state_N)
        df_4Cancers_train_d3 = df_to_read[df_to_read["DRUG_ID"] == 1373].dropna()#.sample(n=N_CellLines,random_state=rand_state_N)
        #N_per_drug = [df_4Cancers_train_d1.shape[0], df_4Cancers_train_d2.shape[0], df_4Cancers_train_d3.shape[0]]
        #Nd1, Nd2, Nd3 = np.clip(N_CellLines_perDrug, 1, N_per_drug[0]), np.clip(N_CellLines_perDrug, 1, N_per_drug[1]), np.clip(N_CellLines_perDrug, 1, N_per_drug[2])
        #df_4Cancers_train = pd.concat([df_4Cancers_train,df_4Cancers_train_d1.sample(n=Nd1,random_state = rand_state_N), df_4Cancers_train_d2.sample(n=Nd2,random_state = rand_state_N),df_4Cancers_train_d3.sample(n=Nd3,random_state = rand_state_N)])
        df_4Cancers_train_Only_d1 = pd.concat([df_4Cancers_train_Only_d1,df_4Cancers_train_d1])
        df_4Cancers_train_Only_d2 = pd.concat([df_4Cancers_train_Only_d2, df_4Cancers_train_d2])
        df_4Cancers_train_Only_d3 = pd.concat([df_4Cancers_train_Only_d3, df_4Cancers_train_d3])
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Here we extract the test dataframe for the one cancer left out"
Name_cancer_test = dict_cancers[indx_cancer_test[0]]
df_to_read = pd.read_csv(_FOLDER + Name_cancer_test)
df_4Cancers_test_d1 = df_to_read[df_to_read["DRUG_ID"]==1036].dropna()
df_4Cancers_test_d2 = df_to_read[df_to_read["DRUG_ID"]==1061].dropna()
df_4Cancers_test_d3 = df_to_read[df_to_read["DRUG_ID"] == 1373].dropna()
df_4Cancers_test = pd.concat([df_4Cancers_test_d1, df_4Cancers_test_d2,df_4Cancers_test_d3])

"This section of code is to allow including few observations from the 5th Cancer as part of the training data"
#a_bit_of_5thCancer = True

if int(config.N_5thCancer_ToBe_Included)!=0:
    N_ToInclude_per_Drug = int(config.N_5thCancer_ToBe_Included)//Num_drugs
    N_ToInclude_per_Drug = np.clip(N_ToInclude_per_Drug, 1, df_4Cancers_test.shape[0] // 3 - 10)
    df_4Cancers_test_d1 = df_to_read[df_to_read["DRUG_ID"]==1036].dropna().reset_index().drop(columns='index')
    df_4Cancers_test_d2 = df_to_read[df_to_read["DRUG_ID"]==1061].dropna().reset_index().drop(columns='index')
    df_4Cancers_test_d3 = df_to_read[df_to_read["DRUG_ID"] == 1373].dropna().reset_index().drop(columns='index')
    N_per_drug = [df_4Cancers_test_d1.shape[0], df_4Cancers_test_d2.shape[0], df_4Cancers_test_d3.shape[0]]
    #indx_test_to_include = [,np.random.permutation(np.arange(0,N_per_drug[1])),np.random.permutation(np.arange(0,N_per_drug[2]))]
    "The seed below is to guarantee always having the same values of 5th cancer to be included in Training regardless"
    "of all the different cross-validations, we do not want them to change at each different cross-validation of MOGP"
    if indx_cancer_test == 0:
        np.random.seed(6)
    elif indx_cancer_test == 1:
        np.random.seed(1)
    elif indx_cancer_test == 2:
        np.random.seed(1)
    elif indx_cancer_test == 3:
        np.random.seed(1)
    elif indx_cancer_test == 4:
        np.random.seed(1)

    Test_drugs_indexes = [np.random.permutation(np.arange(0, N_per_drug[myind])) for myind in range(Num_drugs)]
    indx_test_to_NotInclude = [np.delete(Test_drugs_indexes[myind],np.arange(0,N_ToInclude_per_Drug)) for myind in range(Num_drugs)]
    indx_test_to_include = [Test_drugs_indexes[myind][np.arange(0,N_ToInclude_per_Drug)] for myind in range(Num_drugs)]
    print("Indexes to Include:",indx_test_to_include)
    df_4Cancers_test = pd.concat([df_4Cancers_test_d1.iloc[indx_test_to_NotInclude[0]], df_4Cancers_test_d2.iloc[indx_test_to_NotInclude[1]],df_4Cancers_test_d3.iloc[indx_test_to_NotInclude[2]]])
    df_4Cancers_test_ToInclude = pd.concat([df_4Cancers_test_d1.iloc[indx_test_to_include[0]], df_4Cancers_test_d2.iloc[indx_test_to_include[1]],df_4Cancers_test_d3.iloc[indx_test_to_include[2]]])
    df_4Cancers_test_Only_d1 = df_4Cancers_test_d1.iloc[indx_test_to_include[0]].copy()
    df_4Cancers_test_Only_d2 = df_4Cancers_test_d2.iloc[indx_test_to_include[1]].copy()
    df_4Cancers_test_Only_d3 = df_4Cancers_test_d3.iloc[indx_test_to_include[2]].copy()

drug1_train = df_4Cancers_train_Only_d1[df_4Cancers_train_Only_d1.columns[25:]].values
drug1_test = df_4Cancers_test_Only_d1[df_4Cancers_test_Only_d1.columns[25:]].values
drug2_train = df_4Cancers_train_Only_d2[df_4Cancers_train_Only_d2.columns[25:]].values
drug2_test = df_4Cancers_test_Only_d2[df_4Cancers_test_Only_d2.columns[25:]].values
drug3_train = df_4Cancers_train_Only_d3[df_4Cancers_train_Only_d3.columns[25:]].values
drug3_test = df_4Cancers_test_Only_d3[df_4Cancers_test_Only_d3.columns[25:]].values

from scipy.spatial.distance import minkowski

"Be careful that metric distance is a DISIMILARITY not SIMILARITY"
def Disimilarity_Metric(drug_train,drug_test):
    M_dist = np.zeros((drug_train.shape[0], drug_test.shape[0]))
    for i in range(drug_train.shape[0]):
        for j in range(drug_test.shape[0]):
            M_dist[i,j] = minkowski(drug_train[i,:],drug_test[j,:],1)
    return  M_dist

def My_Categorical(prob,Nsamples):
    prob = prob.flatten()
    cum_prob = np.cumsum(prob, axis=-1)
    myreap = np.repeat(cum_prob[None, :], Nsamples, axis=0)
    r = np.random.uniform(0,1,Nsamples)[:,None]
    sample = np.argmax(myreap>r,axis=-1) #My_Categorical(prob)
    return sample

M_dist_d1 = Disimilarity_Metric(drug1_train,drug1_test).sum(1)
M_dist_d2 = Disimilarity_Metric(drug2_train,drug2_test).sum(1)
M_dist_d3 = Disimilarity_Metric(drug3_train,drug3_test).sum(1)

N_All_Cell = N_CellLines*4   #here it is 4 due to having 4 cancers to cross-val
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"To DO: Create function for each Vec_N_index per drug"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
M_dist_d1_norm = M_dist_d1.max()/M_dist_d1
M_dist_d2_norm = M_dist_d2.max()/M_dist_d2
M_dist_d3_norm = M_dist_d3.max()/M_dist_d3

#prob_drugs = [M_dist_d1_norm/M_dist_d1_norm.sum(0),M_dist_d2_norm/M_dist_d2_norm.sum(0),M_dist_d3_norm/M_dist_d3_norm.sum(0)]
prob_drugs = [np.ones(M_dist_d1_norm.shape[0])*1.0/M_dist_d1_norm.shape[0],np.ones(M_dist_d2_norm.shape[0])*1.0/M_dist_d2_norm.shape[0],np.ones(M_dist_d3_norm.shape[0])*1.0/M_dist_d3_norm.shape[0]]
"Here I just set the different seed to sample from My_Categorical"
np.random.seed(rand_state_N)
ind_d1 = My_Categorical(prob_drugs[0],N_All_Cell//Num_drugs)
ind_d2 = My_Categorical(prob_drugs[1],N_All_Cell//Num_drugs)
ind_d3 = My_Categorical(prob_drugs[2],N_All_Cell//Num_drugs)

df_4Cancers_train_Only_d1 = df_4Cancers_train_Only_d1.reset_index().drop(columns='index')
df_4Cancers_train_Only_d2 = df_4Cancers_train_Only_d2.reset_index().drop(columns='index')
df_4Cancers_train_Only_d3 = df_4Cancers_train_Only_d3.reset_index().drop(columns='index')

df_4Cancers_train = pd.concat([df_4Cancers_train_Only_d1.iloc[ind_d1],df_4Cancers_train_Only_d2.iloc[ind_d2],df_4Cancers_train_Only_d3.iloc[ind_d3]])
df_4Cancers_train = pd.concat([df_4Cancers_train, df_4Cancers_test_ToInclude])

# Here we just check that from the column index 25 the input features start
start_pos_features = 25
print(df_4Cancers_train.columns[start_pos_features])

df_feat_Names = pd.read_csv(_FOLDER + name_feat_file)  # Contain Feature Names
df_feat_Names_nozero = pd.read_csv(_FOLDER + name_feat_file_nozero)  # Contain Feature Names
indx_nozero = df_feat_Names_nozero['index'].values[start_pos_features:]

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# "The lines below are particularly to generate the indexes for Non-zero features"
# scaler = MinMaxScaler().fit(df_4Cancers_train[df_4Cancers_train.columns[25:]])
# X_train_features = scaler.transform(df_4Cancers_train[df_4Cancers_train.columns[25:]])
# ind_all = np.arange(0,X_train_features.shape[1])
# ind_NonZero = ind_all[X_train_features.std(0)!=0.0]
# ind_NonZero_final = np.concatenate((np.arange(0,25),ind_NonZero+25))

scaler = MinMaxScaler().fit(df_4Cancers_train[df_4Cancers_train.columns[indx_nozero]])
X_train_features = scaler.transform(df_4Cancers_train[df_4Cancers_train.columns[indx_nozero]])
X_test_features = scaler.transform(df_4Cancers_test[df_4Cancers_test.columns[indx_nozero]])

"Below we select just 7 concentration since GDSC2 only has such a number"
y_train_drug = np.clip(df_4Cancers_train["norm_cells_" + str(1)].values[:, None], 1.0e-9, np.inf)
y_test_drug = np.clip(df_4Cancers_test["norm_cells_" + str(1)].values[:, None], 1.0e-9, np.inf)
print(y_train_drug.shape)
for i in range(2, 8):
    y_train_drug = np.concatenate((y_train_drug, np.clip(df_4Cancers_train["norm_cells_" + str(i)].values[:, None], 1.0e-9, np.inf)), 1)
    y_test_drug = np.concatenate((y_test_drug, np.clip(df_4Cancers_test["norm_cells_" + str(i)].values[:, None], 1.0e-9, np.inf)), 1)

print("Ytrain size: ", y_train_drug.shape)
print("Ytest size: ", y_test_drug.shape)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
params_4_sig_train = df_4Cancers_train["param_" + str(1)].values[:, None]
params_4_sig_test = df_4Cancers_test["param_" + str(1)].values[:, None]
for i in range(2, 5):
    params_4_sig_train = np.concatenate((params_4_sig_train, df_4Cancers_train["param_" + str(i)].values[:, None]), 1)
    params_4_sig_test = np.concatenate((params_4_sig_test, df_4Cancers_test["param_" + str(i)].values[:, None]), 1)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import matplotlib.pyplot as plt
from sklearn import metrics

plt.close('all')
x_lin = np.linspace(0.111111, 1, 1000)
x_real_dose = np.linspace(0.111111, 1, 7)  #Here is 7 due to using GDSC2 that has 7 doses
def Get_IC50_AUC_Emax(params_4_sig_train,x_lin,x_real_dose):
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
           for dose_j in range(x_lin.shape[0]):
               if(Ydose_res[i][dose_j] < 0.5):
                   break
           Ydose50.append(Ydose_res[i][dose_j])
           aux_IC50 = x_lin[dose_j]  #it has to be a float not an array to avoid bug
           IC50.append(aux_IC50)
        else:
            Ydose50.append(0.5)
            IC50.append(1.5) #IC50.append(x_lin[-1])

    return Ydose50,Ydose_res,IC50,AUC,Emax

Ydose50,Ydose_res,IC50,AUC,Emax = Get_IC50_AUC_Emax(params_4_sig_train,x_lin,x_real_dose)
Ydose50_test,Ydose_res_test,IC50_test,AUC_test,Emax_test = Get_IC50_AUC_Emax(params_4_sig_test,x_lin,x_real_dose)

def my_plot(posy,fig_num,Ydose50,Ydose_res,IC50,AUC,Emax,x_lin,x_real_dose,y_train_drug):
    plt.figure(fig_num)
    plt.plot(x_lin, Ydose_res[posy])
    plt.plot(x_real_dose, y_train_drug[posy, :], '.')
    plt.plot(IC50[posy], Ydose50[posy], 'rx')
    plt.plot(x_lin, np.ones_like(x_lin)*Emax[posy], 'r') #Plot a horizontal line as Emax
    plt.title(f"AUC = {AUC[posy]}")
    print(AUC[posy])

#posy = 0
#my_plot(posy,0,Ydose50,Ydose_res,IC50,AUC,Emax,x_lin,x_real_dose,y_train_drug)
#my_plot(posy,1,Ydose50_test,Ydose_res_test,IC50_test,AUC_test,Emax_test,x_lin,x_real_dose,y_test_drug)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
AUC = np.array(AUC)
IC50 = np.array(IC50)
Emax = np.array(Emax)
AUC_test = np.array(AUC_test)
IC50_test = np.array(IC50_test)
Emax_test = np.array(Emax_test)

"Below we select just the columns with std higher than zero"
Name_Features_5Cancers = df_feat_Names_nozero['feature'].values[start_pos_features:]
Xall = X_train_features.copy()
Yall = y_train_drug.copy()

AUC_all = AUC[:, None].copy()
IC50_all = IC50[:, None].copy()
Emax_all = Emax[:, None].copy()

AUC_test = AUC_test[:, None].copy()
IC50_test = IC50_test[:, None].copy()
Emax_test = Emax_test[:, None].copy()

print("AUC train size:", AUC_all.shape)
print("IC50 train size:", IC50_all.shape)
print("Emax train size:", Emax_all.shape)
print("X all train data size:", Xall.shape)
print("Y all train data size:", Yall.shape)

print("AUC test size:", AUC_test.shape)
print("IC50 test size:", IC50_test.shape)
print("Emax test size:", Emax_test.shape)
print("X all test data size:", X_test_features.shape)
print("Y all test data size:", y_test_drug.shape)

import math
import torch
import gpytorch

#train_x = torch.from_numpy(Xtrain.astype(np.float32))

train_y1 = torch.from_numpy(Yall[0:N_CellLines].astype(np.float32))
train_y2 = torch.from_numpy(Yall[N_CellLines:2*N_CellLines].astype(np.float32))
train_y3 = torch.from_numpy(Yall[2*N_CellLines:3*N_CellLines].astype(np.float32))
train_y4 = torch.from_numpy(Yall[3*N_CellLines:4*N_CellLines].astype(np.float32))
train_y5 = torch.from_numpy(Yall[4*N_CellLines:].astype(np.float32))

train_x1 = torch.from_numpy(Xall[0:N_CellLines,:].astype(np.float32))
train_x2 = torch.from_numpy(Xall[N_CellLines:2*N_CellLines,:].astype(np.float32))
train_x3 = torch.from_numpy(Xall[2*N_CellLines:3*N_CellLines,:].astype(np.float32))
train_x4 = torch.from_numpy(Xall[3*N_CellLines:4*N_CellLines,:].astype(np.float32))
train_x5 = torch.from_numpy(Xall[4*N_CellLines:,:].astype(np.float32))

train_i_task1 = torch.full((train_x1.shape[0],1), dtype=torch.long, fill_value=0)
train_i_task2 = torch.full((train_x2.shape[0],1), dtype=torch.long, fill_value=1)
train_i_task3 = torch.full((train_x3.shape[0],1), dtype=torch.long, fill_value=2)
train_i_task4 = torch.full((train_x4.shape[0],1), dtype=torch.long, fill_value=3)
train_i_task5 = torch.full((train_x5.shape[0],1), dtype=torch.long, fill_value=4)

full_train_x = torch.cat([train_x1, train_x2,train_x3,train_x4,train_x5])
full_train_i = torch.cat([train_i_task1, train_i_task2,train_i_task3,train_i_task4,train_i_task5])
full_train_y = torch.cat([train_y1, train_y2,train_y3,train_y4,train_y5])

test_x = torch.from_numpy(X_test_features.astype(np.float32))
test_y = torch.from_numpy(y_test_drug.astype(np.float32))

myrank = mynum_doses = train_y1.shape[1]

if train_x1.shape.__len__()>1:
    Dim = train_x2.shape[1]
else:
    Dim = 1

torch.manual_seed(11)

from typing import Optional
from gpytorch.priors import Prior

from gpytorch.constraints import Interval, Positive
from gpytorch.lazy import DiagLazyTensor, InterpolatedLazyTensor, PsdSumLazyTensor, RootLazyTensor

from gpytorch.utils.broadcasting import _mul_broadcast_shape

class TL_Kernel(gpytorch.kernels.Kernel):
    def __init__(
            self,
            num_tasks: int,
            rank: Optional[int] = 1,
            prior: Optional[Prior] = None,
            var_constraint: Optional[Interval] = None,
            **kwargs,
    ):
        if rank > num_tasks:
            raise RuntimeError("Cannot create a task covariance matrix larger than the number of tasks")
        super().__init__(**kwargs)

        if var_constraint is None:
            var_constraint = Positive()

        # if lambda_constraint is None:
        #     lambda_constraint = Interval(-100.0,100.0)

        self.register_parameter(
            name="covar_factor", parameter=torch.nn.Parameter(2*torch.rand(*self.batch_shape, num_tasks, rank)-1)
        )
        self.register_parameter(name="raw_var", parameter=torch.nn.Parameter(torch.randn(*self.batch_shape, num_tasks)))
        if prior is not None:
            if not isinstance(prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(prior).__name__)
            self.register_prior("IndexKernelPrior", prior, lambda m: m._eval_covar_matrix())

        self.register_constraint("raw_var", var_constraint)
        #self.register_constraint("covar_factor", lambda_constraint)

    @property
    def var(self):
        return self.raw_var_constraint.transform(self.raw_var)

    @var.setter
    def var(self, value):
        self._set_var(value)

    def _set_var(self, value):
        self.initialize(raw_var=self.raw_var_constraint.inverse_transform(value))

    def _eval_covar_matrix(self):
        cf = 2.0/(1+torch.exp(-self.covar_factor))-1.0
        #print(cf)
        num_tasks = cf.shape[0]
        res_aux = (cf @ cf.transpose(-1, -2)) * (torch.ones((num_tasks,num_tasks)) - torch.eye(num_tasks)) + torch.eye(num_tasks) + torch.diag_embed(self.var)
        #print(res_aux)
        return res_aux

    # @property
    # def covar_matrix(self):
    #     #var = self.var
    #     #res = PsdSumLazyTensor(RootLazyTensor(self.covar_factor), DiagLazyTensor(var))
    #     res = RootLazyTensor(self.covar_factor)
    #     return res

    def forward(self, i1, i2, **params):

        i1, i2 = i1.long(), i2.long()
        covar_matrix = self._eval_covar_matrix()
        batch_shape = _mul_broadcast_shape(i1.shape[:-2], i2.shape[:-2], self.batch_shape)

        res = InterpolatedLazyTensor(
            base_lazy_tensor=covar_matrix,
            left_interp_indices=i1.expand(batch_shape + i1.shape[-2:]),
            right_interp_indices=i2.expand(batch_shape + i2.shape[-2:]),
        )
        return res

class TL_GPModel(gpytorch.models.ExactGP):
    "The num_tasks refers to the total number of source tasks plus target task"
    def __init__(self, source_target_x, source_target_y,target_x,target_y, likelihood, num_tasks,num_doses):
        super(TL_GPModel, self).__init__(target_x, target_y, likelihood)
        #super(TL_GPModel, self).__init__(source_target_x, source_target_y, likelihood)
        self.source_target_x = source_target_x
        self.source_target_y = source_target_y
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()
        self.num_tasks = num_tasks
        self.num_doses = num_doses
        self.N_Target_Train = target_x.shape[0]//num_doses

        # We learn an IndexKernel for 2 tasksT = {Tensor: (1000,)} tensor([0.4005, 0.1535, 0.0994, 0.4777, 0.9611, 0.4716, 0.8745, 0.5570, 0.5175,\n        0.5066, 0.7989, 0.2964, 0.6281, 0.3874, 0.6797, 0.3161, 0.6082, 0.9790,\n        0.8192, 0.8591, 0.2998, 0.0856, 0.3485, 0.5403, 0.7131, 0.7362, 0.9797,\n        0.2136, 0.4425, 0.0171, 0.8181, 0.9462, 0.3013, 0.2165, 0.6909, 0.9003,\n        0.3088, 0.3639, 0.3246, 0.8997, 0.1554, 0.8359, 0.8999, 0.1437, 0.4792,\n        0.8246, 0.3231, 0.9210, 0.9165, 0.1418, 0.0692, 0.7277, 0.3422, 0.4344,\n        0.5298, 0.4167, 0.0920, 0.7126, 0.6037, 0.9610, 0.3977, 0.8877, 0.9717,\n        0.1467, 0.8345, 0.8467, 0.9623, 0.2985, 0.0996, 0.8530, 0.9099, 0.3528,\n        0.0913, 0.8560, 0.2538, 0.5059, 0.9356, 0.4866, 0.6034, 0.7542, 0.3961,\n        0.6874, 0.5145, 0.4162, 0.0784, 0.0827, 0.9629, 0.1104, 0.4972, 0.3561,\n        0.6250, 0.8727, 0.6935, 0.8392, 0.5786, 0.9030, 0.6588, 0.2742, 0.9786,\n        0.9282, 0.9814, 0.1917, 0.8979, 0.0526, 0.1060, 0.6497, 0.8511, 0.4860,\n        0.3762, 0.7926, 0.9123, 0.6993, ...... View
        # (so we'll actually learn 2x2=4 tasks with correlations)
        #self.doses_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=1)
        self.task_covar_module = TL_Kernel(num_tasks=num_tasks, rank=1)    #, lambda_constraint=MyInterval)
        self.doses_covar_module = TL_Kernel(num_tasks=num_doses, rank=1)

    def forward(self,x_in):
        #mean_x = self.mean_module(x)
        #print("doses lamb:",self.doses_covar_module.covar_factor)
        #print("cancers lamb:",self.task_covar_module.covar_factor)
        #x = x_in[0:(x_in.shape[0])//self.num_doses]
        x = x_in[0::self.num_doses]
        Target_N, Target_Dim = x.shape
        label_i_Ttask = torch.full((Target_N, 1), dtype=torch.long, fill_value=self.num_tasks-1)
        i = torch.cat([self.source_target_x[1][:-self.N_Target_Train], label_i_Ttask])
        ToKron = torch.ones(self.num_doses, 1, dtype=torch.long)    #a vector ones (long) with size of num_doses
        #i_doses = torch.kron(i,ToKron)  #This is to replicate the labels by the number of doses
        i_bydoses = torch.kron(i,ToKron)  # This is to replicate the labels by the number of doses
        "The cat below can have problems in D>1, have to change the [:,None]"
        ToKron_input_obs = torch.ones(self.num_doses, 1)    #a vector ones with size of num_doses
        SS_and_T = torch.cat([self.source_target_x[0][:-self.N_Target_Train].reshape(-1,Target_Dim), x])
        All_Tasks_N = SS_and_T.shape[0]
        SS_and_T_doses = torch.kron(SS_and_T,ToKron_input_obs)  #This is to replicate the Xinput by the number of doses
        # Get input-input covariance
        #covar_x = self.covar_module(SS_and_T)
        covar_x = self.covar_module(SS_and_T_doses)
        # Get task-task covariance
        #covar_i = self.task_covar_module(i)
        covar_i_doses = self.task_covar_module(i_bydoses)   #Relatedness among sources and target tasks
        doses = torch.kron(torch.ones(SS_and_T.shape[0], 1, dtype=torch.long),torch.arange(0,self.num_doses,dtype=torch.long)[:,None])
        covar_doses = self.doses_covar_module(doses)        #Relatedness among doses (or outputs)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i_doses).mul(covar_doses)
        Target_Nxdoses = Target_N*self.num_doses
        Target_NTrainxdoses = self.N_Target_Train*self.num_doses
        KTT = covar[-Target_Nxdoses:, -Target_Nxdoses:]
        KSS = covar[:-Target_Nxdoses, :-Target_Nxdoses]
        KTS = covar[-Target_Nxdoses:, :-Target_Nxdoses]
        KST = covar[:-Target_Nxdoses, -Target_Nxdoses:]
        YS = self.source_target_y.reshape(-1)[:-Target_NTrainxdoses]
        YT = self.source_target_y.reshape(-1)[-Target_NTrainxdoses:]
        CSS = KSS  # .clone() #+ Isig_S
        CTT = KTT  # .clone()  # + Isig_T
        # CSS_i = torch.inverse(CSS)#torch.linalg.inv(CSS)
        # CSS_i = CSS.root_inv_decomposition()
        # mean_x = KTS.matmul( torch.linalg.solve(CSS,YS))    #KTS @ CSS_i @ YS
        # Covar_C = CTT - KTS.matmul(torch.linalg.solve(CSS,KST)) #+ 1e-4*torch.eye(CTT.shape[0])   #KTS @ CSS_i @ KST
        # print(torch.linalg.eigvals(KSS))
        mean_x = KTS.matmul(CSS.inv_matmul(YS))  # KTS @ CSS_i @ YS
        Covar_C = CTT - KTS.matmul(CSS.inv_matmul(KST.evaluate()))  # + 1e-4*torch.eye(CTT.shape[0])   #KTS @ CSS_i @ KST

        # mean_x = self.mean_module(x)
        # print("check Cov_C:",Covar_C.evaluate())

        return gpytorch.distributions.MultivariateNormal(mean_x, Covar_C)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
#likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=mynum_doses,rank=myrank)

likelihood.noise = 0.01  # Some small value, but don't make it too small or numerical performance will suffer. I recommend 1e-4.
#likelihood.noise_covar.raw_noise.requires_grad_(False)  # Mark that we don't want to train the noise.

"We need to pass the data to the model where each label identifies the cancer type"
"all data for cancer0 would have label 0, then cancer1 label1 and so on"
"NOTE: the last task cancer that is passed to the model corresponds to the target task"
# Here we have two iterms that we're passing in as train_inputs

if Dim>1:
    ToKron_input_obs = torch.ones(mynum_doses)[:,None]
    replicate_train_x5 = torch.kron(train_x5, ToKron_input_obs)
else:
    ToKron_input_obs = torch.ones(mynum_doses)
    replicate_train_x5 = torch.kron(train_x5,ToKron_input_obs)

if Dim>1:
    #test_y = train_y5
    #test_x = train_x5
    #ToKron_test = torch.ones(mynum_doses)[:,None]
    #replicate_test_x = torch.kron(test_x,ToKron_test)
    replicate_test_x = test_x.repeat(mynum_doses,1) #torch.kron(test_x,ToKron_test)
else:
    #test_x = torch.linspace(0, 1, 100)
    ToKron_test = torch.ones(mynum_doses)
    replicate_test_x = torch.kron(test_x,ToKron_test)

#replicate_train_x2 = train_x2.repeat(mynum_doses)
"Note that we replicate the data observations with repect to the number of doses (outputs),"
"but the way is replicated consists on replicating each observations, then the following:"
"if x = [1,2,3], then the replication for 3 doses would be like x_rep = [1,1,1,2,2,2,3,3,3]."
"Also, we would reshape y, by [y1,y2,y3,y1,y2,y3,y1,y2,y3] to correspond with x, that is because"
"we have different outputs for the same input x"

"Below in the model we have num_tasks, in this case we refer to num_tasks = 5 since it is the number of cancers"
"to interact in the learning, 4 source cancers + 1 target cancer"
model = TL_GPModel((full_train_x, full_train_i), full_train_y, replicate_train_x5,train_y5.reshape(-1),likelihood, num_tasks = 5,num_doses=mynum_doses)

# this is for running the notebook in our testing framework
import os
smoke_test = ('CI' in os.environ)
training_iterations = 2 if smoke_test else 150

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.2)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iterations):
    optimizer.zero_grad()
    #output = model(full_train_x, full_train_i)
    output = model(replicate_train_x5)#, full_train_i)
    #loss = -mll(output, full_train_y)
    if i > 0: loss_old = loss.item();
    else: loss_old = 10000;
    loss = -mll(output, train_y5.reshape(-1))
    loss.backward()
    print('Iter %d/50 - Loss: %.6f' % (i + 1, loss.item()))
    if np.abs(loss_old-loss.item())<1e-2:
        print("Stopped by Epsilon")
        break
    try:
        optimizer.step()
    except:
        print("change step-size")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters


# Set into eval mode
model.eval()
likelihood.eval()

# Initialize plots
#fig = plt.figure(figsize=(8,3))  # create a figure object
#y1_ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure

# Make predictions
with torch.no_grad():#, gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(replicate_test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

mean = mean.reshape(-1,mynum_doses)
lower = lower.reshape(-1,mynum_doses)
upper = upper.reshape(-1,mynum_doses)

print(f"Showing only 20 figure out of {test_y.shape[0]} to avoid crash in plot")
for posy in range(0,20):
    plt.figure(posy)
    #posy = 88
    # Plot training data as black stars
    plt.plot(torch.linspace(0.111111,1.0,7),test_y[posy, :].numpy(), 'k*')
    # Predictive mean as blue line
    plt.plot(torch.linspace(0.111111,1.0,7),mean[posy, :].numpy(), 'b.')
    plt.plot(torch.linspace(0.111111,1.0,7),lower[posy, :].numpy(), 'c.')
    plt.plot(torch.linspace(0.111111,1.0,7),upper[posy, :].numpy(), 'c.')
    plt.ylim([-0.1,1.2])

#y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
#y1_ax.set_title('Observed Values (Likelihood)')
