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
        self.N_CellLines = 12   #Try to put this values as multiple of Num_drugs
        self.sel_cancer = 0
        self.seed_for_N = 1
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
    if int(config.N_5thCancer_ToBe_Included) == 9:
        if indx_cancer_test == 0:
            np.random.seed(6)   #There are 3 IC50s, we put the 9 observations (1 IC50s and 8 non-IC50)
        elif indx_cancer_test == 1:
            np.random.seed(1)   #There are 8 IC50s, we put the 9 observations (3 IC50s and 6 non-IC50)
        elif indx_cancer_test == 2:
            np.random.seed(1)  #There is only 1 IC50s for this Cancer almost all data are non-responsive (Test with no IC50)
        elif indx_cancer_test == 3:
            np.random.seed(1)  #There are 63 IC50s and 68 Non-IC50, we put the 9 observations (5 IC50s and 4 non-IC50)
        elif indx_cancer_test == 4:
            np.random.seed(1)    #There are not IC50s for this Cancer all data are non-responsive
    elif int(config.N_5thCancer_ToBe_Included) == 45:
        if indx_cancer_test == 0:
            np.random.seed(6)   #There are 3 IC50s, we put the 45 observations (1 IC50s and 44 non-IC50)
        elif indx_cancer_test == 1:
            np.random.seed(3)   #There are 8 IC50s, we put the 45 observations (4 IC50s and 40 non-IC50)
        elif indx_cancer_test == 2:
            np.random.seed(1)  #There is only 1 IC50s for this Cancer almost all data are non-responsive (Test with no IC50)
        elif indx_cancer_test == 3:
            np.random.seed(1)  #There are 63 IC50s and 68 Non-IC50, we put the 45 observations (19 IC50s and 26 non-IC50)
        elif indx_cancer_test == 4:
            np.random.seed(1)    #There are not IC50s for this Cancer all data are non-responsive
    else:
        print("SELECTING THE IC50s RANDOMLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

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

prob_drugs = [M_dist_d1_norm/M_dist_d1_norm.sum(0),M_dist_d2_norm/M_dist_d2_norm.sum(0),M_dist_d3_norm/M_dist_d3_norm.sum(0)]
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
#0.142857143
"Be careful that x starts from 0.111111 for 9 drugs,"
"but x starts from 0.142857143 for the case of 7 drugs"
x_lin = np.linspace(0.142857143, 1, 1000)
x_real_dose = np.linspace(0.142857143, 1, 7)  #Here is 7 due to using GDSC2 that has 7 doses
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

#posy = 18
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

"Select the columns that actually are relevant for the experiment of Drugs 1036, 1061 and 1373"
df_4Cancers_train = df_4Cancers_train[df_4Cancers_train.columns[indx_nozero]]
df_4Cancers_test = df_4Cancers_test[df_4Cancers_test.columns[indx_nozero]]

"Adding the AUC, Emax and IC50 to the Training and Testing datasets"
df_4Cancers_train.insert(0, "AUC", AUC, True)
df_4Cancers_train.insert(1, "Emax", Emax, True)
df_4Cancers_train.insert(2, "IC50", IC50, True)

df_4Cancers_test.insert(0, "AUC", AUC_test , True)
df_4Cancers_test.insert(1, "Emax", Emax_test, True)
df_4Cancers_test.insert(2, "IC50", IC50_test, True)

"Reset index and drop old index"
df_4Cancers_train = df_4Cancers_train.reset_index().drop(columns='index')
df_4Cancers_test = df_4Cancers_test.reset_index().drop(columns='index')

Cancer_Names = ['breast','COAD','LUAD','melanoma','SCLC']

"Save the datasets"
#final_path ='N5thCancer_'+str(config.N_5thCancer_ToBe_Included)+'/'+str(config.sel_cancer)+'/N'+str(config.N_CellLines)+'/seed'+str(rand_state_N)+'/'
Ntotal_Cells = int(config.N_CellLines)*4 + int(config.N_5thCancer_ToBe_Included)
final_path = './Datasets_5Cancers_Increasing_TrainingData/'+str(Cancer_Names[int(config.sel_cancer)])+'_cancer/N5th_CancerInTrain_'+str(config.N_5thCancer_ToBe_Included)+'/NTrain_'+str(int(config.N_CellLines)*4)+'plus_'+str(int(config.N_5thCancer_ToBe_Included))+'/seed'+str(int(config.seed_for_N))+'/'
if not os.path.exists(final_path):
   os.makedirs(final_path)

df_4Cancers_train.to_csv(final_path+'Train_N5th_'+str(config.N_5thCancer_ToBe_Included)+'_seed'+str(int(config.seed_for_N))+'.csv')
df_4Cancers_test.to_csv(final_path+'Test_N5th_'+str(config.N_5thCancer_ToBe_Included)+'_seed'+str(int(config.seed_for_N))+'.csv')



