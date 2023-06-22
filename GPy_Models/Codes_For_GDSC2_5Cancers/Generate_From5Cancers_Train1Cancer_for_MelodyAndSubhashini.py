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
        opts, args = getopt.getopt(sys.argv[1:], 'i:s:k:w:r:p:c:a:n:')
        # opts = dict(opts)
        # print(opts)
        self.N_iter_epoch = 1200    #number of iterations
        self.which_seed = 1011    #change seed to initialise the hyper-parameters
        self.rank = 7
        self.scale = 1
        self.weight = 1
        self.bash = "1"
        self.N_CellLines_perc = 100   #Here we treat this variable as percentage. Try to put this values as multiple of Num_drugs?
        self.sel_cancer = 1
        self.seed_for_N = 1

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
                self.N_CellLines_perc = arg
            if op == '-a':
                self.sel_cancer = arg
            if op == '-n':
                self.seed_for_N = arg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
config = commandLine()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
dict_cancers={0:'GDSC2_EGFR_PI3K_MAPK_Breast_1000FR.csv',1:'GDSC2_EGFR_PI3K_MAPK_COAD_1000FR.csv',
              2:'GDSC2_EGFR_PI3K_MAPK_LUAD.csv',3:'GDSC2_EGFR_PI3K_MAPK_melanoma.csv',4:'GDSC2_EGFR_PI3K_MAPK_SCLC.csv'}

#indx_cancer = np.array([0,1,2,3,4])
indx_cancer_train = np.array([int(config.sel_cancer)])
#indx_cancer_train = np.delete(indx_cancer,indx_cancer_test)

name_feat_file = "GDSC2_EGFR_PI3K_MAPK_allfeatures.csv"
name_feat_file_nozero = "GDSC2_EGFR_PI3K_MAPK_features_NonZero_Drugs1036_1061_1373.csv" #"GDSC2_EGFR_PI3K_MAPK_features_NonZero.csv"
Num_drugs = 3

name_for_KLrelevance = dict_cancers[indx_cancer_train[0]]
print(name_for_KLrelevance)

df_to_read = pd.read_csv(_FOLDER + name_for_KLrelevance)#.sample(n=N_CellLines,random_state = rand_state_N)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Split of Training and Testing data"
df_4Cancers_traintest_d1 = df_to_read[df_to_read["DRUG_ID"]==1036]#.sample(n=N_CellLines,random_state = rand_state_N)
df_4Cancers_traintest_d2 = df_to_read[df_to_read["DRUG_ID"]==1061]#.sample(n=N_CellLines,random_state = rand_state_N)
df_4Cancers_traintest_d3 = df_to_read[df_to_read["DRUG_ID"] == 1373]#.sample(n=N_CellLines,random_state=rand_state_N)
N_per_drug = [df_4Cancers_traintest_d1.shape[0],df_4Cancers_traintest_d2.shape[0],df_4Cancers_traintest_d3.shape[0]]
if int(config.sel_cancer) == 3:
    np.random.seed(1)
elif int(config.sel_cancer) == 0:
    np.random.seed(10)
elif int(config.sel_cancer) == 1:
    np.random.seed(3)
elif int(config.sel_cancer) == 2:
    np.random.seed(1)
elif int(config.sel_cancer) == 4:
    np.random.seed(1)

TrainTest_drugs_indexes = [np.random.permutation(np.arange(0, N_per_drug[myind])) for myind in range(Num_drugs)]

indx_train = [TrainTest_drugs_indexes[myind][0:round(TrainTest_drugs_indexes[myind].shape[0]*0.7)] for myind in range(Num_drugs)]
indx_test = [TrainTest_drugs_indexes[myind][round(TrainTest_drugs_indexes[myind].shape[0]*0.7):] for myind in range(Num_drugs)]
"Training data by selecting desired percentage"
df_4Cancers_train_d1 = df_4Cancers_traintest_d1.reset_index().drop(columns='index').iloc[indx_train[0]]
df_4Cancers_train_d2 = df_4Cancers_traintest_d2.reset_index().drop(columns='index').iloc[indx_train[1]]
df_4Cancers_train_d3 = df_4Cancers_traintest_d3.reset_index().drop(columns='index').iloc[indx_train[2]]
N_per_drug_Tr = [df_4Cancers_train_d1.shape[0],df_4Cancers_train_d2.shape[0],df_4Cancers_train_d3.shape[0]]
#N_CellLines_perDrug = int(config.N_CellLines)//Num_drugs
N_CellLines_perc = int(config.N_CellLines_perc)
rand_state_N = int(config.seed_for_N)

"Here we select the percentage of the cancer to be used for trainin; the variable N_CellLines_perc indicates the percentage"
Nd1,Nd2,Nd3 = round(N_per_drug_Tr[0]*N_CellLines_perc/100.0),round(N_per_drug_Tr[1]*N_CellLines_perc/100.0),round(N_per_drug_Tr[2]*N_CellLines_perc/100.0)
df_4Cancers_train = pd.concat([df_4Cancers_train_d1.sample(n=Nd1,random_state = rand_state_N), df_4Cancers_train_d2.sample(n=Nd2,random_state = rand_state_N),df_4Cancers_train_d3.sample(n=Nd3,random_state = rand_state_N)])
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Testing data"
df_4Cancers_test_d1 = df_4Cancers_traintest_d1.reset_index().drop(columns='index').iloc[indx_test[0]]
df_4Cancers_test_d2 = df_4Cancers_traintest_d2.reset_index().drop(columns='index').iloc[indx_test[1]]
df_4Cancers_test_d3 = df_4Cancers_traintest_d3.reset_index().drop(columns='index').iloc[indx_test[2]]
df_4Cancers_test = pd.concat([df_4Cancers_test_d1, df_4Cancers_test_d2,df_4Cancers_test_d3])
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
df_4Cancers_train = df_4Cancers_train.dropna()
df_4Cancers_test = df_4Cancers_test.dropna()

# Here we just check that from the column index 25 the input features start
start_pos_features = 25
print(df_4Cancers_train.columns[start_pos_features])

df_feat_Names = pd.read_csv(_FOLDER + name_feat_file)  # Contain Feature Names
df_feat_Names_nozero = pd.read_csv(_FOLDER + name_feat_file_nozero)  # Contain Feature Names
indx_nozero = df_feat_Names_nozero['index'].values[start_pos_features:]

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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

posy = 0
#my_plot(posy,0,Ydose50,Ydose_res,IC50,AUC,Emax,x_lin,x_real_dose,y_train_drug)
#my_plot(posy,1,Ydose50_test,Ydose_res_test,IC50_test,AUC_test,Emax_test,x_lin,x_real_dose,y_test_drug)

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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Select the columns that actually are relevant for the experiment of Drugs 1036, 1061 and 1373"
CellLinesID_plus_indx_nozero = np.concatenate((np.arange(0,25),indx_nozero))

df_4Cancers_train = df_4Cancers_train[df_4Cancers_train.columns[CellLinesID_plus_indx_nozero]]
df_4Cancers_test = df_4Cancers_test[df_4Cancers_test.columns[CellLinesID_plus_indx_nozero]]


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
# #final_path ='N5thCancer_'+str(config.N_5thCancer_ToBe_Included)+'/'+str(config.sel_cancer)+'/N'+str(config.N_CellLines)+'/seed'+str(rand_state_N)+'/'
#Ntotal_Cells = int(config.N_CellLines)*4 + int(config.N_5thCancer_ToBe_Included)
NInTrain = df_4Cancers_train.shape[0]
NInTest = df_4Cancers_test.shape[0]
final_path = './Datasets_From5Cancers_Train1Cancer_Increasing_TrainingData/'+str(Cancer_Names[int(config.sel_cancer)])+'_cancer/Train'+str(NInTrain)+'/seed'+str(int(config.seed_for_N))+'/'
if not os.path.exists(final_path):
   os.makedirs(final_path)

df_4Cancers_train.to_csv(final_path+'Train'+str(NInTrain)+'_seed'+str(int(config.seed_for_N))+'.csv')

final_path_test = './Datasets_From5Cancers_Train1Cancer_Increasing_TrainingData/'+str(Cancer_Names[int(config.sel_cancer)])+'_cancer/'
if not os.path.exists(final_path_test):
   os.makedirs(final_path_test)
df_4Cancers_test.to_csv(final_path_test+'Test_'+str(Cancer_Names[int(config.sel_cancer)])+'cancer.csv')