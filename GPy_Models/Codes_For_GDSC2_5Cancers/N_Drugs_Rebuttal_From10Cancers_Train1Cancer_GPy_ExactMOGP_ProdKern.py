import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
import os

# _FOLDER = "/rds/general/user/jgiraldo/home/Dataset_5Cancers/GDSC2_10drugs/" #HPC path
_FOLDER = "/home/juanjo/Work_Postdoc/my_codes_postdoc/Dataset_5Cancers/GDSC2_10drugs/"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def sigmoid_4_param(x, x0, L, k, d):
    """ Comparing with Dennis Wang's sigmoid:
    x0 -  p - position, correlation with IC50 or EC50
    L = 1 in Dennis Wang's sigmoid, protect from devision by zero if x is too small
    k = -1/s (s -shape parameter)
    d - determines the vertical position of the sigmoid - shift on y axis - better fitting then Dennis Wang's sigmoid

    """
    return (1 / (L + np.exp(-k * (x - x0))) + d)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import getopt
import sys

warnings.filterwarnings("ignore")


# os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

class commandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'i:s:k:w:r:p:c:a:n:')
        # opts = dict(opts)
        # print(opts)
        self.N_iter = 700  # number of iterations
        self.which_seed = 1011  # change seed to initialise the hyper-parameters
        self.rank = 7
        self.scale = 1
        self.weight = 1
        self.bash = "None"
        self.N_CellLines_perc = 100  # Here we treat this variable as percentage. Try to put this values as multiple of Num_drugs?
        self.sel_cancer = 0
        self.seed_for_N = 1

        for op, arg in opts:
            # print(op,arg)
            if op == '-i':
                self.N_iter = arg
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
# dict_cancers={0:'GDSC2_EGFR_PI3K_MAPK_Breast_1000FR.csv',1:'GDSC2_EGFR_PI3K_MAPK_COAD_1000FR.csv',
#               2:'GDSC2_EGFR_PI3K_MAPK_LUAD.csv',3:'GDSC2_EGFR_PI3K_MAPK_melanoma.csv',4:'GDSC2_EGFR_PI3K_MAPK_SCLC.csv'}

dict_cancers = {0: 'GDSC2_10drugs_SKCM_1000FR.csv', 1: 'GDSC2_10drugs_SCLC_1000FR.csv',
                2: 'GDSC2_10drugs_PAAD_1000FR.csv', 3: 'GDSC2_10drugs_OV_1000FR.csv',
                4: 'GDSC2_10drugs_LUAD_1000FR.csv',
                5: 'GDSC2_10drugs_HNSC_1000FR.csv', 6: 'GDSC2_10drugs_ESCA_1000FR.csv',
                7: 'GDSC2_10drugs_COAD_1000FR.csv',
                8: 'GDSC2_10drugs_BRCA_1000FR.csv', 9: 'GDSC2_10drugs_ALL_1000FR.csv'}

# indx_cancer = np.array([0,1,2,3,4])
indx_cancer_train = np.array([int(config.sel_cancer)])
# indx_cancer_train = np.delete(indx_cancer,indx_cancer_test)

name_feat_file = "GDSC2_10drugresponses_allfeatures.csv"  # "GDSC2_EGFR_PI3K_MAPK_allfeatures.csv"
# name_feat_file_nozero = "GDSC2_EGFR_PI3K_MAPK_features_NonZero_Drugs1036_1061_1373.csv" #"GDSC2_EGFR_PI3K_MAPK_features_NonZero.csv"
Num_drugs = 10

name_for_KLrelevance = dict_cancers[indx_cancer_train[0]]
print(name_for_KLrelevance)

df_to_read = pd.read_csv(_FOLDER + name_for_KLrelevance)  # .sample(n=N_CellLines,random_state = rand_state_N)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Split of Training and Testing data"
df_4Cancers_traintest_d1 = df_to_read[df_to_read["DRUG_ID"] == 1012]  # .sample(n=N_CellLines,random_state = rand_state_N)
df_4Cancers_traintest_d2 = df_to_read[df_to_read["DRUG_ID"] == 1021]  # .sample(n=N_CellLines,random_state = rand_state_N)
df_4Cancers_traintest_d3 = df_to_read[df_to_read["DRUG_ID"] == 1036]  # .sample(n=N_CellLines,random_state=rand_state_N)
df_4Cancers_traintest_d4 = df_to_read[df_to_read["DRUG_ID"] == 1053]  # .sample(n=N_CellLines,random_state=rand_state_N)
df_4Cancers_traintest_d5 = df_to_read[df_to_read["DRUG_ID"] == 1058]  # .sample(n=N_CellLines,random_state=rand_state_N)
df_4Cancers_traintest_d6 = df_to_read[df_to_read["DRUG_ID"] == 1059]  # .sample(n=N_CellLines,random_state=rand_state_N)
df_4Cancers_traintest_d7 = df_to_read[df_to_read["DRUG_ID"] == 1061]  # .sample(n=N_CellLines,random_state=rand_state_N)
df_4Cancers_traintest_d8 = df_to_read[df_to_read["DRUG_ID"] == 1149]  # .sample(n=N_CellLines,random_state=rand_state_N)
df_4Cancers_traintest_d9 = df_to_read[df_to_read["DRUG_ID"] == 1372]  # .sample(n=N_CellLines,random_state=rand_state_N)
df_4Cancers_traintest_d10 = df_to_read[df_to_read["DRUG_ID"] == 1373]  # .sample(n=N_CellLines,random_state=rand_state_N)

N_per_drug = [df_4Cancers_traintest_d1.shape[0], df_4Cancers_traintest_d2.shape[0], df_4Cancers_traintest_d3.shape[0],
              df_4Cancers_traintest_d4.shape[0], df_4Cancers_traintest_d5.shape[0], df_4Cancers_traintest_d6.shape[0],
              df_4Cancers_traintest_d7.shape[0], df_4Cancers_traintest_d8.shape[0], df_4Cancers_traintest_d9.shape[0],
              df_4Cancers_traintest_d10.shape[0]]
np.random.seed(1)
# if int(config.sel_cancer) == 3:
#     np.random.seed(1)
# elif int(config.sel_cancer) == 0:
#     np.random.seed(10)
# elif int(config.sel_cancer) == 1:
#     np.random.seed(3)
# elif int(config.sel_cancer) == 2:
#     np.random.seed(1)
# elif int(config.sel_cancer) == 4:
#     np.random.seed(1)

TrainTest_drugs_indexes = [np.random.permutation(np.arange(0, N_per_drug[myind])) for myind in range(Num_drugs)]

indx_train = [TrainTest_drugs_indexes[myind][0:round(TrainTest_drugs_indexes[myind].shape[0] * 0.7)] for myind in
              range(Num_drugs)]
indx_test = [TrainTest_drugs_indexes[myind][round(TrainTest_drugs_indexes[myind].shape[0] * 0.7):] for myind in
             range(Num_drugs)]
"Training data by selecting desired percentage"
df_4Cancers_train_d1 = df_4Cancers_traintest_d1.reset_index().drop(columns='index').iloc[indx_train[0]]
df_4Cancers_train_d2 = df_4Cancers_traintest_d2.reset_index().drop(columns='index').iloc[indx_train[1]]
df_4Cancers_train_d3 = df_4Cancers_traintest_d3.reset_index().drop(columns='index').iloc[indx_train[2]]
df_4Cancers_train_d4 = df_4Cancers_traintest_d4.reset_index().drop(columns='index').iloc[indx_train[3]]
df_4Cancers_train_d5 = df_4Cancers_traintest_d5.reset_index().drop(columns='index').iloc[indx_train[4]]
df_4Cancers_train_d6 = df_4Cancers_traintest_d6.reset_index().drop(columns='index').iloc[indx_train[5]]
df_4Cancers_train_d7 = df_4Cancers_traintest_d7.reset_index().drop(columns='index').iloc[indx_train[6]]
df_4Cancers_train_d8 = df_4Cancers_traintest_d8.reset_index().drop(columns='index').iloc[indx_train[7]]
df_4Cancers_train_d9 = df_4Cancers_traintest_d9.reset_index().drop(columns='index').iloc[indx_train[8]]
df_4Cancers_train_d10 = df_4Cancers_traintest_d10.reset_index().drop(columns='index').iloc[indx_train[9]]

N_per_drug_Tr = [df_4Cancers_train_d1.shape[0], df_4Cancers_train_d2.shape[0], df_4Cancers_train_d3.shape[0],
                 df_4Cancers_train_d4.shape[0], df_4Cancers_train_d5.shape[0], df_4Cancers_train_d6.shape[0],
                 df_4Cancers_train_d7.shape[0], df_4Cancers_train_d8.shape[0], df_4Cancers_train_d9.shape[0],
                 df_4Cancers_train_d10.shape[0]]

# #N_CellLines_perDrug = int(config.N_CellLines)//Num_drugs
N_CellLines_perc = int(config.N_CellLines_perc)
rand_state_N = int(config.seed_for_N)

"Here we select the percentage of the cancer to be used for trainin; the variable N_CellLines_perc indicates the percentage"
Nd1, Nd2, Nd3 = round(N_per_drug_Tr[0] * N_CellLines_perc / 100.0), round(
    N_per_drug_Tr[1] * N_CellLines_perc / 100.0), round(N_per_drug_Tr[2] * N_CellLines_perc / 100.0)
Nd4, Nd5, Nd6 = round(N_per_drug_Tr[3] * N_CellLines_perc / 100.0), round(
    N_per_drug_Tr[4] * N_CellLines_perc / 100.0), round(N_per_drug_Tr[5] * N_CellLines_perc / 100.0)
Nd7, Nd8, Nd9 = round(N_per_drug_Tr[6] * N_CellLines_perc / 100.0), round(
    N_per_drug_Tr[7] * N_CellLines_perc / 100.0), round(N_per_drug_Tr[8] * N_CellLines_perc / 100.0)
Nd10 = round(N_per_drug_Tr[9] * N_CellLines_perc / 100.0)
df_4Cancers_train = pd.concat([df_4Cancers_train_d1.sample(n=Nd1, random_state=rand_state_N),
                               df_4Cancers_train_d2.sample(n=Nd2, random_state=rand_state_N),
                               df_4Cancers_train_d3.sample(n=Nd3, random_state=rand_state_N),
                               df_4Cancers_train_d4.sample(n=Nd4, random_state=rand_state_N),
                               df_4Cancers_train_d5.sample(n=Nd5, random_state=rand_state_N),
                               df_4Cancers_train_d6.sample(n=Nd6, random_state=rand_state_N),
                               df_4Cancers_train_d7.sample(n=Nd7, random_state=rand_state_N),
                               df_4Cancers_train_d8.sample(n=Nd8, random_state=rand_state_N),
                               df_4Cancers_train_d9.sample(n=Nd9, random_state=rand_state_N),
                               df_4Cancers_train_d10.sample(n=Nd10, random_state=rand_state_N)])
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Testing data"
df_4Cancers_test_d1 = df_4Cancers_traintest_d1.reset_index().drop(columns='index').iloc[indx_test[0]]
df_4Cancers_test_d2 = df_4Cancers_traintest_d2.reset_index().drop(columns='index').iloc[indx_test[1]]
df_4Cancers_test_d3 = df_4Cancers_traintest_d3.reset_index().drop(columns='index').iloc[indx_test[2]]
df_4Cancers_test_d4 = df_4Cancers_traintest_d4.reset_index().drop(columns='index').iloc[indx_test[3]]
df_4Cancers_test_d5 = df_4Cancers_traintest_d5.reset_index().drop(columns='index').iloc[indx_test[4]]
df_4Cancers_test_d6 = df_4Cancers_traintest_d6.reset_index().drop(columns='index').iloc[indx_test[5]]
df_4Cancers_test_d7 = df_4Cancers_traintest_d7.reset_index().drop(columns='index').iloc[indx_test[6]]
df_4Cancers_test_d8 = df_4Cancers_traintest_d8.reset_index().drop(columns='index').iloc[indx_test[7]]
df_4Cancers_test_d9 = df_4Cancers_traintest_d9.reset_index().drop(columns='index').iloc[indx_test[8]]
df_4Cancers_test_d10 = df_4Cancers_traintest_d10.reset_index().drop(columns='index').iloc[indx_test[9]]

df_4Cancers_test = pd.concat([df_4Cancers_test_d1, df_4Cancers_test_d2, df_4Cancers_test_d3,
                              df_4Cancers_test_d4, df_4Cancers_test_d5, df_4Cancers_test_d6,
                              df_4Cancers_test_d7, df_4Cancers_test_d8, df_4Cancers_test_d9,
                              df_4Cancers_test_d10])
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
df_4Cancers_train = df_4Cancers_train.dropna()
df_4Cancers_test = df_4Cancers_test.dropna()

# Here we just check that from the column index 25 the input features start
start_pos_features = 25
print(df_4Cancers_train.columns[start_pos_features])

df_feat_Names = pd.read_csv(_FOLDER + name_feat_file)  # Contain Feature Names
indx_bool = (df_4Cancers_train[df_4Cancers_train.columns[25:]].std(0) != 0.0).values
indx_nozero = np.arange(0, indx_bool.shape[0])[indx_bool] + start_pos_features
# df_feat_Names_nozero = pd.read_csv(_FOLDER + name_feat_file_nozero)  # Contain Feature Names
Name_Features_5Cancers = df_feat_Names['feature'].values[indx_nozero]
# indx_nozero = df_feat_Names_nozero['index'].values[start_pos_features:]

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
    y_train_drug = np.concatenate(
        (y_train_drug, np.clip(df_4Cancers_train["norm_cells_" + str(i)].values[:, None], 1.0e-9, np.inf)), 1)
    y_test_drug = np.concatenate(
        (y_test_drug, np.clip(df_4Cancers_test["norm_cells_" + str(i)].values[:, None], 1.0e-9, np.inf)), 1)

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
# 0.142857143
"Be careful that x starts from 0.111111 for 9 drugs,"
"but x starts from 0.142857143 for the case of 7 drugs"
x_lin = np.linspace(0.142857143, 1, 1000)
x_real_dose = np.linspace(0.142857143, 1, 7)  # Here is 7 due to using GDSC2 that has 7 doses


def Get_IC50_AUC_Emax(params_4_sig_train, x_lin, x_real_dose):
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
        elif Ydose_res[i][-1] < 0.5:
            for dose_j in range(x_lin.shape[0]):
                if (Ydose_res[i][dose_j] < 0.5):
                    break
            Ydose50.append(Ydose_res[i][dose_j])
            aux_IC50 = x_lin[dose_j]  # it has to be a float not an array to avoid bug
            IC50.append(aux_IC50)
        else:
            Ydose50.append(0.5)
            IC50.append(1.5)  # IC50.append(x_lin[-1])

    return Ydose50, Ydose_res, IC50, AUC, Emax


Ydose50, Ydose_res, IC50, AUC, Emax = Get_IC50_AUC_Emax(params_4_sig_train, x_lin, x_real_dose)
Ydose50_test, Ydose_res_test, IC50_test, AUC_test, Emax_test = Get_IC50_AUC_Emax(params_4_sig_test, x_lin, x_real_dose)


def my_plot(posy, fig_num, Ydose50, Ydose_res, IC50, AUC, Emax, x_lin, x_real_dose, y_train_drug):
    plt.figure(fig_num)
    plt.plot(x_lin, Ydose_res[posy])
    plt.plot(x_real_dose, y_train_drug[posy, :], '.')
    plt.plot(IC50[posy], Ydose50[posy], 'rx')
    plt.plot(x_lin, np.ones_like(x_lin) * Emax[posy], 'r')  # Plot a horizontal line as Emax
    plt.title(f"AUC = {AUC[posy]}")
    print(AUC[posy])


posy = 1
my_plot(posy, 0, Ydose50, Ydose_res, IC50, AUC, Emax, x_lin, x_real_dose, y_train_drug)
my_plot(posy, 1, Ydose50_test, Ydose_res_test, IC50_test, AUC_test, Emax_test, x_lin, x_real_dose, y_test_drug)

AUC = np.array(AUC)
IC50 = np.array(IC50)
Emax = np.array(Emax)
AUC_test = np.array(AUC_test)
IC50_test = np.array(IC50_test)
Emax_test = np.array(Emax_test)

"Below we select just the columns with std higher than zero"

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


def bypass_model_params(model1, model2):
    model2.mul.rbf.lengthscale = model1.mul.rbf.lengthscale.copy()
    model2.mul.rbf_1.lengthscale = model1.mul.rbf_1.lengthscale.copy()
    model2.mul.rbf_2.lengthscale = model1.mul.rbf_2.lengthscale.copy()
    model2.mul.rbf_3.lengthscale = model1.mul.rbf_3.lengthscale.copy()
    model2.mul.coregion.W = model1.mul.coregion.W.copy()
    model2.mul.coregion.kappa = model1.mul.coregion.kappa.copy()
    model2.Gaussian_noise.variance = model1.Gaussian_noise.variance


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"This is a function to just select the features related to ID, include IC50, Emax, AUC and get rid of genomics features"


def SummaryMetrics_DataFrameTest(df_4Cancers_test, m_pred_curve, AUC_pred, AUC_val, Emax_pred, Emax_val, IC50_pred,
                                 IC50_val):
    df_4Cancers_test = df_4Cancers_test[df_4Cancers_test.columns[0:25]]

    for i in range(7):
        df_4Cancers_test['norm_cell_' + str(i + 1) + '_MOGP'] = m_pred_curve[:, i]

    df_4Cancers_test['AUC_MOGP'] = AUC_pred
    df_4Cancers_test['AUC_s4'] = AUC_val
    df_4Cancers_test['Emax_MOGP'] = Emax_pred
    df_4Cancers_test['Emax_s4'] = Emax_val
    df_4Cancers_test['IC50_MOGP'] = IC50_pred
    df_4Cancers_test['IC50_s4'] = IC50_val
    return df_4Cancers_test


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Create a K-fold for cross-validation"
from sklearn.model_selection import KFold, cross_val_score

Ndata = Xall.shape[0]
Xind = np.arange(Ndata)
nsplits = 5  # Ndata
k_fold = KFold(n_splits=nsplits, shuffle=True, random_state=1)
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

model_all = []

# for Nfold in range(nsplits,nsplits+1):
for Nfold in range(nsplits, -1, -1):
    model = []
    "The first if below is for the cross-val"
    "Then the else is for using all data to save the model trained over all data"
    if Nfold < nsplits:
        train_ind, test_ind = list_folds[Nfold]
        print(f"{test_ind} to Val in IC50")

        Xval_aux = Xall[test_ind].copy()
        Ylabel_val = np.array([i * np.ones(Xval_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]

        Xval = np.concatenate((np.tile(Xval_aux, (Ntasks, 1)), Ylabel_val), 1)

        # Xval = Xall[train_ind].copy()
        Xtrain_aux = Xall[train_ind].copy()
        Ylabel_train = np.array([i * np.ones(Xtrain_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]
        Xtrain = np.concatenate((np.tile(Xtrain_aux, (Ntasks, 1)), Ylabel_train), 1)

        Yval = Yall[test_ind].T.flatten().copy()[:, None]
        # Yval = Yall[train_ind].copy()
        Ytrain = Yall[train_ind].T.flatten().copy()[:, None]

        Emax_val = Emax_all[test_ind].copy()
        AUC_val = AUC_all[test_ind].copy()
        IC50_val = IC50_all[test_ind].copy()

        # Emax_val = Emax_all[train_ind].copy()
        # AUC_val = AUC_all[train_ind].copy()
        # IC50_val = IC50_all[train_ind].copy()

    else:
        print(f"Train ovell all Data")
        # _, test_ind = list_folds[0] #Just assigning by defaul fold0 as the test (of course not to report it as a result)
        Xval_aux = X_test_features.copy()  # Xall[test_ind].copy()
        Ylabel_val = np.array([i * np.ones(Xval_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]

        Xval = np.concatenate((np.tile(Xval_aux, (Ntasks, 1)), Ylabel_val), 1)

        Xtrain_aux = Xall.copy()
        Ylabel_train = np.array([i * np.ones(Xtrain_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]
        Xtrain = np.concatenate((np.tile(Xtrain_aux, (Ntasks, 1)), Ylabel_train), 1)

        Yval = y_test_drug.T.flatten().copy()[:, None]
        Ytrain = Yall.T.flatten().copy()[:, None]

        Emax_val = Emax_test.copy()
        AUC_val = AUC_test.copy()
        IC50_val = IC50_test.copy()
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
    Dim = Xtrain.shape[1] - 1

    "For each cancer the feature that are actually informative change, so"
    "We created the dict below to properly assign the active features per kernel as per the cancer type"
    AddKern_dict = {0: [95, 389, 407, Dim], 1: [89, 416, 435, Dim],
                    2: [28, 300, 319, Dim], 3: [97, 429, 448, Dim],
                    4: [116, 471, 478, Dim], 5: [56, 307, 349, Dim],
                    6: [52, 335, 344, Dim], 7: [235, 461, 493, Dim],
                    8: [98, 437, 456, Dim], 9: [132, 245, 264, Dim]}

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    myseed = int(config.which_seed)
    np.random.seed(myseed)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "Product Kernels"
    "Below we use the locations:"
    "0:AddKern_loc[0] for Mutation"
    "AddKern_loc[0]:AddKern_loc[1] for PANCAN"
    "AddKern_loc[1]:AddKern_loc[2] for COPY-Number"
    "AddKern_loc[2]:end for Drugs compounds"
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # TODO: Create a particular list AddKern_loc for each cancer type this
    # TODO: due to having different features per cancer. A features can have std = 0 for one cancer, but not for other
    split_dim = 4
    # AddKern_loc = [279, 697,768,Dim]
    AddKern_loc = AddKern_dict[int(config.sel_cancer)]
    # AddKern_loc = [89, 416, 435, Dim]
    mykern = GPy.kern.RBF(AddKern_loc[0], active_dims=list(np.arange(0, AddKern_loc[0])))
    print(list(np.arange(0, AddKern_loc[0])))
    for i in range(1, split_dim):
        mykern = mykern * GPy.kern.RBF(AddKern_loc[i] - AddKern_loc[i - 1],
                                       active_dims=list(np.arange(AddKern_loc[i - 1], AddKern_loc[i])))
        print(list(np.arange(AddKern_loc[i - 1], AddKern_loc[i])))

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    mykern.rbf.lengthscale = float(config.scale) * np.sqrt(Dim) * np.random.rand()
    mykern.rbf.variance.fix()
    for i in range(1, split_dim):
        eval("mykern.rbf_" + str(
            i) + ".lengthscale.setfield(float(config.scale)* np.sqrt(Dim) * np.random.rand(), np.float64)")
        eval("mykern.rbf_" + str(i) + ".variance.fix()")

    kern = mykern ** GPy.kern.Coregionalize(1, output_dim=Ntasks, rank=rank)
    model = GPy.models.GPRegression(Xtrain, Ytrain, kern)
    Init_Ws = float(config.weight) * np.random.randn(Ntasks, rank)
    model.kern.coregion.W = Init_Ws
    if Nfold == nsplits:
        model.optimize(optimizer='adam', messages=True, max_iters=int(config.N_iter), ipython_notebook=False,
                       step_rate=0.01)
        model_all = model
    else:
        bypass_model_params(model_all, model)
    # model[:] = np.load('/home/juanjo/Work_Postdoc/my_codes_postdoc/GPy_Models/Codes_For_GDSC2_5Cancers/Test_Data_ToPlot_GDSC2_5Cancers/Three_drugs/m_924.npy')

    m_pred, v_pred = model.predict(Xval, full_cov=False)
    plt.figure(Nfold + 1)
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
    Val_NMLL = -np.mean(model.log_predictive_density(Xval, Yval))
    print("NegLPD Val", Val_NMLL)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    from scipy.interpolate import interp1d
    from scipy.interpolate import pchip_interpolate

    "Be careful that x starts from 0.111111 for 9 drugs,"
    "but x starts from 0.142857143 for the case of 7 drugs"
    x_dose = np.linspace(0.142857143, 1.0, 7)
    x_dose_new = np.linspace(0.142857143, 1.0, 1000)
    Ydose50_pred = []
    IC50_pred = []
    AUC_pred = []
    Emax_pred = []
    Y_pred_interp_all = []
    std_upper_interp_all = []
    std_lower_interp_all = []
    for i in range(Yval_curve.shape[0]):
        y_resp = m_pred_curve[i, :].copy()
        std_upper = y_resp + 2 * np.sqrt(v_pred_curve[i, :])
        std_lower = y_resp - 2 * np.sqrt(v_pred_curve[i, :])
        f = interp1d(x_dose, y_resp)
        # f2 = interp1d(x_dose, y_resp, kind='cubic')
        y_resp_interp = pchip_interpolate(x_dose, y_resp, x_dose_new)
        std_upper_interp = pchip_interpolate(x_dose, std_upper, x_dose_new)
        std_lower_interp = pchip_interpolate(x_dose, std_lower, x_dose_new)

        # y_resp_interp = f2(x_dose_new)
        Y_pred_interp_all.append(y_resp_interp)
        std_upper_interp_all.append(std_upper_interp)
        std_lower_interp_all.append(std_lower_interp)
        AUC_pred.append(metrics.auc(x_dose_new, y_resp_interp))
        Emax_pred.append(y_resp_interp[-1])

        res1 = y_resp_interp < 0.507
        res2 = y_resp_interp > 0.493
        res_aux = np.where(res1 & res2)[0]
        if (res1 & res2).sum() > 0:
            res_IC50 = np.arange(res_aux[0], res_aux[0] + res_aux.shape[0]) == res_aux
            res_aux = res_aux[res_IC50].copy()
        else:
            res_aux = res1 & res2

        if (res1 & res2).sum() > 0:
            Ydose50_pred.append(y_resp_interp[res_aux].mean())
            IC50_pred.append(x_dose_new[res_aux].mean())
        elif y_resp_interp[-1] < 0.5:
            for dose_j in range(x_dose_new.shape[0]):
                if (y_resp_interp[dose_j] < 0.5):
                    break
            Ydose50_pred.append(y_resp_interp[dose_j])
            aux_IC50 = x_dose_new[dose_j]  # it has to be a float not an array to avoid bug
            IC50_pred.append(aux_IC50)
        else:
            Ydose50_pred.append(0.5)
            IC50_pred.append(1.5)

    Ydose50_pred = np.array(Ydose50_pred)
    IC50_pred = np.array(IC50_pred)[:, None]
    AUC_pred = np.array(AUC_pred)[:, None]
    Emax_pred = np.array(Emax_pred)[:, None]

    "The lines belos plot the dose response prediction over test set uncomment to plot!"
    if Nfold == nsplits:
        for posy in range(Y_pred_interp_all.__len__()):
            if(posy == 40 or posy == 63):  #87 #95
                plt.figure(Nfold + nsplits + 2 + posy)
                plt.plot(x_dose_new, Y_pred_interp_all[posy],'r')
                plt.plot(x_dose_new, std_upper_interp_all[posy], 'r--')
                plt.plot(x_dose_new, std_lower_interp_all[posy], 'r--')
                plt.plot(x_dose, Yval_curve[posy, :], 'k.',markersize=10)
                plt.plot(IC50_pred[posy], Ydose50_pred[posy], 'bx',markersize=10)
                #plt.plot(x_dose_new, np.ones_like(x_dose_new) * Emax_pred[posy], 'g')  # Plot a horizontal line as Emax
                offset = 0.0
                if posy==87:
                    offset = 0.05
                plt.plot(np.linspace(0,1,100), np.ones(100) * Emax_pred[posy]-offset, 'g')  # Plot a horizontal line as Emax
                plt.plot(IC50_pred[posy]*np.ones(10)-0.001,np.linspace(-0.1,0.5,10),'b|',markersize=5,linewidth=3)
                plt.plot(IC50_pred[posy] * np.ones(10)+0.001, np.linspace(-0.1, 0.5, 10), 'b|', markersize=5, linewidth=3)
                # plt.plot(x_dose_new, np.ones_like(x_dose_new) * Emax_val[posy], 'r')  # Plot a horizontal line as Emax
                plt.title(f"Response Prediction Cell-Line: 753561; AUC = {AUC_pred[posy][0]:0.4f}")
                plt.ylim([-0.05, 1.2])
                plt.xlim([0.135, 1.05])
                if posy == 40:
                    plt.xlabel('Drug Concentration (DrugID:1058)',fontsize=14)
                if posy == 63:
                    plt.xlabel('Drug Concentration (DrugID:1059)', fontsize=14)
                plt.ylabel('Cell Viability',fontsize=14)
                plt.legend(['MOGP-pred','2-std','_nolegend_','Real Data','IC50','Emax'])

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
    spear_corr_actualIC50, p_value_actual = spearmanr(IC50_val[pos_Actual_IC50], IC50_pred[pos_Actual_IC50])
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
    else:
        print(f"Nfold:{Nfold} and nsplits:{nsplits}")
        df_4Cancers_test = SummaryMetrics_DataFrameTest(df_4Cancers_test, m_pred_curve, AUC_pred, AUC_val, Emax_pred,
                                                        Emax_val, IC50_pred, IC50_val)
        IC50_MSE_test = IC50_MSE
        AUC_abs_test = AUC_abs
        Emax_abs_test = Emax_abs
    break    #This break avoids running for all cross-validations
    print("Yval shape", Yval.shape)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Save model and predictions"

##path_cancer = '/rds/general/user/jgiraldo/home/PaperPrecisionOncology_Rebuttal/Experiments_Results/'
##path_cancer =path_cancer + './N_drugs_' + str(Num_drugs) + '/Cancer_' + str(config.sel_cancer) + '/Train' + str(N_CellLines_perc) + '/seed' + str(rand_state_N) + '/'

# path_cancer = 'Models_5Cancers/N_drugs_' + str(Num_drugs) + '/Cancer_' + str(config.sel_cancer) + '/Train' + str(N_CellLines_perc) + '/seed' + str(rand_state_N) + '/'
# if not os.path.exists(path_cancer):
#     os.makedirs(path_cancer)
# f = open(path_cancer + "Metrics.txt", "a+")
# f.write("bash" + str(
#     config.bash) + f" Med_MSE={np.mean(Med_MSE_AllFolds):0.5f}({np.std(Med_MSE_AllFolds):0.5f}) Mean_MSE={np.mean(Mean_MSE_AllFolds):0.5f}({np.std(Mean_MSE_AllFolds):0.5f}) NegLPD={np.mean(NegMLL_AllFolds):0.5f}({np.std(NegMLL_AllFolds):0.5f}) IC50_MSE={np.mean(IC50_MSE_AllFolds):0.5f}({np.std(IC50_MSE_AllFolds):0.5f}) AUC_abs={np.mean(AUC_abs_AllFolds):0.5f}({np.std(AUC_abs_AllFolds):0.5f}) Emax_abs={np.mean(Emax_abs_AllFolds):0.5f}({np.std(Emax_abs_AllFolds):0.5f})\n")
# f.close()
#
# f = open(path_cancer + "Average_Metrics_IC50_AUC_Emax.txt", "a+")
# Aver_IC50_AUC_Emax_MSECurve = np.array(
#     [np.mean(IC50_MSE_AllFolds), np.mean(AUC_abs_AllFolds), np.mean(Emax_abs_AllFolds), np.mean(Mean_MSE_AllFolds)])
# f.write("bash" + str(config.bash) + f", {np.mean(Aver_IC50_AUC_Emax_MSECurve):0.5f} \n")
# f.close()
#
# f = open(path_cancer + "Test_Metrics_IC50_AUC_Emax.txt", "a+")
# f.write("bash" + str(
#     config.bash) + f" IC50_MSE={IC50_MSE_test:0.5f} AUC_abs={AUC_abs_test:0.5f} Emax_abs={Emax_abs_test:0.5f}\n")
# f.close()
#
# "The last model should have been trained over all dataset without splitting"
#
# ##np.save(path_cancer+'m_'+str(config.bash)+'.npy', model.param_array)  #This is to save the model if desired!
#
# df_4Cancers_test.to_csv(path_cancer+'MOGP_Predict_C'+str(config.sel_cancer)+'_Train'+str(N_CellLines_perc)+'_m_'+str(config.bash)+'.csv')
