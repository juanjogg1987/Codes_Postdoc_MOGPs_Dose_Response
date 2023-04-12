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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

_FOLDER = "/home/juanjo/Work_Postdoc/my_codes_postdoc/Dataset_5Cancers/GDSC2_GDSC1_common3drugs-cell-line_Top5cancers/"

indx_cancer = np.array([0,1,2,3,4])
indx_cancer_test = np.array([int(config.sel_cancer)])
indx_cancer_train = np.delete(indx_cancer,indx_cancer_test)

Diff_AUC_5Cancers = []
Diff_Emax_5Cancers = []
Diff_IC50_5Cancers = []

which_cancer = 0

# for which_cancer in range(0):
AUC_GDSC1_All = []; AUC_GDSC2_All = []
Emax_GDSC1_All = []; Emax_GDSC2_All = []
IC50_GDSC1_All = []; IC50_GDSC2_All = []
for fold in range(2,5,2):
#for fold in range(2,3):
    #fold = 4  #fold = 2 one is used for 9 doses, fold = 4 one is used for 5 doses (GDSC1)
    Cancer_Names = ['breast','COAD','LUAD','melanoma','SCLC']
    Sel_Cancer = Cancer_Names[which_cancer]
    df_Cancer = pd.read_csv(_FOLDER + 'GDSC2_GDSC1_'+str(fold)+'-fold_'+Sel_Cancer+'_3drugs_uM.csv')

    def Extract_Dose_Response(df_Cancer,sel_dataset = "GDSC1", fold = 2):
        if sel_dataset=="GDSC2":
            "Below we select 7 concentration since GDSC2 has such a number"
            norm_cell = "_x"
            Ndoses_lim = 7+1
        elif sel_dataset=="GDSC1":
            "Below we select 9 or 5 concentration since GDSC1 has such a number"
            norm_cell = "_y"
            if fold == 2:
                Ndoses_lim = 9 + 1
            elif fold == 4:
                Ndoses_lim = 5 + 1

        y_drug_GDSC = np.clip(df_Cancer["norm_cells_" + str(1)+norm_cell].values[:, None], 1.0e-9, np.inf)
        x_dose_uM = df_Cancer["fd_uM_" + str(1) + norm_cell].values[:, None]
        print(y_drug_GDSC.shape)
        for i in range(2, Ndoses_lim):  #Here until 8 for GDSC2
            y_drug_GDSC = np.concatenate((y_drug_GDSC, np.clip(df_Cancer["norm_cells_" + str(i)+norm_cell].values[:, None], 1.0e-9, np.inf)), 1)
            x_dose_uM = np.concatenate((x_dose_uM, df_Cancer["fd_uM_" + str(i) + norm_cell].values[:, None]), 1)
        print("Y size: ", y_drug_GDSC.shape)
        print("X size: ", x_dose_uM.shape)

        params_4_sigmoid = df_Cancer["param_" + str(1)+norm_cell].values[:, None]
        for i in range(2, 5):
            params_4_sigmoid = np.concatenate((params_4_sigmoid, df_Cancer["param_" + str(i)+norm_cell].values[:, None]),1)

        return  y_drug_GDSC,x_dose_uM, params_4_sigmoid

    y_drug_GDSC1, x_dose_GDSC1_uM, params_4_sig_GDSC1 = Extract_Dose_Response(df_Cancer,sel_dataset="GDSC1",fold=fold)
    y_drug_GDSC2, x_dose_GDSC2_uM, params_4_sig_GDSC2 = Extract_Dose_Response(df_Cancer,sel_dataset="GDSC2")   #GDSC2 does not need fold always 7 doses

    Ndoses_GDSC1 = y_drug_GDSC1.shape[1]

    import matplotlib.pyplot as plt

    #plt.close('all')

    posy = 5
    plt.figure(1)
    x_dose_GDSC2_uM_log10 = np.log10(x_dose_GDSC2_uM)
    x_dose_GDSC1_uM_log10 = np.log10(x_dose_GDSC1_uM)
    plt.plot(x_dose_GDSC2_uM_log10[posy],y_drug_GDSC2[posy],'ro')
    plt.plot(x_dose_GDSC1_uM_log10[posy],y_drug_GDSC1[posy],'bo')
    plt.ylim([-0.1,1.5])

    plt.figure(2)
    x_lin_GDSC2 = np.linspace(0.142857143,1,7)
    x_lin_GDSC1 = np.linspace(0.111111,1,Ndoses_GDSC1)
    plt.plot(x_lin_GDSC2,y_drug_GDSC2[posy],'ro')
    plt.plot(x_lin_GDSC1,y_drug_GDSC1[posy],'bo')
    plt.ylim([-0.1,1.5])

    my_prop_log = (x_dose_GDSC2_uM_log10[:,0]-x_dose_GDSC2_uM_log10[:,-1])/(x_dose_GDSC1_uM_log10[:,0]-x_dose_GDSC1_uM_log10[:,-1])
    "I realised that all my_prop_log for all data become exactly the same, so I just use one unique scalar instead of vector"
    my_prop_log = my_prop_log[0]
    my_prop_orig = (x_lin_GDSC1[0]-x_lin_GDSC1[-1])/(x_lin_GDSC2[0]-x_lin_GDSC2[-1])

    "Here we scale to have GDSC2 in same range (of normalized doses, i.e., [0.111111,1]) of GDSC1"
    x_lin_GDSC2_scaled = x_lin_GDSC2*my_prop_orig-(my_prop_orig-1.0)
    "Here we scale to guarantee GDSC2 gets a bigger range than GDSC1."
    "This is due to having for GDSC2 a minimum dose of uM smaller than GDSC1"
    "The scaling in meant to be in the normalized doses space."
    x_lin_GDSC2_scaled = x_lin_GDSC2_scaled*my_prop_log-(my_prop_log-1.0)
    #x_lin_GDSC2_scaled = np.repeat(x_lin_GDSC2_scaled[None,:],my_prop_log.shape[0],axis=0)*my_prop_log-(my_prop_log-1.0)

    plt.figure(3)
    plt.plot(x_lin_GDSC2_scaled,y_drug_GDSC2[posy],'ro')
    plt.plot(x_lin_GDSC1,y_drug_GDSC1[posy],'bo')
    plt.ylim([-0.1,1.5])

    from sklearn import metrics
    def Get_IC50_AUC_Emax(params_4_sig_train,x_lin,x_lin_scaled=None,cut_GDSC2=False,my_prop_log=None):
        x_lin_tile = np.tile(x_lin, (params_4_sig_train.shape[0], 1))
        if cut_GDSC2:
            assert x_lin.shape == x_lin_scaled.shape
            indx_cut = int(x_lin.shape[0]-x_lin.shape[0]/my_prop_log)
            x_lin = x_lin_scaled[indx_cut:].copy()
        # (x_lin,params_4_sig_train.shape[0],1).shape
        Ydose_res = []
        AUC = []
        IC50 = []
        Ydose50 = []
        Emax = []
        for i in range(params_4_sig_train.shape[0]):
            if cut_GDSC2:
                "Here we compute the curve in the original space, but then the metrics are taken with x_scaled"
                Ydose_res.append(sigmoid_4_param(x_lin_tile[i, :], *params_4_sig_train[i, :])[indx_cut:])
                "Here we use x_lin_scaled to account for the new curve scaled in a new range"
                AUC.append(metrics.auc(x_lin_scaled[indx_cut:], Ydose_res[i]))
            else:
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

        #return Ydose50,Ydose_res,IC50,AUC,Emax, x_lin
        return np.array(Ydose50),np.array(Ydose_res),np.array(IC50)[:,None],np.array(AUC)[:,None],np.array(Emax)[:,None], x_lin

    "Here we define x_interpol in the range of GDSC1 with a big number of points to plot sigmoid_4"
    x_for_sig4_GDSC1 = np.linspace(0.111111,1,1000)
    x_for_sig4_GDSC2 = np.linspace(0.142857143,1,1248)
    x_for_sig4_GDSC2_scaled = np.linspace(x_lin_GDSC2_scaled[0],1,1248)
    Ydose50_GDSC1,Ydose_res_GDSC1,IC50_GDSC1,AUC_GDSC1,Emax_GDSC1,_ = Get_IC50_AUC_Emax(params_4_sig_GDSC1,x_for_sig4_GDSC1)
    Ydose50_GDSC2,Ydose_res_GDSC2,IC50_GDSC2,AUC_GDSC2,Emax_GDSC2, x_for_sig4_GDSC2_cut = Get_IC50_AUC_Emax(params_4_sig_GDSC2,x_for_sig4_GDSC2,x_for_sig4_GDSC2_scaled,cut_GDSC2=True,my_prop_log=my_prop_log)
    Ydose50_GDSC2_ori,Ydose_res_GDSC2_ori,IC50_GDSC2_ori,AUC_GDSC2_ori,Emax_GDSC2_ori, _ = Get_IC50_AUC_Emax(params_4_sig_GDSC2,x_for_sig4_GDSC2)

    def my_plot(posy,fig_num,Ydose50,Ydose_res,IC50,AUC,Emax,x_lin,x_real_dose,y_train_drug):
        plt.figure(fig_num)
        plt.plot(x_lin, Ydose_res[posy])
        plt.plot(x_real_dose, y_train_drug[posy, :], '.')
        plt.plot(IC50[posy], Ydose50[posy], 'rx')
        plt.plot(x_lin, np.ones_like(x_lin)*Emax[posy], 'r') #Plot a horizontal line as Emax
        plt.title(f"AUC = {AUC[posy]}")
        plt.ylim([-0.1,1.2])
        print("AUC:",AUC[posy])
        print("Emax:",Emax[posy])
        print("IC50:", IC50[posy])

    #posy = 22
    my_plot(posy,0,Ydose50_GDSC1,Ydose_res_GDSC1,IC50_GDSC1,AUC_GDSC1,Emax_GDSC1,x_for_sig4_GDSC1,x_lin_GDSC1,y_drug_GDSC1)
    my_plot(posy,0,Ydose50_GDSC2,Ydose_res_GDSC2,IC50_GDSC2,AUC_GDSC2,Emax_GDSC2,x_for_sig4_GDSC2_cut,x_lin_GDSC2_scaled,y_drug_GDSC2)
    my_plot(posy,10,Ydose50_GDSC2_ori,Ydose_res_GDSC2_ori,IC50_GDSC2_ori,AUC_GDSC2_ori,Emax_GDSC2_ori,x_for_sig4_GDSC2,x_lin_GDSC2,y_drug_GDSC2)

    AUC_GDSC1_All.append(AUC_GDSC1.flatten())
    AUC_GDSC2_All.append(AUC_GDSC2.flatten())
    Emax_GDSC1_All.append(Emax_GDSC1.flatten())
    Emax_GDSC2_All.append(Emax_GDSC2.flatten())
    IC50_GDSC1_All.append(IC50_GDSC1.flatten())
    IC50_GDSC2_All.append(IC50_GDSC2.flatten())

AUC_GDSC1_cancer = np.concatenate((AUC_GDSC1_All[0],AUC_GDSC1_All[1]),axis=0)
AUC_GDSC2_cancer = np.concatenate((AUC_GDSC2_All[0],AUC_GDSC2_All[1]),axis=0)
Emax_GDSC1_cancer = np.concatenate((Emax_GDSC1_All[0],Emax_GDSC1_All[1]),axis=0)
Emax_GDSC2_cancer = np.concatenate((Emax_GDSC2_All[0],Emax_GDSC2_All[1]),axis=0)
IC50_GDSC1_cancer = np.concatenate((IC50_GDSC1_All[0],IC50_GDSC1_All[1]),axis=0)
IC50_GDSC2_cancer = np.concatenate((IC50_GDSC2_All[0],IC50_GDSC2_All[1]),axis=0)

    # diff_AUC_GDSC1_GDSC2 = AUC_GDSC1_cancer-AUC_GDSC2_cancer
    # diff_Emax_GDSC1_GDSC2 = Emax_GDSC1_cancer - Emax_GDSC2_cancer
    # diff_IC50_GDSC1_GDSC2 = IC50_GDSC1_cancer - IC50_GDSC2_cancer
    #
    # Diff_AUC_5Cancers.append(diff_AUC_GDSC1_GDSC2)
    # Diff_Emax_5Cancers.append(diff_Emax_GDSC1_GDSC2)
    # Diff_IC50_5Cancers.append(diff_IC50_GDSC1_GDSC2)
    #
    # print(f"{Sel_Cancer} MAE AUC: {np.mean(np.abs(diff_AUC_GDSC1_GDSC2))} ({np.std(np.abs(diff_AUC_GDSC1_GDSC2))})")
    # print(f"{Sel_Cancer} MAE Emax: {np.mean(np.abs(diff_Emax_GDSC1_GDSC2))} ({np.std(np.abs(diff_Emax_GDSC1_GDSC2))})")
    # print(f"{Sel_Cancer} MAE IC50: {np.mean(np.abs(diff_IC50_GDSC1_GDSC2))} ({np.std(np.abs(diff_IC50_GDSC1_GDSC2))})")

# plt.close('all')
# fig, axs = plt.subplots(2,3)
# for i in range(5):
#     plt.figure(20)
#     axs[1,0].errorbar(Cancer_Names[i],np.mean(np.abs(Diff_AUC_5Cancers[i])),np.std(np.abs(Diff_AUC_5Cancers[i])) , linestyle='None', marker='^',capsize=3)
#     #axs[1, 0].set_title("Bench Mark for MAE-AUC")
#     #plt.title("Bench Mark for MAE-AUC")
#     axs[1, 0].set_title("Mean and Std Deviation")
#     axs[1,0].set_ylim([-0.02,0.47])
#     axs[1,0].grid()
#     #plt.figure(21)
#     axs[1,1].errorbar(Cancer_Names[i], np.mean(np.abs(Diff_Emax_5Cancers[i])), np.std(np.abs(Diff_Emax_5Cancers[i])),linestyle='None', marker='^', capsize=3)
#     axs[1, 1].set_title("Mean and Std Deviation")
#     #plt.title("Bench Mark for MAE-Emax")
#     axs[1,1].set_ylim([-0.02, 0.86])
#     axs[1, 1].grid()
#     #plt.grid()
#     #plt.figure(22)
#     axs[1,2].errorbar(Cancer_Names[i], np.mean(np.abs(Diff_IC50_5Cancers[i])), np.std(np.abs(Diff_IC50_5Cancers[i])),linestyle='None', marker='^', capsize=3)
#     axs[1, 2].set_title("Mean and Std Deviation")
#     #plt.title("Bench Mark for MAE-IC50")
#     axs[1,2].set_ylim([-0.02, 1.45])
#     axs[1,2].grid()
#
#
# #plt.figure(23)
# data = [np.abs(Diff_AUC_5Cancers[i]) for i in range(5)]
# axs[0,0].boxplot(data,1,showmeans=True)
# plt.xticks([1, 2, 3, 4,5], Cancer_Names)
# axs[0,0].grid()
# axs[0,0].set_title("Benchmark for AE-AUC (Boxplot)")
#
# #plt.figure(24)
# data = [np.abs(Diff_Emax_5Cancers[i]) for i in range(5)]
# axs[0,1].boxplot(data,1,showmeans=True)
# plt.xticks([1, 2, 3, 4,5], Cancer_Names)
# axs[0,1].grid()
# axs[0,1].set_title("Benchmark for AE-Emax (Boxplot)")
#
# #plt.figure(25)
# data = [np.abs(Diff_IC50_5Cancers[i]) for i in range(5)]
# axs[0,2].boxplot(data,1,showmeans=True)
# plt.xticks([1, 2, 3, 4,5], Cancer_Names)
# axs[0,2].grid()
# axs[0,2].set_title("Benchmark for AE-IC50 (Boxplot)")

#df_4Cancers_test.loc[(df_4Cancers_test['DRUG_ID'].isin(myarray_d)) & df_4Cancers_test['COSMIC_ID'].isin(myarray)]