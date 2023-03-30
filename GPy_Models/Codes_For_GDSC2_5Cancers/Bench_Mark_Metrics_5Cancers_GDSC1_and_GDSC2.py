import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
import os

#_FOLDER = "/home/ac1jjgg/Dataset_5Cancers/GDSC2_EGFR_PI3K_MAPK_Top5cancers/"
_FOLDER = "/home/juanjo/Work_Postdoc/my_codes_postdoc/Dataset_5Cancers/GDSC2_GDSC1_common3drugs-cell-line_Top5cancers/"
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
        self.N_CellLines = 24   #Try to put this values as multiple of Num_drugs
        self.sel_cancer = 0
        self.seed_for_N = 5
        self.N_5thCancer_ToBe_Included = 10 #Try to put this values as multiple of Num_drugs

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

#GDSC2_GDSC1_2-fold_breast_3drugs

indx_cancer = np.array([0,1,2,3,4])
indx_cancer_test = np.array([int(config.sel_cancer)])
indx_cancer_train = np.delete(indx_cancer,indx_cancer_test)

#name_feat_file = "GDSC2_EGFR_PI3K_MAPK_allfeatures.csv"
#name_feat_file_nozero = "GDSC2_EGFR_PI3K_MAPK_features_NonZero_Drugs1036_1061_1373.csv" #"GDSC2_EGFR_PI3K_MAPK_features_NonZero.csv"
#Num_drugs = 3
#N_CellLines_perDrug = int(config.N_CellLines)//Num_drugs
#N_CellLines = int(config.N_CellLines)
#rand_state_N = int(config.seed_for_N)

fold = 4  #fold = 2 one is used for 9 doses, fold = 4 one is used for 5 doses (GDSC1)
Cancer_Names = ['breast','COAD','LUAD','melanoma','SCLC']
Sel_Cancer = Cancer_Names[4]
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
    return  y_drug_GDSC,x_dose_uM

y_drug_GDSC1, x_dose_GDSC1_uM = Extract_Dose_Response(df_Cancer,sel_dataset="GDSC1",fold=fold)
y_drug_GDSC2, x_dose_GDSC2_uM = Extract_Dose_Response(df_Cancer,sel_dataset="GDSC2")   #GDSC2 does not need fold always 7 doses

Ndoses_GDSC1 = y_drug_GDSC1.shape[1]

import matplotlib.pyplot as plt

plt.close('all')

posy = 2
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

my_prop_log = (x_dose_GDSC2_uM_log10[posy][0]-x_dose_GDSC2_uM_log10[posy][-1])/(x_dose_GDSC1_uM_log10[posy][0]-x_dose_GDSC1_uM_log10[posy][-1])
my_prop_orig = (x_lin_GDSC1[0]-x_lin_GDSC1[-1])/(x_lin_GDSC2[0]-x_lin_GDSC2[-1])

x_lin_GDSC2_scaled = x_lin_GDSC2*my_prop_orig-(my_prop_orig-1.0)
x_lin_GDSC2_scaled = x_lin_GDSC2_scaled*my_prop_log-(my_prop_log-1.0)

plt.figure(3)
plt.plot(x_lin_GDSC2_scaled,y_drug_GDSC2[posy],'ro')
plt.plot(x_lin_GDSC1,y_drug_GDSC1[posy],'bo')
plt.ylim([-0.1,1.5])