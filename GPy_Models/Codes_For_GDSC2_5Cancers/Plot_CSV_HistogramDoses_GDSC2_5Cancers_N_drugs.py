import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from scipy.interpolate import interp1d
from scipy.interpolate import pchip_interpolate
#plt.close('all')

_FOLDER = '/home/juanjo/Work_Postdoc/my_codes_postdoc/FilesCSV_Predict_5Cancers_IncreasingCellLines/'
_FOLDER_Train = '/home/juanjo/Work_Postdoc/GDSC2_Datasets_5Cancers_Increasing_TrainingData/'
"/home/juanjo/Work_Postdoc/GDSC2_Datasets_5Cancers_Increasing_TrainingData/breast_cancer/N5th_CancerInTrain_9/NTrain_48_plus_9/seed1"
cancer_names = {0:'breast_cancer',1:'COAD_cancer',2:'LUAD_cancer',3:'melanoma_cancer',4:'SCLC_cancer'}

rc('font', weight='bold')
plt.close('all')
#sel_cancer = 0
cancers = [1]
"NOTE: if using 45 below you have to organise code to access 45 cases"
N5th_cancer = 9    #CAREFUL HERE, TODO for 45
All_Nseed = [1]
All_N_cells = np.array([144]) * 4 #np.array([0,12,24,48,96,144]) * 4
#Ntotal_Cells = int(N_cells)*4 + int(N5th_cancer)

Nth_dose = 7

fig = []
axs = []
for i in range(5):
    fig_i, axs_i = plt.subplots(3, 1,figsize = (15,10))
    fig.append(fig_i); axs.append(axs_i)

for sel_cancer in cancers:
    cancer = cancer_names[sel_cancer]
    AE_per_dose_Ncells = []
    MAE_per_dose_Ncells = []
    AE_AUC_Res_Ncells = [] ; AE_Emax_Res_Ncells = [] ; AE_IC50_Res_Ncells = []
    MAE_AUC_Res_Ncells = [] ; MAE_Emax_Res_Ncells = [] ; MAE_IC50_Res_Ncells = []
    AE_AUC_NoRes_Ncells = [] ; AE_Emax_NoRes_Ncells = [] ; AE_IC50_NoRes_Ncells = []
    MAE_AUC_NoRes_Ncells = [] ; MAE_Emax_NoRes_Ncells = [] ; MAE_IC50_NoRes_Ncells = []
    for N_cells in All_N_cells:
        for Nseed in All_Nseed:
            path_to_read = _FOLDER + cancer+'/N5th_CancerInTrain_'+str(N5th_cancer)+'/NTrain_'+str(N_cells)+'_plus_'+str(N5th_cancer)+'/MOGP_Predict_C'+str(sel_cancer)+'_Train_'+str(N_cells)+'_plus_'+str(N5th_cancer)+'_seed'+str(Nseed)+'.csv'
            if N_cells>0:
                path_to_read_Train = _FOLDER_Train + cancer + '/N5th_CancerInTrain_' + str(N5th_cancer) + '/NTrain_'+str(N_cells)+'_plus_'+ str(N5th_cancer)+'/seed'+str(Nseed)+'/Train_'+str(N_cells) + '_plus_' + str(N5th_cancer) + '_seed' + str(Nseed) + '.csv'
            else:
                path_to_read_Train = _FOLDER + cancer + '/N5th_CancerInTrain_' + str(N5th_cancer) + '/Train_9_c'+str(sel_cancer+1)+'.csv'
            path_to_read_9Canc = _FOLDER + cancer + '/N5th_CancerInTrain_' + str(N5th_cancer) + '/Train_9_c'+str(sel_cancer+1)+'.csv'
            df_pred = pd.read_csv(path_to_read)
            #Y_dose_Dth = df_pred[df_pred.columns[15:22]].values   #these are all the 7 dose values
            #df_Ypred = df_pred[df_pred.columns[26:33]].values     #these are all the 7 dose prediction values
            Y_dose_Dth = df_pred['norm_cells_'+str(Nth_dose)].values
            #Ypred_dose_Dth = np.clip(df_pred['norm_cell_7_MOGP'].values, 0, 1)
            Ypred_dose_Dth = df_pred['norm_cell_'+str(Nth_dose)+'_MOGP'].values
            df_Y9 = pd.read_csv(path_to_read_9Canc)
            Y_9InTrain_dose_Dth = df_Y9['norm_cells_'+str(Nth_dose)].values
            Yall_dose_Dth = np.concatenate((Y_dose_Dth,Y_9InTrain_dose_Dth))
            df_Train = pd.read_csv(path_to_read_Train)
            YTrain_dose_Dth = df_Train['norm_cells_'+str(Nth_dose)].values
            #plt.figure(1,figsize=(12,10))
            axs[sel_cancer][0].hist(Yall_dose_Dth, bins=20)
            axs[sel_cancer][0].bar(Yall_dose_Dth[-9:],2*np.ones(9),width=0.01,color='r')
            axs[sel_cancer][0].bar(Yall_dose_Dth[-9:], 2 * np.ones(9), width=0.001, color='black')
            axs[sel_cancer][0].set_xlim([0,1.1])
            myfont = 14
            axs[sel_cancer][0].set_title(f"Histogram of Dose {Nth_dose} (True values)",fontsize=myfont)
            axs[sel_cancer][0].set_ylabel(cancer,fontsize=myfont)
            axs[sel_cancer][0].legend(["Histogram","9 Values in Train"])
            #plt.figure(2,figsize=(12,10))
            axs[sel_cancer][1].hist(Ypred_dose_Dth, bins=20)
            axs[sel_cancer][1].set_xlim([0, 1.1])
            axs[sel_cancer][1].set_title(f"Histogram of Dose {Nth_dose} (MOGP Prediction)",fontsize=myfont)
            axs[sel_cancer][1].set_ylabel(cancer, fontsize=myfont)
            if N_cells>0: use_bin = 20;
            else: use_bin=9;
            axs[sel_cancer][2].hist(YTrain_dose_Dth, bins=use_bin)
            axs[sel_cancer][2].set_xlim([0, 1.1])
    #plt.figure(5)
    Num_cells = All_N_cells.__len__()

from scipy import stats
from fitter import Fitter

plt.figure(7)
dist_fitter = Fitter(Yall_dose_Dth,
                   distributions = ["cauchy",
                                    "rayleigh",
                                    "gamma",
                                    "beta",
                                    "lognorm",
                                    "norm",
                                    "skewnorm"])
dist_fitter.fit()
dist_fitter.summary()