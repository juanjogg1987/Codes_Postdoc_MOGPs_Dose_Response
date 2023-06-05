import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from Utils_to_Extract_BestModel import model_pred_test_csv
import os

def r2_score_pearson(y_true,y_pred_MOGP):
    r2_metric = np.corrcoef(y_true,y_pred_MOGP)[0,1]**2
    return r2_metric

def lm_rsqr(y_true,y_pred_MOGP):
    y_true = y_true.reshape(-1,1)
    y_pred_MOGP = y_pred_MOGP.reshape(-1,1)
    reg = LinearRegression().fit(y_pred_MOGP, y_true)   #Here fit(X,y) i.e., fit(ind_var, dep_var)
    y_pred = reg.predict(y_pred_MOGP)
    return r2_score(y_true,y_pred)

def my_r2_score(y_true,y_pred):
    y_bar = np.mean(y_true)
    SS_res = np.sum((y_true-y_pred)**2)
    SS_tot = np.sum((y_true-y_bar)**2)
    r2_metric = 1.0 - SS_res/SS_tot
    return r2_metric

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

sel_cancers = [3]
Num_drugs = 3
N_CellLines = [20,40,60,80,100]
Seeds_for_NCells = [1,2,3,4,5,6]

for Sel_cancer in sel_cancers:
    for cell_k,N_cells in enumerate(N_CellLines):
        for Seed_N in Seeds_for_NCells:
            _FOLDER_Cancer = '/home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_5Cancers/bash_Cancer'+str(Sel_cancer)+'_Train1Cancer_GPyjobs_N_Drugs_GPy_ExactMOGP_ProdKern/N_drugs_'+str(Num_drugs)+'/Cancer_'+str(Sel_cancer)+'/'
            path_to_aver = _FOLDER_Cancer+'Train'+str(N_cells)+'/seed'+str(Seed_N)+'/Average_Metrics_IC50_AUC_Emax.txt'
            try:
                df_Aver_Metrics = pd.read_csv(path_to_aver,header=None)
                print("Reading Path: ",path_to_aver)
                myloc_winner = df_Aver_Metrics[1]==df_Aver_Metrics[1].min()
                winner_bash = df_Aver_Metrics[0][myloc_winner].values[0]
                _FOLDER_Cancer_csv_test = '/data/ac1jjgg/Data_Marina/GPy_results/Codes_for_GDSC2_5Cancers/Train1Cancer/N_drugs_'+str(Num_drugs)+ '/Cancer_' + str(Sel_cancer)+'/'+'Train'+str(N_cells) + '/seed' + str(Seed_N) + '/'
                Nth_bash = int(winner_bash.split('h')[1])
                print("Nth_bash:",Nth_bash)

                "Read the csv file of Test results"
                #path_to_test = _FOLDER_Cancer_csv_test + 'Results_Test_IC50_AUC_Emax_' + 'm_' + str(Nth_bash) + '.csv'
                path_to_model = _FOLDER_Cancer_csv_test + 'm_' + str(Nth_bash) + '.npy'
                df_test_pred = pd.read_csv(_FOLDER_Cancer_csv_test + 'MOGP_Predict_C'+str(Sel_cancer)+'_Train'+str(N_cells)+'_m_'+str(Nth_bash)+'.csv')

                Cancer_Names = ['breast','COAD','LUAD','melanoma','SCLC']

                "Save the Predictions"
                final_path = '/data/ac1jjgg/Data_Marina/GPy_results/Codes_for_GDSC2_5Cancers/Train1Cancer/N_drugs_3/FilesCSV_Predict_Train1Cancer_IncreasingCellLines/'+str(Cancer_Names[int(Sel_cancer)]) + '_cancer/Train'+str(N_cells)+'/'
                if not os.path.exists(final_path):
                   os.makedirs(final_path)

                df_test_pred.to_csv(final_path+ 'MOGP_Predict_C'+str(Sel_cancer)+'_Train'+str(N_cells)+'_seed'+str(int(Seed_N))+'.csv')

            except:
                print('Probably Non existent path: ',path_to_aver)
                print('Probably Non existent model path: ', path_to_model)
