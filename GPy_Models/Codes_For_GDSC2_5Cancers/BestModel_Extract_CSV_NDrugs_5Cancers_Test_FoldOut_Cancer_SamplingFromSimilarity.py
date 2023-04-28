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

sel_cancers = [0,1,2,3,4]
Ns_for_5thCancer = [9,45]
Num_drugs = 3
N_CellLines = [0,12,24,48,96,144]
Seeds_for_NCells = [1,2,3,4,5,6]

for Sel_cancer in sel_cancers:
    for N5th_cancer in Ns_for_5thCancer:
        for cell_k,N_cells in enumerate(N_CellLines):
            for Seed_N in Seeds_for_NCells:
                _FOLDER_Cancer = '/home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_5Cancers/bash_Cancer'+str(Sel_cancer)+'_SamplingFromSimilarity_GPyjobs_N_Drugs_5Cancers_GPy_ExactMOGP_ProdKern_N5thCancer_'+str(N5th_cancer)+'/N_drugs_'+str(Num_drugs)+'/N5thCancer_'+str(N5th_cancer)+'/Cancer_'+str(Sel_cancer)+'/'
                path_to_aver = _FOLDER_Cancer+'N'+str(N_cells)+'/seed'+str(Seed_N)+'/Average_Metrics_IC50_AUC_Emax.txt'
                #path_to_test = _FOLDER_Cancer+'N'+str(N_cells)+'/seed'+str(Seed_N)+'/Test_Metrics_IC50_AUC_Emax.txt'
                #path_to_CV = _FOLDER_Cancer + 'N' + str(N_cells) + '/seed' + str(Seed_N) + '/Metrics.txt'
                try:
                    df_Aver_Metrics = pd.read_csv(path_to_aver,header=None)
                    print("Reading Path: ",path_to_aver)
                    myloc_winner = df_Aver_Metrics[1]==df_Aver_Metrics[1].min()
                    winner_bash = df_Aver_Metrics[0][myloc_winner].values[0]
                    _FOLDER_Cancer_csv_test = '/data/ac1jjgg/Data_Marina/GPy_results/Codes_for_GDSC2_5Cancers/SamplingFromSimilarity/N_drugs_'+str(Num_drugs)+'/N5thCancer_' + str(N5th_cancer) + '/Cancer_' + str(Sel_cancer)+'/'+'N'+str(N_cells) + '/seed' + str(Seed_N) + '/'
                    Nth_bash = int(winner_bash.split('h')[1])
                    print("Nth_bash:",Nth_bash)

                    "Read the csv file of Test results"
                    #path_to_test = _FOLDER_Cancer_csv_test + 'Results_Test_IC50_AUC_Emax_' + 'm_' + str(Nth_bash) + '.csv'
                    path_to_model = _FOLDER_Cancer_csv_test + 'm_' + str(Nth_bash) + '.npy'
                    #print("Reading Path Test: ", path_to_test)

                    path_to_bash = '/home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_5Cancers/bash_Cancer' + str(Sel_cancer) + '_SamplingFromSimilarity_GPyjobs_N_Drugs_5Cancers_GPy_ExactMOGP_ProdKern_N5thCancer_' + str(N5th_cancer)+ '/bash'+str(Nth_bash)+'.sh'
                    df_test_pred = model_pred_test_csv(path_to_read_bash=path_to_bash,path_to_model=path_to_model)

                    Cancer_Names = ['breast','COAD','LUAD','melanoma','SCLC']

                    "Save the Predictions"
                    #final_path ='N5thCancer_'+str(config.N_5thCancer_ToBe_Included)+'/'+str(config.sel_cancer)+'/N'+str(config.N_CellLines)+'/seed'+str(rand_state_N)+'/'
                    Ntotal_Cells = int(N_cells)*4 + int(N5th_cancer)
                    final_path = '/data/ac1jjgg/Data_Marina/GPy_results/Codes_for_GDSC2_5Cancers/SamplingFromSimilarity/N_drugs_3/FilesCSV_Predict_5Cancers_IncreasingCellLines/'+str(Cancer_Names[int(Sel_cancer)])+'_cancer/N5th_CancerInTrain_'+str(N5th_cancer)+'/NTrain_'+str(int(N_cells)*4)+'_plus_'+str(int(N5th_cancer))+'/'
                    if not os.path.exists(final_path):
                       os.makedirs(final_path)

                    df_test_pred.to_csv(final_path+'MOGP_Predict_C'+str(Sel_cancer)+'_Train_'+str(int(N_cells)*4)+'_plus_'+str(int(N5th_cancer))+'_seed'+str(int(Seed_N))+'.csv')

                except:
                    print('Probably Non existent path: ',path_to_aver)
                    print('Probably Non existent bash path: ', path_to_bash)
                    print('Probably Non existent model path: ', path_to_model)
