import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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

sel_cancers = [0]
N5th_cancer = 9
Num_drugs = 3
N_CellLines = [12,24,48,96,144]
Seeds_for_NCells = [1,2,3,4,5,6]

AUC_per_cell = [[i] for i in range(N_CellLines.__len__())]
Emax_per_cell = [[i] for i in range(N_CellLines.__len__())]
IC50_per_cell = [[i] for i in range(N_CellLines.__len__())]

AUCR2_per_cell = [[i] for i in range(N_CellLines.__len__())]
EmaxR2_per_cell = [[i] for i in range(N_CellLines.__len__())]
IC50R2_per_cell = [[i] for i in range(N_CellLines.__len__())]

AUC_CV_per_cell = [[i] for i in range(N_CellLines.__len__())]
Emax_CV_per_cell = [[i] for i in range(N_CellLines.__len__())]
IC50_CV_per_cell = [[i] for i in range(N_CellLines.__len__())]

print("Becareful we you try using more sel_cancers, you should reset AUC_per_cell each iterations of Sel_cancer")
for Sel_cancer in sel_cancers:
    for cell_k,N_cells in enumerate(N_CellLines):
        for Seed_N in Seeds_for_NCells:
            #Sel_cancer = 3
            _FOLDER_Cancer = '/home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_5Cancers/bash_Cancer'+str(Sel_cancer)+'_SamplingFromSimilarity_GPyjobs_N_Drugs_5Cancers_GPy_ExactMOGP_ProdKern_N5thCancer_'+str(N5th_cancer)+'/N_drugs_'+str(Num_drugs)+'/N5thCancer_'+str(N5th_cancer)+'/Cancer_'+str(Sel_cancer)+'/'
            path_to_aver = _FOLDER_Cancer+'N'+str(N_cells)+'/seed'+str(Seed_N)+'/Average_Metrics_IC50_AUC_Emax.txt'
            #path_to_test = _FOLDER_Cancer+'N'+str(N_cells)+'/seed'+str(Seed_N)+'/Test_Metrics_IC50_AUC_Emax.txt'
            path_to_CV = _FOLDER_Cancer + 'N' + str(N_cells) + '/seed' + str(Seed_N) + '/Metrics.txt'
            try:
                df_Aver_Metrics = pd.read_csv(path_to_aver,header=None)
                print("Reading Path: ",path_to_aver)
                #df_Test_Metrics = pd.read_csv(path_to_test,header=None,sep=' ')
                myloc_winner = df_Aver_Metrics[1]==df_Aver_Metrics[1].min()
                winner_bash = df_Aver_Metrics[0][myloc_winner].values[0]
                #df_Test_Metrics[0]==winner_bash
                #indx_test_by_winner = df_Test_Metrics[0]==winner_bash
                #df_test_winner = df_Test_Metrics[indx_test_by_winner]
                #_FOLDER_Cancer_csv_test = '/data/ac1jjgg/Data_Marina/GPy_results/Codes_for_GDSC2_5Cancers/N_drugs_'+str(Num_drugs)+'/N5thCancer_'+str(N5th_cancer)+'/Cancer_'+str(Sel_cancer)+'/'+'N'+str(N_cells)+'/seed'+str(Seed_N)+'/'
                _FOLDER_Cancer_csv_test = '/data/ac1jjgg/Data_Marina/GPy_results/Codes_for_GDSC2_5Cancers/SamplingFromSimilarity/N_drugs_'+str(Num_drugs)+'/N5thCancer_' + str(N5th_cancer) + '/Cancer_' + str(Sel_cancer)+'/'+'N'+str(N_cells) + '/seed' + str(Seed_N) + '/'
                #print('Results_Test_IC50_AUC_Emax_'+'m_'+str(winner_bash)+'.csv')
                Nth_bash = int(winner_bash.split('h')[1])
                print("Nth_bash:",Nth_bash)
                "Read the csv file of Test results"
                path_to_test = _FOLDER_Cancer_csv_test + 'Results_Test_IC50_AUC_Emax_' + 'm_' + str(Nth_bash) + '.csv'
                print("Reading Path Test: ", path_to_test)
                df_Test_Metrics = pd.read_csv(path_to_test)
                print(df_Test_Metrics)
                AUC_aux = np.mean(np.abs(df_Test_Metrics['AUC_MOGP'].values - df_Test_Metrics['AUC_s4'].values))
                print("AUC_aux:",AUC_aux)
                AUC_per_cell[cell_k].append(AUC_aux)
                Emax_aux = np.mean(np.abs(df_Test_Metrics['Emax_MOGP'].values - df_Test_Metrics['Emax_s4'].values))
                print("Emax_aux:", Emax_aux)
                Emax_per_cell[cell_k].append(Emax_aux)
                IC50_aux = np.mean(np.abs(df_Test_Metrics['IC50_MOGP'].values - df_Test_Metrics['IC50_s4'].values))
                print("IC50_aux:", IC50_aux)
                IC50_per_cell[cell_k].append(IC50_aux)

                df_CrossVal = pd.read_csv(path_to_CV, header=None, sep=' ')
                IC50_CV = np.array([float(df_CrossVal[4].values[i].split("=")[1].split("(")[0]) for i in range(df_CrossVal.shape[0])])
                AUC_CV = np.array([float(df_CrossVal[5].values[i].split("=")[1].split("(")[0]) for i in range(df_CrossVal.shape[0])])
                Emax_CV = np.array([float(df_CrossVal[7].values[i].split("=")[1].split("(")[0]) for i in range(df_CrossVal.shape[0])])
                bash_CV = np.array([int(df_CrossVal[0].values[i].split("h")[1]) for i in range(df_CrossVal.shape[0])])
                indx_CV = np.where(bash_CV == Nth_bash)[0][0]
                AUC_CV_per_cell[cell_k].append(AUC_CV[indx_CV])
                Emax_CV_per_cell[cell_k].append(Emax_CV[indx_CV])
                IC50_CV_per_cell[cell_k].append(IC50_CV[indx_CV])
                print("AUC_CV:", AUC_CV[indx_CV])
                print("Emax_CV:", Emax_CV[indx_CV])
                print("IC50_CV:", IC50_CV[indx_CV])

                "R^2 Metrics"
                AUCR2_aux = r2_score(df_Test_Metrics['AUC_s4'].values,df_Test_Metrics['AUC_MOGP'].values)
                print("AUC_aux_r2:", AUCR2_aux)
                AUCR2_per_cell[cell_k].append(AUCR2_aux)
                EmaxR2_aux = r2_score(df_Test_Metrics['Emax_s4'].values,df_Test_Metrics['Emax_MOGP'].values)
                print("Emax_aux_r2:", EmaxR2_aux)
                EmaxR2_per_cell[cell_k].append(EmaxR2_aux)
                IC50R2_aux = r2_score(df_Test_Metrics['IC50_s4'].values,df_Test_Metrics['IC50_MOGP'].values)
                print("IC50_aux_r2:", IC50R2_aux)
                IC50R2_per_cell[cell_k].append(IC50R2_aux)

            except:
                print('Non existent path: ',path_to_test)

with open('Test_Metrics_To_Plot_N_drugs_'+str(Num_drugs)+'_SamplingFromSimilarity_Cancer_'+str(sel_cancers[0])+'.pkl', 'wb') as f:
    pickle.dump([AUC_per_cell,Emax_per_cell,IC50_per_cell,AUCR2_per_cell,EmaxR2_per_cell,IC50R2_per_cell,AUC_CV_per_cell,Emax_CV_per_cell,IC50_CV_per_cell], f)

"To load use:"
#np.load('Test_Metrics_To_Plot.pkl',allow_pickle=True)