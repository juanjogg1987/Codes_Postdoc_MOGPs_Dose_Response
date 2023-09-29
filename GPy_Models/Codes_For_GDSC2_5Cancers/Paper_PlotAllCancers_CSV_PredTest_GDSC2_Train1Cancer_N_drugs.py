import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from scipy.interpolate import interp1d
from scipy.interpolate import pchip_interpolate
plt.close('all')

_FOLDER = '/home/juanjo/Work_Postdoc/my_codes_postdoc/FilesCSV_Predict_Train1Cancer_IncreasingCellLines/'
_FOLDER_subha = _FOLDER + 'Subhashini_QSAR/'
_FOLDER_melody = _FOLDER + 'Melody_SRMF/'
cancer_names = {0:'breast_cancer',1:'COAD_cancer',2:'LUAD_cancer',3:'melanoma_cancer',4:'SCLC_cancer'}
cancer_Ntrain = {0:round(110*0.7),1:round(108*0.7),2:round(139*0.7),3:round(142*0.7),4:round(133*0.7)}

rc('font', weight='bold')

#sel_cancer = 0
cancers = [0,1,2,3,4]
#N5th_cancer = 9
All_Nseed = [1,2,3,4,5,6]
All_N_cells = np.array([10,20,35,55,75,95])
#Ntotal_Cells = int(N_cells)*4 + int(N5th_cancer)

fig, axs = plt.subplots(5, 7,figsize = (20,17))
fig_all = []
axs_all = []
for i in range(5):
    if i == 2 or i == 4:
        figaux, axsaux = plt.subplots(3, 1,figsize = (10,20))   #(10,20)
    else:
        figaux, axsaux = plt.subplots(3, 2, figsize=(7, 10))   #(15,20)
    fig_all.append(figaux); axs_all.append(axsaux)

def Compute_Metric(df_pred, Ini_Metrics=None,metric_name='IC50', thresh=0.5,sel_res='Juan', Squared=False):
    Val_Squared = 1
    if Squared: Val_Squared = 2;

    if sel_res == 'Juan':
        Metric_Res_indx = df_pred[metric_name + '_s4'].values < thresh
        Metric_NoRes_indx = df_pred[metric_name + '_s4'].values >= thresh

        AE_Metric_Res_aux = np.abs(df_pred[metric_name + '_MOGP'].values[Metric_Res_indx] - df_pred[metric_name + '_s4'].values[Metric_Res_indx]) ** Val_Squared
        AE_Metric_NoRes_aux = np.abs(df_pred[metric_name + '_MOGP'].values[Metric_NoRes_indx] - df_pred[metric_name + '_s4'].values[Metric_NoRes_indx]) ** Val_Squared
    elif sel_res == 'Subha':
        Metric_Res_indx = df_pred[metric_name].values < thresh
        Metric_NoRes_indx = df_pred[metric_name].values >= thresh

        AE_Metric_Res_aux = np.abs(df_pred['Prediction ('+metric_name+')'].values[Metric_Res_indx] - df_pred[metric_name].values[Metric_Res_indx]) ** Val_Squared
        AE_Metric_NoRes_aux = np.abs(df_pred['Prediction ('+metric_name+')'].values[Metric_NoRes_indx] - df_pred[metric_name].values[Metric_NoRes_indx]) ** Val_Squared
    elif sel_res == 'Melody':
        Metric_Res_indx = df_pred[metric_name].values < thresh
        Metric_NoRes_indx = df_pred[metric_name].values >= thresh

        AE_Metric_Res_aux = np.abs(df_pred['SRMF_'+metric_name].values[Metric_Res_indx] - df_pred[metric_name].values[Metric_Res_indx]) ** Val_Squared
        AE_Metric_NoRes_aux = np.abs(df_pred['SRMF_'+metric_name].values[Metric_NoRes_indx] - df_pred[metric_name].values[Metric_NoRes_indx]) ** Val_Squared
    if (Ini_Metrics is None):
        AE_Metric_Res = AE_Metric_Res_aux.copy()
        AE_Metric_NoRes = AE_Metric_NoRes_aux.copy()
        MAE_Metric_Res = np.array([np.mean(AE_Metric_Res_aux)])
        MAE_Metric_NoRes = np.array([np.mean(AE_Metric_NoRes_aux)])
    else:
        Ini_Metric_Res = Ini_Metrics[0];Ini_Metric_NoRes = Ini_Metrics[1];InitMError_Metr_Res = Ini_Metrics[2];InitMError_Metr_NoRes = Ini_Metrics[3];
        AE_Metric_Res = np.concatenate((Ini_Metric_Res, AE_Metric_Res_aux), 0)
        MAE_Metric_Res = np.concatenate((InitMError_Metr_Res, np.array([np.mean(AE_Metric_Res_aux)])), 0)
        AE_Metric_NoRes = np.concatenate((Ini_Metric_NoRes, AE_Metric_NoRes_aux))
        MAE_Metric_NoRes = np.concatenate((InitMError_Metr_NoRes, np.array([np.mean(AE_Metric_NoRes_aux)])),0)

    return AE_Metric_Res, AE_Metric_NoRes, MAE_Metric_Res, MAE_Metric_NoRes, Metric_Res_indx, Metric_NoRes_indx

for sel_cancer in cancers:
    cancer = cancer_names[sel_cancer]
    AE_per_dose_Ncells = []
    MAE_per_dose_Ncells = []
    AE_AUC_Res_Ncells = [] ; AE_Emax_Res_Ncells = [] ; AE_IC50_Res_Ncells = []
    MAE_AUC_Res_Ncells = [] ; MAE_Emax_Res_Ncells = [] ; MAE_IC50_Res_Ncells = []
    AE_AUC_NoRes_Ncells = [] ; AE_Emax_NoRes_Ncells = [] ; AE_IC50_NoRes_Ncells = []
    MAE_AUC_NoRes_Ncells = [] ; MAE_Emax_NoRes_Ncells = [] ; MAE_IC50_NoRes_Ncells = []
    "Variables for Subhashini and Melody results"
    AE_IC50_Res_Ncells_Subha = [] ; AE_IC50_Res_Ncells_Melody = []
    MAE_IC50_Res_Ncells_Subha = [] ; MAE_IC50_Res_Ncells_Melody = []
    AE_IC50_NoRes_Ncells_Subha = [] ; AE_IC50_NoRes_Ncells_Melody = []
    MAE_IC50_NoRes_Ncells_Subha = [] ; MAE_IC50_NoRes_Ncells_Melody = []
    AE_AUC_Res_Ncells_Melody = [] ; AE_Emax_Res_Ncells_Melody = []
    MAE_AUC_Res_Ncells_Melody = [] ; MAE_Emax_Res_Ncells_Melody = []
    AE_AUC_NoRes_Ncells_Melody = [] ; AE_Emax_NoRes_Ncells_Melody = []
    MAE_AUC_NoRes_Ncells_Melody = [] ; MAE_Emax_NoRes_Ncells_Melody = []
    for N_cells in All_N_cells:
        for Nseed in All_Nseed:
            path_to_read = _FOLDER + cancer+'/Train'+str(N_cells)+'/MOGP_Predict_C'+str(sel_cancer)+'_Train'+str(N_cells)+'_seed'+str(Nseed)+'.csv'
            path_to_read_Subha = _FOLDER_subha + cancer + '/Train' + str(N_cells) + '/s'+str(Nseed)+'_prediction.csv'
            path_to_read_Melody_IC50 = _FOLDER_melody + cancer + '/Train' + str(N_cells) + '/seed'+str(Nseed)+'/IC50_s' + str(Nseed) + '_prediction.csv'
            path_to_read_Melody_AUC = _FOLDER_melody + cancer + '/Train' + str(N_cells) + '/seed' + str(Nseed) + '/AUC_s' + str(Nseed) + '_prediction.csv'
            path_to_read_Melody_Emax = _FOLDER_melody + cancer + '/Train' + str(N_cells) + '/seed' + str(Nseed) + '/Emax_s' + str(Nseed) + '_prediction.csv'

            df_pred = pd.read_csv(path_to_read)
            #df_pred_Subha = pd.read_csv(path_to_read_Subha)
            df_pred_Melody_IC50 = pd.read_csv(path_to_read_Melody_IC50)

            "Replace the values higher than 1.0 to 1.5 to be fair with the method"
            #indxSubha_higher_than_1 = np.where(df_pred_Subha['Prediction (IC50)']>1.0)[0]
            indxMelody_higher_than_1 = np.where(df_pred_Melody_IC50['SRMF_IC50'] > 1.0)[0]
            df_pred_Melody_IC50.iloc[indxMelody_higher_than_1, [df_pred_Melody_IC50.columns.__len__() - 1]] = 1.5
            #df_pred_Subha.iloc[indxSubha_higher_than_1, [df_pred_Subha.columns.__len__() - 1]] = 1.5
            print("Melody df:",df_pred_Melody_IC50)
            #print("Subha df:",df_pred_Subha)

            df_pred_Melody_AUC = pd.read_csv(path_to_read_Melody_AUC)
            df_pred_Melody_Emax = pd.read_csv(path_to_read_Melody_Emax)
            cols_label = ["norm_cells_" + str(i) for i in range(1, 8)]
            cols_pred = ["norm_cell_" + str(i)+"_MOGP" for i in range(1, 8)]
            if Nseed == 1:
                AE_per_dose = np.abs(df_pred[cols_label].values-df_pred[cols_pred].values)
                MAE_per_dose = np.mean(AE_per_dose,0)[None,:]

                AE_AUC_Res,AE_AUC_NoRes,MAE_AUC_Res, MAE_AUC_NoRes,AUC_Res_indx,AUC_NoRes_indx = Compute_Metric(df_pred,Ini_Metrics= None,metric_name = 'AUC',thresh = 0.55,sel_res='Juan',Squared=False)
                AE_AUC_Res_Melody, AE_AUC_NoRes_Melody, MAE_AUC_Res_Melody, MAE_AUC_NoRes_Melody, AUC_Res_indx_Melody, AUC_NoRes_indx_Melody = Compute_Metric(df_pred_Melody_AUC, Ini_Metrics=None, metric_name='AUC', thresh=0.55, sel_res='Melody', Squared=False)

                AE_Emax_Res, AE_Emax_NoRes, MAE_Emax_Res, MAE_Emax_NoRes, Emax_Res_indx, Emax_NoRes_indx = Compute_Metric(df_pred,Ini_Metrics= None, metric_name='Emax', thresh=0.5, sel_res='Juan',Squared=False)
                AE_Emax_Res_Melody, AE_Emax_NoRes_Melody, MAE_Emax_Res_Melody, MAE_Emax_NoRes_Melody, Emax_Res_indx_Melody, Emax_NoRes_indx_Melody = Compute_Metric(df_pred_Melody_Emax, Ini_Metrics=None, metric_name='Emax', thresh=0.5, sel_res='Melody', Squared=False)

                AE_IC50_Res, AE_IC50_NoRes, MAE_IC50_Res, MAE_IC50_NoRes, IC50_Res_indx, IC50_NoRes_indx = Compute_Metric(df_pred,Ini_Metrics= None, metric_name='IC50', thresh=1.5, sel_res='Juan',Squared=True)
                #AE_IC50_Res_Subha, AE_IC50_NoRes_Subha, MAE_IC50_Res_Subha, MAE_IC50_NoRes_Subha, IC50_Res_indx_Subha, IC50_NoRes_indx_Subha = Compute_Metric(df_pred_Subha, Ini_Metrics=None, metric_name='IC50', thresh=1.5,sel_res='Subha', Squared=True)
                AE_IC50_Res_Melody, AE_IC50_NoRes_Melody, MAE_IC50_Res_Melody, MAE_IC50_NoRes_Melody, IC50_Res_indx_Melody, IC50_NoRes_indx_Melody = Compute_Metric(df_pred_Melody_IC50, Ini_Metrics=None, metric_name='IC50', thresh=1.5, sel_res='Melody', Squared=True)
            else:
                AE_per_dose_aux = np.abs(df_pred[cols_label].values - df_pred[cols_pred].values)
                AE_per_dose = np.concatenate((AE_per_dose,AE_per_dose_aux))
                MAE_per_dose = np.concatenate((MAE_per_dose,np.mean(AE_per_dose_aux,0)[None,:]),0)

                AE_AUC_Res, AE_AUC_NoRes, MAE_AUC_Res, MAE_AUC_NoRes, AUC_Res_indx, AUC_NoRes_indx = Compute_Metric(df_pred, Ini_Metrics=[AE_AUC_Res.copy(),AE_AUC_NoRes.copy(),MAE_AUC_Res.copy(),MAE_AUC_NoRes.copy()], metric_name='AUC', thresh=0.55, Squared=False)
                AE_AUC_Res_Melody, AE_AUC_NoRes_Melody, MAE_AUC_Res_Melody, MAE_AUC_NoRes_Melody, AUC_Res_indx_Melody, AUC_NoRes_indx_Melody = Compute_Metric(df_pred_Melody_AUC, Ini_Metrics=[AE_AUC_Res_Melody.copy(), AE_AUC_NoRes_Melody.copy(), MAE_AUC_Res_Melody.copy(), MAE_AUC_NoRes_Melody.copy()],metric_name='AUC', thresh=0.55,sel_res='Melody',Squared=False)

                AE_Emax_Res, AE_Emax_NoRes, MAE_Emax_Res, MAE_Emax_NoRes, Emax_Res_indx, Emax_NoRes_indx = Compute_Metric(df_pred,Ini_Metrics=[AE_Emax_Res.copy(), AE_Emax_NoRes.copy(), MAE_Emax_Res.copy(), MAE_Emax_NoRes.copy()],metric_name='Emax', thresh=0.5, Squared=False)
                AE_Emax_Res_Melody, AE_Emax_NoRes_Melody, MAE_Emax_Res_Melody, MAE_Emax_NoRes_Melody, Emax_Res_indx_Melody, Emax_NoRes_indx_Melody = Compute_Metric(df_pred_Melody_Emax,Ini_Metrics=[AE_Emax_Res_Melody.copy(), AE_Emax_NoRes_Melody.copy(), MAE_Emax_Res_Melody.copy(), MAE_Emax_NoRes_Melody.copy()],metric_name='Emax', thresh=0.5,sel_res='Melody',Squared=False)

                AE_IC50_Res, AE_IC50_NoRes, MAE_IC50_Res, MAE_IC50_NoRes, IC50_Res_indx, IC50_NoRes_indx = Compute_Metric(df_pred,Ini_Metrics=[AE_IC50_Res.copy(), AE_IC50_NoRes.copy(), MAE_IC50_Res.copy(), MAE_IC50_NoRes.copy()],metric_name='IC50', thresh=1.5, Squared=True)
                #AE_IC50_Res_Subha, AE_IC50_NoRes_Subha, MAE_IC50_Res_Subha, MAE_IC50_NoRes_Subha, IC50_Res_indx_Subha, IC50_NoRes_indx_Subha = Compute_Metric(df_pred_Subha,Ini_Metrics=[AE_IC50_Res_Subha.copy(), AE_IC50_NoRes_Subha.copy(), MAE_IC50_Res_Subha.copy(), MAE_IC50_NoRes_Subha.copy()], metric_name='IC50', thresh=1.5, sel_res='Subha', Squared=True)
                AE_IC50_Res_Melody, AE_IC50_NoRes_Melody, MAE_IC50_Res_Melody, MAE_IC50_NoRes_Melody, IC50_Res_indx_Melody, IC50_NoRes_indx_Melody = Compute_Metric(df_pred_Melody_IC50,Ini_Metrics=[AE_IC50_Res_Melody.copy(), AE_IC50_NoRes_Melody.copy(), MAE_IC50_Res_Melody.copy(),MAE_IC50_NoRes_Melody.copy()], metric_name='IC50', thresh=1.5, sel_res='Melody', Squared=True)

        AE_per_dose_Ncells.append(AE_per_dose)
        MAE_per_dose_Ncells.append(MAE_per_dose)

        AE_AUC_Res_Ncells.append(AE_AUC_Res)
        MAE_AUC_Res_Ncells.append(MAE_AUC_Res)
        AE_AUC_NoRes_Ncells.append(AE_AUC_NoRes)
        MAE_AUC_NoRes_Ncells.append(MAE_AUC_NoRes)

        AE_Emax_Res_Ncells.append(AE_Emax_Res)
        MAE_Emax_Res_Ncells.append(MAE_Emax_Res)
        AE_Emax_NoRes_Ncells.append(AE_Emax_NoRes)
        MAE_Emax_NoRes_Ncells.append(MAE_Emax_NoRes)

        AE_IC50_Res_Ncells.append(AE_IC50_Res)
        MAE_IC50_Res_Ncells.append(MAE_IC50_Res)
        AE_IC50_NoRes_Ncells.append(AE_IC50_NoRes)
        MAE_IC50_NoRes_Ncells.append(MAE_IC50_NoRes)

        # AE_IC50_Res_Ncells_Subha.append(AE_IC50_Res_Subha)
        # MAE_IC50_Res_Ncells_Subha.append(MAE_IC50_Res_Subha)
        # AE_IC50_NoRes_Ncells_Subha.append(AE_IC50_NoRes_Subha)
        # MAE_IC50_NoRes_Ncells_Subha.append(MAE_IC50_NoRes_Subha)

        AE_IC50_Res_Ncells_Melody.append(AE_IC50_Res_Melody)
        MAE_IC50_Res_Ncells_Melody.append(MAE_IC50_Res_Melody)
        AE_IC50_NoRes_Ncells_Melody.append(AE_IC50_NoRes_Melody)
        MAE_IC50_NoRes_Ncells_Melody.append(MAE_IC50_NoRes_Melody)

        AE_AUC_Res_Ncells_Melody.append(AE_AUC_Res_Melody)
        MAE_AUC_Res_Ncells_Melody.append(MAE_AUC_Res_Melody)
        AE_AUC_NoRes_Ncells_Melody.append(AE_AUC_NoRes_Melody)
        MAE_AUC_NoRes_Ncells_Melody.append(MAE_AUC_NoRes_Melody)

        AE_Emax_Res_Ncells_Melody.append(AE_Emax_Res_Melody)
        MAE_Emax_Res_Ncells_Melody.append(MAE_Emax_Res_Melody)
        AE_Emax_NoRes_Ncells_Melody.append(AE_Emax_NoRes_Melody)
        MAE_Emax_NoRes_Ncells_Melody.append(MAE_Emax_NoRes_Melody)

    #plt.figure(5)
    Num_cells = All_N_cells.__len__()

    def plot_Nth_dose(indx_plot,axs,sel_dose,Num_cells,AE_per_dose_Ncells,MAE_per_dose_Ncells,mycolor = ['blue','blue'],Responsive = True,my_ylim=None,my_title=None,force_title=False):
        AE_per_Nthdose_Ncells = np.zeros((AE_per_dose_Ncells[0].shape[0], Num_cells))
        MeanAE_per_Nthdose_Ncells = np.zeros((MAE_per_dose_Ncells[0].shape[0], Num_cells))
        Nth_dose = sel_dose - 1  #start with dose 1, i.e., 1 - 1 = 0
        if AE_per_dose_Ncells[0].shape.__len__() == 1:
            for i in range(Num_cells):
                AE_per_Nthdose_Ncells[:, i] = AE_per_dose_Ncells[i].copy()
                MeanAE_per_Nthdose_Ncells[:, i] = MAE_per_dose_Ncells[i].copy()
        else:
            for i in range(Num_cells):
                AE_per_Nthdose_Ncells[:,i] = AE_per_dose_Ncells[i][:,Nth_dose]
                MeanAE_per_Nthdose_Ncells[:, i] = MAE_per_dose_Ncells[i][:, Nth_dose]

        MAE_per_Nthdose_Ncells = AE_per_Nthdose_Ncells.mean(0)
        stdAE_per_Nthdose_Ncells = AE_per_Nthdose_Ncells.std(0)
        Total_Ncell = All_N_cells * cancer_Ntrain[sel_cancer] * 0.01 #Here 0.01 is just like divide by 100 to obtain % value
        print("Total_Ncell",Total_Ncell)
        New_X = np.linspace(Total_Ncell[0],Total_Ncell[-1],1000)
        f_MAE_per_Nthdose_Ncells = pchip_interpolate(Total_Ncell, MAE_per_Nthdose_Ncells,New_X)
        f_stdAE_per_Nthdose_Ncells = pchip_interpolate(Total_Ncell, stdAE_per_Nthdose_Ncells,New_X)
        Ntest = AE_per_Nthdose_Ncells.shape[0]/All_Nseed.__len__()
        MAE_per_seed = [AE_per_Nthdose_Ncells[int(Ntest)*nseed:int(Ntest)*nseed+int(Ntest), :].mean(0)[:,None] for nseed in range(All_Nseed.__len__())]
        MAE_per_seed = np.array(MAE_per_seed)[:,:,0]
        print(f"{sel_cancer} MAE_per_seed",MAE_per_seed)
        New_X_per_seed = Total_Ncell.reshape(1, -1).repeat(6, axis=0)

        Mean_MAE_per_Nthdose_Ncells = MeanAE_per_Nthdose_Ncells.mean(0)
        std_MAE_per_Nthdose_Ncells = MeanAE_per_Nthdose_Ncells.std(0)
        f_Mean_MAE_per_Nthdose_Ncells = pchip_interpolate(Total_Ncell, Mean_MAE_per_Nthdose_Ncells,New_X)
        f_std_MAE_per_Nthdose_Ncells = pchip_interpolate(Total_Ncell, std_MAE_per_Nthdose_Ncells,New_X)

        global line_averMAE, line_Seeds

        if Responsive is True:
            #axs[sel_cancer, Nth_dose].boxplot(AE_per_Nthdose_Ncells,medianprops = dict(color = "orangered", linewidth = 1.8),positions=Total_Ncell,widths=4.0,notch=True,flierprops={'marker': 'o', 'markersize': 1, 'markerfacecolor': 'black'})
            #plt.fill_between(New_X, f_MAE_per_Nthdose_Ncells - f_stdAE_per_Nthdose_Ncells, f_MAE_per_Nthdose_Ncells + f_stdAE_per_Nthdose_Ncells,alpha=0.2)
            #AE_per_Nthdose_Ncells
            axs[indx_plot, Nth_dose].fill_between(New_X, f_Mean_MAE_per_Nthdose_Ncells - f_std_MAE_per_Nthdose_Ncells, f_Mean_MAE_per_Nthdose_Ncells + f_std_MAE_per_Nthdose_Ncells,color=mycolor[0],alpha=0.07)
            line3 = axs[indx_plot, Nth_dose].plot(New_X_per_seed, MAE_per_seed, '.', color=mycolor[0], alpha=0.4,label='Mean-Error per seed')
            line2, = axs[indx_plot, Nth_dose].plot(New_X, f_Mean_MAE_per_Nthdose_Ncells,'-',color=mycolor[1],linewidth=1.0,label = 'Avg. Mean-Error ± Std')
            if my_ylim is None:
                axs[indx_plot, Nth_dose].set_ylim([-0.01,0.27])
            else:
                axs[indx_plot, Nth_dose].set_ylim(my_ylim)
            axs[indx_plot, Nth_dose].grid(False)
            if indx_plot==0 or force_title==True:
                if Nth_dose == 1:
                    line_averMAE = line2
                    line_Seeds = line3
                    #global line_boxAll
                    #line_boxAll = line_box
                #axs[sel_cancer, Nth_dose].legend(handles=[line2])
                #Nole = '_nolegend_'
                #axs[sel_cancer, Nth_dose].legend(labels=[Nole,Nole,Nole,Nole,Nole,Nole,Nole,Nole,"abd"])
                if my_title is None:
                    axs[indx_plot, Nth_dose].set_title(f"Dose {Nth_dose+1}",fontsize=15)
                else:
                    axs[indx_plot, Nth_dose].set_title(my_title)
        else:
            #axs[sel_cancer].boxplot(AE_per_Nthdose_Ncells, medianprops=dict(color="orangered", linewidth=1.8),positions=Total_Ncell, widths=4.0, notch=True,flierprops={'marker': 'o', 'markersize': 1, 'markerfacecolor': 'black'})
            # plt.fill_between(New_X, f_MAE_per_Nthdose_Ncells - f_stdAE_per_Nthdose_Ncells, f_MAE_per_Nthdose_Ncells + f_stdAE_per_Nthdose_Ncells,alpha=0.2)
            axs[indx_plot].fill_between(New_X, f_Mean_MAE_per_Nthdose_Ncells - f_std_MAE_per_Nthdose_Ncells,f_Mean_MAE_per_Nthdose_Ncells + f_std_MAE_per_Nthdose_Ncells,color=mycolor[0],alpha=0.07)
            line3 = axs[indx_plot].plot(New_X_per_seed, MAE_per_seed, '.', color=mycolor[0], alpha=0.4,label='Mean-Error per seed')
            line2, = axs[indx_plot].plot(New_X, f_Mean_MAE_per_Nthdose_Ncells, '-', color=mycolor[1], linewidth=1.0, label='Avg. Mean-Error ± Std')
            if my_ylim is None:
                axs[indx_plot].set_ylim([-0.01, 0.27])
            else:
                axs[indx_plot].set_ylim(my_ylim)
            axs[indx_plot].grid(False)
            if indx_plot == 0 or force_title == True:
                if Nth_dose == 1:
                    #global line_averMAE
                    line_averMAE = line2
                    line_Seeds = line3
                    # global line_boxAll
                    # line_boxAll = line_box
                # axs[sel_cancer, Nth_dose].legend(handles=[line2])
                # Nole = '_nolegend_'
                # axs[sel_cancer, Nth_dose].legend(labels=[Nole,Nole,Nole,Nole,Nole,Nole,Nole,Nole,"abd"])
                if my_title is None:
                    axs[indx_plot].set_title(f"Dose {Nth_dose + 1}")
                else:
                    axs[indx_plot].set_title(my_title)

    for Ndose in range(1,8):
        plot_Nth_dose(sel_cancer,axs,Ndose,Num_cells,AE_per_dose_Ncells,MAE_per_dose_Ncells)

    axs[0,0].legend(handles=[line_averMAE,line_Seeds[0]], loc='upper right', bbox_to_anchor=(1.0, 1.5), ncol=1, fancybox=True, shadow=True,fontsize=13)

cancer_name_plot = {0:'BRCA (Error)',1:'COAD (Error)',2:'LUAD (Error)',3:'SKCM (Error)',4:'SCLC (Error)'}
cancer_name_plot_abs = {0:'BRCA (Abs. Error)',1:'COAD (Abs. Error)',2:'LUAD (Abs. Error)',3:'SKCM (Abs. Error)',4:'SCLC (Abs. Error)'}

for i in range(5):
    axs[i, 0].set_ylabel(cancer_name_plot_abs[i], fontsize=17)

#fig.tight_layout(pad=6.0, w_pad=0.4, h_pad=-0.1)
fig.tight_layout(pad=3.0,w_pad=-2.2,h_pad=0.28)
for i in range(1,7):
    for j in range(5):
        axs[j,i].yaxis.set_tick_params(labelleft=False)

fig.supxlabel('Number of dose response curves in training', fontsize=18,x=0.5,y=0.003)
axs[0][0].legend(["Avg.±std (MOGP)","MAE-seed (MOGP)"]+["_nolegend_"]*6+["Avg.±std (SRMF)","MAE-seed (SRMF)"]+["_nolegend_"]*6+["Median (BERK)","Mean (BERK)"],loc='upper right',bbox_to_anchor=(2.34, 1.35), ncol=2, fancybox=True, shadow=True,fontsize='x-large')
