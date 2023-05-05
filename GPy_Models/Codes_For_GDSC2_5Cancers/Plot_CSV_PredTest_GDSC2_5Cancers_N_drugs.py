import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import pchip_interpolate
#plt.close('all')

_FOLDER = '/home/juanjo/Work_Postdoc/my_codes_postdoc/FilesCSV_Predict_5Cancers_IncreasingCellLines/'
cancer_names = {0:'breast_cancer',1:'COAD_cancer',2:'LUAD_cancer',3:'melanoma_cancer',4:'SCLC_cancer'}

#sel_cancer = 0
cancers = [0,1,2,3,4]
N5th_cancer = 45
All_Nseed = [1,2,3,4,5,6]
All_N_cells = np.array([0,12,24,48,96,144]) * 4
#Ntotal_Cells = int(N_cells)*4 + int(N5th_cancer)

fig, axs = plt.subplots(5, 7,figsize = (15,10))
for sel_cancer in cancers:
    cancer = cancer_names[sel_cancer]
    AE_per_dose_Ncells = []
    MAE_per_dose_Ncells = []
    for N_cells in All_N_cells:
        for Nseed in All_Nseed:
            path_to_read = _FOLDER + cancer+'/N5th_CancerInTrain_'+str(N5th_cancer)+'/NTrain_'+str(N_cells)+'_plus_'+str(N5th_cancer)+'/MOGP_Predict_C'+str(sel_cancer)+'_Train_'+str(N_cells)+'_plus_'+str(N5th_cancer)+'_seed'+str(Nseed)+'.csv'

            df_pred = pd.read_csv(path_to_read)

            if Nseed == 1:
                AE_per_dose = np.abs(df_pred[df_pred.columns[15:22]].values-df_pred[df_pred.columns[26:33]].values)
                MAE_per_dose = np.mean(AE_per_dose,0)[None,:]
            else:
                AE_per_dose_aux = np.abs(df_pred[df_pred.columns[15:22]].values - df_pred[df_pred.columns[26:33]].values)
                AE_per_dose = np.concatenate((AE_per_dose,AE_per_dose_aux))
                MAE_per_dose = np.concatenate((MAE_per_dose,np.mean(AE_per_dose_aux,0)[None,:]),0)

        AE_per_dose_Ncells.append(AE_per_dose)
        MAE_per_dose_Ncells.append(MAE_per_dose)

    #plt.figure(5)
    Num_cells = All_N_cells.__len__()

    def plot_Nth_dose(sel_cancer,axs,sel_dose,Num_cells,AE_per_dose_Ncells,MAE_per_dose_Ncells):
        AE_per_Nthdose_Ncells = np.zeros((AE_per_dose_Ncells[0].shape[0], Num_cells))
        MeanAE_per_Nthdose_Ncells = np.zeros((MAE_per_dose_Ncells[0].shape[0], Num_cells))
        Nth_dose = sel_dose - 1  #start with dose 1, i.e., 1 - 1 = 0
        for i in range(Num_cells):
            AE_per_Nthdose_Ncells[:,i] = AE_per_dose_Ncells[i][:,Nth_dose]
            MeanAE_per_Nthdose_Ncells[:, i] = MAE_per_dose_Ncells[i][:, Nth_dose]

        MAE_per_Nthdose_Ncells = AE_per_Nthdose_Ncells.mean(0)
        stdAE_per_Nthdose_Ncells = AE_per_Nthdose_Ncells.std(0)
        Total_Ncell = All_N_cells + N5th_cancer
        New_X = np.linspace(Total_Ncell[0],Total_Ncell[-1],1000)
        f_MAE_per_Nthdose_Ncells = pchip_interpolate(Total_Ncell, MAE_per_Nthdose_Ncells,New_X)
        f_stdAE_per_Nthdose_Ncells = pchip_interpolate(Total_Ncell, stdAE_per_Nthdose_Ncells,New_X)

        Mean_MAE_per_Nthdose_Ncells = MeanAE_per_Nthdose_Ncells.mean(0)
        std_MAE_per_Nthdose_Ncells = MeanAE_per_Nthdose_Ncells.std(0)
        f_Mean_MAE_per_Nthdose_Ncells = pchip_interpolate(Total_Ncell, Mean_MAE_per_Nthdose_Ncells,New_X)
        f_std_MAE_per_Nthdose_Ncells = pchip_interpolate(Total_Ncell, std_MAE_per_Nthdose_Ncells,New_X)

        axs[sel_cancer, Nth_dose].boxplot(AE_per_Nthdose_Ncells,positions=Total_Ncell,widths=25.0,notch=True,flierprops={'marker': 'o', 'markersize': 2, 'markerfacecolor': 'black'})
        #plt.fill_between(New_X, f_MAE_per_Nthdose_Ncells - f_stdAE_per_Nthdose_Ncells, f_MAE_per_Nthdose_Ncells + f_stdAE_per_Nthdose_Ncells,alpha=0.2)
        axs[sel_cancer, Nth_dose].fill_between(New_X, f_Mean_MAE_per_Nthdose_Ncells - f_std_MAE_per_Nthdose_Ncells, f_Mean_MAE_per_Nthdose_Ncells + f_std_MAE_per_Nthdose_Ncells,alpha=0.2)
        axs[sel_cancer, Nth_dose].plot(New_X, f_Mean_MAE_per_Nthdose_Ncells,'-',color='blue',linewidth=0.5)
        axs[sel_cancer, Nth_dose].set_ylim([-0.01,0.3])
        axs[sel_cancer, Nth_dose].grid()
        if sel_cancer==0:
            axs[sel_cancer, Nth_dose].set_title(f"Dose {Nth_dose+1}")

    for Ndose in range(1,8):
        plot_Nth_dose(sel_cancer,axs,Ndose,Num_cells,AE_per_dose_Ncells,MAE_per_dose_Ncells)