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
N5th_cancer = 9
All_Nseed = [1,2,3,4,5,6]
All_N_cells = np.array([0,12,24,48,96,144]) * 4
#Ntotal_Cells = int(N_cells)*4 + int(N5th_cancer)

fig, axs = plt.subplots(5, 7,figsize = (25,20))
fig2, axs2 = plt.subplots(5, 6,figsize = (25,20))
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

            df_pred = pd.read_csv(path_to_read)

            if Nseed == 1:
                AE_per_dose = np.abs(df_pred[df_pred.columns[15:22]].values-df_pred[df_pred.columns[26:33]].values)
                MAE_per_dose = np.mean(AE_per_dose,0)[None,:]

                AUC_Res_indx = df_pred['AUC_s4'].values < 0.55
                AUC_NoRes_indx = df_pred['AUC_s4'].values >= 0.55
                "TODO: REMEMBER THE CASE WHEN THERE ARE NOT RESPONSIVE EVER"
                AE_AUC_Res = np.abs(df_pred['AUC_MOGP'].values[AUC_Res_indx] - df_pred['AUC_s4'].values[AUC_Res_indx])
                AE_AUC_NoRes = np.abs(df_pred['AUC_MOGP'].values[AUC_NoRes_indx] - df_pred['AUC_s4'].values[AUC_NoRes_indx])
                MAE_AUC_Res = np.array( [np.mean(AE_AUC_Res)] )
                MAE_AUC_NoRes = np.array( [np.mean(AE_AUC_NoRes)] )

                Emax_Res_indx = df_pred['Emax_s4'].values < 0.5
                Emax_NoRes_indx = df_pred['Emax_s4'].values >= 0.5
                "TODO: REMEMBER THE CASE WHEN THERE ARE NOT RESPONSIVE EVER"
                AE_Emax_Res = np.abs(df_pred['Emax_MOGP'].values[Emax_Res_indx] - df_pred['Emax_s4'].values[Emax_Res_indx])
                AE_Emax_NoRes = np.abs(df_pred['Emax_MOGP'].values[Emax_NoRes_indx] - df_pred['Emax_s4'].values[Emax_NoRes_indx])
                MAE_Emax_Res = np.array([np.mean(AE_Emax_Res)])
                MAE_Emax_NoRes = np.array([np.mean(AE_Emax_NoRes)])

                IC50_Squared = 2   #use 2 to square or 1 for Absolute
                IC50_Res_indx = df_pred['IC50_s4'].values < 1.5
                IC50_NoRes_indx = df_pred['IC50_s4'].values >= 1.5
                "TODO: REMEMBER THE CASE WHEN THERE ARE NOT RESPONSIVE EVER"
                AE_IC50_Res = np.abs(df_pred['IC50_MOGP'].values[IC50_Res_indx] - df_pred['IC50_s4'].values[IC50_Res_indx])**IC50_Squared
                AE_IC50_NoRes = np.abs( df_pred['IC50_MOGP'].values[IC50_NoRes_indx] - df_pred['IC50_s4'].values[IC50_NoRes_indx])**IC50_Squared
                MAE_IC50_Res = np.array([np.mean(AE_IC50_Res)])
                MAE_IC50_NoRes = np.array([np.mean(AE_IC50_NoRes)])
            else:
                AE_per_dose_aux = np.abs(df_pred[df_pred.columns[15:22]].values - df_pred[df_pred.columns[26:33]].values)
                AE_per_dose = np.concatenate((AE_per_dose,AE_per_dose_aux))
                MAE_per_dose = np.concatenate((MAE_per_dose,np.mean(AE_per_dose_aux,0)[None,:]),0)

                AE_AUC_Res_aux = np.abs(df_pred['AUC_MOGP'].values[AUC_Res_indx] - df_pred['AUC_s4'].values[AUC_Res_indx])
                AE_AUC_NoRes_aux = np.abs(df_pred['AUC_MOGP'].values[AUC_NoRes_indx] - df_pred['AUC_s4'].values[AUC_NoRes_indx])
                AE_AUC_Res = np.concatenate((AE_AUC_Res, AE_AUC_Res_aux),0)
                MAE_AUC_Res = np.concatenate((MAE_AUC_Res, np.array([np.mean(AE_AUC_Res_aux)])), 0)
                AE_AUC_NoRes = np.concatenate((AE_AUC_NoRes, AE_AUC_NoRes_aux))
                MAE_AUC_NoRes = np.concatenate((MAE_AUC_NoRes, np.array([np.mean(AE_AUC_NoRes_aux)])), 0)

                AE_Emax_Res_aux = np.abs(df_pred['Emax_MOGP'].values[Emax_Res_indx] - df_pred['Emax_s4'].values[Emax_Res_indx])
                AE_Emax_NoRes_aux = np.abs(df_pred['Emax_MOGP'].values[Emax_NoRes_indx] - df_pred['Emax_s4'].values[Emax_NoRes_indx])
                AE_Emax_Res = np.concatenate((AE_Emax_Res, AE_Emax_Res_aux), 0)
                MAE_Emax_Res = np.concatenate((MAE_Emax_Res, np.array([np.mean(AE_Emax_Res_aux)])), 0)
                AE_Emax_NoRes = np.concatenate((AE_Emax_NoRes, AE_Emax_NoRes_aux))
                MAE_Emax_NoRes = np.concatenate((MAE_Emax_NoRes, np.array([np.mean(AE_Emax_NoRes_aux)])), 0)

                AE_IC50_Res_aux = np.abs(df_pred['IC50_MOGP'].values[IC50_Res_indx] - df_pred['IC50_s4'].values[IC50_Res_indx])**IC50_Squared
                AE_IC50_NoRes_aux = np.abs(df_pred['IC50_MOGP'].values[IC50_NoRes_indx] - df_pred['IC50_s4'].values[IC50_NoRes_indx])**IC50_Squared
                AE_IC50_Res = np.concatenate((AE_IC50_Res, AE_IC50_Res_aux), 0)
                MAE_IC50_Res = np.concatenate((MAE_IC50_Res, np.array([np.mean(AE_IC50_Res_aux)])), 0)
                AE_IC50_NoRes = np.concatenate((AE_IC50_NoRes, AE_IC50_NoRes_aux))
                MAE_IC50_NoRes = np.concatenate((MAE_IC50_NoRes, np.array([np.mean(AE_IC50_NoRes_aux)])), 0)

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

    #plt.figure(5)
    Num_cells = All_N_cells.__len__()

    def plot_Nth_dose(sel_cancer,axs,sel_dose,Num_cells,AE_per_dose_Ncells,MAE_per_dose_Ncells,my_ylim=None,my_title=None):
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
        Total_Ncell = All_N_cells + N5th_cancer
        New_X = np.linspace(Total_Ncell[0],Total_Ncell[-1],1000)
        f_MAE_per_Nthdose_Ncells = pchip_interpolate(Total_Ncell, MAE_per_Nthdose_Ncells,New_X)
        f_stdAE_per_Nthdose_Ncells = pchip_interpolate(Total_Ncell, stdAE_per_Nthdose_Ncells,New_X)

        Mean_MAE_per_Nthdose_Ncells = MeanAE_per_Nthdose_Ncells.mean(0)
        std_MAE_per_Nthdose_Ncells = MeanAE_per_Nthdose_Ncells.std(0)
        f_Mean_MAE_per_Nthdose_Ncells = pchip_interpolate(Total_Ncell, Mean_MAE_per_Nthdose_Ncells,New_X)
        f_std_MAE_per_Nthdose_Ncells = pchip_interpolate(Total_Ncell, std_MAE_per_Nthdose_Ncells,New_X)

        axs[sel_cancer, Nth_dose].boxplot(AE_per_Nthdose_Ncells,medianprops = dict(color = "orangered", linewidth = 1.8),positions=Total_Ncell,widths=25.0,notch=True,flierprops={'marker': 'o', 'markersize': 1, 'markerfacecolor': 'black'})
        #plt.fill_between(New_X, f_MAE_per_Nthdose_Ncells - f_stdAE_per_Nthdose_Ncells, f_MAE_per_Nthdose_Ncells + f_stdAE_per_Nthdose_Ncells,alpha=0.2)
        axs[sel_cancer, Nth_dose].fill_between(New_X, f_Mean_MAE_per_Nthdose_Ncells - f_std_MAE_per_Nthdose_Ncells, f_Mean_MAE_per_Nthdose_Ncells + f_std_MAE_per_Nthdose_Ncells,alpha=0.2)
        line2, = axs[sel_cancer, Nth_dose].plot(New_X, f_Mean_MAE_per_Nthdose_Ncells,'-',color='blue',linewidth=0.7,label = 'Average Mean-Error ± Std')
        if my_ylim is None:
            axs[sel_cancer, Nth_dose].set_ylim([-0.01,0.3])
        else:
            axs[sel_cancer, Nth_dose].set_ylim(my_ylim)
        axs[sel_cancer, Nth_dose].grid()
        if sel_cancer==0:
            if Nth_dose == 0:
                global line_averMAE
                line_averMAE = line2
                #global line_boxAll
                #line_boxAll = line_box
            #axs[sel_cancer, Nth_dose].legend(handles=[line2])
            #Nole = '_nolegend_'
            #axs[sel_cancer, Nth_dose].legend(labels=[Nole,Nole,Nole,Nole,Nole,Nole,Nole,Nole,"abd"])
            if my_title is None:
                axs[sel_cancer, Nth_dose].set_title(f"Dose {Nth_dose+1}")
            else:
                axs[sel_cancer, Nth_dose].set_title(my_title)

    for Ndose in range(1,8):
        plot_Nth_dose(sel_cancer,axs,Ndose,Num_cells,AE_per_dose_Ncells,MAE_per_dose_Ncells)


    def plot_benchmark(axs, loc, N_Cells_lin, data,alpha = 0.5):
        line_Q1, = axs[loc[0], loc[1]].plot(N_Cells_lin, np.percentile(data, 25) * np.ones_like(N_Cells_lin), 'm--',linewidth=1.1,alpha=alpha,label='Benchmark Q1')
        line_Q2, = axs[loc[0], loc[1]].plot(N_Cells_lin, np.percentile(data, 50) * np.ones_like(N_Cells_lin), 'm',linewidth=1.1,alpha=alpha,label='Benchmark Q2')
        line_Q3, = axs[loc[0], loc[1]].plot(N_Cells_lin, np.percentile(data, 75) * np.ones_like(N_Cells_lin), 'm--',linewidth=1.1,alpha=alpha,label='Benchmark Q3')
        line_mean, = axs[loc[0], loc[1]].plot(N_Cells_lin, np.mean(data) * np.ones_like(N_Cells_lin), 'green',linewidth=1.1,alpha=alpha,label='Benchmark Mean')
        #line_median, = axs[loc[0], loc[1]].plot(1000*N_Cells_lin, np.mean(data) * np.ones_like(N_Cells_lin), 'm',alpha=1.0, label='Boxplot Median')
        axs[0, 0].legend(handles=[line_mean,line_averMAE,line_Q3,line_Q2,line_Q1],loc='upper right', bbox_to_anchor=(5.0, 1.5),
          ncol=5, fancybox=True, shadow=True)

    data_AUC, data_Emax, data_IC50, data_IC50_Res, data_IC50_NoRes,data_AUC_Res,data_AUC_NoRes,data_Emax_Res,data_Emax_NoRes, data_Ydose_res = np.load('Bench_Mark_AUC_Emax_IC50.pkl', allow_pickle=True)
    N_Cells_lin = 4 * np.linspace(0, 144, 1000) + N5th_cancer

    print(f"AUC Cancer {sel_cancer}:",AE_AUC_Res_Ncells)
    if AE_AUC_Res_Ncells[0].shape[0] != 0:
        plot_Nth_dose(sel_cancer, axs2, 1, Num_cells, AE_AUC_Res_Ncells, MAE_AUC_Res_Ncells,my_ylim=[-0.01,0.8],my_title="AUC Responsive (AE)")
        if data_AUC_Res[sel_cancer].shape[0] != 0:
            plot_benchmark(axs2, [sel_cancer, 0], N_Cells_lin, data_AUC_Res[sel_cancer],alpha=0.5)
    plot_Nth_dose(sel_cancer, axs2, 2, Num_cells, AE_AUC_NoRes_Ncells, MAE_AUC_NoRes_Ncells,my_ylim=[-0.01,0.8],my_title="AUC Non-Responsive (AE)")
    plot_benchmark(axs2, [sel_cancer, 1], N_Cells_lin, data_AUC_NoRes[sel_cancer], alpha=0.5)

    print(f"Emax Cancer {sel_cancer}:", AE_Emax_Res_Ncells)
    if AE_Emax_Res_Ncells[0].shape[0] != 0:
        plot_Nth_dose(sel_cancer, axs2, 3, Num_cells, AE_Emax_Res_Ncells, MAE_Emax_Res_Ncells,my_ylim=[-0.01,0.8],my_title="Emax Responsive (AE)")
        if data_Emax_Res[sel_cancer].shape[0] != 0:
            plot_benchmark(axs2, [sel_cancer, 2], N_Cells_lin, data_Emax_Res[sel_cancer],alpha=0.5)
    plot_Nth_dose(sel_cancer, axs2, 4, Num_cells, AE_Emax_NoRes_Ncells, MAE_Emax_NoRes_Ncells,my_ylim=[-0.01,0.8],my_title="Emax Non-Responsive (AE)")
    plot_benchmark(axs2, [sel_cancer, 3], N_Cells_lin, data_Emax_NoRes[sel_cancer], alpha=0.5)

    print(f"IC50 Cancer {sel_cancer}:", AE_IC50_Res_Ncells)
    if AE_IC50_Res_Ncells[0].shape[0] != 0:
        plot_Nth_dose(sel_cancer, axs2, 5, Num_cells, AE_IC50_Res_Ncells, MAE_IC50_Res_Ncells, my_ylim=[-0.01, 1.2],my_title="IC50 Responsive (SE)")
        plot_benchmark(axs2, [sel_cancer, 4], N_Cells_lin, data_IC50_Res[sel_cancer],alpha=0.5)
    plot_Nth_dose(sel_cancer, axs2, 6, Num_cells, AE_IC50_NoRes_Ncells, MAE_IC50_NoRes_Ncells, my_ylim=[-0.01, 1.2],my_title="IC50 Non-Responsive (SE)")
    plot_benchmark(axs2, [sel_cancer, 5], N_Cells_lin, data_IC50_NoRes[sel_cancer],alpha=0.5)

#cancer_names = {0:'breast_cancer',1:'COAD_cancer',2:'LUAD_cancer',3:'melanoma_cancer',4:'SCLC_cancer'}
cancer_name_plot = {0:'Breast (Error)',1:'COAD (Error)',2:'LUAD (Error)',3:'Melanoma (Error)',4:'SCLC (Error)'}
cancer_name_plot_abs = {0:'Breast (Abs. Error)',1:'COAD (Abs. Error)',2:'LUAD (Abs. Error)',3:'Melanoma (Abs. Error)',4:'SCLC (Abs. Error)'}
for i in range(5):
    axs[i, 0].set_ylabel(cancer_name_plot_abs[i], fontsize=12)
    axs2[i, 0].set_ylabel(cancer_name_plot[i], fontsize=12)

axs[4, 3].set_xlabel("Number of dose response curves in training", fontsize=13)
axs2[4, 2].set_xlabel("                                                        Number of dose response curves in training", fontsize=13)
