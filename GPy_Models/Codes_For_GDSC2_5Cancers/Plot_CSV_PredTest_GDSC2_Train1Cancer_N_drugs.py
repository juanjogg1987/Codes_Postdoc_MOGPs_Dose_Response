import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from scipy.interpolate import interp1d
from scipy.interpolate import pchip_interpolate
plt.close('all')

_FOLDER = '/home/juanjo/Work_Postdoc/my_codes_postdoc/FilesCSV_Predict_Train1Cancer_IncreasingCellLines/'
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
        figaux, axsaux = plt.subplots(3, 1,figsize = (10,20))
    else:
        figaux, axsaux = plt.subplots(3, 2, figsize=(15, 20))
    fig_all.append(figaux); axs_all.append(axsaux)
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
            path_to_read = _FOLDER + cancer+'/Train'+str(N_cells)+'/MOGP_Predict_C'+str(sel_cancer)+'_Train'+str(N_cells)+'_seed'+str(Nseed)+'.csv'

            df_pred = pd.read_csv(path_to_read)
            cols_label = ["norm_cells_" + str(i) for i in range(1, 8)]
            cols_pred = ["norm_cell_" + str(i)+"_MOGP" for i in range(1, 8)]
            if Nseed == 1:
                AE_per_dose = np.abs(df_pred[cols_label].values-df_pred[cols_pred].values)
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
                AE_per_dose_aux = np.abs(df_pred[cols_label].values - df_pred[cols_pred].values)
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

    def plot_Nth_dose(indx_plot,axs,sel_dose,Num_cells,AE_per_dose_Ncells,MAE_per_dose_Ncells,Responsive = True,my_ylim=None,my_title=None,force_title=False):
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
        Total_Ncell = All_N_cells * cancer_Ntrain[sel_cancer] * 0.01
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
            axs[indx_plot, Nth_dose].fill_between(New_X, f_Mean_MAE_per_Nthdose_Ncells - f_std_MAE_per_Nthdose_Ncells, f_Mean_MAE_per_Nthdose_Ncells + f_std_MAE_per_Nthdose_Ncells,alpha=0.2)
            line3 = axs[indx_plot, Nth_dose].plot(New_X_per_seed, MAE_per_seed, '.', color='black', alpha=0.4,label='Mean-Error per seed')
            line2, = axs[indx_plot, Nth_dose].plot(New_X, f_Mean_MAE_per_Nthdose_Ncells,'-',color='blue',linewidth=0.7,label = 'Avg. Mean-Error ± Std')
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
            #axs[indx_plot, Nth_dose].set_xticklabels(np.array([0,20,40,60,80]), fontsize=13)
        else:
            #axs[sel_cancer].boxplot(AE_per_Nthdose_Ncells, medianprops=dict(color="orangered", linewidth=1.8),positions=Total_Ncell, widths=4.0, notch=True,flierprops={'marker': 'o', 'markersize': 1, 'markerfacecolor': 'black'})
            # plt.fill_between(New_X, f_MAE_per_Nthdose_Ncells - f_stdAE_per_Nthdose_Ncells, f_MAE_per_Nthdose_Ncells + f_stdAE_per_Nthdose_Ncells,alpha=0.2)
            axs[indx_plot].fill_between(New_X, f_Mean_MAE_per_Nthdose_Ncells - f_std_MAE_per_Nthdose_Ncells,
                                                   f_Mean_MAE_per_Nthdose_Ncells + f_std_MAE_per_Nthdose_Ncells,
                                                   alpha=0.2)
            line3 = axs[indx_plot].plot(New_X_per_seed, MAE_per_seed, '.', color='black', alpha=0.4,label='Mean-Error per seed')
            line2, = axs[indx_plot].plot(New_X, f_Mean_MAE_per_Nthdose_Ncells, '-', color='blue',
                                                    linewidth=0.7, label='Avg. Mean-Error ± Std')
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
            axs[indx_plot].set_xticklabels(New_X,fontsize=20)

    for Ndose in range(1,8):
        plot_Nth_dose(sel_cancer,axs,Ndose,Num_cells,AE_per_dose_Ncells,MAE_per_dose_Ncells)


    def plot_benchmark(axs, loc, N_Cells_lin, data,alpha = 0.5, Responsive = True):
        if Responsive is True:
            line_Q1, = axs[loc[0], loc[1]].plot(N_Cells_lin, np.percentile(data, 25) * np.ones_like(N_Cells_lin), 'm--',linewidth=1.1,alpha=alpha,label='BERK Q1')
            line_Q2, = axs[loc[0], loc[1]].plot(N_Cells_lin, np.percentile(data, 50) * np.ones_like(N_Cells_lin), 'm',linewidth=1.1,alpha=alpha,label='BERK Q2')
            line_Q3, = axs[loc[0], loc[1]].plot(N_Cells_lin, np.percentile(data, 75) * np.ones_like(N_Cells_lin), 'm--',linewidth=1.1,alpha=alpha,label='BERK Q3')
            line_mean, = axs[loc[0], loc[1]].plot(N_Cells_lin, np.mean(data) * np.ones_like(N_Cells_lin), 'green',linewidth=1.1,alpha=alpha,label='BERK Mean-Error')
            axs[0, 0].legend(handles=[line_averMAE,line_Seeds[0],line_mean,line_Q3,line_Q2,line_Q1],loc='upper right', bbox_to_anchor=(2.03, 1.3),ncol=6, fancybox=True, shadow=True)
        else:
            line_Q1, = axs[loc[0]].plot(N_Cells_lin, np.percentile(data, 25) * np.ones_like(N_Cells_lin), 'm--',linewidth=1.1, alpha=alpha, label='BERK Q1')
            line_Q2, = axs[loc[0]].plot(N_Cells_lin, np.percentile(data, 50) * np.ones_like(N_Cells_lin), 'm',linewidth=1.1, alpha=alpha, label='BERK Q2')
            line_Q3, = axs[loc[0]].plot(N_Cells_lin, np.percentile(data, 75) * np.ones_like(N_Cells_lin), 'm--',linewidth=1.1, alpha=alpha, label='BERK Q3')
            line_mean, = axs[loc[0]].plot(N_Cells_lin, np.mean(data) * np.ones_like(N_Cells_lin), 'green',linewidth=1.1, alpha=alpha, label='BERK Mean-Error')
            axs[0].legend(handles=[line_averMAE,line_Seeds[0],line_mean, line_Q3, line_Q2, line_Q1], loc='upper right',bbox_to_anchor=(1.1, 1.25), ncol=6, fancybox=True, shadow=True)

    data_AUC, data_Emax, data_IC50, data_IC50_Res, data_IC50_NoRes,data_AUC_Res,data_AUC_NoRes,data_Emax_Res,data_Emax_NoRes, data_Ydose_res = np.load('Bench_Mark_AUC_Emax_IC50.pkl', allow_pickle=True)
    N_Cells_lin = np.linspace(6, 95, 1000)

    IsRes = True
    if sel_cancer == 2 or sel_cancer == 4: IsRes = False;
    print(f"AUC Cancer {sel_cancer}:",AE_AUC_Res_Ncells)
    if AE_AUC_Res_Ncells[0].shape[0] != 0:
        plot_Nth_dose(0, axs_all[sel_cancer], 2, Num_cells, AE_AUC_Res_Ncells, MAE_AUC_Res_Ncells,Responsive = IsRes,my_ylim=[-0.01,0.42],my_title="AUC Responsive (AE)")
        if data_AUC_Res[sel_cancer].shape[0] != 0:
            plot_benchmark(axs_all[sel_cancer], [0, 1], N_Cells_lin, data_AUC_Res[sel_cancer],alpha=0.5,Responsive = IsRes)
    plot_Nth_dose(0, axs_all[sel_cancer], 1, Num_cells, AE_AUC_NoRes_Ncells, MAE_AUC_NoRes_Ncells,Responsive = IsRes,my_ylim=[-0.01,0.42],my_title="AUC Non-Responsive (AE)")
    plot_benchmark(axs_all[sel_cancer], [0, 0], N_Cells_lin, data_AUC_NoRes[sel_cancer], alpha=0.5,Responsive = IsRes)

    print(f"Emax Cancer {sel_cancer}:", AE_Emax_Res_Ncells)
    if AE_Emax_Res_Ncells[0].shape[0] != 0:
        plot_Nth_dose(1, axs_all[sel_cancer], 2, Num_cells, AE_Emax_Res_Ncells, MAE_Emax_Res_Ncells,Responsive = IsRes,my_ylim=[-0.01,0.9],my_title="Emax Responsive (AE)",force_title=True)
        if data_Emax_Res[sel_cancer].shape[0] != 0:
            plot_benchmark(axs_all[sel_cancer], [1, 1], N_Cells_lin, data_Emax_Res[sel_cancer],alpha=0.5,Responsive = IsRes)
    plot_Nth_dose(1, axs_all[sel_cancer], 1, Num_cells, AE_Emax_NoRes_Ncells, MAE_Emax_NoRes_Ncells,Responsive = IsRes,my_ylim=[-0.01,0.9],my_title="Emax Non-Responsive (AE)",force_title=True)
    plot_benchmark(axs_all[sel_cancer], [1, 0], N_Cells_lin, data_Emax_NoRes[sel_cancer], alpha=0.5,Responsive = IsRes)

    print(f"IC50 Cancer {sel_cancer}:", AE_IC50_Res_Ncells)
    if AE_IC50_Res_Ncells[0].shape[0] != 0:
        plot_Nth_dose(2, axs_all[sel_cancer], 2, Num_cells, AE_IC50_Res_Ncells, MAE_IC50_Res_Ncells,Responsive = IsRes, my_ylim=[-0.01, 1.2],my_title="IC50 Responsive (SE)",force_title=True)
        plot_benchmark(axs_all[sel_cancer], [2, 1], N_Cells_lin, data_IC50_Res[sel_cancer],alpha=0.5,Responsive = IsRes)
    plot_Nth_dose(2, axs_all[sel_cancer], 1, Num_cells, AE_IC50_NoRes_Ncells, MAE_IC50_NoRes_Ncells,Responsive = IsRes, my_ylim=[-0.01, 1.2],my_title="IC50 Non-Responsive (SE)",force_title=True)
    plot_benchmark(axs_all[sel_cancer], [2, 0], N_Cells_lin, data_IC50_NoRes[sel_cancer],alpha=0.5,Responsive = IsRes)

#cancer_names = {0:'breast_cancer',1:'COAD_cancer',2:'LUAD_cancer',3:'melanoma_cancer',4:'SCLC_cancer'}
cancer_name_plot = {0:'Breast (Error)',1:'COAD (Error)',2:'LUAD (Error)',3:'Melanoma (Error)',4:'SCLC (Error)'}
cancer_name_plot_abs = {0:'Breast (Abs. Error)',1:'COAD (Abs. Error)',2:'LUAD (Abs. Error)',3:'Melanoma (Abs. Error)',4:'SCLC (Abs. Error)'}

for i in range(5):
    axs[i, 0].set_ylabel(cancer_name_plot_abs[i], fontsize=15)
axs[4, 3].set_xlabel("Number of dose response curves in training\n", fontsize=16)

for i in range(5):
    for j in range(3):
        if i == 2 or i == 4:
            if j < 2:
                axs_all[i][j].set_ylabel(cancer_name_plot_abs[i], fontsize=12)
            else:
                axs_all[i][j].set_ylabel(cancer_name_plot_abs[i][:-12] + '(Squared Error)', fontsize=12)
        else:
            if j<2:
                axs_all[i][j, 0].set_ylabel(cancer_name_plot_abs[i], fontsize=14)
            else:
                axs_all[i][j, 0].set_ylabel(cancer_name_plot_abs[i][:-12]+'(Squared Error)', fontsize=14)

for i in range(5):
    if i == 2 or i == 4:
        axs_all[i][2].set_xlabel("    Number of dose response curves in training",fontsize=15)
    else:
        axs_all[i][2,0].set_xlabel("                                                                                      Number of dose response curves in training", fontsize=15)

fig.tight_layout(pad=5.0, w_pad=-2.2, h_pad=1.0)
#fig.tight_layout(w_pad=-2.2)
for i in range(1,7):
    for j in range(5):
        axs[j,i].yaxis.set_tick_params(labelleft=False)

#axs[0,0].legend(handles=[line_averMAE,line_Seeds[0]], loc='upper right', bbox_to_anchor=(1.0, 1.5), ncol=1, fancybox=True, shadow=True,fontsize=13)
axs[0,3].legend(["Average Mean-Error ± std","Mean-Error per seed"],loc='upper right',bbox_to_anchor=(1.35, 1.35), ncol=2, fancybox=True, shadow=True)
#axs[0,3].set_xticklabels(fontsize=20)