import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import pchip_interpolate
plt.close('all')
sel_cancer = 0
N5th_cancer = 9
#path_to_load = '/home/juanjo/Work_Postdoc/my_codes_postdoc/GPy_Models/Codes_For_GDSC2_5Cancers/Test_Data_ToPlot_GDSC2_5Cancers/Three_drugs/'
#AUC_per_cell,Emax_per_cell,IC50_per_cell,AUCR2_per_cell,EmaxR2_per_cell,IC50R2_per_cell = np.load(path_to_load+'Test_Metrics_To_Plot_Three_drugs.pkl',allow_pickle=True)

#path_to_load = '/home/juanjo/Work_Postdoc/my_codes_postdoc/GPy_Models/Codes_For_GDSC2_5Cancers/Test_Data_ToPlot_GDSC2_5Cancers/N_drugs_3/'
path_to_load = '/home/juanjo/Work_Postdoc/my_codes_postdoc/GPy_Models/Codes_For_GDSC2_5Cancers/Test_Data_ToPlot_GDSC2_5Cancers/N_drugs_3/SamplingFromSimilarity/Cancer_'+str(sel_cancer)+'/'
#AUC_per_cell,Emax_per_cell,IC50_per_cell,AUCR2_per_cell,EmaxR2_per_cell,IC50R2_per_cell,AUC_CV_per_cell,Emax_CV_per_cell,IC50_CV_per_cell = np.load(path_to_load+'Test_Metrics_To_Plot_N_drugs_3_SamplingFromSimilarity_Cancer_'+str(sel_cancer)+'.pkl',allow_pickle=True)
AUC_per_cell,Emax_per_cell,IC50_per_cell,IC50_Resp_MAE_per_cell, IC50_NoResp_MAE_per_cell,IC50_Resp_r2_per_cell,IC50_NoResp_r2_per_cell,curve_MAE_per_cell,AUC_CV_per_cell,Emax_CV_per_cell,IC50_CV_per_cell = np.load(path_to_load+'Test_Metrics_To_Plot_N_drugs_3_SamplingFromSimilarity_Cancer_'+str(sel_cancer)+'_N5thCancer_'+str(N5th_cancer)+'.pkl',allow_pickle=True)
#IC50R2_per_cell[0][2]=0.0
def get_mean_std(data_per_cell):
    data_std = np.array([np.std(data_per_cell[i][1:]) for i in range(data_per_cell.__len__())])
    data_mean = np.array([np.mean(data_per_cell[i][1:]) for i in range(data_per_cell.__len__())])
    return data_mean,data_std

N_Cells = 4*np.array([12,24,48,96,144])
#N_Cells = 4*np.array([12,24,48,96])
AUC_mean, AUC_std = get_mean_std(AUC_per_cell)
IC50_mean, IC50_std = get_mean_std(IC50_per_cell)
Emax_mean, Emax_std = get_mean_std(Emax_per_cell)

IC50_Resp_MAE_mean, IC50_Resp_MAE_std = get_mean_std(IC50_Resp_MAE_per_cell)
IC50_NoResp_MAE_mean, IC50_NoResp_MAE_std = get_mean_std(IC50_NoResp_MAE_per_cell)
IC50_Resp_r2_mean, IC50_Resp_r2_std = get_mean_std(IC50_Resp_r2_per_cell)
IC50_NoResp_r2_mean, IC50_NoResp_r2_std = get_mean_std(IC50_NoResp_r2_per_cell)

curve_MAE_mean, curve_MAE_std = get_mean_std(curve_MAE_per_cell)

AUC_CV_mean, AUC_CV_std = get_mean_std(AUC_CV_per_cell)
IC50_CV_mean, IC50_CV_std = get_mean_std(IC50_CV_per_cell)
Emax_CV_mean, Emax_CV_std = get_mean_std(Emax_CV_per_cell)

N_Cells_lin = 4*np.linspace(12,144,1000)
#N_Cells_lin = 4*np.linspace(12,96,1000)
f_meanAUC = pchip_interpolate(N_Cells, AUC_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdAUC = pchip_interpolate(N_Cells, AUC_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')
f_meanEmax = pchip_interpolate(N_Cells, Emax_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdEmax = pchip_interpolate(N_Cells, Emax_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')
f_meanIC50 = pchip_interpolate(N_Cells, IC50_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdIC50 = pchip_interpolate(N_Cells, IC50_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')

f_meancurve_MAE = pchip_interpolate(N_Cells, curve_MAE_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdcurve_MAE = pchip_interpolate(N_Cells,curve_MAE_std , N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')

f_meanIC50_Res_MAE = pchip_interpolate(N_Cells, IC50_Resp_MAE_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdIC50_Res_MAE = pchip_interpolate(N_Cells,IC50_Resp_MAE_std , N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')

f_meanIC50_NoRes_MAE = pchip_interpolate(N_Cells, IC50_NoResp_MAE_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdIC50_NoRes_MAE = pchip_interpolate(N_Cells,IC50_NoResp_MAE_std , N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')

f_meanIC50_Res_r2 = pchip_interpolate(N_Cells, IC50_Resp_r2_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdIC50_Res_r2 = pchip_interpolate(N_Cells,IC50_Resp_r2_std , N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')

f_meanIC50_NoRes_r2 = pchip_interpolate(N_Cells, IC50_NoResp_r2_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdIC50_NoRes_r2 = pchip_interpolate(N_Cells,IC50_NoResp_r2_std , N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')

f_meanAUC_CV = pchip_interpolate(N_Cells, AUC_CV_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdAUC_CV = pchip_interpolate(N_Cells, AUC_CV_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')
f_meanEmax_CV = pchip_interpolate(N_Cells, Emax_CV_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdEmax_CV = pchip_interpolate(N_Cells, Emax_CV_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')
f_meanIC50_CV = pchip_interpolate(N_Cells, IC50_CV_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdIC50_CV = pchip_interpolate(N_Cells, IC50_CV_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')

# f_meanAUCR2 = pchip_interpolate(N_Cells, AUCR2_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
# f_stdAUCR2 = pchip_interpolate(N_Cells, AUCR2_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')
# f_meanEmaxR2 = pchip_interpolate(N_Cells, EmaxR2_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
# f_stdEmaxR2 = pchip_interpolate(N_Cells, EmaxR2_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')
# f_meanIC50R2 = pchip_interpolate(N_Cells, IC50R2_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
# f_stdIC50R2 = pchip_interpolate(N_Cells, IC50R2_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')

# def myplot_fillbetween(Nfig,N_Cells_lin,f_mean,f_std,title='my title'):
#     plt.figure(Nfig)
#     plt.plot(N_Cells_lin, f_mean,'-',color='black')
#     plt.fill_between(N_Cells_lin, f_mean - f_std, f_mean + f_std,alpha=0.5)
#     plt.title(title)

def myplot_fillbetween(axs,loc,N_Cells_lin,f_mean,f_std,title='my title',ylabel = False,ylim=None):
    axs[loc[0], loc[1]].plot(N_Cells_lin, f_mean,'-',color='black')
    axs[loc[0], loc[1]].fill_between(N_Cells_lin, f_mean - f_std, f_mean + f_std,alpha=0.5)
    axs[loc[0], loc[1]].set_title(title)
    axs[loc[0], loc[1]].set_xlabel(f"N Data (4-Cancers + {N5th_cancer} obs. {cancer})")
    if ylabel:
        axs[loc[0], loc[1]].set_ylabel("Mean Absolute Error")
    if ylim:
        axs[loc[0], loc[1]].set_ylim(ylim)

def plot_benchmark(axs,loc,N_Cells_lin,data):
    axs[loc[0], loc[1]].plot(N_Cells_lin, np.percentile(data, 25) * np.ones_like(N_Cells_lin), 'r--')
    axs[loc[0], loc[1]].plot(N_Cells_lin, np.percentile(data, 50) * np.ones_like(N_Cells_lin), 'r')
    axs[loc[0], loc[1]].plot(N_Cells_lin, np.percentile(data, 75) * np.ones_like(N_Cells_lin), 'r--')
    axs[loc[0], loc[1]].plot(N_Cells_lin, np.mean(data) * np.ones_like(N_Cells_lin), 'green')

data_AUC,data_Emax,data_IC50,data_IC50_Res,data_IC50_NoRes = np.load('Bench_Mark_AUC_Emax_IC50.pkl',allow_pickle=True)

cancer_names = {0:'Breast Cancer',1:'Coad Cancer',2:'Luad Cancer',3:'Melanoma Cancer',4:'SCLC Cancer'}
cancer = cancer_names[sel_cancer]
# if sel_cancer ==0:
#     ylimAUC = [0.04,0.13]; ylimAUC_CV = [0.04,0.095]
#     ylimEmax = [0.1,0.18]; ylimEmax_CV = [0.085,0.17]
#     ylimIC50 = [0.0073,0.0343]; ylimIC50_CV = [0.04,0.13]
# elif sel_cancer == 1:
#     ylimAUC = [0.06, 0.15];   ylimAUC_CV = [0.035, 0.095]
#     ylimEmax = [0.16, 0.27];   ylimEmax_CV = [0.075, 0.16]
#     ylimIC50 = [0.04, 0.24];   ylimIC50_CV = [0.03, 0.14]
# elif sel_cancer == 2:
#     ylimAUC = [0.03, 0.085];    ylimAUC_CV = [0.03, 0.085]
#     ylimEmax = [0.08, 0.135];    ylimEmax_CV = [0.075, 0.14]
#     ylimIC50 = [-0.0001, 0.025];    ylimIC50_CV = [0.04, 0.13]
# elif sel_cancer == 3:
#     ylimAUC = [0.12, 0.16];    ylimAUC_CV = [0.013, 0.085]
#     ylimEmax = [0.18, 0.22];    ylimEmax_CV = [0.05, 0.15]
#     ylimIC50 = [0.35, 0.48];    ylimIC50_CV = [0.0, 0.08]
# elif sel_cancer == 4:
#     ylimAUC = [0.04, 0.13];    ylimAUC_CV = [0.04, 0.095]
#     ylimEmax = [0.1, 0.17];    ylimEmax_CV = [0.1, 0.17]
#     ylimIC50 = [0.0073, 0.0343];    ylimIC50_CV = [0.04, 0.13]

ylimAUC = [0.0,0.3]; ylimAUC_CV = [0.0,0.3]
ylimEmax = [0.0,0.3]; ylimEmax_CV = [0.0,0.3]
ylimIC50 = [0.0,0.3]; ylimIC50_CV = [0.0,0.3]

fig, axs = plt.subplots(2, 3,figsize = (15,10))
myplot_fillbetween(axs,[0,0],N_Cells_lin,f_meanAUC,f_stdAUC,title='AUC-Interp. '+cancer,ylabel = True,ylim=ylimAUC)
myplot_fillbetween(axs,[1,0],N_Cells,AUC_mean,AUC_std,title='AUC '+cancer,ylabel = True,ylim=ylimAUC)
plot_benchmark(axs,[0,0],N_Cells_lin,data_AUC[sel_cancer])
myplot_fillbetween(axs,[0,1],N_Cells_lin,f_meanEmax,f_stdEmax,title='Emax-Interp. '+cancer,ylabel = True,ylim=ylimEmax)
myplot_fillbetween(axs,[1,1],N_Cells,Emax_mean,Emax_std,title='Emax '+cancer,ylabel = True,ylim=ylimEmax)
plot_benchmark(axs,[0,1],N_Cells_lin,data_Emax[sel_cancer])
myplot_fillbetween(axs,[0,2],N_Cells_lin,f_meanIC50,f_stdIC50,title='IC50-Interp. '+cancer,ylabel = True,ylim=ylimIC50)
myplot_fillbetween(axs,[1,2],N_Cells,IC50_mean,IC50_std,title='IC50 '+cancer,ylabel = True,ylim=ylimIC50)
plot_benchmark(axs,[0,2],N_Cells_lin,data_IC50[sel_cancer])

fig, axs1 = plt.subplots(2, 3,figsize = (15,10))
myplot_fillbetween(axs1,[0,0],N_Cells_lin,f_meanAUC_CV,f_stdAUC_CV,title='Validation AUC-Interp. ('+cancer+')',ylabel = True,ylim=ylimAUC_CV)
myplot_fillbetween(axs1,[1,0],N_Cells,AUC_CV_mean,AUC_CV_std,title='Validation AUC ('+cancer+')',ylabel = True,ylim=ylimAUC_CV)
myplot_fillbetween(axs1,[0,1],N_Cells_lin,f_meanEmax_CV,f_stdEmax_CV,title='Validation Emax-Interp. ('+cancer+')',ylabel = True,ylim=ylimEmax_CV)
myplot_fillbetween(axs1,[1,1],N_Cells,Emax_CV_mean,Emax_CV_std,title='Validation Emax ('+cancer+')',ylabel = True,ylim=ylimEmax_CV)
myplot_fillbetween(axs1,[0,2],N_Cells_lin,f_meanIC50_CV,f_stdIC50_CV,title='Validation IC50-Interp. ('+cancer+')',ylabel = True,ylim=ylimIC50_CV)
myplot_fillbetween(axs1,[1,2],N_Cells,IC50_CV_mean,IC50_CV_std,title='Validation IC50 ('+cancer+')',ylabel = True,ylim=ylimIC50_CV)

fig, axs2 = plt.subplots(2, 2,figsize = (15,10))
myplot_fillbetween(axs2,[0,0],N_Cells_lin,f_meanIC50_Res_MAE,f_stdIC50_Res_MAE,title='MAE IC50-Interpolated '+cancer+' Responsive')
myplot_fillbetween(axs2,[1,0],N_Cells,IC50_Resp_MAE_mean,IC50_Resp_MAE_std,title='MAE IC50 '+cancer+' Responsive')
plot_benchmark(axs2,[0,0],N_Cells_lin,data_IC50_Res[sel_cancer])
myplot_fillbetween(axs2,[0,1],N_Cells_lin,f_meanIC50_NoRes_MAE,f_stdIC50_NoRes_MAE,title='MAE IC50-Interpolated '+cancer+' Non-Responsive')
myplot_fillbetween(axs2,[1,1],N_Cells,IC50_NoResp_MAE_mean,IC50_NoResp_MAE_std,title='MAE IC50 '+cancer+' Non-Responsive')
plot_benchmark(axs2,[0,1],N_Cells_lin,data_IC50_NoRes[sel_cancer])

fig, axs3 = plt.subplots(2, 2,figsize = (15,10))
myplot_fillbetween(axs3,[0,0],N_Cells_lin,f_meanIC50_Res_r2,f_stdIC50_Res_r2,title='R² IC50-Interpolated '+cancer+' Responsive')
myplot_fillbetween(axs3,[1,0],N_Cells,IC50_Resp_r2_mean,IC50_Resp_r2_std,title='R² IC50 '+cancer+' Responsive')
myplot_fillbetween(axs3,[0,1],N_Cells_lin,f_meanIC50_NoRes_r2,f_stdIC50_NoRes_r2,title='R² IC50-Interpolated '+cancer+' Non-Responsive')
myplot_fillbetween(axs3,[1,1],N_Cells,IC50_NoResp_r2_mean,IC50_NoResp_r2_std,title='R² IC50 '+cancer+' Non-Responsive')

fig, axs4 = plt.subplots(2, 2,figsize = (15,10))
myplot_fillbetween(axs4,[0,0],N_Cells_lin,f_meancurve_MAE,f_stdcurve_MAE,title='MAE All Predictions - Interpolated '+cancer)
myplot_fillbetween(axs4,[1,0],N_Cells,curve_MAE_mean,curve_MAE_std,title='MAE All Predictions '+cancer)
