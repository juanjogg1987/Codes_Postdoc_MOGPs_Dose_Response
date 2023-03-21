import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import pchip_interpolate
plt.close('all')
#path_to_load = '/home/juanjo/Work_Postdoc/my_codes_postdoc/GPy_Models/Codes_For_GDSC2_5Cancers/Test_Data_ToPlot_GDSC2_5Cancers/Three_drugs/'
#AUC_per_cell,Emax_per_cell,IC50_per_cell,AUCR2_per_cell,EmaxR2_per_cell,IC50R2_per_cell = np.load(path_to_load+'Test_Metrics_To_Plot_Three_drugs.pkl',allow_pickle=True)

#path_to_load = '/home/juanjo/Work_Postdoc/my_codes_postdoc/GPy_Models/Codes_For_GDSC2_5Cancers/Test_Data_ToPlot_GDSC2_5Cancers/N_drugs_3/'
path_to_load = '/home/juanjo/Work_Postdoc/my_codes_postdoc/GPy_Models/Codes_For_GDSC2_5Cancers/Test_Data_ToPlot_GDSC2_5Cancers/N_drugs_3/SamplingFromSimilarity/'
AUC_per_cell,Emax_per_cell,IC50_per_cell,AUCR2_per_cell,EmaxR2_per_cell,IC50R2_per_cell,AUC_CV_per_cell,Emax_CV_per_cell,IC50_CV_per_cell = np.load(path_to_load+'Test_Metrics_To_Plot_N_drugs_3_SamplingFromSimilarity.pkl',allow_pickle=True)
#IC50R2_per_cell[0][2]=0.0
def get_mean_std(data_per_cell):
    data_std = np.array([np.std(data_per_cell[i][1:]) for i in range(data_per_cell.__len__())])
    data_mean = np.array([np.mean(data_per_cell[i][1:]) for i in range(data_per_cell.__len__())])
    return data_mean,data_std

N_Cells = 4*np.array([12,24,48,96,144])
AUC_mean, AUC_std = get_mean_std(AUC_per_cell)
IC50_mean, IC50_std = get_mean_std(IC50_per_cell)
Emax_mean, Emax_std = get_mean_std(Emax_per_cell)

AUCR2_mean, AUCR2_std = get_mean_std(AUCR2_per_cell)
IC50R2_mean, IC50R2_std = get_mean_std(IC50R2_per_cell)
EmaxR2_mean, EmaxR2_std = get_mean_std(EmaxR2_per_cell)

AUC_CV_mean, AUC_CV_std = get_mean_std(AUC_CV_per_cell)
IC50_CV_mean, IC50_CV_std = get_mean_std(IC50_CV_per_cell)
Emax_CV_mean, Emax_CV_std = get_mean_std(Emax_CV_per_cell)

N_Cells_lin = 4*np.linspace(12,144,1000)
f_meanAUC = pchip_interpolate(N_Cells, AUC_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdAUC = pchip_interpolate(N_Cells, AUC_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')
f_meanEmax = pchip_interpolate(N_Cells, Emax_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdEmax = pchip_interpolate(N_Cells, Emax_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')
f_meanIC50 = pchip_interpolate(N_Cells, IC50_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdIC50 = pchip_interpolate(N_Cells, IC50_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')

f_meanAUC_CV = pchip_interpolate(N_Cells, AUC_CV_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdAUC_CV = pchip_interpolate(N_Cells, AUC_CV_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')
f_meanEmax_CV = pchip_interpolate(N_Cells, Emax_CV_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdEmax_CV = pchip_interpolate(N_Cells, Emax_CV_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')
f_meanIC50_CV = pchip_interpolate(N_Cells, IC50_CV_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdIC50_CV = pchip_interpolate(N_Cells, IC50_CV_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')

f_meanAUCR2 = pchip_interpolate(N_Cells, AUCR2_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdAUCR2 = pchip_interpolate(N_Cells, AUCR2_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')
f_meanEmaxR2 = pchip_interpolate(N_Cells, EmaxR2_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdEmaxR2 = pchip_interpolate(N_Cells, EmaxR2_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')
f_meanIC50R2 = pchip_interpolate(N_Cells, IC50R2_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdIC50R2 = pchip_interpolate(N_Cells, IC50R2_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')

# def myplot_fillbetween(Nfig,N_Cells_lin,f_mean,f_std,title='my title'):
#     plt.figure(Nfig)
#     plt.plot(N_Cells_lin, f_mean,'-',color='black')
#     plt.fill_between(N_Cells_lin, f_mean - f_std, f_mean + f_std,alpha=0.5)
#     plt.title(title)

def myplot_fillbetween(axs,loc,N_Cells_lin,f_mean,f_std,title='my title',ylabel = False,ylim=None):
    axs[loc[0], loc[1]].plot(N_Cells_lin, f_mean,'-',color='black')
    axs[loc[0], loc[1]].fill_between(N_Cells_lin, f_mean - f_std, f_mean + f_std,alpha=0.5)
    axs[loc[0], loc[1]].set_title(title)
    axs[loc[0], loc[1]].set_xlabel("N Data (4-Cancers)")
    if ylabel:
        axs[loc[0], loc[1]].set_ylabel("Mean Absolute Error")
    if ylim:
        axs[loc[0], loc[1]].set_ylim(ylim)

cancer = 'Breast Cancer'
fig, axs = plt.subplots(2, 3,figsize = (15,10))
myplot_fillbetween(axs,[0,0],N_Cells_lin,f_meanAUC,f_stdAUC,title='AUC-Interp. '+cancer+' (9-data in Training)',ylabel = True,ylim=[0.04,0.13])
myplot_fillbetween(axs,[1,0],N_Cells,AUC_mean,AUC_std,title='AUC '+cancer+' (9-data in Training)',ylabel = True,ylim=[0.04,0.13])
myplot_fillbetween(axs,[0,1],N_Cells_lin,f_meanEmax,f_stdEmax,title='Emax-Interp. '+cancer+' (9-data in Training)',ylabel = True,ylim=[0.1,0.17])
myplot_fillbetween(axs,[1,1],N_Cells,Emax_mean,Emax_std,title='Emax '+cancer+' (9-data in Training)',ylabel = True,ylim=[0.1,0.17])
myplot_fillbetween(axs,[0,2],N_Cells_lin,f_meanIC50,f_stdIC50,title='IC50-Interp. '+cancer+' (9-data in Training)',ylabel = True,ylim=[0.0073,0.0343])
myplot_fillbetween(axs,[1,2],N_Cells,IC50_mean,IC50_std,title='IC50 '+cancer+' (9-data in Training)',ylabel = True,ylim=[0.0073,0.0343])

fig, axs1 = plt.subplots(2, 3,figsize = (15,10))
myplot_fillbetween(axs1,[0,0],N_Cells_lin,f_meanAUC_CV,f_stdAUC_CV,title='Validation AUC-Interp. ('+cancer+' 9-data in Train)',ylabel = True,ylim=[0.04,0.095])
myplot_fillbetween(axs1,[1,0],N_Cells,AUC_CV_mean,AUC_CV_std,title='Validation AUC ('+cancer+' 9-data in Training)',ylabel = True,ylim=[0.04,0.095])
myplot_fillbetween(axs1,[0,1],N_Cells_lin,f_meanEmax_CV,f_stdEmax_CV,title='Validation Emax-Interp. ('+cancer+' 9-data in Train)',ylabel = True,ylim=[0.1,0.17])
myplot_fillbetween(axs1,[1,1],N_Cells,Emax_CV_mean,Emax_CV_std,title='Validation Emax ('+cancer+' 9-data in Training)',ylabel = True,ylim=[0.1,0.17])
myplot_fillbetween(axs1,[0,2],N_Cells_lin,f_meanIC50_CV,f_stdIC50_CV,title='Validation IC50-Interp. ('+cancer+' 9-data in Train)',ylabel = True,ylim=[0.04,0.13])
myplot_fillbetween(axs1,[1,2],N_Cells,IC50_CV_mean,IC50_CV_std,title='Validation IC50 ('+cancer+' 9-data in Training)',ylabel = True,ylim=[0.04,0.13])

fig, axs2 = plt.subplots(2, 3,figsize = (15,10))
myplot_fillbetween(axs2,[0,0],N_Cells_lin,f_meanAUCR2,f_stdAUCR2,title='R^2 AUC-Interpolated '+cancer)
myplot_fillbetween(axs2,[1,0],N_Cells,AUCR2_mean,AUCR2_std,title='R^2 AUC '+cancer)
myplot_fillbetween(axs2,[0,1],N_Cells_lin,f_meanEmaxR2,f_stdEmaxR2,title='R^2 Emax-Interpolated '+cancer)
myplot_fillbetween(axs2,[1,1],N_Cells,EmaxR2_mean,EmaxR2_std,title='R^2 Emax '+cancer)
myplot_fillbetween(axs2,[0,2],N_Cells_lin,f_meanIC50R2,f_stdIC50R2,title='R^2 IC50-Interpolated '+cancer)
myplot_fillbetween(axs2,[1,2],N_Cells,IC50R2_mean,IC50R2_std,title='R^2 IC50 '+cancer)

