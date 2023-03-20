import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import pchip_interpolate
plt.close('all')
path_to_load = '/home/juanjo/Work_Postdoc/my_codes_postdoc/GPy_Models/Codes_For_GDSC2_5Cancers/Test_Data_ToPlot_GDSC2_5Cancers/'
AUC_per_cell,Emax_per_cell,IC50_per_cell,AUCR2_per_cell,EmaxR2_per_cell,IC50R2_per_cell = np.load(path_to_load+'Test_Metrics_To_Plot_LM.pkl',allow_pickle=True)
#IC50R2_per_cell[0][2]=0.0
def get_mean_std(data_per_cell):
    data_std = np.array([np.std(data_per_cell[i][1:]) for i in range(data_per_cell.__len__())])
    data_mean = np.array([np.mean(data_per_cell[i][1:]) for i in range(data_per_cell.__len__())])
    return data_mean,data_std

N_Cells = np.array([20,40,80,160])
AUC_mean, AUC_std = get_mean_std(AUC_per_cell)
IC50_mean, IC50_std = get_mean_std(IC50_per_cell)
Emax_mean, Emax_std = get_mean_std(Emax_per_cell)

AUCR2_mean, AUCR2_std = get_mean_std(AUCR2_per_cell)
IC50R2_mean, IC50R2_std = get_mean_std(IC50R2_per_cell)
EmaxR2_mean, EmaxR2_std = get_mean_std(EmaxR2_per_cell)

N_Cells_lin = np.linspace(20,160,1000)
f_meanAUC = pchip_interpolate(N_Cells, AUC_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdAUC = pchip_interpolate(N_Cells, AUC_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')
f_meanEmax = pchip_interpolate(N_Cells, Emax_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdEmax = pchip_interpolate(N_Cells, Emax_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')
f_meanIC50 = pchip_interpolate(N_Cells, IC50_mean, N_Cells_lin) #interp1d(N_Cells, AUC_mean, kind='quadratic')
f_stdIC50 = pchip_interpolate(N_Cells, IC50_std, N_Cells_lin) #interp1d(N_Cells, AUC_std, kind='quadratic')

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

def myplot_fillbetween(axs,loc,N_Cells_lin,f_mean,f_std,title='my title'):
    axs[loc[0], loc[1]].plot(N_Cells_lin, f_mean,'-',color='black')
    axs[loc[0], loc[1]].fill_between(N_Cells_lin, f_mean - f_std, f_mean + f_std,alpha=0.5)
    axs[loc[0], loc[1]].set_title(title)

cancer = 'Breast Cancer'
fig, axs = plt.subplots(2, 3,figsize = (15,10))
myplot_fillbetween(axs,[0,0],N_Cells_lin,f_meanAUC,f_stdAUC,title='MAE AUC-Interpolated '+cancer)
myplot_fillbetween(axs,[1,0],N_Cells,AUC_mean,AUC_std,title='MAE AUC '+cancer)
myplot_fillbetween(axs,[0,1],N_Cells_lin,f_meanEmax,f_stdEmax,title='MAE Emax-Interpolated '+cancer)
myplot_fillbetween(axs,[1,1],N_Cells,Emax_mean,Emax_std,title='MAE Emax '+cancer)
myplot_fillbetween(axs,[0,2],N_Cells_lin,f_meanIC50,f_stdIC50,title='MSE IC50-Interpolated '+cancer)
myplot_fillbetween(axs,[1,2],N_Cells,IC50_mean,IC50_std,title='MSE IC50 '+cancer)

fig, axs2 = plt.subplots(2, 3,figsize = (15,10))
myplot_fillbetween(axs2,[0,0],N_Cells_lin,f_meanAUCR2,f_stdAUCR2,title='R^2 AUC-Interpolated '+cancer)
myplot_fillbetween(axs2,[1,0],N_Cells,AUCR2_mean,AUCR2_std,title='R^2 AUC '+cancer)
myplot_fillbetween(axs2,[0,1],N_Cells_lin,f_meanEmaxR2,f_stdEmaxR2,title='R^2 Emax-Interpolated '+cancer)
myplot_fillbetween(axs2,[1,1],N_Cells,EmaxR2_mean,EmaxR2_std,title='R^2 Emax '+cancer)
myplot_fillbetween(axs2,[0,2],N_Cells_lin,f_meanIC50R2,f_stdIC50R2,title='R^2 IC50-Interpolated '+cancer)
myplot_fillbetween(axs2,[1,2],N_Cells,IC50R2_mean,IC50R2_std,title='R^2 IC50 '+cancer)
