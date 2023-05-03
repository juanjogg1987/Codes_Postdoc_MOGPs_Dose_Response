import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import pchip_interpolate
#plt.close('all')

_FOLDER = '/home/juanjo/Work_Postdoc/my_codes_postdoc/FilesCSV_Predict_5Cancers_IncreasingCellLines/'
cancer_names = {0:'breast_cancer',1:'COAD_cancer',2:'LUAD_cancer',3:'melanoma_cancer',4:'SCLC_cancer'}

sel_cancer = 2
N5th_cancer = 45
cancer = cancer_names[sel_cancer]
Nseed = 1
N_cells = 144 * 4
#Ntotal_Cells = int(N_cells)*4 + int(N5th_cancer)

path_to_read = _FOLDER + cancer+'/N5th_CancerInTrain_'+str(N5th_cancer)+'/NTrain_'+str(N_cells)+'_plus_'+str(N5th_cancer)+'/MOGP_Predict_C'+str(sel_cancer)+'_Train_'+str(N_cells)+'_plus_'+str(N5th_cancer)+'_seed'+str(Nseed)+'.csv'

df_pred = pd.read_csv(path_to_read)
AE_per_dose = np.abs(df_pred[df_pred.columns[15:22]].values-df_pred[df_pred.columns[26:33]].values)

plt.figure(3)
plt.boxplot(AE_per_dose)
plt.ylim([-0.05,1.1])