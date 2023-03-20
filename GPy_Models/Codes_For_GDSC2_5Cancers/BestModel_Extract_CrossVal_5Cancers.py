import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df_CrossVal = pd.read_csv("Metrics_CrossVal.txt",header=None,sep=' ')
IC50_CV = np.array([float(df_CrossVal[4].values[i].split("=")[1].split("(")[0]) for i in range(df_CrossVal.shape[0])])
AUC_CV = np.array([float(df_CrossVal[5].values[i].split("=")[1].split("(")[0]) for i in range(df_CrossVal.shape[0])])
Emax_CV = np.array([float(df_CrossVal[7].values[i].split("=")[1].split("(")[0]) for i in range(df_CrossVal.shape[0])])
bash_CV = np.array([int(df_CrossVal[0].values[i].split("h")[1]) for i in range(df_CrossVal.shape[0])])