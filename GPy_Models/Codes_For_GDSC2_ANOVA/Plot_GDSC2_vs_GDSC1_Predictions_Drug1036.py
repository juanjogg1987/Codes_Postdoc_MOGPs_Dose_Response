import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.close('all')

MyFolder = "/home/juanjo/Work_Postdoc/my_codes_postdoc/GPy_Models/Codes_For_GDSC2_ANOVA/"
df_GDSC1_ToPred_GDSC2 = pd.read_csv(MyFolder+"Metrics_Drug1036_ExactMOGP_TrainedGDSC1_PredictGDSC2.csv")
df_GDSC2_ToPred_GDSC1 = pd.read_csv(MyFolder+"Metrics_Drug1036_ExactMOGP_TrainedGDSC2_PredictGDSC1.csv")

df_GDSC2_ToPred_GDSC1_Sorted = df_GDSC2_ToPred_GDSC1.sort_values(by="COSMIC_ID")
df_GDSC1_ToPred_GDSC2_Sorted = df_GDSC1_ToPred_GDSC2.sort_values(by="COSMIC_ID")

df_G2_To_G1_ID = df_GDSC2_ToPred_GDSC1_Sorted["COSMIC_ID"].values
df_G1_To_G2_ID = df_GDSC1_ToPred_GDSC2_Sorted["COSMIC_ID"].values

IC50_G2_To_G1 = df_GDSC2_ToPred_GDSC1_Sorted["IC50_MOGP"].values
IC50_G1_To_G2 = df_GDSC1_ToPred_GDSC2_Sorted["IC50_MOGP"].values
AUC_G2_To_G1 = df_GDSC2_ToPred_GDSC1_Sorted["AUC_MOGP"].values
AUC_G1_To_G2 = df_GDSC1_ToPred_GDSC2_Sorted["AUC_MOGP"].values
Emax_G2_To_G1 = df_GDSC2_ToPred_GDSC1_Sorted["Emax_MOGP"].values
Emax_G1_To_G2 = df_GDSC1_ToPred_GDSC2_Sorted["Emax_MOGP"].values
"Below the label values"
IC50_G2_To_G1_true = df_GDSC2_ToPred_GDSC1_Sorted["IC50_s4"].values
IC50_G1_To_G2_true = df_GDSC1_ToPred_GDSC2_Sorted["IC50_s4"].values
AUC_G2_To_G1_true = df_GDSC2_ToPred_GDSC1_Sorted["AUC_s4"].values
AUC_G1_To_G2_true = df_GDSC1_ToPred_GDSC2_Sorted["AUC_s4"].values
Emax_G2_To_G1_true = df_GDSC2_ToPred_GDSC1_Sorted["Emax_s4"].values
Emax_G1_To_G2_true = df_GDSC1_ToPred_GDSC2_Sorted["Emax_s4"].values

ind_G2 = []
ind_G1 = []

for i in range(df_G2_To_G1_ID.shape[0]):
    myaux = np.where(df_G2_To_G1_ID[i] == df_G1_To_G2_ID)
    if myaux[0].shape[0] != 0:
        ind_G2.append(myaux[0][0])
        ind_G1.append(i)

from sklearn.metrics import r2_score
import scipy

R2_IC50 = r2_score(np.clip(IC50_G1_To_G2[ind_G2],0,1),np.clip(IC50_G2_To_G1[ind_G1],0,1))
R2_AUC = r2_score(AUC_G1_To_G2[ind_G2],AUC_G2_To_G1[ind_G1])
R2_Emax = r2_score(Emax_G1_To_G2[ind_G2],Emax_G2_To_G1[ind_G1])

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
plt.figure(1,figsize=(6.5*1.1,5*1.1))
#plt.scatter(IC50_G1_To_G2[ind_G2],IC50_G2_To_G1[ind_G1])
plt.scatter(np.clip(IC50_G1_To_G2[ind_G2],0,1),np.clip(IC50_G2_To_G1[ind_G1],0,1),linewidths=1.5,color='blue')
plt.ylabel("Prediction over GDSC1 (Trained on GDSC2)")
plt.xlabel("Prediction over GDSC2 (Trained on GDSC1)")
plt.xlim([0.2,1.1])
plt.ylim([0.2,1.1])
plt.legend([f"R2={R2_IC50:.2f}\nPearson = {scipy.stats.pearsonr(IC50_G1_To_G2[ind_G2],IC50_G2_To_G1[ind_G1])[0]:.2f}\nSpearman = {scipy.stats.spearmanr(IC50_G1_To_G2[ind_G2],IC50_G2_To_G1[ind_G1])[0]:.2f}"])
plt.figure(2,figsize=(6.5*1.1,5*1.1))
plt.scatter(AUC_G1_To_G2[ind_G2],AUC_G2_To_G1[ind_G1],linewidths=1.5,color='blue')
plt.legend([f"R2={R2_AUC:.2f}\nPearson = {scipy.stats.pearsonr(AUC_G1_To_G2[ind_G2],AUC_G2_To_G1[ind_G1])[0]:.2f}\nSpearman = {scipy.stats.spearmanr(AUC_G1_To_G2[ind_G2],AUC_G2_To_G1[ind_G1])[0]:.2f}"])
plt.ylabel("Prediction over GDSC1 (Trained on GDSC2)")
plt.xlabel("Prediction over GDSC2 (Trained on GDSC1)")
plt.xlim([0.2,1])
plt.ylim([0.2,1])
plt.figure(3,figsize=(6.5*1.1,5*1.1))
plt.scatter(Emax_G1_To_G2[ind_G2],Emax_G2_To_G1[ind_G1],linewidths=1.5,color='blue')
plt.legend([f"R2={R2_Emax:.2f}\nPearson = {scipy.stats.pearsonr(Emax_G1_To_G2[ind_G2],Emax_G2_To_G1[ind_G1])[0]:.2f}\nSpearman = {scipy.stats.spearmanr(Emax_G1_To_G2[ind_G2],Emax_G2_To_G1[ind_G1])[0]:.2f}"])
plt.ylabel("Prediction over GDSC1 (Trained on GDSC2)")
plt.xlabel("Prediction over GDSC2 (Trained on GDSC1)")
plt.xlim([0.2,1])
plt.ylim([0.2,1])

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
R2_IC50_true_G2_G2 = r2_score(np.clip(IC50_G1_To_G2[ind_G2],0,1),np.clip(IC50_G1_To_G2_true[ind_G2],0,1))
R2_AUC_true_G2_G2 = r2_score(AUC_G1_To_G2[ind_G2],AUC_G1_To_G2_true[ind_G2])
R2_Emax_true_G2_G2 = r2_score(Emax_G1_To_G2[ind_G2],Emax_G1_To_G2_true[ind_G2])
plt.figure(1)
#plt.scatter(IC50_G1_To_G2[ind_G2],IC50_G2_To_G1[ind_G1])
plt.scatter(np.clip(IC50_G1_To_G2[ind_G2],0,1),np.clip(IC50_G1_To_G2_true[ind_G2],0,1),linewidths=0.7,marker='x',color='orangered')
plt.ylabel("Prediction over GDSC1 (Trained on GDSC2)")
plt.xlabel("Prediction over GDSC2 (Trained on GDSC1)")
plt.xlim([0.15,1.1])
plt.ylim([0.15,1.1])
plt.legend([f"R2={R2_IC50:.2f}\nPearson = {scipy.stats.pearsonr(IC50_G1_To_G2[ind_G2],IC50_G2_To_G1[ind_G1])[0]:.2f}\nSpearman = {scipy.stats.spearmanr(IC50_G1_To_G2[ind_G2],IC50_G2_To_G1[ind_G1])[0]:.2f}",f"R2={R2_IC50_true_G2_G2:.2f}\nPearson = {scipy.stats.pearsonr(IC50_G1_To_G2[ind_G2],IC50_G1_To_G2_true[ind_G2])[0]:.2f}\nSpearman = {scipy.stats.spearmanr(IC50_G1_To_G2[ind_G2],IC50_G1_To_G2_true[ind_G2])[0]:.2f}"])
plt.figure(2)
plt.scatter(AUC_G1_To_G2[ind_G2],AUC_G1_To_G2_true[ind_G2],linewidths=0.7,marker='x',color='orangered')
plt.legend([f"R2={R2_AUC:.2f}\nPearson = {scipy.stats.pearsonr(AUC_G1_To_G2[ind_G2],AUC_G2_To_G1[ind_G1])[0]:.2f}\nSpearman = {scipy.stats.spearmanr(AUC_G1_To_G2[ind_G2],AUC_G2_To_G1[ind_G1])[0]:.2f}",f"R2={R2_AUC_true_G2_G2:.2f}\nPearson = {scipy.stats.pearsonr(AUC_G1_To_G2[ind_G2],AUC_G1_To_G2_true[ind_G2])[0]:.2f}\nSpearman = {scipy.stats.spearmanr(AUC_G1_To_G2[ind_G2],AUC_G1_To_G2_true[ind_G2])[0]:.2f}"])
plt.ylabel("Prediction over GDSC1 (Trained on GDSC2)")
plt.xlabel("Prediction over GDSC2 (Trained on GDSC1)")
plt.xlim([0.15,1])
plt.ylim([0.15,1])
plt.figure(3)
plt.scatter(Emax_G1_To_G2[ind_G2],Emax_G1_To_G2_true[ind_G2],linewidths=0.9,marker='x',color='orangered')
plt.legend([f"R2={R2_Emax:.2f}\nPearson = {scipy.stats.pearsonr(Emax_G1_To_G2[ind_G2],Emax_G2_To_G1[ind_G1])[0]:.2f}\nSpearman = {scipy.stats.spearmanr(Emax_G1_To_G2[ind_G2],Emax_G2_To_G1[ind_G1])[0]:.2f}",f"R2={R2_Emax_true_G2_G2:.2f}\nPearson = {scipy.stats.pearsonr(Emax_G1_To_G2[ind_G2],Emax_G1_To_G2_true[ind_G2])[0]:.2f}\nSpearman = {scipy.stats.spearmanr(Emax_G1_To_G2[ind_G2],Emax_G1_To_G2_true[ind_G2])[0]:.2f}"])
plt.ylabel("Prediction over GDSC1 (Trained on GDSC2)")
plt.xlabel("Prediction over GDSC2 (Trained on GDSC1)")
plt.xlim([0.15,1.1])
plt.ylim([0.15,1.1])

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
R2_IC50_true_G1_G1 = r2_score(np.clip(IC50_G2_To_G1_true[ind_G1],0,1),np.clip(IC50_G2_To_G1[ind_G1],0,1))
R2_AUC_true_G1_G1 = r2_score(AUC_G2_To_G1_true[ind_G1],AUC_G2_To_G1[ind_G1])
R2_Emax_true_G1_G1 = r2_score(Emax_G2_To_G1_true[ind_G1],Emax_G2_To_G1[ind_G1])
plt.figure(1)
#plt.scatter(IC50_G1_To_G2[ind_G2],IC50_G2_To_G1[ind_G1])
plt.scatter(np.clip(IC50_G2_To_G1_true[ind_G1],0,1),np.clip(IC50_G2_To_G1[ind_G1],0,1),linewidths=0.9,marker='x',color='limegreen')
plt.ylabel("Prediction over GDSC1 (Trained on GDSC2)")
plt.xlabel("Prediction over GDSC2 (Trained on GDSC1)")
plt.title("IC50 Prediction performance of MOGP Trained on GDSC2 vs GDSC1")
plt.xlim([-0.1,1.15])
plt.ylim([-0.1,1.15])
#plt.legend([f"R2={R2_IC50:.2f}\nPearson = {scipy.stats.pearsonr(IC50_G1_To_G2[ind_G2],IC50_G2_To_G1[ind_G1])[0]:.2f}\nSpearman = {scipy.stats.spearmanr(IC50_G1_To_G2[ind_G2],IC50_G2_To_G1[ind_G1])[0]:.2f}",f"R2={R2_IC50_true_G2_G2:.2f}\nPearson = {scipy.stats.pearsonr(IC50_G1_To_G2[ind_G2],IC50_G1_To_G2_true[ind_G2])[0]:.2f}\nSpearman = {scipy.stats.spearmanr(IC50_G1_To_G2[ind_G2],IC50_G1_To_G2_true[ind_G2])[0]:.2f}"])
plt.legend([f"R2={R2_IC50:.2f}\nPearson = {scipy.stats.pearsonr(IC50_G1_To_G2[ind_G2],IC50_G2_To_G1[ind_G1])[0]:.2f}\nSpearman = {scipy.stats.spearmanr(IC50_G1_To_G2[ind_G2],IC50_G2_To_G1[ind_G1])[0]:.2f}",f"GDSC1-True",f"GDSC2-True"])
plt.figure(2)
plt.scatter(AUC_G2_To_G1_true[ind_G1],AUC_G2_To_G1[ind_G1],linewidths=0.9,marker='x',color='limegreen')
#plt.legend([f"R2={R2_AUC:.2f}\nPearson = {scipy.stats.pearsonr(AUC_G1_To_G2[ind_G2],AUC_G2_To_G1[ind_G1])[0]:.2f}\nSpearman = {scipy.stats.spearmanr(AUC_G1_To_G2[ind_G2],AUC_G2_To_G1[ind_G1])[0]:.2f}",f"R2={R2_AUC_true_G2_G2:.2f}\nPearson = {scipy.stats.pearsonr(AUC_G1_To_G2[ind_G2],AUC_G1_To_G2_true[ind_G2])[0]:.2f}\nSpearman = {scipy.stats.spearmanr(AUC_G1_To_G2[ind_G2],AUC_G1_To_G2_true[ind_G2])[0]:.2f}"])
plt.legend([f"R2={R2_AUC:.2f}\nPearson = {scipy.stats.pearsonr(AUC_G1_To_G2[ind_G2],AUC_G2_To_G1[ind_G1])[0]:.2f}\nSpearman = {scipy.stats.spearmanr(AUC_G1_To_G2[ind_G2],AUC_G2_To_G1[ind_G1])[0]:.2f}",f"GDSC1-True",f"GDSC2-True"])
plt.ylabel("Prediction over GDSC1 (Trained on GDSC2)")
plt.xlabel("Prediction over GDSC2 (Trained on GDSC1)")
plt.title("AUC Prediction performance of MOGP Trained on GDSC2 vs GDSC1")
plt.xlim([-0.1,1.15])
plt.ylim([-0.1,1.15])
plt.figure(3)
plt.scatter(Emax_G2_To_G1_true[ind_G1],Emax_G2_To_G1[ind_G1],linewidths=0.9,marker='x',color='limegreen')  #lime
#plt.legend([f"R2={R2_Emax:.2f}\nPearson = {scipy.stats.pearsonr(Emax_G1_To_G2[ind_G2],Emax_G2_To_G1[ind_G1])[0]:.2f}\nSpearman = {scipy.stats.spearmanr(Emax_G1_To_G2[ind_G2],Emax_G2_To_G1[ind_G1])[0]:.2f}",f"R2={R2_Emax_true_G2_G2:.2f}\nPearson = {scipy.stats.pearsonr(Emax_G1_To_G2[ind_G2],Emax_G1_To_G2_true[ind_G2])[0]:.2f}\nSpearman = {scipy.stats.spearmanr(Emax_G1_To_G2[ind_G2],Emax_G1_To_G2_true[ind_G2])[0]:.2f}"])
plt.legend([f"R2={R2_Emax:.2f}\nPearson = {scipy.stats.pearsonr(Emax_G1_To_G2[ind_G2],Emax_G2_To_G1[ind_G1])[0]:.2f}\nSpearman = {scipy.stats.spearmanr(Emax_G1_To_G2[ind_G2],Emax_G2_To_G1[ind_G1])[0]:.2f}",f"GDSC1-True",f"GDSC2-True"])
plt.ylabel("Prediction over GDSC1 (Trained on GDSC2)")
plt.xlabel("Prediction over GDSC2 (Trained on GDSC1)")
plt.title("Emax Prediction performance of MOGP Trained on GDSC2 vs GDSC1")
plt.xlim([-0.1,1.15])
plt.ylim([-0.1,1.15])