import numpy as np
import pandas as pd

dir_cancer_length = {0:51,1:47,2:65,3:54,4:60}  # This is a dictionary to indicate how many cell-line are available per cancer

sel_cancer_Target = [0]
which_drug = [1036,1061,1373,1039,1560,1057,1059,1062,2096,2045]
idx_CID_Target = 0 #np.arange(0,dir_cancer_length[sel_cancer_Target[0]])  # Array to index each cell-line of the Target cancer.

dict_cancers={0:'GDSC2_EGFR_PI3K_MAPK_Breast_1000FR.csv',1:'GDSC2_EGFR_PI3K_MAPK_COAD_1000FR.csv',
              2:'GDSC2_EGFR_PI3K_MAPK_LUAD.csv',3:'GDSC2_EGFR_PI3K_MAPK_melanoma.csv',4:'GDSC2_EGFR_PI3K_MAPK_SCLC.csv'}

name_file_cancer_target = dict_cancers[int(sel_cancer_Target[0])]
print("Target Cancer:",name_file_cancer_target)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
_FOLDER = "/home/juanjo/Work_Postdoc/my_codes_postdoc/Dataset_5Cancers/GDSC2_EGFR_PI3K_MAPK_Top5cancers/"
#_FOLDER = "/rds/general/user/jgiraldo/home/Dataset_5Cancers/GDSC2_EGFR_PI3K_MAPK_Top5cancers/" #HPC path

df_to_read_target = pd.read_csv(_FOLDER + name_file_cancer_target)#.sample(n=N_CellLines,random_state = rand_state_N)

Index_sel_target = (df_to_read_target["DRUG_ID"] == 1036) | (df_to_read_target["DRUG_ID"] == 1061)| (df_to_read_target["DRUG_ID"] == 1373) \
            | (df_to_read_target["DRUG_ID"] == 1039) | (df_to_read_target["DRUG_ID"] == 1560) | (df_to_read_target["DRUG_ID"] == 1057) \
            | (df_to_read_target["DRUG_ID"] == 1059)| (df_to_read_target["DRUG_ID"] == 1062) | (df_to_read_target["DRUG_ID"] == 2096) \
            | (df_to_read_target["DRUG_ID"] == 2045)

df_TargetCancer_all = df_to_read_target[Index_sel_target]
df_all_target = df_TargetCancer_all.reset_index().drop(columns=['index'])
df_all_target = df_all_target.dropna()

myset_target = set(df_all_target['COSMIC_ID'].values)
myLabels = np.arange(0,myset_target.__len__())
CosmicIDs_All_Target = list(myset_target)
"Here we order the list of target COSMIC_IDs from smallest CosmicID to biggest"
CosmicIDs_All_Target.sort()
CellLine_pos = int(idx_CID_Target) #37
print(f"The CosmicID of the selected Target Cell-line: {CosmicIDs_All_Target[CellLine_pos]}")
CosmicID_target = CosmicIDs_All_Target[CellLine_pos]


#path_home = '/rds/general/user/jgiraldo/home/TransferLearning_Results/'
#path_val = path_home+'Jobs_TLMOGP_OneCell_OneDrug_Testing/TargetCancer'+str(sel_cancer_Target)+'/Drug_'+str(which_drug)+'/CellLine'+str(idx_CID_Target)+'_*/'

path_val = '/home/juanjo/Work_Postdoc/my_codes_postdoc/GPyTorch_Models/GaussianProcess_TransferLearning/GaussianProcess_TransferLearning_CancerExperiments/CellLine0_CID'+str(CosmicID_target)+'/'

#df = pd.read_csv(path_val+'Validation.txt', delimiter=",")
df_val = pd.read_csv(path_val+'Validation.txt', delimiter=",",header=None)
df_test = pd.read_csv(path_val+'Test.txt', delimiter=",",header=None)
myvals = df_val[1].values
mybash = df_val[0].values
ValLogLoss = np.array([float(val[12:]) for val in myvals])
idx_BestModel = np.where(ValLogLoss==ValLogLoss.min())[0][0]
print(ValLogLoss)
print(f"Best Model {mybash[idx_BestModel]} ValLogLoss:{ValLogLoss[idx_BestModel]}")
df_test_best = df_test.iloc[idx_BestModel:idx_BestModel+1].reset_index().drop(columns=['index'])
df_test_best['COSMIC_ID'] = CosmicID_target

#TODO: Create properly the df_test_best file!


