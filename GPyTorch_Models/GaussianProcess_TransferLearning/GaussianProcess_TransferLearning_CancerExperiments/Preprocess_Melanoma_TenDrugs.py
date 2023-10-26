import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
import os

#_FOLDER = "/home/ac1jjgg/Dataset_5Cancers/GDSC2_EGFR_PI3K_MAPK_Top5cancers/"
_FOLDER = "/home/juanjo/Work_Postdoc/my_codes_postdoc/Dataset_5Cancers/GDSC2_EGFR_PI3K_MAPK_Top5cancers/"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def sigmoid_4_param(x, x0, L, k, d):
    """ Comparing with Dennis Wang's sigmoid:
    x0 -  p - position, correlation with IC50 or EC50
    L = 1 in Dennis Wang's sigmoid, protect from devision by zero if x is too small
    k = -1/s (s -shape parameter)
    d - determines the vertical position of the sigmoid - shift on y axis - better fitting then Dennis Wang's sigmoid

    """
    return ( 1/ (L + np.exp(-k*(x-x0))) + d)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import getopt
import sys

warnings.filterwarnings("ignore")
#os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

class commandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'i:s:k:w:r:p:c:a:n:')
        # opts = dict(opts)
        # print(opts)
        self.N_iter_epoch = 1200    #number of iterations
        self.which_seed = 1011    #change seed to initialise the hyper-parameters
        self.rank = 7
        self.scale = 1
        self.weight = 1
        self.bash = "1"
        self.N_CellLines_perc = 100   #Here we treat this variable as percentage. Try to put this values as multiple of Num_drugs?
        self.sel_cancer = 3
        self.seed_for_N = 1

        for op, arg in opts:
            # print(op,arg)
            if op == '-i':
                self.N_iter_epoch = arg
            if op == '-r':  # (r)and seed
                self.which_seed = arg
            if op == '-k':  # ran(k)
                self.rank = arg
            if op == '-s':  # (r)and seed
                self.scale = arg
            if op == '-p':  # (p)ython bash
                self.bash = arg
            if op == '-w':
                self.weight = arg
            if op == '-c':
                self.N_CellLines_perc = arg
            if op == '-a':
                self.sel_cancer = arg
            if op == '-n':
                self.seed_for_N = arg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
config = commandLine()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
dict_cancers={0:'GDSC2_EGFR_PI3K_MAPK_Breast_1000FR.csv',1:'GDSC2_EGFR_PI3K_MAPK_COAD_1000FR.csv',
              2:'GDSC2_EGFR_PI3K_MAPK_LUAD.csv',3:'GDSC2_EGFR_PI3K_MAPK_melanoma.csv',4:'GDSC2_EGFR_PI3K_MAPK_SCLC.csv'}

#indx_cancer = np.array([0,1,2,3,4])
indx_cancer_train = np.array([int(config.sel_cancer)])
#indx_cancer_train = np.delete(indx_cancer,indx_cancer_test)

name_feat_file = "GDSC2_EGFR_PI3K_MAPK_allfeatures.csv"
name_feat_file_nozero = "GDSC2_EGFR_PI3K_MAPK_features_NonZero_Drugs1036_1061_1373.csv" #"GDSC2_EGFR_PI3K_MAPK_features_NonZero.csv"
Num_drugs = 3

name_for_KLrelevance = dict_cancers[indx_cancer_train[0]]
print(name_for_KLrelevance)

df_to_read = pd.read_csv(_FOLDER + name_for_KLrelevance)#.sample(n=N_CellLines,random_state = rand_state_N)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Split of Training and Testing data"
df_4Cancers_traintest_d1 = df_to_read[df_to_read["DRUG_ID"]==1036]#.sample(n=N_CellLines,random_state = rand_state_N)
df_4Cancers_traintest_d2 = df_to_read[df_to_read["DRUG_ID"]==1061]#.sample(n=N_CellLines,random_state = rand_state_N)
df_4Cancers_traintest_d3 = df_to_read[df_to_read["DRUG_ID"] == 1373]#.sample(n=N_CellLines,random_state=rand_state_N)
df_4Cancers_traintest_d4 = df_to_read[df_to_read["DRUG_ID"] == 1549]
df_4Cancers_traintest_d5 = df_to_read[df_to_read["DRUG_ID"] == 1039]
df_4Cancers_traintest_d6 = df_to_read[df_to_read["DRUG_ID"] == 1560]
df_4Cancers_traintest_d7 = df_to_read[df_to_read["DRUG_ID"] == 1561]
df_4Cancers_traintest_d8 = df_to_read[df_to_read["DRUG_ID"] == 1053]
df_4Cancers_traintest_d9 = df_to_read[df_to_read["DRUG_ID"] == 1057]
df_4Cancers_traintest_d10 = df_to_read[df_to_read["DRUG_ID"] == 1058]
df_4Cancers_traintest_d11 = df_to_read[df_to_read["DRUG_ID"] == 1059]
df_4Cancers_traintest_d12 = df_to_read[df_to_read["DRUG_ID"] == 1060]
df_4Cancers_traintest_d13 = df_to_read[df_to_read["DRUG_ID"] == 1062]
df_4Cancers_traintest_d14 = df_to_read[df_to_read["DRUG_ID"] == 2096]
df_4Cancers_traintest_d15 = df_to_read[df_to_read["DRUG_ID"] == 2045]

for i in range(1, 16):
    print(f"{i}:")
    print(eval("df_4Cancers_traintest_d" + str(i) + "[df_4Cancers_traintest_d" + str(i) + "['COSMIC_ID']==906793]"))

Index_sel = (df_to_read["DRUG_ID"] == 1036) | (df_to_read["DRUG_ID"] == 1061)| (df_to_read["DRUG_ID"] == 1373) \
            | (df_to_read["DRUG_ID"] == 1039) | (df_to_read["DRUG_ID"] == 1560) | (df_to_read["DRUG_ID"] == 1057) \
            | (df_to_read["DRUG_ID"] == 1059)| (df_to_read["DRUG_ID"] == 1062) | (df_to_read["DRUG_ID"] == 2096) \
            | (df_to_read["DRUG_ID"] == 2045)

df_4Cancers_traintest_all = df_to_read[Index_sel]
df_all = df_4Cancers_traintest_all.reset_index().drop(columns=['index'])
df_all = df_all.dropna()

# cell 906793

df_source = df_all[df_all['COSMIC_ID']!=906793].reset_index().drop(columns=['index'])
df_target = df_all[df_all['COSMIC_ID']==906793].reset_index().drop(columns=['index'])
idx_train = np.array([3,4,9])
idx_test = np.delete(np.arange(0,df_target.shape[0]),idx_train)

df_target_test = df_target.iloc[idx_test]
df_target_train = df_target.iloc[idx_train]

df_source_and_target = pd.concat([df_source,df_target_train])

# Here we just check that from the column index 25 the input features start
start_pos_features = 25
print(df_all.columns[start_pos_features])

df_feat = df_all[df_all.columns[start_pos_features:]]
Names_All_features = df_all.columns[start_pos_features:]
Idx_Non_ZeroStd = np.where(df_feat.std()!=0.0)
Names_features_NonZeroStd = Names_All_features[Idx_Non_ZeroStd]


scaler = MinMaxScaler().fit(df_source_and_target[Names_features_NonZeroStd])
X_source_train_feat = scaler.transform(df_source[Names_features_NonZeroStd])
X_target_train_feat = scaler.transform(df_target_train[Names_features_NonZeroStd])
X_target_test_feat = scaler.transform(df_target_test[Names_features_NonZeroStd])

