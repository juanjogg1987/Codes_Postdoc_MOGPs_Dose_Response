import numpy as np
import os

my_n = [4000]   # Number of iteration
#my_d = [1036,1061,1373,1039,1560,1057,1059,1062,2096,2045]
my_i = [0] #np.arange(0,dir_cancer_length[my_t[0]])  # Array to index each cell-line of the Target cancer.
my_r = list(np.arange(1,50)) #[10,15,30,35,40]  # Random seeds
"Template"
"TLMOGP_OverGDSC2_AnyDrugConcentrations_MeanInPrior_AlternativeVersion.py -n 5 -r 35 -p 101"

#path_file = '/rds/general/user/jgiraldo/home/TransferLearning_Results/Jobs_TLMOGP_OneCell_OneDrug_Testing/'
path_file = './Jobs_TLMOGP_OneCell_MultiDrug_Testing/'
if not os.path.exists(path_file):
  #os.mkdir(path_file)   #Use this for a single dir
  os.makedirs(path_file) #Use this for a multiple sub dirs

mycount = 0
for the_i in my_i:
    for the_n in my_n:
        for the_r in my_r:
            "Here we save the Validation Log Loss in path_val in order to have a list of different bashes to select the best model"
            f = open(path_file + 'bash' + str(mycount) + '.sh', "w")
            f.write(f"\n#PBS -l walltime=01:30:00")
            f.write(f"\n#PBS -l select=1:ncpus=40:mem=15gb")
            f.write(f"\n\nmodule load anaconda3/personal\nsource activate py38_pytorch")
            f.write(f"\n\npython /rds/general/user/jgiraldo/home/TLMOGP_MeanInPrior/TLMOGP_OverGDSC2_AnyDrugConcentrations_MeanInPrior_AlternativeVersion.py")
            f.write(f" -n {the_n} -r {the_r} -p {mycount}")
            f.close()
            mycount = mycount + 1