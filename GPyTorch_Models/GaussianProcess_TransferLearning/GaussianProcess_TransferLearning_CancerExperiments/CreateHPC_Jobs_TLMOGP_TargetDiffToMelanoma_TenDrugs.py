import numpy as np
import os

dir_cancer_length = {0:51,1:47,2:65,3:54,4:60}  # This is a dictionary to indicate how many cell-line are available per cancer

my_n = [180]   # Number of iteration
my_r = [10,15,30,35,40]  # Random seeds
my_w = [0.3, 0.5, 1.0, 2.0]   # Weights for hyper-parameters initializations
my_s = [3]   # This is the Source cancer domain. The labels are 0:Breast, 1:Coad, 2:Luad, 3:Melanoma and 4:SCLC
my_t = [0]   # This is the Target cancer domain. The same labels as per the Source.
my_i = np.arange(0,dir_cancer_length[my_t[0]])  # Array to index each cell-line of the Target cancer.
my_d = [1036,1061,1373,1039,1560,1057,1059,1062,2096,2045]

"Template"
"TLMOGP_TargetDiffToMelanoma_TenDrugs_CrossVal.py -n 5 -r 35 -p 101 -w 0.3 -s 3 -t 0 -i 5 -d 1059"

#path_file = '/rds/general/user/jgiraldo/home/TransferLearning_Results/Jobs_TLMOGP_OneCell_OneDrug_Testing/'
path_file = './Jobs_TLMOGP_OneCell_OneDrug_Testing/'
if not os.path.exists(path_file):
  #os.mkdir(path_file)   #Use this for a single dir
  os.makedirs(path_file) #Use this for a multiple sub dirs

mycount = 0
for the_n in my_n:
    for the_r in my_r:
        for the_w in my_w:
            for the_s in my_s:
                for the_t in my_t:
                    for the_i in my_i:
                        for the_d in my_d:
                            "Here we save the Validation Log Loss in path_val in order to have a list of different bashes to select the best model"
                            f = open(path_file + 'bash' + str(mycount) + '.sh', "w")
                            f.write(f"\n#PBS -l walltime=01:30:00")
                            f.write(f"\n#PBS -l select=1:ncpus=32:mem=8gb")
                            f.write(f"\n\nmodule load anaconda3/personal\nsource activate py38_pytorch")
                            f.write(f"\n\npython /rds/general/user/jgiraldo/home/TransferLearning_Codes/TLMOGP_TargetDiffToMelanoma_TenDrugs_CrossVal.py")
                            f.write(f" -n {the_n} -r {the_r} -p {mycount} -w {the_w} -s {the_s} -t {the_t} -i {the_i} -d {the_d}")
                            f.close()
                            mycount = mycount + 1