
import os

import numpy as np

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import getopt
import sys

class commandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'c:')
        self.Test_cancer = 0

        for op, arg in opts:
            # print(op,arg)
            if op == '-c':
                self.Test_cancer = arg
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
config = commandLine()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

seeds = [1012,1013,1014,1015,1016]
weights = [0.05,0.3,1.5]
scales = [0.05,0.3,1.5]
ranks = [7]   #Since GDSC2 has 7 outputs the maximun rank is 7.
N_CellLines = [100]   #This is the percentage of dose responses I want to use for Training
Seeds_for_N = [1]
Test_cancers = [int(config.Test_cancer)]

#path_file = '/rds/general/user/jgiraldo/home/TransferLearning_Results/Jobs_TLMOGP_OneCell_OneDrug_Testing/'
path_file = './JobsRebuttal_Cancer'+str(Test_cancers[0])+'_Train1Cancer_GPyjobs_N_Drugs_GPy_ExactMOGP_ProdKern/'
if not os.path.exists(path_file):
    # os.mkdir(path_file)   #Use this for a single dir
    os.makedirs(path_file)  #Use this for a multiple sub dirs

mycount = 0
for myseed in seeds:
    for scale in scales:
        for w in weights:
            for k in ranks:
                for N_Cells in N_CellLines:
                    for Seed_N in Seeds_for_N:
                        for sel_cancer in Test_cancers:
                            "Here we save the Validation Log Loss in path_val in order to have a list of different bashes to select the best model"
                            f = open(path_file + 'bash' + str(mycount) + '.sh', "w")
                            f.write(f"\n#PBS -l walltime=01:30:00")
                            f.write(f"\n#PBS -l select=1:ncpus=32:mem=24gb")
                            f.write(f"\n\nmodule load anaconda3/personal\nsource activate py38_gpy")
                            f.write(
                                f"\n\npython /rds/general/user/jgiraldo/home/PaperPrecisionOncology_Rebuttal/N_Drugs_Rebuttal_From10Cancers_Train1Cancer_GPy_ExactMOGP_ProdKern.py")
                            f.write(f" -i {800} -r {myseed} -k {k} -s {scale} -p {mycount} -w {w} -c {N_Cells} -a {sel_cancer} -n {Seed_N}")
                            f.close()
                            mycount = mycount + 1
