
import os
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

seeds = [1012,1013]
weights = [0.05,0.3,1.5]
scales = [0.05,0.3,1.5]
ranks = [2,3,4,5,6,7]   #Since GDSC2 has 7 outputs the maximun rank is 7.
N_CellLines = [12,24,48,96,144]
Seeds_for_N = [1,2,3,4,5,6]
Test_cancers = [int(config.Test_cancer)]

ram = 35
which_HPC = 'sharc'

count = 0

path_to_save = './bash_Cancer'+str(Test_cancers[0])+'_GPyjobs_Three_Drugs_5Cancers_GPy_ExactMOGP_ProdKern/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

for myseed in seeds:
    for scale in scales:
        for w in weights:
            for k in ranks:
                for N_Cells in N_CellLines:
                    for Seed_N in Seeds_for_N:
                        for sel_cancer in Test_cancers:
                            f = open(path_to_save + "bash" + str(count) + ".sh", "w+")
                            if which_HPC == 'sharc' and count%1 == 0:
                                f.write("#!/bin/bash\n#$ -P rse\n#$ -l rmem=%dG #Ram memory\n\n"  
                                        "module load apps/python/conda\nsource activate py38_gpflow\npython /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_5Cancers/Three_Drugs_5Cancers_GPy_ExactMOGP_ProdKern.py -i 1500 -s %.4f -k %d -w %.4f -r %d -p %d -c %d -a %d -n %d" % (
                                        ram, scale,k,w, myseed,count,N_Cells,sel_cancer,Seed_N))

                            elif which_HPC == 'bessemer':
                                "ToDo"
                                pass
                            f.close()
                            count = count + 1
