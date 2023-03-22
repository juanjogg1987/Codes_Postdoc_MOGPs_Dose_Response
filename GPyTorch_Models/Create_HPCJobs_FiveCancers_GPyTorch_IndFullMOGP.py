
import os

seeds = [1010,1011,1012,1013,1014] #,1015,1016,1017,1018,1019]
scales = [0.01,0.1,1.0,3]
spli_dim = [2,5,10,15,20]
ram = 35
which_HPC = 'sharc'

count = 0

path_to_save = './bash_torchjobs_FiveCancers_IndFullMOGP/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

for myseed in seeds:
        for s in scales:
            for d in spli_dim:
                f = open(path_to_save + "bash" + str(count) + ".sh", "w+")
                if which_HPC == 'sharc' and count%1 == 0:
                    f.write("#!/bin/bash\n#$ -P rse\n#$ -l rmem=%dG #Ram memory\n\n"  
                            "module load apps/python/conda\nsource activate py38_gpflow\npython /home/ac1jjgg/MOGP_GPyTorch/All_Drugs_FiveCancers_SmallData_GPyTorch_IndFullMOGP.py -i 300 -s %f -r %d -d %d -p %d" % (
                            ram, s, myseed,d,count))

                elif which_HPC == 'bessemer':
                    "ToDo"
                    pass
                f.close()
                count = count + 1
