
import os

seeds = [1010,1011,1012] #,1015,1016,1017,1018,1019]
myranks = [2,3,5,10]
scales = [0.01,0.1,1.0,3]
spli_dim = [2,5,10,20]
ram = 35
which_HPC = 'sharc'

count = 0

path_to_save = './bash_torchjobs_SmallData_ExactMOGP/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

for myseed in seeds:
        for rank in myranks:
            for s in scales:
                for d in spli_dim:
                    f = open(path_to_save + "bash" + str(count) + ".sh", "w+")
                    if which_HPC == 'sharc' and count%1 == 0:
                        f.write("#!/bin/bash\n#$ -P rse\n#$ -l rmem=%dG #Ram memory\n\n"  
                                "module load apps/python/conda\nsource activate py38_gpflow\npython /home/ac1jjgg/MOGP_GPyTorch/All_Drugs_LungCancer_SmallData_GPyTorch_ExactMOGP.py -i 1500 -s %.4f -k %d -r %d -d %d -p %d" % (
                                ram, s, rank, myseed,d,count))

                    elif which_HPC == 'bessemer':
                        "ToDo"
                        pass
                    f.close()
                    count = count + 1
