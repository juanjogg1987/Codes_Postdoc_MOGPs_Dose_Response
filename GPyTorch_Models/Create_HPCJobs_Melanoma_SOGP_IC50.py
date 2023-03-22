
import os

seeds = [1010,1011,1012,1015,1016,1017,1018,1019]
spli_dim = [2,5,10,15,20]
scales = [0.01,0.1,1,3]
ram = 20
which_HPC = 'sharc'

count = 0

path_to_save = './bash_torchjobs_Melanoma_SOGP_IC50/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

for myseed in seeds:
    for scale in scales:
        for d in spli_dim:
            f = open(path_to_save + "bash" + str(count) + ".sh", "w+")
            if which_HPC == 'sharc' and count%1 == 0:
                f.write("#!/bin/bash\n#$ -P rse\n#$ -l rmem=%dG #Ram memory\n\n"  
                        "module load apps/python/conda\nsource activate py38_gpflow\npython /home/ac1jjgg/SOGP_GPyTorch/All_Drugs_Melanoma_GPyTorch_SOGP_IC50.py -i 1200 -s %.4f -r %d -d %d -p %d" % (
                        ram, scale, myseed,d,count))
            # if which_HPC == 'sharc' and count%2 != 0:
            #     f.write("#!/bin/bash\n#$ -P bioinf-core\n#$ -l rmem=%dG #Ram memory\n\n"
            #             "module load apps/python/conda\nsource activate py38_gpflow\npython /home/ac1jjgg/GP_Models_MarinaData/All_Drugs_GPAdditiveKern_Test1_0_HPC.py -l %d -b %d -m %d -i 1500 -w %.4f -s %.4f -r %d" % (
            #             ram, l, minbatch, m, w, s, myseed))

            elif which_HPC == 'bessemer':
                "ToDo"
                pass
            f.close()
            count = count + 1
