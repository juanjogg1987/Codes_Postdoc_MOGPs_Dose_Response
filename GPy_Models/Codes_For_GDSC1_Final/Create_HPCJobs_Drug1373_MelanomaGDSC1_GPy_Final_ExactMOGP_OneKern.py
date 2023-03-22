
import os

seeds = [1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020]
weights = [0.01,0.1,1.0]
scales = [0.01,0.1,1,3]
ranks = [1,2,3,4,5]   #Since GDSC1 Drug1373 has 5 outputs the maximun rank is 5.

ram = 35
which_HPC = 'sharc'

count = 0

path_to_save = './bash_torchjobs_Drug1373_MelanomaGDSC1_GPy_Final_ExactMOGP_OneKern/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

for myseed in seeds:
    for scale in scales:
        for w in weights:
            for k in ranks:
                f = open(path_to_save + "bash" + str(count) + ".sh", "w+")
                if which_HPC == 'sharc' and count%1 == 0:
                    f.write("#!/bin/bash\n#$ -P rse\n#$ -l rmem=%dG #Ram memory\n\n"  
                            "module load apps/python/conda\nsource activate py38_gpflow\npython /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC1_Final/Drug1373_MelanomaGDSC1_GPy_Final_ExactMOGP_OneKern.py -i 1500 -s %.4f -k %d -w %.4f -r %d -p %d" % (
                            ram, scale,k,w, myseed,count))
                # if which_HPC == 'sharc' and count%2 != 0:
                #     f.write("#!/bin/bash\n#$ -P bioinf-core\n#$ -l rmem=%dG #Ram memory\n\n"
                #             "module load apps/python/conda\nsource activate py38_gpflow\npython /home/ac1jjgg/GP_Models_MarinaData/All_Drugs_GPAdditiveKern_Test1_0_HPC.py -l %d -b %d -m %d -i 1500 -w %.4f -s %.4f -r %d" % (
                #             ram, l, minbatch, m, w, s, myseed))

                elif which_HPC == 'bessemer':
                    "ToDo"
                    pass
                f.close()
                count = count + 1