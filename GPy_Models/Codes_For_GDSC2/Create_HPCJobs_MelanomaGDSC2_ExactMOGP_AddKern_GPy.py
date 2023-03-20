
import os

seeds = [1010,1011]
weights = [0.01,0.1,1.0]
scales = [0.01,0.1,1,3]
ranks = [1,2,3,4,5,6,7]   #Since GDSC2 has 7 outputs the maximun rank is 7.
split_dims = [2,3,4,5,10]

ram = 35
which_HPC = 'sharc'

count = 0

path_to_save = './bash_torchjobs_MelanomaGDSC2_GPy_ExactMOGP_AddKern/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

for myseed in seeds:
    for scale in scales:
        for w in weights:
            for d in split_dims:
                for k in ranks:
                    f = open(path_to_save + "bash" + str(count) + ".sh", "w+")
                    if which_HPC == 'sharc' and count%1 == 0:
                        f.write("#!/bin/bash\n#$ -P rse\n#$ -l rmem=%dG #Ram memory\n\n"  
                                "module load apps/python/conda\nsource activate py38_gpflow\npython /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2/All_Drugs_MelanomaGDSC2_GPy_ExactMOGP_AddKern.py -i 1500 -s %.4f -k %d -w %.4f -r %d -d %d -p %d" % (
                                ram, scale,k,w, myseed,d,count))
                    # if which_HPC == 'sharc' and count%2 != 0:
                    #     f.write("#!/bin/bash\n#$ -P bioinf-core\n#$ -l rmem=%dG #Ram memory\n\n"
                    #             "module load apps/python/conda\nsource activate py38_gpflow\npython /home/ac1jjgg/GP_Models_MarinaData/All_Drugs_GPAdditiveKern_Test1_0_HPC.py -l %d -b %d -m %d -i 1500 -w %.4f -s %.4f -r %d" % (
                    #             ram, l, minbatch, m, w, s, myseed))

                    elif which_HPC == 'bessemer':
                        "ToDo"
                        pass
                    f.close()
                    count = count + 1
