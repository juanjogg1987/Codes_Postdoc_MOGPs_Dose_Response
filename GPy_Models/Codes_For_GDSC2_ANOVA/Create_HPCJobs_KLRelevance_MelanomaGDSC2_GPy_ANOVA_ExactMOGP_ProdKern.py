
import os
which_HPC = 'sharc'
ram = 10
path_to_save = './bash_torchjobs_KLRelevance_GPy_ANOVA_MelanomaGDSC2_ProdKern/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)


"The winner model should be loaded as"
"KLRelevance_MelanomaGDSC2_GPy_ANOVA_ExactMOGP_ProdKern.py -i 1500 -s 3.0000 -k 2 -w 1.0000 -r 1011 -p 162 -e %d"
"just the -e should be the iterable over the P features"
Pfeatures = 24
for count in range(Pfeatures):
    f = open(path_to_save + "bash" + str(count) + ".sh", "w+")
    if which_HPC == 'sharc' and count%1 == 0:
        f.write("#!/bin/bash\n#$ -P rse\n#$ -l rmem=%dG #Ram memory\n\n"  
                "module load apps/python/conda\nsource activate py38_gpflow\npython /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_ANOVA/KLRelevance_All_Drugs_MelanomaGDSC2_GPy_ANOVA_ExactMOGP_ProdKern.py -i 1500 -s 3.0000 -k 2 -w 1.0000 -r 1011 -p 162 -e %d" % (
                ram, count))
    # if which_HPC == 'sharc' and count%2 != 0:
    #     f.write("#!/bin/bash\n#$ -P bioinf-core\n#$ -l rmem=%dG #Ram memory\n\n"
    #             "module load apps/python/conda\nsource activate py38_gpflow\npython /home/ac1jjgg/GP_Models_MarinaData/All_Drugs_GPAdditiveKern_Test1_0_HPC.py -l %d -b %d -m %d -i 1500 -w %.4f -s %.4f -r %d" % (
    #             ram, l, minbatch, m, w, s, myseed))

    elif which_HPC == 'bessemer':
        "ToDo"
        pass
    f.close()
    #count = count + 1

#All_Drugs_GP_Test1_0_HPC -l 1 -b 500 -m 100 -i 100 -w 0.1 -s 0.03 -r 1010