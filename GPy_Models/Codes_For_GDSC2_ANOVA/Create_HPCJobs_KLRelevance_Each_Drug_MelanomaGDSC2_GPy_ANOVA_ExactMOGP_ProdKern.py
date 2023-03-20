
import os
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import getopt
import sys

class commandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'd:')
        self.drug_name = "1036"

        for op, arg in opts:
            # print(op,arg)
            if op == '-d':
                self.drug_name = arg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
config = commandLine()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Best Model Drug1036: Each_Drug_MelanomaGDSC2_GPy_ANOVA_ExactMOGP_ProdKern.py -i 1500 -s 1.0000 -k 2 -w 0.0100 -r 1016 -p 547 -d 1036 -e %d"
"Best Model Drug1061: Each_Drug_MelanomaGDSC2_GPy_ANOVA_ExactMOGP_ProdKern.py -i 1500 -s 3.0000 -k 5 -w 1.0000 -r 1019 -p 837 -d 1061 -e %d"
"Best Model Drug1373: Each_Drug_MelanomaGDSC2_GPy_ANOVA_ExactMOGP_ProdKern.py -i 1500 -s 3.0000 -k 2 -w 0.0100 -r 1015 -p 484 -d 1373 -e %d"

if config.drug_name == "1036":
    drug_setting = "-i 1500 -s 1.0000 -k 2 -w 0.0100 -r 1016 -p 547 -d 1036"
elif config.drug_name == "1061":
    drug_setting = "-i 1500 -s 3.0000 -k 5 -w 1.0000 -r 1019 -p 837 -d 1061"
elif config.drug_name == "1373":
    drug_setting = "-i 1500 -s 3.0000 -k 2 -w 0.0100 -r 1015 -p 484 -d 1373"
else:
    print(f"The drug {config.drug_name} has not been tested")

which_HPC = 'sharc'
ram = 10
path_to_save = './bash_torchjobs_KLRelevance_Drug'+config.drug_name+'_GPy_ANOVA_MelanomaGDSC2_ProdKern/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)


"The winner model should be loaded using drug_setting variable"

"just the -e should be the iterable over the P features"
Pfeatures = 24  #Number of features for ANOVA and our experiments in order to compare
for count in range(Pfeatures):
    f = open(path_to_save + "bash" + str(count) + ".sh", "w+")
    if which_HPC == 'sharc' and count%1 == 0:
        f.write("#!/bin/bash\n#$ -P rse\n#$ -l rmem=%dG #Ram memory\n\n"  
                "module load apps/python/conda\nsource activate py38_gpflow\npython /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_ANOVA/KLRelevance_Each_Drug_MelanomaGDSC2_GPy_ANOVA_ExactMOGP_ProdKern.py %s -e %d" % (
                ram, drug_setting,count))
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