
import os
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import getopt
import sys

class commandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'd:')
        self.drug_name = "1061"

        for op, arg in opts:
            # print(op,arg)
            if op == '-d':
                self.drug_name = arg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
config = commandLine()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

seeds = [1010,1011,1012,1013,1014,1015,1016,1017,1018,1019]
weights = [0.01,0.1,1.0]
scales = [0.01,0.1,1,3]

if config.drug_name == "1061" or config.drug_name == "1036":
    ranks = [1,2,3,4,5,6,7,8,9]   #Since GDSC1 has 9 outputs for 1061 the maximum rank is 9.
elif config.drug_name == "1373":
    ranks = [1, 2, 3, 4, 5]  # Since GDSC1 has 5 outputs for 1373 the maximum rank is 5.

ram = 35
which_HPC = 'sharc'

count = 0

path_to_save = './bash_torchjobs_Drug'+config.drug_name+'_MelanomaGDSC1_GPy_ANOVA_ExactMOGP_ProdKern/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

for myseed in seeds:
    for scale in scales:
        for w in weights:
            for k in ranks:
                f = open(path_to_save + "bash" + str(count) + ".sh", "w+")
                if which_HPC == 'sharc' and count%1 == 0:
                    f.write("#!/bin/bash\n#$ -P rse\n#$ -l rmem=%dG #Ram memory\n\n"  
                            "module load apps/python/conda\nsource activate py38_gpflow\npython /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC1_ANOVA/Each_Drug_MelanomaGDSC1_GPy_ANOVA_ExactMOGP_ProdKern.py -i 1500 -s %.4f -k %d -w %.4f -r %d -p %d -d %s" % (
                            ram, scale,k,w, myseed,count,config.drug_name))
                # if which_HPC == 'sharc' and count%2 != 0:
                #     f.write("#!/bin/bash\n#$ -P bioinf-core\n#$ -l rmem=%dG #Ram memory\n\n"
                #             "module load apps/python/conda\nsource activate py38_gpflow\npython /home/ac1jjgg/GP_Models_MarinaData/All_Drugs_GPAdditiveKern_Test1_0_HPC.py -l %d -b %d -m %d -i 1500 -w %.4f -s %.4f -r %d" % (
                #             ram, l, minbatch, m, w, s, myseed))

                elif which_HPC == 'bessemer':
                    "ToDo"
                    pass
                f.close()
                count = count + 1
