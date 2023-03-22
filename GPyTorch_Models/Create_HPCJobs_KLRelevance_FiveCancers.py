
import os
#KLRelevance_FiveCancers_SmallData_GPyTorch_SparseMOGP.py -l 15 -b 200 -m 1000 -i 0 -w 0.0100 -s 3.0000 -r 1010 -d 10 -p 80
seeds = [1010]#,1011,1012] #,1015,1016,1017,1018,1019]
weights = [0.01]
scales = [3]
latent_GPs = [15]
num_inducing = [1000]
spli_dim = [10]
ram = 35
minbatch = 200
which_HPC = 'sharc'

#count = 0
Nfold = 0
bashwin = 80

path_to_save = './bash_torchjobs_KLRelevances_FiveCancers/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)


for myseed in seeds:
        for w in weights:
            for s in scales:
                for l in latent_GPs:
                    for m in num_inducing:
                        for d in spli_dim:
                            for count in range(826):
                                f = open(path_to_save + "bash" + str(count) + ".sh", "w+")
                                if which_HPC == 'sharc' and count%1 == 0:
                                    f.write("#!/bin/bash\n#$ -P rse\n#$ -l rmem=%dG #Ram memory\n\n"  
                                            "module load apps/python/conda\nsource activate py38_gpflow\npython /home/ac1jjgg/MOGP_GPyTorch/KLRelevance_FiveCancers_SmallData_GPyTorch_SparseMOGP.py -l %d -b %d -m %d -i 0 -w %.4f -s %.4f -r %d -d %d -p %d -f %d -e %d -c melanoma" % (
                                            ram, l, minbatch, m, w, s, myseed,d,bashwin,Nfold,count))
                                # if which_HPC == 'sharc' and count%2 != 0:
                                #     f.write("#!/bin/bash\n#$ -P bioinf-core\n#$ -l rmem=%dG #Ram memory\n\n"
                                #             "module load apps/python/conda\nsource activate py38_gpflow\npython /home/ac1jjgg/GP_Models_MarinaData/All_Drugs_GPAdditiveKern_Test1_0_HPC.py -l %d -b %d -m %d -i 1500 -w %.4f -s %.4f -r %d" % (
                                #             ram, l, minbatch, m, w, s, myseed))

                                elif which_HPC == 'bessemer':
                                    "ToDo"
                                    pass
                                f.close()
                                count = count + 1

#All_Drugs_GP_Test1_0_HPC -l 1 -b 500 -m 100 -i 100 -w 0.1 -s 0.03 -r 1010