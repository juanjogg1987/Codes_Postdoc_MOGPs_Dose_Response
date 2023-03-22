
import os

seeds = [1010]#,1011,1012] #,1015,1016,1017,1018,1019]
weights = [0.01,0.1,1.0]
scales = [0.1,1,3]
latent_GPs = [5,9,15]
num_inducing = [500,1000]
spli_dim = [5,10,20]
Nfold = [0,1,2,3,4]
ram = 35
minbatch = 100
which_HPC = 'sharc'

count = 0

path_to_save = './bash_torchjobs_AdditiveCorrectRealY/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)


for myseed in seeds:
        for w in weights:
            for s in scales:
                for l in latent_GPs:
                    for m in num_inducing:
                        for d in spli_dim:
                            for myfold in Nfold:
                                f = open(path_to_save + "bash" + str(count) + ".sh", "w+")
                                if which_HPC == 'sharc' and count%1 == 0:
                                    f.write("#!/bin/bash\n#$ -P rse\n#$ -l rmem=%dG #Ram memory\n\n"  
                                            "module load apps/python/conda\nsource activate py38_gpflow\npython /home/ac1jjgg/MOGP_GPyTorch/All_Drugs_LungCancer_GPAdditiveRealY_GPyTorch.py -l %d -b %d -m %d -i 50 -w %.4f -s %.4f -r %d -d %d -p %d -f %d" % (
                                            ram, l, minbatch, m, w, s, myseed,d,count,myfold))
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