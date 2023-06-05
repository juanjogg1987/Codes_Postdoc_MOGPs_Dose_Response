#!/bin/bash
#$ -P rse
#$ -l rmem=18G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_5Cancers/N_Drugs_From5Cancers_Train1Cancer_GPy_ExactMOGP_ProdKern.py -i 1500 -s 0.3000 -k 3 -w 1.5000 -r 1012 -p 931 -c 20 -a 3 -n 2