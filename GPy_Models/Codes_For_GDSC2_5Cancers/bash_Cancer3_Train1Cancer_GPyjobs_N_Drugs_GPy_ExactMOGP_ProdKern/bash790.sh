#!/bin/bash
#$ -P rse
#$ -l rmem=18G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_5Cancers/N_Drugs_From5Cancers_Train1Cancer_GPy_ExactMOGP_ProdKern.py -i 1500 -s 0.3000 -k 4 -w 0.3000 -r 1012 -p 790 -c 40 -a 3 -n 5