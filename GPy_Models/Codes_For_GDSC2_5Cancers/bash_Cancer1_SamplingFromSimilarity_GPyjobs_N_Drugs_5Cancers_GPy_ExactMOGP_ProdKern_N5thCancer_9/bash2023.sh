#!/bin/bash
#$ -P rse
#$ -l rmem=25G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_5Cancers/N_Drugs_5Cancers_GPy_ExactMOGP_ProdKern_SamplingFromSimilarity.py -i 1500 -s 0.0500 -k 3 -w 1.5000 -r 1013 -p 2023 -c 48 -a 1 -n 2 -t 9