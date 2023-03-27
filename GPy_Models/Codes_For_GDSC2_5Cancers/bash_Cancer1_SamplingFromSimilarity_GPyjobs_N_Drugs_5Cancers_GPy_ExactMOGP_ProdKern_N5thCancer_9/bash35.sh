#!/bin/bash
#$ -P rse
#$ -l rmem=25G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_5Cancers/N_Drugs_5Cancers_GPy_ExactMOGP_ProdKern_SamplingFromSimilarity.py -i 1500 -s 0.0500 -k 3 -w 0.0500 -r 1012 -p 35 -c 12 -a 1 -n 6 -t 9