#!/bin/bash
#$ -P rse
#$ -l rmem=10G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC1/KLRelevance_Drug1371_MelanomaGDSC1_GPy_ExactMOGP.py -i 1500 -s 1.0000 -k 3 -w 0.0100 -r 1012 -p 152 -e 182