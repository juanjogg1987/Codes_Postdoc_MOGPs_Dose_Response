#!/bin/bash
#$ -P rse
#$ -l rmem=10G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC1/KLRelevance_Drug1036_MelanomaGDSC1_GPy_ExactMOGP.py -i 1500 -s 1.0000 -k 4 -w 1.0000 -r 1012 -p 291 -e 160