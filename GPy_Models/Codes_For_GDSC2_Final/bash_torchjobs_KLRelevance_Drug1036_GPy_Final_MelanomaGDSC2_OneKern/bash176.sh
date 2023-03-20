#!/bin/bash
#$ -P rse
#$ -l rmem=10G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_Final/KLRelevance_Drug1036_MelanomaGDSC2_GPy_Final_ExactMOGP_OneKern.py -i 1500 -s 1.0000 -k 2 -w 0.0100 -r 1010 -p 43 -e 176