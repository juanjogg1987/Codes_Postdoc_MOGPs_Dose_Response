#!/bin/bash
#$ -P rse
#$ -l rmem=10G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_Final/KLRelevance_MelanomaGDSC2_GPy_Final_ExactMOGP_ProdKern.py -i 1500 -s 1.0000 -k 6 -w 0.1000 -r 1011 -p 138 -e 180