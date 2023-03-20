#!/bin/bash
#$ -P rse
#$ -l rmem=10G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_Final/KLRelevance_Drug1373_MelanomaGDSC2_GPy_Final_ExactMOGP_ProdKern.py -i 1500 -s 1.0000 -k 3 -w 0.0100 -r 1011 -p 128 -e 152