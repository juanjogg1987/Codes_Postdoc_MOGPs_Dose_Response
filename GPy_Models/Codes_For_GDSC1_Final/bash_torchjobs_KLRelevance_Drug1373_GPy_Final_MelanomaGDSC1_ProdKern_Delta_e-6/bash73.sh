#!/bin/bash
#$ -P rse
#$ -l rmem=10G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC1_Final/KLRelevance_Drug1373_MelanomaGDSC1_GPy_Final_ExactMOGP_ProdKern.py -i 1500 -s 3.0000 -k 2 -w 1.0000 -r 1013 -p 236 -e 73