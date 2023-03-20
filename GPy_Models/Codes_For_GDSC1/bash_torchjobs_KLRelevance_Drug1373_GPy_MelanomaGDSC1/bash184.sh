#!/bin/bash
#$ -P rse
#$ -l rmem=10G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC1/KLRelevance_Drug1373_MelanomaGDSC1_GPy_ExactMOGP.py -i 1500 -s 3.0000 -k 2 -w 0.1000 -r 1019 -p 471 -e 184