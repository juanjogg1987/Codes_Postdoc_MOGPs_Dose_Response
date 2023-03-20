#!/bin/bash
#$ -P rse
#$ -l rmem=35G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC1_Final/Drug1373_MelanomaGDSC1_GPy_Final_ExactMOGP_ProdKern.py -i 1500 -s 1.0000 -k 3 -w 0.1000 -r 1010 -p 37