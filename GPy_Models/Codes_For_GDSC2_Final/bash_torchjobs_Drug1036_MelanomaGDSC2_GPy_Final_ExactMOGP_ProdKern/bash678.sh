#!/bin/bash
#$ -P rse
#$ -l rmem=35G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_Final/Drug1036_MelanomaGDSC2_GPy_Final_ExactMOGP_ProdKern.py -i 1500 -s 0.0100 -k 7 -w 0.0100 -r 1018 -p 678