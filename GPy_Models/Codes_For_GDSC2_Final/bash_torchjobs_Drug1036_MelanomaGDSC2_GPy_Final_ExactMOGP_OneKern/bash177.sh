#!/bin/bash
#$ -P rse
#$ -l rmem=35G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_Final/Drug1036_MelanomaGDSC2_GPy_Final_ExactMOGP_OneKern.py -i 1500 -s 0.0100 -k 3 -w 0.1000 -r 1012 -p 177