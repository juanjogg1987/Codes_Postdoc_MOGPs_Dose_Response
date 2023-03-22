#!/bin/bash
#$ -P rse
#$ -l rmem=35G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC1_Final/Drug1061_MelanomaGDSC1_GPy_Final_ExactMOGP_OneKern.py -i 1500 -s 0.0100 -k 1 -w 0.0100 -r 1013 -p 324