#!/bin/bash
#$ -P rse
#$ -l rmem=35G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC1_Final/Drug1061_MelanomaGDSC1_GPy_Final_ExactMOGP_OneKern.py -i 1500 -s 3.0000 -k 5 -w 1.0000 -r 1011 -p 211