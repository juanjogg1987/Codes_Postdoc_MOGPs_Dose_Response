#!/bin/bash
#$ -P rse
#$ -l rmem=35G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_Final/Drug1061_MelanomaGDSC2_GPy_Final_ExactMOGP_OneKern.py -i 1500 -s 3.0000 -k 6 -w 0.0100 -r 1018 -p 740