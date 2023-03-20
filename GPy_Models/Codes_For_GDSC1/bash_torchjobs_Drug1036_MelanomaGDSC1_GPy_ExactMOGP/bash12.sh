#!/bin/bash
#$ -P rse
#$ -l rmem=35G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC1/Drug1036_MelanomaGDSC1_GPy_ExactMOGP.py -i 1500 -s 0.0100 -k 4 -w 0.1000 -r 1010 -p 12