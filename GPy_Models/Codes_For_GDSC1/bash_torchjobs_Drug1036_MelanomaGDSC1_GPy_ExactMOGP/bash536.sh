#!/bin/bash
#$ -P rse
#$ -l rmem=35G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC1/Drug1036_MelanomaGDSC1_GPy_ExactMOGP.py -i 1500 -s 3.0000 -k 6 -w 1.0000 -r 1016 -p 536