#!/bin/bash
#$ -P rse
#$ -l rmem=35G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC1/Drug1373_MelanomaGDSC1_GPy_ExactMOGP.py -i 1500 -s 1.0000 -k 5 -w 1.0000 -r 1012 -p 164