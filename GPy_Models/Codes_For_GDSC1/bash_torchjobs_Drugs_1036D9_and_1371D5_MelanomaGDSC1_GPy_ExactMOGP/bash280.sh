#!/bin/bash
#$ -P rse
#$ -l rmem=35G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC1/Drugs_1036Dose9_and_1371Dose5_MelanomaGDSC1_ToPredict_MelanomaGDSC2.py -i 1500 -s 1.0000 -k 2 -w 0.1000 -r 1012 -p 280