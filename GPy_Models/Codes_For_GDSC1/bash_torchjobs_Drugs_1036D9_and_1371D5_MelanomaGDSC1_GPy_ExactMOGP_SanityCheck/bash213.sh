#!/bin/bash
#$ -P rse
#$ -l rmem=35G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC1/Drugs_1036Dose9_and_1371Dose5_MelanomaGDSC1_ToPredict_MelanomaGDSC2_SanityCheck.py -i 1500 -s 3.0000 -k 7 -w 1.0000 -r 1011 -p 213