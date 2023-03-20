#!/bin/bash
#$ -P rse
#$ -l rmem=10G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC1/KLRelevance_Drugs_1036D9_and_1371D5_MelanomaGDSC1_ToPredict_MelanomaGDSC2_SanityCheck.py -i 1500 -s 3.0000 -k 2 -w 0.0100 -r 1017 -p 622 -e 383