#!/bin/bash
#$ -P rse
#$ -l rmem=35G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC1_ANOVA/Each_Drug_MelanomaGDSC1_GPy_ANOVA_ExactMOGP_ProdKern.py -i 1500 -s 0.0100 -k 4 -w 1.0000 -r 1019 -p 993 -d 1036