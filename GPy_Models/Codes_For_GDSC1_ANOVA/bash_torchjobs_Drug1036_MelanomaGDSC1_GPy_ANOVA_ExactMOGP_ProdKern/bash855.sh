#!/bin/bash
#$ -P rse
#$ -l rmem=35G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC1_ANOVA/Each_Drug_MelanomaGDSC1_GPy_ANOVA_ExactMOGP_ProdKern.py -i 1500 -s 3.0000 -k 1 -w 1.0000 -r 1017 -p 855 -d 1036