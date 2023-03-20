#!/bin/bash
#$ -P rse
#$ -l rmem=10G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_ANOVA/KLRelevance_All_Drugs_MelanomaGDSC2_GPy_ANOVA_ExactMOGP_ProdKern.py -i 1500 -s 3.0000 -k 2 -w 1.0000 -r 1011 -p 162 -e 10