#!/bin/bash
#$ -P rse
#$ -l rmem=35G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_5Cancers/All_Drugs_5Cancers_GPy_ExactMOGP_ProdKern.py -i 1500 -s 0.3000 -k 7 -w 0.3000 -r 1013 -p 2006 -c 80 -a 0 -n 3