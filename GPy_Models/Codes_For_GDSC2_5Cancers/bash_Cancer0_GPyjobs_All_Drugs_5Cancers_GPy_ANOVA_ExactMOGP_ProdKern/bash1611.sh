#!/bin/bash
#$ -P rse
#$ -l rmem=35G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_5Cancers/All_Drugs_5Cancers_GPy_ExactMOGP_ProdKern.py -i 1500 -s 0.0500 -k 3 -w 1.5000 -r 1013 -p 1611 -c 20 -a 0 -n 4