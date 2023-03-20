#!/bin/bash
#$ -P rse
#$ -l rmem=35G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/MOGP_GPy/Codes_for_GDSC2_5Cancers/All_Drugs_5Cancers_GPy_ExactMOGP_ProdKern.py -i 1500 -s 0.3000 -k 7 -w 1.5000 -r 1012 -p 851 -c 40 -a 0 -n 6