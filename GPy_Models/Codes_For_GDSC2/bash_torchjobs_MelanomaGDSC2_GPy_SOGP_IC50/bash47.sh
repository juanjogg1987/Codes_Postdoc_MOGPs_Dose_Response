#!/bin/bash
#$ -P rse
#$ -l rmem=35G #Ram memory

module load apps/python/conda
source activate py38_gpflow
python /home/ac1jjgg/SOGP_GPy/Codes_for_GDSC2/All_Drugs_MelanomaGDSC2_GPy_SOGP_IC50.py -i 1500 -s 3.0000 -r 1019 -p 47