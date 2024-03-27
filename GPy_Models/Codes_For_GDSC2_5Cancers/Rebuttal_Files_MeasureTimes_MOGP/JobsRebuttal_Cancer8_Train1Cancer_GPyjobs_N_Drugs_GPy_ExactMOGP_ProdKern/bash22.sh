
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=32:mem=24gb

module load anaconda3/personal
source activate py38_gpy

python /rds/general/user/jgiraldo/home/PaperPrecisionOncology_Rebuttal/N_Drugs_Rebuttal_From10Cancers_Train1Cancer_GPy_ExactMOGP_ProdKern.py -i 800 -r 1014 -k 7 -s 0.3 -p 22 -w 0.3 -c 100 -a 8 -n 1