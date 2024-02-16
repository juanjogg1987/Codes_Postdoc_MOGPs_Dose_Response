
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=32:mem=24gb

module load anaconda3/personal
source activate py38_gpy

python /rds/general/user/jgiraldo/home/PaperPrecisionOncology_Rebuttal/N_Drugs_Rebuttal_From10Cancers_Train1Cancer_GPy_ExactMOGP_ProdKern.py -i 800 -r 1012 -k 7 -s 1.5 -p 6 -w 0.05 -c 100 -a 6 -n 1