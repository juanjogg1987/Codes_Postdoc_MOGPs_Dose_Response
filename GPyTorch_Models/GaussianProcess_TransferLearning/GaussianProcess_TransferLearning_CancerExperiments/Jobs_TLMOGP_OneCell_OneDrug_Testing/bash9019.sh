
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=32:mem=8gb

module load anaconda3/personal
source activate py38_pytorch

python /rds/general/user/jgiraldo/home/TransferLearning_Codes/TLMOGP_TargetDiffToMelanoma_TenDrugs_CrossVal.py -n 180 -r 40 -p 9019 -w 0.5 -s 3 -t 0 -i 34 -d 2045