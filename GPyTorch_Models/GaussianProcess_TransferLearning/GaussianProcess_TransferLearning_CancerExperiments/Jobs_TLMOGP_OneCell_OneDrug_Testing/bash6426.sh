
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=32:mem=8gb

module load anaconda3/personal
source activate py38_pytorch

python /rds/general/user/jgiraldo/home/TransferLearning_Codes/TLMOGP_TargetDiffToMelanoma_TenDrugs_CrossVal.py -n 180 -r 35 -p 6426 -w 0.3 -s 3 -t 0 -i 30 -d 1059