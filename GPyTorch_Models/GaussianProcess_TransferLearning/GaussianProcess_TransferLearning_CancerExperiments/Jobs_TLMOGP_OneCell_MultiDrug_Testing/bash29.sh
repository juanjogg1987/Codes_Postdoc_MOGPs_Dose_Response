
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=40:mem=15gb

module load anaconda3/personal
source activate py38_pytorch

python /rds/general/user/jgiraldo/home/TLMOGP_MeanInPrior/TLMOGP_OverGDSC2_AnyDrugConcentrations_MeanInPrior_AlternativeVersion.py -n 4000 -r 30 -p 29