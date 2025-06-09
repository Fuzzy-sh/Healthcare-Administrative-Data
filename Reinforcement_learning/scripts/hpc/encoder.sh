#!/bin/bash -v
# ====================================
#SBATCH --nodes=1
#SBATCH --mem=128GB
#SBATCH --time=160:00:00
#SBATCH --ntasks=10
#SBATCH --job-name=encoder_identification
#SBATCH --output=encoder_%A_%a.out
#SBATCH --cpus-per-task=1
# ====================================

# Check the GPUs with the nvidia-smi command.



date
id






echo start initialization



which python
conda env list

# Print some job information

echo
echo "Clustering file with task Id: $SLURM_ARRAY_TASK_ID"
echo "My hostname is: $(hostname -s)"
echo

# run the python program 

index=0
while read line ; do
        LINEARRAY[$index]="$line"
        index=$(($index+1))
done < params_encoder.txt

echo $((${SLURM_ARRAY_TASK_ID}-1))
echo ${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]}

echo starting main program python code for clustering

echo python 4-encoder_preprocessing.py ${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]} 

CUDA_LAUNCH_BLOCKING=1 python 4-encoder_preprocessing.py ${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]} 



echo ending slurm script to do training for main 