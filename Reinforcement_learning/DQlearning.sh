#!/bin/bash
# ====================================
#SBATCH --nodes=1
#SBATCH --mem=128GB
#SBATCH --time=165:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=DQlearning
#SBATCH --output=/work/messier_lab/fuzzy/DQlearning_%A_%a.out
#SBATCH --cpus-per-task=2
# ====================================

# Check the GPUs with the nvidia-smi command.



date
id






echo start initialization



which python
conda env list

# Print some job information

echo
echo "DQlearning file with task Id: $SLURM_ARRAY_TASK_ID"
echo "My hostname is: $(hostname -s)"
echo

# run the python program 

index=0
while read line ; do
        LINEARRAY[$index]="$line"
        index=$(($index+1))
done < params_dqlearning.txt

echo $((${SLURM_ARRAY_TASK_ID}-1))
echo ${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]}

echo starting main program python code for DQlearning

# echo python 6-DQlearning_2layer.py  ${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]} 

CUDA_LAUNCH_BLOCKING=1 python src/Qlearning/7-DQlearning_score.py  ${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]} 
# CUDA_LAUNCH_BLOCKING=1 python 6-DQlearning_2layer.py


echo ending slurm script to do DQlearning