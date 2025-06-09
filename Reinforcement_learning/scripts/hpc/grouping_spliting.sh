#!/bin/bash -v
# ====================================
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=10               
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --job-name=Qlearning
#SBATCH --output=Qlearning_%A.out
#SBATCH --cpus-per-task=1
# ====================================

# Check the GPUs with the nvidia-smi command.
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"


date
id






echo start initialization



which python
conda env list

# Print some job information

# echo
# echo "Clustering file with task Id: $SLURM_ARRAY_TASK_ID"
# echo "My hostname is: $(hostname -s)"
# echo

# # run the python program 

# index=0
# while read line ; do
#         LINEARRAY[$index]="$line"
#         index=$(($index+1))
# done < params_cluster_best.txt

# echo $((${SLURM_ARRAY_TASK_ID}-1))
# echo ${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]}

echo starting main program python code for clustering

# echo python 4-add_cluster_preprocessing.py --obs_n_clusters 20 --act_n_clusters 15 #${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]} 

# python 4-add_cluster_preprocessing.py --obs_n_clusters 20 --act_n_clusters 15 #${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]} 

echo python 6-DQlearning.py

python 6-DQlearning.py 


# echo python 5-add_cluster_preprocessing.py --file_name val

# python 5-add_cluster_preprocessing.py --file_name val


echo ending slurm script to do training for main 