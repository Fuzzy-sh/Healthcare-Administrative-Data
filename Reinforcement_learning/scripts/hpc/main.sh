#!/bin/bash -v
# ====================================

#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=a100:4
#SBATCH --ntasks-per-node=32
#SBATCH --gpus-per-node=1
#SBATCH --job-name=autoencoder_GPU
#SBATCH --output=autoencoder_GPU_%A.out


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

echo python 4-add_cluster_preprocessing.py --obs_n_clusters 20 --act_n_clusters 15 #${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]} 

python 4-add_cluster_preprocessing.py --obs_n_clusters 20 --act_n_clusters 15 #${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]} 

# echo python 5-load_episodes.py

# python 5-load_episodes.py  


echo ending slurm script to do training for main 