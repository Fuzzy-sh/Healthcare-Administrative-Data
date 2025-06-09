#!/bin/bash
# ====================================
#SBATCH --nodes=1
#SBATCH --mem=128GB
#SBATCH --time=160:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=merg
#SBATCH --output=merging data_%A.out
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
echo "merge file with task Id: $SLURM_ARRAY_TASK_ID"
echo "My hostname is: $(hostname -s)"
echo




echo "Calling Mergingfiles ..."
echo "python 2-merge_preprocessing.py"
# Call the Python preprocessing script
python 2-merge_preprocessing.py 







# if [ $1 == 0 ]
# then

# # run the python program 

# echo starting preprocess program python code for preprocessing_traj

# echo python preprocessing_traj.py 

# CUDA_LAUNCH_BLOCKING=1 python preprocessing_traj.py



# echo ending slurm script to do preprocessing preprocessing_traj

# else

# # run the python program 

# echo starting preprocess program python code for preprocessing_split

# echo python preprocessing_split.py 

# CUDA_LAUNCH_BLOCKING=1 python preprocessing_split.py



# echo ending slurm script to do preprocessing preprocessing_split

# fi
