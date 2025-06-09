#!/bin/bash
# ====================================
#SBATCH --nodes=1
#SBATCH --mem=128GB
#SBATCH --time=160:00:00
#SBATCH --ntasks=10
#SBATCH --job-name=preprocessing
#SBATCH --output=preprocess_%A_%a.out
#SBATCH --cpus-per-task=10
# ====================================

# Check the GPUs with the nvidia-smi command.



date
id


echo start initialization



which python
conda env list

# Print some job information

echo
echo "main preprocess file with task Id: $SLURM_ARRAY_TASK_ID"
echo "My hostname is: $(hostname -s)"
echo


# Accept arguments from top.sh
FILE_PATH=$1
FILE_NAME=$2

echo "Calling preprocessing.py for $FILE_NAME..."
echo "python 1-main_preprocessing.py --file_path $FILE_PATH --file_name $FILE_NAME"
# Call the Python preprocessing script
python 1-main_preprocessing.py --file_path "$FILE_PATH" --file_name "$FILE_NAME"







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
