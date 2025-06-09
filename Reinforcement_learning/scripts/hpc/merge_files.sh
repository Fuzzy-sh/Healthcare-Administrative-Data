#!/bin/bash
#SBATCH --job-name=merge_h5    # Job name
#SBATCH --output=merge_h5_%j.out  # Output log file (with job ID)
#SBATCH --error=merge_h5_%j.err   # Error log file (with job ID)
#SBATCH --time=02:00:00      # Time limit (hh:mm:ss)
#SBATCH --mem=64G
#SBATCH --ntasks=1           # Number of tasks
#SBATCH --cpus-per-task=4    # Number of CPU cores per task


# Load Python module (if needed)
# module load python/3.8  # Adjust based on your cluster

# Activate virtual environment (if needed)
# source ~/your_virtual_env/bin/activate  # Change to your environment path

# Run Python script
python merge_files.py  # Make sure your script is named merge_h5.py
# python knn_predictoin.py