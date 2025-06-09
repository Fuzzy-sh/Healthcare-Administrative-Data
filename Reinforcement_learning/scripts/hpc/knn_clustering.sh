#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --job-name=cluster_encoder_identification
#SBATCH --output=cluster_encoder_%A_%a.out
#SBATCH --cpus-per-task=1
#SBATCH --job-name=knn-sweep       # Job name
#SBATCH --output=/work/messier_lab/knn_%A_%a.out  # Standard output (%%A=jobid, %%a=array index)

#SBATCH --array=2-2000              # Range of k values: 2..2000

# The array index corresponds to the value of k
k=$SLURM_ARRAY_TASK_ID

echo "Running KNN for k=$k"


# Run the Python script, specifying k
python knn_clustering.py --k_min $k --k_max $k

echo "Done with k=$k"


# # After all array jobs finish, gather them:
# cd results_knn/
# head -n 1 knn_sweep_500_clusters_k_2_*.csv > combined_knn_results.csv
# # For each file, skip the header (tail -n +2) and append:
# for f in knn_sweep_500_clusters_k_*.csv; do
#     tail -n +2 "$f" >> combined_knn_results.csv
# done