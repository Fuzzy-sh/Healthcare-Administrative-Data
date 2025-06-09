#!/bin/bash
echo Start identifing the best cluster number  from top level Fuzzy script
echo Today is 
date
echo ---------------------------------------------------

#############################################################################
# send the parameters and write them down to the main_GRU.txt file
# We send the parameters from the top level part of the program to know the number of the job array counter 


#Remove the file if it exists
rm params_qlearning.txt

echo ---------------------------------------------------
# ##################################################################################
jobCounter=-1  # Correcting the starting point
echo "startig point for the jobcounter for main_GRU is: $jobCounter"
# create the file name array

# clusters=($(seq 70 50 271))
# clusters_tes=($(seq 70 50 271))
# train_encoder_clusters={20..500..50}
# test_clusters={20..500..50}
# test_encoder_clusters={20..500..50}
# val_clusters={20..500..50}
# val_encoder_clusters={20..500..50}


dimentios=(64)
clusters=(70)
clusters_tes=(70)

    for TRC in ${clusters[@]}; do
    # for TREC in ${clusters[@]}; do
    for TEC in ${clusters_tes[@]}; do
    # for TEEC in ${clusters[@]}; do

    for DIM in ${dimentios[@]}; do


        echo "Running jobs with $TRC $TEC $DIM dimentions for clustering..."
        echo "--latent_dim $DIM --n_clusters $TRC --n_encoder_clusters $TRC --n_clusters_test $TEC --n_encoder_clusters_test $TEC" >> "params_qlearning.txt"
        jobCounter=$((jobCounter+1))

    done
    done
    done
    # done
    # done
    # echo "Running clustering with $CLUSTERS clusters..."

    # # Call the cluster.sh script with the file and cluster number
    # # bash "$CLUSTER_SCRIPT" "$FILE_PATH" "$CLUSTERS" > "$OUTPUT_DIR/results_$CLUSTERS.txt"

    # echo "--n_clusters $CLUSTERS" >> "params_cluster.txt"
    # jobCounter=$((jobCounter+1))


jobID1=$(sbatch --array=0-$jobCounter  Qlearning.sh)
echo $jobID1




