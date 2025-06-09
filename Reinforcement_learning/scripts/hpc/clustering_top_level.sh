#!/bin/bash
echo Start identifing the best cluster number  from top level Fuzzy script
echo Today is 
date
echo ---------------------------------------------------

#############################################################################
# send the parameters and write them down to the main_GRU.txt file
# We send the parameters from the top level part of the program to know the number of the job array counter 


#Remove the file if it exists
rm params_dimention.txt

echo ---------------------------------------------------
# ##################################################################################
jobCounter=-1  # Correcting the starting point
echo "startig point for the jobcounter for main_GRU is: $jobCounter"
# create the file name array

# file_name=("train" "val" "test")
file_name=("train")
clusters=($(seq 2 1 2000))
# experiment=("autoencoder" "clustering")
dimentios=(64)

    for FILE in ${file_name[@]}; do
        for DIM in ${dimentios[@]}; do
            for CLUSTERS in ${clusters[@]}; do


        echo "Running $FILE with $DIM dimentions and $CLUSTERS for clustering..."
        echo "--latent_dim_obs $DIM --file_name $FILE --n_cluster $CLUSTERS" >> "params_dimention.txt"
        jobCounter=$((jobCounter+1))
    done
    done
    done

    # echo "Running clustering with $CLUSTERS clusters..."

    # # Call the cluster.sh script with the file and cluster number
    # # bash "$CLUSTER_SCRIPT" "$FILE_PATH" "$CLUSTERS" > "$OUTPUT_DIR/results_$CLUSTERS.txt"

    # echo "--n_clusters $CLUSTERS" >> "params_cluster.txt"
    # jobCounter=$((jobCounter+1))


jobID1=$(sbatch --array=0-$jobCounter  cluster.sh)
echo $jobID1




