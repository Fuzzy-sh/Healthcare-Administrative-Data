#!/bin/bash

# Top-level preprocessing script for submitting SLURM jobs per data file and merging results

# Set variables
DATA_PATH="data/AHS_data/"        # Directory containing data files
DATA_TYPE=".h5"              # File extension to filter (e.g., .h5)
DATA_NAME="filter_string"    # Optional: substring to filter file names (set to "" to disable)

echo "Preprocessing data from $DATA_PATH..."
echo "Filtering by type: $DATA_TYPE"

# List and filter files by extension and optional substring, then sort
FILES=$(find "$DATA_PATH" -maxdepth 1 -type f -name "*$DATA_TYPE" | grep "$DATA_NAME" | sort)

echo "Found files:"
echo "$FILES"

jobIDs=()  # Array to store submitted job IDs
jobCounter=0

# Submit a SLURM job for each file
while IFS= read -r FILE; do
    # Extract just the filename for passing to the script
    BASENAME=$(basename "$FILE")
    # Submit the job and capture its ID
    jobID=$(sbatch --array=$jobCounter preprocessing.sh "$DATA_PATH" "$BASENAME" | awk '{print $4}')
    jobIDs+=($jobID)
    echo "Job submitted for $BASENAME with ID: $jobID"
    jobCounter=$((jobCounter + 1))
done <<< "$FILES"

# Combine job IDs into a comma-separated dependency list
jobDependency=$(IFS=','; echo "${jobIDs[*]}")

# Submit the merge job, dependent on all preprocessing jobs finishing successfully
echo "Submitting merge.sh with dependency on jobs: $jobDependency"
sbatch --dependency=afterok:$jobDependency merge.sh









# #!/bin/bash

# echo Starting inclusion exclusion and cohort creation from top level Fuzzy script
# # print out the date and time for now. 
# date

# echo ------------------------------------------------------------------------
# # ########################################################################################
# # Submit the job array 

# # send the add traj parameter as zero to the preprocessing.sh so the preprocessing.sh will call the preprocessing.py with zero to run the preprocessing_traj.py

# jobID1=$(sbatch preprocessing.sh 0 | awk '{print $4}')

# echo $jobID1

# # send the 1 to the preprocessing.sh, so the preprocessing.sh will call the preprocessing_split.py

# jobID2=$(sbatch --dependency=aftercorr:$jobID1 preprocessing.sh 1)

# echo $jobID2



# # Just call the preprocessing to do the precporsiing for each dataset in the AHS_data 
# # save the preporcessed data in the preprocessed_data folder.
# # concat all the datasets in the proporcessed datasets folder and then normalize the data and create the columsn with the names of the features.
# # aftere haveing whole of the dataset, we need to split the data into train and test data. and then save them into train_val_test_data folder.
# # we need to run several jobs to find the best cluster number 


# # it is not obvious that we need to run the clustering for each datasets, like train, test and validation or we need to run the clustering for the whole dataset ?
# # I think we need to run the clustering for the whole dataset, because we need to find the best cluster number for the whole dataset.
# # Also, it is not obvious that we need to run the clustering for the normalized data or the original data. I think we need to run the clustering for the normalized data.
# # I also need to figure it out if we need to run the clustering for the X features as the observaiotns, actions separately? without considering Y as the target variable.
# # I think we need to run the clustering for the X features as the observations and actions separately.

# # https://realpython.com/k-means-clustering-python/