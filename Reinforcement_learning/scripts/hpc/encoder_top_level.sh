#!/bin/bash
echo Start identifing the best cluster number  from top level Fuzzy script
echo Today is 
date
echo ---------------------------------------------------

#############################################################################
# send the parameters and write them down to the main_GRU.txt file
# We send the parameters from the top level part of the program to know the number of the job array counter 


#Remove the file if it exists
rm params_encoder.txt

echo ---------------------------------------------------
# ##################################################################################
jobCounter=-1  # Correcting the starting point
echo "startig point for the jobcounter for main_GRU is: $jobCounter"
# create the file name array

latent_dim=(8 16 32 64 128 225)
prefix=("observation_prefix" "action_prefix")

for DIM in ${latent_dim[@]}; do
    for PREFIX in ${prefix[@]}; do
    
        echo "Running encoder with $DIM latent dimention for $PREFIX..."
        echo "--n_dim $DIM --prefix $PREFIX" >> "params_encoder.txt"
        jobCounter=$((jobCounter+1))

    done
    done
    # echo "Running clustering with $CLUSTERS clusters..."



jobID1=$(sbatch --array=0-$jobCounter  encoder.sh)
echo $jobID1




