#!/bin/bash
echo Start identifing the best cluster number  from top level Fuzzy script
echo Today is 
date
echo ---------------------------------------------------

#############################################################################
# send the parameters and write them down to the main_GRU.txt file
# We send the parameters from the top level part of the program to know the number of the job array counter 


#Remove the file if it exists
rm params_dqlearning.txt

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



algo_list=('UHA-SAC' 'UA-SAC' 'UHA-CQL' 'UA-CQL' 'BQL' 'SAC' 'CQL' 'DDQN' 'DQN')
num_layers_list=(2)
num_hidden_neurons_list=(64)
activation_func_list=('relu')
learning_rate_list=(0.0001)
n_steps_list=(1000000)

conservative_alpha_list=(0.5)
clusters=(70)
clusters_tes=(70)
dimentios=(64)

# Only used for UA-SAC or UHA-SAC
use_attention_1=(True, False)
use_attention_2=(True, False)
num_attention_heads_1=(1 2 4)
num_attention_heads_2=(1 2 4)

use_uncertainty_actor_loss_1=(False, True)
use_uncertainty_actor_loss_2=(False, True)
use_uncertainty_critic_loss_1=(True, False)
use_uncertainty_critic_loss_2=(True, False)
uncertainty_actor_weight_1=(0.001)
uncertainty_actor_weight_2=(0.001)
uncertainty_critic_weight_1=(0.001)
uncertainty_critic_weight_2=(0.001)

jobCounter=-1  # Correcting the starting point
echo "startig point for the jobcounter for main_GRU is: $jobCounter"
# create the file name array

for AGO in "${algo_list[@]}"; do
  for NL in "${num_layers_list[@]}"; do
    for NH in "${num_hidden_neurons_list[@]}"; do
      for ACT in "${activation_func_list[@]}"; do
        for LR in "${learning_rate_list[@]}"; do
          for NS in "${n_steps_list[@]}"; do
            for DIM in "${dimentios[@]}"; do

              # ----------------------------------------------------------
              # 1) Special case for CQL/DCQL: loop over conservative_alpha
              # ----------------------------------------------------------
              if [[ "$AGO" == "BQL" || "$AGO" == "SAC" || "$AGO" == "DQN" || "$AGO" == "CQL"||  "$AGO" == "DDQN" ]]; then
                for CA in "${conservative_alpha_list[@]}"; do
                  # echo "Running job #$jobCounter with $AGO $NL $NH $ACT $LR $NS $DIM for DQlearning (CQL/DCQL)..."
                  echo "--algo $AGO --num_layers $NL --num_hidden_neurons $NH --activation_func $ACT --learning_rate $LR --n_steps $NS --conservative_alpha $CA --latent_dim $DIM" >> "params_dqlearning.txt"
                  jobCounter=$((jobCounter + 1))
                done

              # -----------------------------------------------------------
              # 2) Special case for UA-SAC / UHA-SAC
              #    We loop over attention flags and uncertainty flags
              # -----------------------------------------------------------
              elif [[ "$AGO" == "UHA-SAC" || "$AGO" == "UHA-CQL" ]]; then
                CA=0  # not used for UA-SAC/UHA-SAC, presumably

                for att1 in "${use_attention_1[@]}"; do
                  for att2 in "${use_attention_2[@]}"; do

                    # If att1 is True, loop over num_attention_heads_1
                    # If att1 is False, we can set heads to a default of 1 (or skip).
                    if [[ "$att1" == "True" ]]; then
                      heads1_list=("${num_attention_heads_1[@]}")
                    else
                      heads1_list=(1)  # or skip
                    fi

                    # Similarly for att2
                    if [[ "$att2" == "True" ]]; then
                      heads2_list=("${num_attention_heads_2[@]}")
                    else
                      heads2_list=(1)
                    fi

                    for NAH1 in "${heads1_list[@]}"; do
                      for NAH2 in "${heads2_list[@]}"; do

                        # Now handle uncertainty flags for both levels:
                        for uActLoss1 in "${use_uncertainty_actor_loss_1[@]}"; do
                          # If True, loop over all actor weights; if False, use a single [0.0] or skip
                          if [[ "$uActLoss1" == "True" ]]; then
                            actorW1_list=("${uncertainty_actor_weight_1[@]}")
                          else
                            actorW1_list=(0.0)
                          fi

                          for uActLoss2 in "${use_uncertainty_actor_loss_2[@]}"; do
                            if [[ "$uActLoss2" == "True" ]]; then
                              actorW2_list=("${uncertainty_actor_weight_2[@]}")
                            else
                              actorW2_list=(0.0)
                            fi

                            for uCritLoss1 in "${use_uncertainty_critic_loss_1[@]}"; do
                              if [[ "$uCritLoss1" == "True" ]]; then
                                criticW1_list=("${uncertainty_critic_weight_1[@]}")
                              else
                                criticW1_list=(0.0)
                              fi

                              for uCritLoss2 in "${use_uncertainty_critic_loss_2[@]}"; do
                                if [[ "$uCritLoss2" == "True" ]]; then
                                  criticW2_list=("${uncertainty_critic_weight_2[@]}")
                                else
                                  criticW2_list=(0.0)
                                fi

                                # Finally, nest loops for the actual weight combos
                                for aW1 in "${actorW1_list[@]}"; do
                                  for aW2 in "${actorW2_list[@]}"; do
                                    for cW1 in "${criticW1_list[@]}"; do
                                      for cW2 in "${criticW2_list[@]}"; do
                                        # echo "Running job #$jobCounter with $AGO, layers=$NL, hidden=$NH, actf=$ACT, LR=$LR, steps=$NS, dim=$DIM"
                                        # echo "  attention1=$att1, heads1=$NAH1, attention2=$att2, heads2=$NAH2"
                                        # echo "  uncActor1=$uActLoss1, wActor1=$aW1, uncActor2=$uActLoss2, wActor2=$aW2"
                                        # echo "  uncCrit1=$uCritLoss1, wCrit1=$cW1, uncCrit2=$uCritLoss2, wCrit2=$cW2"

                                        # Append to file
                                        echo "--algo $AGO --num_layers $NL --num_hidden_neurons $NH --activation_func $ACT --learning_rate $LR --n_steps $NS --conservative_alpha 0 --latent_dim $DIM --use_attention_1 $att1 --num_attention_heads_1 $NAH1 --use_attention_2 $att2 --num_attention_heads_2 $NAH2 --use_uncertainty_actor_loss_1 $uActLoss1 --uncertainty_actor_weight_1 $aW1 --use_uncertainty_actor_loss_2 $uActLoss2 --uncertainty_actor_weight_2 $aW2 --use_uncertainty_critic_loss_1 $uCritLoss1 --uncertainty_critic_weight_1 $cW1 --use_uncertainty_critic_loss_2 $uCritLoss2 --uncertainty_critic_weight_2 $cW2" >> "params_dqlearning.txt"

                                        jobCounter=$((jobCounter + 1))
                                      done
                                    done
                                  done
                                done

                              done
                            done
                          done
                        done

                      done
                    done
                  done
                done


              elif [[ "$AGO" == "UA-SAC" || "$AGO" == "UA-CQL" ]]; then
                CA=0  # not used for UA-SAC/UHA-SAC, presumably

               
                  for att2 in "${use_attention_2[@]}"; do

                    # If att1 is True, loop over num_attention_heads_1
                    # If att1 is False, we can set heads to a default of 1 (or skip).
                   

                    # Similarly for att2
                    if [[ "$att2" == "True" ]]; then
                      heads2_list=("${num_attention_heads_2[@]}")
                    else
                      heads2_list=(1)
                    fi

                  
                      for NAH2 in "${heads2_list[@]}"; do


                          for uActLoss2 in "${use_uncertainty_actor_loss_2[@]}"; do
                            if [[ "$uActLoss2" == "True" ]]; then
                              actorW2_list=("${uncertainty_actor_weight_2[@]}")
                            else
                              actorW2_list=(0.0)
                            fi

              

                              for uCritLoss2 in "${use_uncertainty_critic_loss_2[@]}"; do
                                if [[ "$uCritLoss2" == "True" ]]; then
                                  criticW2_list=("${uncertainty_critic_weight_2[@]}")
                                else
                                  criticW2_list=(0.0)
                                fi

                                # Finally, nest loops for the actual weight combos
                               
                                  for aW2 in "${actorW2_list[@]}"; do
                                   
                                      for cW2 in "${criticW2_list[@]}"; do
                                        # echo "Running job #$jobCounter with $AGO, layers=$NL, hidden=$NH, actf=$ACT, LR=$LR, steps=$NS, dim=$DIM"
                                        # echo "  attention1=$att1, heads1=$NAH1, attention2=$att2, heads2=$NAH2"
                                        # echo "  uncActor1=$uActLoss1, wActor1=$aW1, uncActor2=$uActLoss2, wActor2=$aW2"
                                        # echo "  uncCrit1=$uCritLoss1, wCrit1=$cW1, uncCrit2=$uCritLoss2, wCrit2=$cW2"

                                        # Append to file
                                        echo "--algo $AGO --num_layers $NL --num_hidden_neurons $NH --activation_func $ACT --learning_rate $LR --n_steps $NS --conservative_alpha 0 --latent_dim $DIM  --use_attention_2 $att2 --num_attention_heads_2 $NAH2  --use_uncertainty_actor_loss_2 $uActLoss2 --uncertainty_actor_weight_2 $aW2 --use_uncertainty_critic_loss_2 $uCritLoss2 --uncertainty_critic_weight_2 $cW2" >> "params_dqlearning.txt"

                                        jobCounter=$((jobCounter + 1))
                                      done
                                    done
                                  done
                                done

                   
                          done
                        done

               
               
               
             
             

              # -----------------------------------------
              # -----------------------------------------
              # 3) All other algos: standard one-liner
              # -----------------------------------------
              else
                CA=0
                echo "Running job #$jobCounter with $AGO $NL $NH $ACT $LR $NS $DIM for DQlearning..."
                echo "--algo $AGO --num_layers $NL --num_hidden_neurons $NH --activation_func $ACT --learning_rate $LR --n_steps $NS --conservative_alpha $CA --latent_dim $DIM" >> "params_dqlearning.txt"
                jobCounter=$((jobCounter + 1))
              fi

            done
          done
        done
      done
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


jobID1=$(sbatch --array=0-$jobCounter  DQlearning.sh)
echo $jobID1




