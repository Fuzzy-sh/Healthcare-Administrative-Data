import numpy as np
import pickle
import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
print(sys.path)
from utils import (
    LoadEpisodes,
    PreprocessData,
    PolicyResolver,
    Encoder,
    DatasetFromEpisodes,
    LoadEncoder,
)
from utils import WeightedImportanceSampling
from config.config_param import config_param
import argparse
import datetime
import torch
import mdpdataloader

from RLModels_2layer import *


from collections import defaultdict
from utils import add_columns_level_1, add_columns_level_2, add_columns_level_3
import pandas as pd
import os
import multiprocessing
from d3rlpy.dataset import MDPDataset


# Define function for parallel evaluation
def evaluate_policy(eval_episodes, eval_name, policy_name, policy, physician_policy):
    wis, ci, f1 = WI.weighted_importance_sampling_with_bootstrap(
        eval_episodes,
        0.99,
        policy,
        physician_policy,
        num_bootstrap_samples=1000,
        N=1000,  # Reduced samples for speedup
    )
    return {
        f"{policy_name}_{eval_name}_wis": wis,
        f"{policy_name}_{eval_name}_ci": str(ci),
        f"{policy_name}_{eval_name}_f1": f1,
    }


if __name__ == "__main__":

    # 1- setting all the parameters
    ############################################################################################################
    config_params = config_param()
    # read the data from config
    param_dic = config_params.get_params_updated()
    encoder = Encoder()

    pred = PreprocessData(param_dic)
    WI = WeightedImportanceSampling()

    model_path = param_dic["models_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # encoder factory

    parser = argparse.ArgumentParser(
        description="Run the clustering-based inference using Q-Learning"
    )
    parser.add_argument(
        "--experiment_type",
        type=str,
        default="autoencoder",
        help="The type of the experiment (autoencoder_kmeans, autoencoder_sigmoid_8_kmeans, vanilla_kmeans)",
    )
    # parser.add_argument('--autoencoder_name', type=str, default='autoencoder_sigmoid_64', help='The name of the autoencoder model')
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=70,
        help="The number of clusters for the observation",
    )
    parser.add_argument(
        "--n_encoder_clusters",
        type=int,
        default=70,
        help="The number of clusters for the encoder observation",
    )
    parser.add_argument(
        "--n_clusters_test",
        type=int,
        default=70,
        help="The number of clusters for the observation",
    )
    parser.add_argument(
        "--n_encoder_clusters_test",
        type=int,
        default=70,
        help="The number of clusters for the encoder observation",
    )
    parser.add_argument(
        "--n_clusters_val",
        type=int,
        default=70,
        help="The number of clusters for the observation",
    )
    parser.add_argument(
        "--n_encoder_clusters_val",
        type=int,
        default=70,
        help="The number of clusters for the encoder observation",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=64,
        help="The latent dimension of the autoencoder",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="SAC",
        help="The name of the reinforcement learning algorithm",
    )
    # parser.add_argument('--act_n_clusters', type=str, default='15', help='The name of the reward function')
    parser.add_argument(
        "--experiment_name", type=str, default="", help="The name of the experiment"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="The number of layers in the neural network",
    )
    parser.add_argument(
        "--num_hidden_neurons",
        type=int,
        default=64,
        help="The number of hidden neurons in the neural network",
    )
    parser.add_argument(
        "--activation_func",
        type=str,
        default="relu",
        help="The activation function of the neural network",
    )
    parser.add_argument(
        "--learning_rate_list",
        type=float,
        default=0.0001,
        help="The learning rate of the neural network",
    )
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        default=0.00001,
        help="The learning rate of the neural network",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=1000000,
        help="The number of steps in the neural network",
    )
    parser.add_argument(
        "--n_steps_per_epoch",
        type=int,
        default=100000,
        help="The number of steps per epoch in the neural network",
    )
    parser.add_argument(
        "--conservative_alpha",
        type=float,
        default=0.5,
        help="The conservative alpha parameter for CQL",
    )
    parser.add_argument(
        "--cluster_type",
        type=str,
        default="kmeans",
        help="The type of the clustering algorithm",
    )

    parser.add_argument(
        "--use_attention_1",
        type=str,
        default="True",
        help="The use of attention mechanism in the first level",
    )
    parser.add_argument(
        "--use_attention_2",
        type=str,
        default="True",
        help="The use of attention mechanism in the second level",
    )
    parser.add_argument(
        "--num_attention_heads_1",
        type=int,
        default=4,
        help="The number of attention heads in the first level",
    )
    parser.add_argument(
        "--num_attention_heads_2",
        type=int,
        default=4,
        help="The number of attention heads in the second level",
    )
    parser.add_argument(
        "--use_uncertainty_actor_loss_1",
        type=str,
        default="True",
        help="The use of uncertainty actor loss in the first level",
    )
    parser.add_argument(
        "--use_uncertainty_actor_loss_2",
        type=str,
        default="True",
        help="The use of uncertainty actor loss in the second level",
    )
    parser.add_argument(
        "--use_uncertainty_critic_loss_1",
        type=str,
        default="True",
        help="The use of uncertainty critic loss in the first level",
    )
    parser.add_argument(
        "--use_uncertainty_critic_loss_2",
        type=str,
        default="True",
        help="The use of uncertainty critic loss in the second level",
    )
    parser.add_argument(
        "--uncertainty_actor_weight_1",
        type=float,
        default=1.0,
        help="The weight of the uncertainty actor loss in the first level",
    )
    parser.add_argument(
        "--uncertainty_actor_weight_2",
        type=float,
        default=1.0,
        help="The weight of the uncertainty actor loss in the second level",
    )
    parser.add_argument(
        "--uncertainty_critic_weight_1",
        type=float,
        default=1.0,
        help="The weight of the uncertainty critic loss in the first level",
    )
    parser.add_argument(
        "--uncertainty_critic_weight_2",
        type=float,
        default=1.0,
        help="The weight of the uncertainty critic loss in the second level",
    )
    parser.add_argument(
        "--arc",
        type=str,
        default="False",
        help="The use of attention mechanism in the second level",
    )

    args = parser.parse_args()
    experiment_type = args.experiment_type
    n_clusters = args.n_clusters
    n_encoder_clusters = args.n_encoder_clusters
    n_clusters_test = args.n_clusters_test
    n_encoder_clusters_test = args.n_encoder_clusters_test
    n_clusters_val = args.n_clusters_val
    n_encoder_clusters_val = args.n_encoder_clusters_val
    latent_dim = args.latent_dim
    cluster_type = args.cluster_type
    algo = args.algo
    ARC = True if args.arc == "True" else False
    use_attention_1 = False if args.use_attention_1 == "False" else True
    use_attention_2 = False if args.use_attention_2 == "False" else True

    num_attention_heads_1 = args.num_attention_heads_1
    num_attention_heads_2 = args.num_attention_heads_2
    use_uncertainty_actor_loss_1 = (
        False if args.use_uncertainty_actor_loss_1 == "False" else True
    )
    use_uncertainty_actor_loss_2 = (
        False if args.use_uncertainty_actor_loss_2 == "False" else True
    )
    use_uncertainty_critic_loss_1 = (
        False if args.use_uncertainty_critic_loss_1 == "False" else True
    )
    use_uncertainty_critic_loss_2 = (
        False if args.use_uncertainty_critic_loss_2 == "False" else True
    )
    uncertainty_actor_weight_1 = args.uncertainty_actor_weight_1
    uncertainty_actor_weight_2 = args.uncertainty_actor_weight_2
    uncertainty_critic_weight_1 = args.uncertainty_critic_weight_1
    uncertainty_critic_weight_2 = args.uncertainty_critic_weight_2

    behavior_policy_path = "./results/rl_policies_autoencoder_train_cluster70_70_test_cluster_70_70_20250112-224409.pkl"
    best_physician_policy = pickle.load(open(behavior_policy_path, "rb"))
    best_physician_policy = best_physician_policy["physician"]
    num_layers = args.num_layers
    num_hidden_neurons = args.num_hidden_neurons
    activation_func = args.activation_func
    critic_learning_rate = args.learning_rate_list

    actor_learning_rate = args.actor_learning_rate
    n_steps = args.n_steps
    if "CQL" in algo:
        conservative_alpha = float(args.conservative_alpha)
    else:
        conservative_alpha = None

    ob_prefix = param_dic["observation_prefix"]

    use_autoencoder = True

    autoencoder_type = "autoencoder_sigmoid"
    # if use_autoencoder:
    #     experiment_name = f"{algo}-with-{autoencoder_type}-{latent_dim}__{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}" if args.experiment_name == '' else args.experiment_name
    # else:
    #     experiment_name = f"{algo}__{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}" if args.experiment_name == '' else args.experiment_name
    experiment_name = (
        f"{algo}"
        f"_cl{n_clusters}"
        f"_ec{n_encoder_clusters}"
        f"_ld{latent_dim}"
        f"_ly{num_layers}"
        f"_hn{num_hidden_neurons}"
        f"_af{activation_func}"
        f"_lrA{actor_learning_rate}"
        f"_lrC{critic_learning_rate}"
        f"_st{n_steps}"
        # Combine actor flags & weights for level 1 and 2 into a single token:
        f"_uA({use_uncertainty_actor_loss_1}-{uncertainty_actor_weight_1}"
        f"|{use_uncertainty_actor_loss_2}-{uncertainty_actor_weight_2})"
        f"_uC({use_uncertainty_critic_loss_1}-{uncertainty_critic_weight_1}"
        f"|{use_uncertainty_critic_loss_2}-{uncertainty_critic_weight_2})"
        # Example merging attention flags & heads as well:
        f"_att1({use_attention_1}-h{num_attention_heads_1})"
        f"_att2({use_attention_2}-h{num_attention_heads_2})"
        # Shorter timestamp
        f"_{datetime.datetime.now().strftime('%y%m%d-%H%M')}"
    )

    os.makedirs(os.path.join(model_path, experiment_name), exist_ok=True)

    args = parser.parse_args()
    experiment_type = args.experiment_type

    # Call mdpdataloader.py to load the episodes
    ###################################################

    mdpdataloader_parameters = {
        "latent_dim": latent_dim,
        "ob_prefix": ob_prefix,
        "cluster_type": cluster_type,
        "n_clusters": n_clusters,
        "n_encoder_clusters": n_encoder_clusters,
        "n_clusters_test": n_clusters_test,
        "n_encoder_clusters_test": n_encoder_clusters_test,
        "n_clusters_val": n_clusters_val,
        "n_encoder_clusters_val": n_encoder_clusters_val
    }
    mdpdata_dict = mdpdataloader.load_episodes(
        mdpdataloader_parameters
    )


    #####################################################
    # put the behavior policy inside the reference episodes

    # Initialize dictionary to store action counts
    train_action_counts = defaultdict(int)
    test_action_counts = defaultdict(int)
    validation_action_counts = defaultdict(int)

    # Iterate over all episodes in the training dataset
    for episode in mdpdata_dict['train_episodes']:
        for i in range(
            len(episode.transitions)
        ):  # Iterate over transitions in each episode
            action = episode.transitions[
                i
            ].action_level_2  # Extract action from transition
            train_action_counts[action] += 1  # Increment count for that action

    # Iterate over all episodes in the training dataset
    for episode in mdpdata_dict['test_episodes']:
        for i in range(
            len(episode.transitions)
        ):  # Iterate over transitions in each episode
            action = episode.transitions[
                i
            ].action_level_2  # Extract action from transition
            test_action_counts[action] += 1  # Increment count for that action

    for episode in mdpdata_dict['validation_episodes']:
        for i in range(len(episode.transitions)):
            action = episode.transitions[i].action_level_2
            validation_action_counts[action] += 1
    # Print final action counts
    print("Action Counts in Train Dataset:", dict(train_action_counts))
    print("Action Counts in Test Dataset:", dict(test_action_counts))
    print("Action Counts in Validation Dataset:", dict(validation_action_counts))

    # only for the action_dict_level_1 for now
    for dataset, behavior_dataset in [
        (mdpdata_dict['train_episodes'], mdpdata_dict['behavior_train_episodes']),
        (mdpdata_dict['test_episodes'], mdpdata_dict['behavior_test_episodes']),
        (mdpdata_dict['validation_episodes'], mdpdata_dict['behavior_validation_episodes']),
    ]:
        for episode, behavior_episode in zip(dataset, behavior_dataset):
            for transition, behavior_transition in zip(episode, behavior_episode):
                behavior_state = behavior_transition.state
                behavior_state = int(behavior_state[0])
                transition.prediction_probs["best_physician_policy"] = (
                    best_physician_policy[behavior_state, :]
                )

                # single action policies
                for action in mdpdata_dict['action_dict_level_2'].keys():
                    transition.prediction_probs[f"single_action_policy_{action}"] = (
                        np.zeros(len(mdpdata_dict['action_dict_level_2']))
                    )
                    transition.prediction_probs[f"single_action_policy_{action}"][
                        action
                    ] = 1.0

    n_steps = args.n_steps
    n_steps_per_epoch = args.n_steps_per_epoch

    model_params = {
        "num_layers": num_layers,
        "num_hidden_neurons": num_hidden_neurons,
        "activation_func": activation_func,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "learning_rate": critic_learning_rate,
        "n_steps": n_steps,
        "conservative_alpha": conservative_alpha,
        "batch_size": 64,
        "gamma": 0.99,
        "n_epochs": n_steps // n_steps_per_epoch,
        "n_steps_per_epoch": n_steps_per_epoch,
        "n_updates_per_epoch": 1000,
        "target_update_interval": 1000,
        "eval_interval": 1000,
        "log_interval": 1000,
        # Actor parameters for the encoder factory
        "num_actor_layers": num_layers,
        "num_actor_hidden_neurons": num_hidden_neurons,
        "actor_activation_func": activation_func,
        # UHA_SAC leavel 1 parameters
        "use_uncertainty_actor_loss_1": use_uncertainty_actor_loss_1,
        "use_uncertainty_critic_loss_1": use_uncertainty_critic_loss_1,
        "uncertainty_critic_weight_1": uncertainty_critic_weight_1,
        "uncertainty_actor_weight_1": uncertainty_actor_weight_1,
        "use_attention_1": use_attention_1,
        "num_attention_heads_1": num_attention_heads_1,
        # UHA_SAC leavel 2 parameters
        "use_uncertainty_actor_loss_2": use_uncertainty_actor_loss_2,
        "use_uncertainty_critic_loss_2": use_uncertainty_critic_loss_2,
        "uncertainty_critic_weight_2": uncertainty_critic_weight_2,
        "uncertainty_actor_weight_2": uncertainty_actor_weight_2,
        "use_attention_2": use_attention_2,
        "num_attention_heads_2": num_attention_heads_2,
    }

    # # for CQL
    # if conservative_alpha is not None:

    DQmodel = DQNModel(algo, model_params, str(device))
    # experiment_name = algo
    # fit the model
    rl_model = DQmodel.fit_model(mdpdata_dict['train_d3r_dataset'], experiment_name)
    # model_path = os.path.join("d3rlpy_logs/", experiment_name + ".d3")
    # model_path_pt = os.path.join("d3rlpy_logs/", experiment_name + ".pt")



    # evaluate the model
    rl_policy = PolicyResolver(rl_model, list(mdpdata_dict['action_dict_level_2'].keys()))
    rl_greedy_policy = PolicyResolver(rl_model, list(mdpdata_dict['action_dict_level_2'].keys()), greedy=True)
    physician_policy = PolicyResolver('best_physician_policy', list(mdpdata_dict['action_dict_level_2'].keys()))
    physician_policy_greedy = PolicyResolver('best_physician_policy', list(mdpdata_dict['action_dict_level_2'].keys()), greedy=True)
    rl_policy = PolicyResolver(rl_model, list(mdpdata_dict['action_dict_level_2'].keys()))
    
    data_dict = {
        "test_rl": (mdpdata_dict['test_episodes'], rl_policy),
        "test_rl_greedy": (mdpdata_dict['test_episodes'], rl_greedy_policy),
        # "val_rl": (mdpdata_dict['validation_episodes'], rl_policy),
        # "val_rl_greedy": (mdpdata_dict['validation_episodes'], rl_greedy_policy),

        # "train_rl": (mdpdata_dict['train_episodes'], rl_policy),
        # "train_rl_greedy": (mdpdata_dict['train_episodes'], rl_greedy_policy),
    }
    WI = WeightedImportanceSampling()
    policy_eval_results={}
    results,df = WI.multi_weighted_importance_sampling_with_bootstrap(
        data_dict=data_dict,
        behavior_policy=physician_policy,  # one shared behavior policy
        gamma=0.99,
        num_bootstrap_samples=1000,
        N=1000,
        confidence_level=95,
        reward_name=None
    )
    EXPERIMENTS_RESULTS = "d3rlpy_logs/"        
    for name, res in results.items():
        wis = res["wis"]
        ci = res["ci"]
        f1 = res["f1"]
        dr = res["dr"]
        dr_ci = res["dr_ci"]
        print(f"{name}: WIS={wis}, CI={ci}, F1={f1} , DR={dr:.2f} , DR_CI={dr_ci}")
        policy_eval_results[name + "_wis"] = wis
        policy_eval_results[name + "_ci"] = str(ci)
        policy_eval_results[name + "_f1"] = f1
        policy_eval_results[name + "_dr"] = round(dr , 2)
        policy_eval_results[name + "_dr_ci"] = str(dr_ci)
        # with pd.ExcelWriter(EXPERIMENTS_RESULTS+experiment_name+".xlsx", engine='openpyxl') as writer:
        #     for subject_id, tb in df.items():
        #         tb.to_excel(writer, sheet_name=f"{subject_id}", index=False)


    evaluation_results = pd.concat([pd.DataFrame(policy_eval_results, index=[0])], ignore_index=True)
    # evaluation_results = pd.concat([evaluation_results, evaluation_results_], ignore_index=True, axis=0)
    evaluation_results.to_csv(os.path.join(EXPERIMENTS_RESULTS, f"{experiment_name}.csv"), index=False)
    file_path = "./results/rl_policies_autoencoder_train_cluster70_70_test_cluster_70_70_20250112-224409.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    # Extract policies
    physician = np.array(data['physician'])
    physician_state_action = {}
    s=0
    for state in physician:
        physician_state_action[s] = np.argmax(state)
        s += 1




    states = []
    behavior_states = []
    actions = []
    observed_action = []
    predicted_values = {}
    episode_count = 0
    transition_count = 0
    predicted_values_df = {}
    physician_policy = []
    transitions = []
   
    for Episode, behavior in zip(mdpdata_dict['test_episodes'], mdpdata_dict['behavior_test_episodes']):
        for transition, behavior_transition in zip(Episode.transitions, behavior.transitions):
            states.append (transition.state)
            behavior_states.append(behavior_transition.state)
            actions.append(transition.action_level_2)
            observed_action.append(behavior_transition.action_level_2)
            physician_policy.append(physician_state_action[behavior_transition.state[0]])
            transitions.append(transition_count)  
            transition_count += 1

        predicted = rl_model.predict(
            np.array(states),
        )

        for i in range(len(predicted)):
            
            predicted_values[i] = {
            
                "episode": episode_count,
                "transition": transitions[i],
                "predicted": predicted[i],
                "action": actions[i],
                "behavior_state": behavior_states[i][0],
                "physician_policy": physician_policy[i],
            }
        predicted_values_df[episode_count] = pd.DataFrame.from_dict(predicted_values, orient='index')
        episode_count += 1
    # concatenate the predicted_values_df which is a dictionary of dataframes into a single dataframe
    df = pd.concat(predicted_values_df.values(), ignore_index=True)
    # save the predicted values as a csv file
    df.to_csv(os.path.join(EXPERIMENTS_RESULTS, f"{experiment_name}_predicted_values.csv"), index=False)
    # save the predicted values as the list of dictionaries in csv file

    # predicted_values_df = pd.concat([pd.DataFrame.from_dict(predicted_values[i], orient='index') for i in range(len(predicted_values))])
    # predicted_values_df.to_csv(os.path.join(EXPERIMENTS_RESULTS, f"{experiment_name}_predicted_values.csv"), index=False)







    # rl_model.save(model_path)
    # rl_model.save(model_path_pt)
    # with open("logs/test_d3r_dataset.pkl", "wb") as f:
    #     pickle.dump(mdpdata_dict['test_d3r_dataset'], f)

    # Epoch-wise training with direct access


#     offlinerl_dataset = MDPDataset(
#     observations=train_d3r_dataset["state"],
#     actions=train_d3r_dataset["action"],
#     rewards=train_d3r_dataset["reward"],
#     terminals=train_d3r_dataset["done"],
# )
#     print(offlinerl_dataset)

#     exit()


# for epoch, metrics in rl_algo.fitter(
#     dataset=train_d3r_dataset,
#     n_steps=n_steps,
#     n_steps_per_epoch=n_steps_per_epoch,
#     experiment_name=experiment_name,
# ):
#     # After each epoch completes, you have direct access to your trained model
#     rl_model = rl_algo  # this is your trained algorithm instance at current epoch


#     # if algo =='SAC':
#     #     print(f"Epoch {epoch}:")
#     #     print(f"  SAC1 - Critic Loss: {metrics.get('sac1_critic_loss', 0.0):.2f}, "
#     #         f"Actor Loss: {metrics.get('sac1_actor_loss', 0.0):.2f}")
#     #     print(f"  SAC2 - Critic Loss: {metrics.get('sac2_critic_loss', 0.0):.2f}, "
#     #         f"Actor Loss: {metrics.get('sac2_actor_loss', 0.0):.2f}")


#     # else:
#     #     print(f"Epoch: {epoch}, Metrics: {metrics}")
#     # Now you can do whatever you want after each epoch, for example:
#     # - Evaluate your model
#     # - Save the current snapshot of the model
#     # - Use the model for inference or further experiments
#     # DQmodel.save_model(f"rl_model_epoch_{epoch}.pt")


#     # save full parameters and configurations in a single file.
#     if conservative_alpha is None:
#         # for normal DQN
#         model_name = f"{num_layers}__{num_hidden_neurons}__{activation_func}__{actor_learning_rate}__{critic_learning_rate}__{n_steps}"
#     else:
#         # for CQL
#         model_name = f"{num_layers}__{num_hidden_neurons}__{activation_func}__{actor_learning_rate}__{critic_learning_rate}__{n_steps}__{conservative_alpha}"


#     if ARC == True:
#          folder_name= '/work/messier_lab/fuzzy/DQ_resutls/results'
#     else:
#         folder_name = 'results'


#     policy_eval_results['epoch'] = epoch
#     policy_eval_results['experiment_name'] = experiment_name

#     EXPERIMENTS_RESULTS = folder_name+algo
#     if not os.path.exists(EXPERIMENTS_RESULTS):
#         os.makedirs(EXPERIMENTS_RESULTS, exist_ok=True)


#     model_path_ = os.path.join(EXPERIMENTS_RESULTS, experiment_name + '.d3')

#     rl_model.save(model_path_)
#     print('Model saved at:', model_path_)

#     # # # TODO: remove this line
#     # model_import_folder = 'models_obstructive_cad5/DQN-all-caths-reward-mace__20240415-155033'
#     # model_import_path = os.path.join(EXPERIMENTS_RESULTS, model_name + '.d3')

#     # rl_model = d3rlpy.load_learnable(model_path_)

#     # evaluate the model
#     rl_policy = PolicyResolver(rl_model, list(action_dict_level_2.keys()))
#     rl_greedy_policy = PolicyResolver(rl_model, list(action_dict_level_2.keys()), greedy=True)
#     physician_policy = PolicyResolver('best_physician_policy', list(action_dict_level_2.keys()))
#     physician_policy_greedy = PolicyResolver('best_physician_policy', list(action_dict_level_2.keys()), greedy=True)


#     for param_name, param_value in model_params.items():
#         policy_eval_results[param_name] = param_value

#     # # Run all policy evaluations in parallel
#     # results = Parallel(n_jobs=n_cores , backend="threading")(
#     #     delayed(evaluate_policy)(eval_episodes, eval_name, policy_name, policy, physician_policy)
#     #     for eval_episodes, eval_name in zip([test_episodes, validation_episodes, train_episodes], ['test', 'validation', 'train'])
#     #     for policy_name, policy in zip(['rl_policy', 'rl_greedy_policy'], [rl_policy, rl_greedy_policy])
#     # )

#     # # Update results
#     # for res in results:
#     #     policy_eval_results.update(res)

#     # # Best physician policy evaluations (parallelized)
#     # best_physician_results = Parallel(n_jobs=n_cores , backend="threading")(
#     #     delayed(evaluate_policy)(eval_episodes, eval_name, 'best_physician_policy', physician_policy, physician_policy)
#     #     for eval_episodes, eval_name in zip([test_episodes, validation_episodes, train_episodes], ['test', 'validation', 'train'])
#     # )

#     # for res in best_physician_results:
#     #     policy_eval_results.update(res)

#     # Best greedy physician policy
#     # best_physician_greedy_results = Parallel(n_jobs=n_cores , backend="threading")(
#     #     delayed(evaluate_policy)(eval_episodes, eval_name, 'best_physician_policy_greedy', physician_policy_greedy, physician_policy)
#     #     for eval_episodes, eval_name in zip([test_episodes, validation_episodes, train_episodes], ['test', 'validation', 'train'])
#     # )

#     # for res in best_physician_greedy_results:
#     #     policy_eval_results.update(res)

#     # Single Action Policies
#     # single_action_results = Parallel(n_jobs=n_cores , backend="threading")(
#     #     delayed(evaluate_policy)(eval_episodes, eval_name, f'single_action_policy_{action_id}',
#     #                             PolicyResolver(f'single_action_policy_{action_id}', list(action_dict_level_2.keys())),
#     #                             physician_policy)
#     #     for eval_episodes, eval_name in zip([test_episodes, validation_episodes, train_episodes], ['test', 'validation', 'train'])
#     #     for action_id in action_dict_level_2.keys()
#     # )

#     # for res in single_action_results:
#     #     policy_eval_results.update(res)

#     # # evaluate the policies
#     # for eval_episodes, eval_name in zip([test_episodes, validation_episodes, train_episodes], ['test', 'validation', 'train']):
#     #     for policy_name, policy in tqdm(zip(['rl_policy', 'rl_greedy_policy'], [rl_policy, rl_greedy_policy])):
#     #         wis, ci, f1 = WI.weighted_importance_sampling_with_bootstrap(eval_episodes, 0.99, policy, physician_policy, num_bootstrap_samples=1000, N=1000)
#     #         policy_eval_results[f'{policy_name}_{eval_name}_wis'] = wis
#     #         policy_eval_results[f'{policy_name}_{eval_name}_ci'] = str(ci)
#     #         print(f"Weighted importance sampling for the {policy_name} on the {eval_name} data: {wis:.2f}, CI: {ci} -- F1: {f1:.2f}")

#     data_dict = {
#         "test_rl": (test_episodes, rl_policy),
#         "test_rl_greedy": (test_episodes, rl_greedy_policy),
#         "val_rl": (validation_episodes, rl_policy),
#         "val_rl_greedy": (validation_episodes, rl_greedy_policy),

#         "train_rl": (train_episodes, rl_policy),
#         "train_rl_greedy": (train_episodes, rl_greedy_policy),
#     }
#     WI = WeightedImportanceSampling()

#     results,df = WI.multi_weighted_importance_sampling_with_bootstrap(
#         data_dict=data_dict,
#         behavior_policy=physician_policy,  # one shared behavior policy
#         gamma=0.99,
#         num_bootstrap_samples=1000,
#         N=1000,
#         confidence_level=95,
#         reward_name=None
#     )

#     for name, res in results.items():
#         wis = res["wis"]
#         ci = res["ci"]
#         f1 = res["f1"]
#         print(f"{name}: WIS={wis}, CI={ci}, F1={f1}")
#         policy_eval_results[name + "_wis"] = wis
#         policy_eval_results[name + "_ci"] = str(ci)
#         policy_eval_results[name + "_f1"] = f1
#         with pd.ExcelWriter(EXPERIMENTS_RESULTS+"/e_"+str(epoch)+name+experiment_name+".xlsx", engine='openpyxl') as writer:
#             for subject_id, tb in df.items():
#                 tb.to_excel(writer, sheet_name=f"{subject_id}", index=False)


#     # # evaluate the best physician policy
#     # for eval_episodes, eval_name in zip([test_episodes, validation_episodes, train_episodes], ['test', 'validation', 'train']):
#     #     wis, ci , f1= WI.weighted_importance_sampling_with_bootstrap(eval_episodes, 0.99, physician_policy, physician_policy, num_bootstrap_samples=1000, N=1000)
#     #     policy_eval_results[f'best_physician_policy_{eval_name}_wis'] = wis
#     #     policy_eval_results[f'best_physician_policy_{eval_name}_ci'] = str(ci)
#     #     # print(f"Weighted importance sampling for the best physician policy on the {eval_name} data: {wis}, CI: {ci} -- F1: {f1:.2f}")

#     #     # evaluate the greedy best physician policy
#     #     wis, ci, f1 = WI.weighted_importance_sampling_with_bootstrap(eval_episodes, 0.99, physician_policy_greedy, physician_policy, num_bootstrap_samples=1000, N=1000)
#     #     policy_eval_results[f'best_physician_policy_greedy_{eval_name}_wis'] = wis
#     #     policy_eval_results[f'best_physician_policy_greedy_{eval_name}_ci'] = str(ci)
#     #     # print(f"Weighted importance sampling for the greedy best physician policy on the {eval_name} data: {wis}, CI: {ci} -- F1: {f1:.2f}")

#     #     # evaluate the single action policies
#     #     for action_id in action_dict_level_2.keys():

#     #         single_action_policy = PolicyResolver(f'single_action_policy_{action_id}', list(action_dict_level_2.keys()))
#     #         wis, ci, f1 = WI.weighted_importance_sampling_with_bootstrap(eval_episodes, 0.99, single_action_policy, physician_policy, num_bootstrap_samples=1000, N=1000)
#     #         policy_eval_results[f'single_action_policy_{action_id}_{action_id}_wis'] = wis
#     #         policy_eval_results[f'single_action_policy_{action_id}_{action_id}_ci'] = str(ci)
#     #         # print(f"Weighted importance sampling for the single action policy {action_id} on the {eval_name} data: {wis}, CI: {ci} -- F1: {f1:.2f}")


#     evaluation_results_ = pd.concat([pd.DataFrame(policy_eval_results, index=[0])] , ignore_index=True)
#     evaluation_results = pd.concat([evaluation_results, evaluation_results_], ignore_index=True, axis=0)
#     evaluation_results.to_csv(os.path.join(EXPERIMENTS_RESULTS, f"{experiment_name}.csv"), index=False)
