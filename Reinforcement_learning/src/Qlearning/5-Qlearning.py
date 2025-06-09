from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, classification_report
from utils import (
    LoadEpisodes,
    Autoencoder,
    TrainAutoencoder,
    PolicyResolver,
    WeightedImportanceSampling,
    Encoder,
    QLearning,
)
from utils import PreprocessData, UpdateData
from config.config_param import config_param
import argparse
import datetime
import torch
from torch.utils.data import DataLoader, TensorDataset

import logging
from dataloader import DataLoaderClass
import mdptoolbox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.cluster import KMeans

from tqdm import tqdm

from itertools import product
from utils import add_columns_level_1, add_columns_level_2, add_columns_level_3


def do_q_learning_and_evaluate_policy(
    train_episodes_cluster,
    test_episodes_cluster,
    train_episodes_raw,
    test_episodes_raw,
    actions_dict_cluster,
    n_clusters_actions,
    n_clusters_state,
    only_train=False,
):

    behavior_plicy_name = "behavior_physician_policy"
    WI = WeightedImportanceSampling()

    # n_states =  len(observation_column)
    if only_train:
        print(
            "Training the Q-Learning model to gain the physician policy based on the clustering of the actions and states"
        )

    transition_matrix = np.zeros(
        (n_clusters_actions, n_clusters_state + 1, n_clusters_state + 1), dtype=float
    )
    reward_matrix = np.zeros(
        (n_clusters_actions, n_clusters_state + 1, n_clusters_state + 1), dtype=float
    )
    max_episode_length = max([len(episode) for episode in train_episodes_cluster])
    print("max episode length:", max_episode_length)

    for episode in tqdm(train_episodes_cluster):

        if len(episode) == 1:  # only one transition (reached to the terminal state)
            s = episode.transitions[0].state[0]
            a = episode.transitions[0].action_level_2

            r = episode.transitions[0].reward
            transition_matrix[a, s, n_clusters_state] += 1
            reward_matrix[a, s, n_clusters_state] += r

        else:
            #
            for i in range(len(episode) - 1):

                s = episode.transitions[i].state[0]
                a = episode.transitions[i].action_level_2

                r = episode.transitions[i].reward

                s_next = episode.transitions[i + 1].state[0]
                transition_matrix[a, s, s_next] += 1
                reward_matrix[a, s, s_next] += r

            # last transition
            s = episode.transitions[-1].state[0]
            a = episode.transitions[-1].action_level_2

            r = episode.transitions[-1].reward
            transition_matrix[a, s, n_clusters_state] += 1
            reward_matrix[a, s, n_clusters_state] += r

    # calculate physician's policy (probability of each action for each state)
    sum_over_states = transition_matrix.sum(axis=2)
    sum_over_actions = sum_over_states.sum(axis=0)
    sum_over_states[:, sum_over_actions == 0] = (
        1  # set the states with no actions to 1 to avoid division by 0
    )
    sum_over_actions = sum_over_states.sum(axis=0, keepdims=True)

    physician_policy = sum_over_states / sum_over_actions
    physician_policy = physician_policy.T

    print("make sure that the transition matrix is stochastic (sum of each row is 1)")
    print("then normalize the rows of the transition matrix")
    for i in tqdm(range(transition_matrix.shape[0])):
        unreachable_states = np.where(transition_matrix[i, :, :].sum(axis=1) == 0)[0]
        transition_matrix[i, unreachable_states, :] = 1.0
        transition_matrix[i, :, :] = transition_matrix[i, :, :] / transition_matrix[
            i, :, :
        ].sum(axis=1, keepdims=True)

    print("Initialize Q-Learning")
    # # Initialize Q-Learning
    # q_learning = QLearning(transition_matrix, reward_matrix, discount=0.99, n_iter=1000000 * int(1 + n_clusters_state/100))
    # ql = mdptoolbox.mdp.QLearning(transition_matrix, reward_matrix, discount=0.99, n_iter=1000000 * int(1 + n_clusters_state/100))
    # print("Train Q-Learning")
    # q_learning.train()
    # optimal_policy = q_learning.get_optimal_policy()
    # Q_matrix = q_learning.get_q_matrix()

    ql = mdptoolbox.mdp.QLearning(
        transition_matrix,
        reward_matrix,
        discount=0.99,
        n_iter=max_episode_length * 1000,
    )  # * int(1 + n_clusters_state/100))

    print("Run Q-Learning")
    ql.setVerbose()
    ql.run()

    # Get the optimal policy and Q-matrix
    optimal_policy = ql.policy
    Q_matrix = ql.Q

    # calculate the optimal policy probabilities
    Q_matrix_sum = np.sum(Q_matrix, axis=1, keepdims=True)
    Q_matrix_sum[Q_matrix_sum == 0] = (
        1  # set the states with no actions to 1 to avoid division by 0
    )
    optimal_policy_probs = Q_matrix / Q_matrix_sum

    print("calculate the greedy policy for the physician", len(actions_dict_cluster))
    greedy_physician_policy = np.zeros((len(optimal_policy), len(actions_dict_cluster)))
    for i in range(physician_policy.shape[0]):
        greedy_physician_policy[i, np.argmax(physician_policy[i, :])] = 1.0

    print("calculate the greedy policy for the optimal policy")
    greedy_optimal_policy = np.zeros((len(optimal_policy), len(actions_dict_cluster)))
    for i in range(optimal_policy_probs.shape[0]):
        greedy_optimal_policy[i, np.argmax(optimal_policy_probs[i, :])] = 1.0

    # save the policies to a file
    policies = {
        "physician": physician_policy,
        "optimal": optimal_policy_probs,
        "greedy_physician": greedy_physician_policy,
    }

    if only_train:
        print(
            "Physician policy has been trained successfully! and getting back to the main function"
        )
        return None, policies

    # EVALUATION with the Behavior Policy: Same Clustering Model
    # resolve the policies
    print("---------------------------------------------------------------------------")
    print("EVALUATION with the Behavior Policy: Same Clustering Model")

    physician_policy_resolver = PolicyResolver(
        physician_policy, list(actions_dict_cluster.keys())
    )
    greedy_physician_policy_resolver = PolicyResolver(
        greedy_physician_policy, list(actions_dict_cluster.keys())
    )
    optimal_policy_resolver = PolicyResolver(
        optimal_policy_probs, list(actions_dict_cluster.keys())
    )
    greedy_optimal_policy_resolver = PolicyResolver(
        greedy_optimal_policy, list(actions_dict_cluster.keys())
    )

    # Evaluate the policies
    results = {}
    for dataset_name, dataset in [
        ("test", test_episodes_cluster),
        ("train", train_episodes_cluster),
    ]:
        for eval_policy_name, eval_policy_resolver in [
            ("physician", physician_policy_resolver),
            ("greedy_physician", greedy_physician_policy_resolver),
            ("optimal", optimal_policy_resolver),
            ("greedy_optimal", greedy_optimal_policy_resolver),
        ]:

            try:
                wis, ci, f1 = WI.weighted_importance_sampling_with_bootstrap(
                    dataset,
                    0.99,
                    eval_policy_resolver,
                    physician_policy_resolver,
                    num_bootstrap_samples=1000,
                    N=1000,
                    confidence_level=0.95,
                )
                # wis, ci = WI.weighted_importance_sampling_with_bootstrap(dataset, 0.99, eval_policy_resolver,num_bootstrap_samples=1000, N=1000, confidence_level=0.95)
                results[f"{eval_policy_name}_{dataset_name}"] = [wis, ci[0], ci[1], f1]

            except Exception as e:
                print("Error:", e)

    print("---------------------------------------------------------------------------")
    print(
        "EVALUATION with the Behavior Policy: Best Clustering Model for the Physician Imitation"
    )

    current_physician_policy_probs = policies["physician"]
    current_physician_policy_greedy_probs = policies["greedy_physician"]

    # make greedy optimal policy
    greedy_optimal_policy_probs = np.zeros(
        (optimal_policy_probs.shape[0], optimal_policy_probs.shape[1])
    )
    greedy_optimal_policy_probs[
        np.arange(optimal_policy_probs.shape[0]),
        np.argmax(optimal_policy_probs, axis=1),
    ] = 1.0

    print("put the best physician policy inside the episodes")
    for dataset, raw_dataset in [
        (train_episodes_cluster, train_episodes_raw),
        (test_episodes_cluster, test_episodes_raw),
    ]:
        for ep, episode in enumerate(dataset):
            raw_episode = raw_dataset[ep]
            for tr, transition in enumerate(episode):
                state = transition.state
                state = int(state[0])
                transition.prediction_probs["optimal"] = optimal_policy_probs[
                    state, :
                ]  # put the optimal policy inside the episode
                transition.prediction_probs["greedy_optimal"] = (
                    greedy_optimal_policy_probs[state, :]
                )  # put the greedy optimal policy inside the episode
                transition.prediction_probs["current_physician_policy"] = (
                    current_physician_policy_probs[state, :]
                )  # put the current physician policy inside the episode
                transition.prediction_probs["current_greedy_physician_policy"] = (
                    current_physician_policy_greedy_probs[state, :]
                )  # put the current physician policy inside the episode
                transition.prediction_probs[
                    behavior_plicy_name
                ] = raw_episode.transitions[tr].prediction_probs[
                    behavior_plicy_name
                ]  # put the physician's policy inside the episode

    print("evaluate using the imitator model as the behavior policy")
    behavior_plicy_resolver = PolicyResolver(
        behavior_plicy_name, list(actions_dict_cluster.keys())
    )
    for dataset_name, dataset in [
        ("test", test_episodes_cluster),
        ("train", train_episodes_cluster),
    ]:
        for eval_policy_name in [
            "optimal",
            "greedy_optimal",
            "current_physician_policy",
            "current_greedy_physician_policy",
        ]:
            try:
                eval_policy_resolver = PolicyResolver(
                    eval_policy_name, list(actions_dict_cluster.keys())
                )
                wis, ci, f1 = WI.weighted_importance_sampling_with_bootstrap(
                    dataset,
                    0.99,
                    eval_policy_resolver,
                    behavior_plicy_resolver,
                    num_bootstrap_samples=1000,
                    N=1000,
                    confidence_level=0.95,
                )
                results[
                    f"{eval_policy_name}_{dataset_name}_vs_{behavior_plicy_name}"
                ] = [wis, ci[0], ci[1], f1]
            except Exception as e:
                print("Error:", e)
                results[
                    f"{eval_policy_name}_{dataset_name}_vs_{behavior_plicy_name}"
                ] = [0, 0, 0, 0]
    print(results)

    return results, policies


if __name__ == "__main__":

    config_params = config_param()
    # read the data from config
    param_dic = config_params.get_params_updated()
    encoder = Encoder()

    pred = PreprocessData(param_dic)
    WI = WeightedImportanceSampling()

    model_path = param_dic["models_path"]
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
        "--latent_dim",
        type=int,
        default=64,
        help="The latent dimension of the autoencoder",
    )
    parser.add_argument(
        "--cluster_type",
        type=str,
        default="kmeans",
        help="The name of the reward function",
    )

    args = parser.parse_args()
    experiment_type = args.experiment_type
    n_clusters = args.n_clusters
    n_encoder_clusters = args.n_encoder_clusters
    n_clusters_test = args.n_clusters_test
    n_encoder_clusters_test = args.n_encoder_clusters_test
    latent_dim = args.latent_dim
    cluster_type = args.cluster_type

    # print("create aciton train encoded")
    # train_actions_encoded = [ encoder.action_to_index(train_actions_dict, action, param_dic['colgroup_treatment']) for action in train_actions]
    # actions_dict_update = train_actions_dict #{key:value for key,value in train_actions_dict.items() if key in set(train_actions_encoded)}
    # print("Actions dictionary:", actions_dict_update)
    ob_prefix = param_dic["observation_prefix"]

    print("Loading the train, test, and validation data")

    train_path, test_path, val_path = (
        param_dic["split_files"]["train"],
        param_dic["split_files"]["test"],
        param_dic["split_files"]["val"],
    )
    train_data, test_data, val_data = (
        pred.read_data(train_path),
        pred.read_data(test_path),
        pred.read_data(val_path),
    )
    print("Data loaded successfully!")

    # print("Loading the actions dictionary")
    # train_actions_dict = pred.read_data_pickle(param_dic['action_dic_filename_treatment_group']['train'])
    # train_action_dic_cluster = train_actions_dict #pred.read_data_pickle(param_dic['action_dic_filename_cluster']['train'])
    # train_action_dic_encoder_cluster = train_actions_dict #pred.read_data_pickle(param_dic['action_dic_filename_encoder']['train'])

    folder_dim = f"{cluster_type}_latend_dim_{latent_dim}/"
    folder_cluster = os.path.join(param_dic["split_data_path"], folder_dim)
    train_cluster_path = os.path.join(folder_cluster, train_path.split("/")[-1])
    test_cluster_path = os.path.join(folder_cluster, test_path.split("/")[-1])
    val_cluster_path = os.path.join(folder_cluster, val_path.split("/")[-1])
    train_data_cluster, test_data_cluster, val_data_cluster = (
        pred.read_data(train_cluster_path),
        pred.read_data(test_cluster_path),
        pred.read_data(val_cluster_path),
    )
    cluster_col_train = f"obs_cluster_{n_clusters}"
    cluster_encoder_col_train = f"obs_encoder_cluster_{n_encoder_clusters}"
    cluster_col_test = f"obs_cluster_{n_clusters_test}"
    cluster_encoder_col_test = f"obs_encoder_cluster_{n_encoder_clusters_test}"
    train_data = pd.concat(
        [
            train_data,
            train_data_cluster[[cluster_col_train, cluster_encoder_col_train]],
        ],
        axis=1,
    )
    test_data = pd.concat(
        [test_data, test_data_cluster[[cluster_col_test, cluster_encoder_col_test]]],
        axis=1,
    )
    # val_data = pd.concat([val_data, val_data_cluster[[cluster_col, cluster_encoder_col]]], axis=1)

    ##################################################################
    # actions_dict = param_dic['actions_dict']
    # print ("Actions dictionary:", actions_dict)
    # print(param_dic['action_columns'])

    # 2- Loading the data and set the actions based on the level of the actions
    ############################################################################################################

    # print("Loading the train, test, and validation data")

    # train_path ,test_path, val_path = param_dic['split_with_cluster_files']['train'], param_dic['split_with_cluster_files']['test'], param_dic['split_with_cluster_files']['val']
    # train_data, test_data, val_data = pred.read_data(train_path), pred.read_data(test_path), pred.read_data(val_path)
    # print("Data loaded successfully!")

    action_columns_level_1 = param_dic["action_columns_level_1"][0]
    action_columns_level_2 = param_dic["action_columns_level_2"]
    action_columns_level_3 = param_dic["action_columns_level_3"]

    train_data[action_columns_level_1] = add_columns_level_1(train_data)
    test_data[action_columns_level_1] = add_columns_level_1(test_data)
    val_data[action_columns_level_1] = add_columns_level_1(val_data)

    train_data = add_columns_level_2(train_data, action_columns_level_2)
    test_data = add_columns_level_2(test_data, action_columns_level_2)
    val_data = add_columns_level_2(val_data, action_columns_level_2)

    train_data = add_columns_level_3(train_data)
    test_data = add_columns_level_3(test_data)
    val_data = add_columns_level_3(val_data)

    action_dict_level_1 = param_dic["actions_level_1_dict"]
    action_dict_level_2 = param_dic["actions_level_2_dict"]
    action_dict_level_3 = param_dic["actions_level_3_dict"]

    ##############################################################
    # Group treatments and save results
    # group_treatments = UpdateData(param_dic)
    # train_data = group_treatments.get_group_treatments(train_data, param_dic)
    # test_data = group_treatments.get_group_treatments(test_data, param_dic)
    # val_data = group_treatments.get_group_treatments(val_data, param_dic)
    # for col in train_data.columns:
    #     print (col)
    # exit()
    ##############################################################
    print("loadig episodes (train, test and valiadtion) based on raw data")
    le = LoadEpisodes(param_dic)

    # the clustering observation and action for the encoder has been added to the train and test sets, now we can evaluate the policies

    train_episodes_raw, train_episodes_cluster, train_episodes_encoder_cluster = (
        le.load_episodes(train_data, cluster_col_train, cluster_encoder_col_train)
    )  # train_episodes_raw , train_episodes_cluster , train_episodes_encoder_cluster#le.load_episodes('train')
    test_episodes_raw, test_episodes_cluster, test_episodes_encoder_cluster = (
        le.load_episodes(test_data, cluster_col_test, cluster_encoder_col_test)
    )  # train_episodes_raw , train_episodes_cluster , train_episodes_encoder_cluster#le.load_episodes('test')
    # val_episodes_raw, val_episodes_cluster, val_episodes_encoder_cluster =le.load_episodes(val_data, cluster_col, cluster_encoder_col)
    # train_episodes_raw , train_episodes_cluster , train_episodes_encoder_cluster#le.load_episodes('val')

    print("load states and actions")
    # train_states, train_actions = le.load_states_actions(train_episodes_raw)
    # train_states_cluster, train_actions_cluster = le.load_states_actions(train_episodes_cluster)
    # train_states_encoder, train_actions_encoder = le.load_states_actions(train_episodes_encoder_cluster)
    # for action in train_actions:
    # train_actions_encoded = [encoder.action_to_index(train_actions_dict, action, param_dic['colgroup_treatment']) for action in train_actions]
    # print("Actions dictionary:", train_actions)

    # 1- for the first time the q learning will be train and return the physician policy based on the clustering of the actions and states

    # print("create aciton train encoded", train_actions_encoded)
    # update the action dict based on the train_action_encoded unque values
    # print ("Actions dictionary:", train_action_dic_cluster)
    train_n_clusters_state = n_clusters
    n_actions = len(action_dict_level_2)

    _, behavior_physician_policy = do_q_learning_and_evaluate_policy(
        train_episodes_cluster,
        test_episodes_cluster,
        train_episodes_raw,
        test_episodes_raw,
        action_dict_level_2,
        n_actions,
        train_n_clusters_state,
        only_train=True,
    )

    behavior_physician_policy = behavior_physician_policy["physician"]
    behavior_train_episodes, behavior_test_episodes = (
        train_episodes_cluster,
        test_episodes_cluster,
    )

    # 2- once we have the physician policy, we can evaluate the policies using the physician policy as the behavior policy
    for dataset, behavior_dataset in [
        (train_episodes_raw, behavior_train_episodes),
        (test_episodes_raw, behavior_test_episodes),
    ]:
        for episode, behavior_episode in zip(dataset, behavior_dataset):
            for transition, behavior_transition in zip(episode, behavior_episode):
                behavior_state = behavior_transition.state
                behavior_state = int(behavior_state[0])

                transition.prediction_probs["behavior_physician_policy"] = (
                    behavior_physician_policy[behavior_state, :]
                )

                # random action policies
                random_action = np.random.choice(list(action_dict_level_2.keys()))
                transition.prediction_probs["random"] = np.zeros(
                    len(action_dict_level_2.keys())
                )
                transition.prediction_probs["random"][random_action] = 1.0

                # single action policies
                for action_id in action_dict_level_2.keys():
                    transition.prediction_probs[str(action_id)] = np.zeros(
                        len(action_dict_level_2.keys())
                    )
                    transition.prediction_probs[str(action_id)][action_id] = 1.0

    train_n_encoder_clusters_state = n_encoder_clusters

    results, policies = do_q_learning_and_evaluate_policy(
        train_episodes_encoder_cluster,
        test_episodes_encoder_cluster,
        train_episodes_raw,
        test_episodes_raw,
        action_dict_level_2,
        n_actions,
        train_n_encoder_clusters_state,
        only_train=False,
    )

    experiment_name = f"{experiment_type}_train_cluster{n_clusters}_{n_encoder_clusters}_test_cluster_{n_clusters_test}_{n_encoder_clusters_test}"

    # policies_file = os.path.join(tc.models_path, f'{experiment_type}_models', f'policies_{n_clusters}.pkl')

    # results_row = {'type': 'encoder_clustering'}
    results_row = {}
    results_df = pd.DataFrame(columns=["policy_name", "wis", "LB", "UB", "f1"])
    for key, value in results.items():

        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame(
                    {
                        "policy_name": key,
                        "wis": value[0],
                        "LB": value[1],
                        "UB": value[2],
                        "f1": value[3],
                    },
                    index=[0],
                ),
            ],
            axis=0,
        )

    print(
        "Evaluate the policies with the behavior policy: Behavior Policy itself, a Random Policy, and Single Action Policies"
    )
    behavior_plicy = PolicyResolver(
        "behavior_physician_policy", list(action_dict_level_2.keys())
    )
    for dataset_name, dataset in [
        ("test", test_episodes_raw),
        ("train", train_episodes_raw),
    ]:
        for policy_name in ["behavior_physician_policy", "random"] + [
            str(k) for k in action_dict_level_2.keys()
        ]:
            policy_resolver = PolicyResolver(
                policy_name, list(action_dict_level_2.keys())
            )

            try:
                wis, ci, f1 = WI.weighted_importance_sampling_with_bootstrap(
                    dataset,
                    0.99,
                    policy_resolver,
                    behavior_plicy,
                    num_bootstrap_samples=1000,
                    N=1000,
                    confidence_level=0.95,
                )
            except Exception as e:
                print("Error:", e)
                wis, ci = 0, [0, 0]

            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        {
                            "policy_name": f"{policy_name}_{dataset_name}",
                            "wis": wis,
                            "LB": ci[0],
                            "UB": ci[1],
                            "f1": f1,
                        },
                        index=[0],
                    ),
                ],
                axis=0,
            )

    EXPERIMENTS_RESULTS = "results/"
    # Save the results to a CSV file
    file_name = os.path.join(EXPERIMENTS_RESULTS, f"{experiment_name}.csv")
    # results_df.sort_values(by='n_clusters', inplace=True)
    results_df.to_csv(file_name, index=False)
    policies_file = os.path.join(
        EXPERIMENTS_RESULTS, f"rl_policies_{experiment_name}_latest.pkl"
    )
    pred.save_data_pickle(policies, policies_file)
