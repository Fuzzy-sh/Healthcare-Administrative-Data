import numpy as np
import pickle
import os
import pandas as pd
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


from Qlearning.RLModels_2layer import *


from collections import defaultdict
from utils import add_columns_level_1, add_columns_level_2, add_columns_level_3
import pandas as pd
import os
import multiprocessing


# num_cores = multiprocessing.cpu_count()
# n_cores = max(1, num_cores // 8)  # Use half of available cores


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
        default="UHA-SAC",
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
        default=1000,
        help="The number of steps in the neural network",
    )
    parser.add_argument(
        "--n_steps_per_epoch",
        type=int,
        default=1000,
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
        default="False",
        help="The use of attention mechanism in the first level",
    )
    parser.add_argument(
        "--use_attention_2",
        type=str,
        default="False",
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
        default="False",
        help="The use of uncertainty actor loss in the first level",
    )
    parser.add_argument(
        "--use_uncertainty_actor_loss_2",
        type=str,
        default="False",
        help="The use of uncertainty actor loss in the second level",
    )
    parser.add_argument(
        "--use_uncertainty_critic_loss_1",
        type=str,
        default="False",
        help="The use of uncertainty critic loss in the first level",
    )
    parser.add_argument(
        "--use_uncertainty_critic_loss_2",
        type=str,
        default="False",
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

    behavior_policy_path = "results/rl_policies_autoencoder_train_cluster70_70_test_cluster_70_70_20250112-224409.pkl"
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

        # dataframe to store the results
    # evaluation_results = pd.DataFrame(columns=['experiment_name', 'num_layers', 'num_hidden_neurons', 'activation_func', 'learning_rate', 'n_steps',
    #                                            'rl_policy_test_wis', 'rl_policy_test_ci',
    #                                            'rl_policy_validation_wis', 'rl_policy_validation_ci',
    #                                            'rl_policy_train_wis', 'rl_policy_train_ci',
    #                                            'rl_greedy_policy_test_wis', 'rl_greedy_policy_test_ci',
    #                                            'rl_greedy_policy_validation_wis', 'rl_greedy_policy_validation_ci',
    #                                             'rl_greedy_policy_train_wis', 'rl_greedy_policy_train_ci'])

    # num_layers, num_hidden_neurons, activation_func, learning_rate, n_steps, conservative_alpha

    # print("create aciton train encoded")
    # train_actions_encoded = [ encoder.action_to_index(train_actions_dict, action, param_dic['colgroup_treatment']) for action in train_actions]
    # actions_dict_update = train_actions_dict #{key:value for key,value in train_actions_dict.items() if key in set(train_actions_encoded)}
    # print("Actions dictionary:", actions_dict_update)
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

    # experiment_name = 'DQN-all-caths' + '-reward-' + 'survival-mace' + '__' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # experiment_name = 'CQL-all-caths-obstructiveCAD' + '-reward-' + 'mace' + '__' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    os.makedirs(os.path.join(model_path, experiment_name), exist_ok=True)

    args = parser.parse_args()
    experiment_type = args.experiment_type

    # 2- Loading the data and set the actions based on the level of the actions
    ############################################################################################################

    print("Loading the train, test, and validation data")

    train_path, test_path, val_path = (
        param_dic["split_with_cluster_files"]["train"],
        param_dic["split_with_cluster_files"]["test"],
        param_dic["split_with_cluster_files"]["val"],
    )
    train_data, test_data, val_data = (
        pred.read_data(train_path),
        pred.read_data(test_path),
        pred.read_data(val_path),
    )
    print("Data loaded successfully!")

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

    # 3- Adding the clustering and for clumns and encoded columsn results for the observation data to the train, test, and validation data
    ############################################################################################################
    # adding the clustering restuls to the train test and validation data
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
    cluster_col_val = f"obs_cluster_{n_clusters_val}"
    cluster_encoder_col_val = f"obs_encoder_cluster_{n_encoder_clusters_val}"

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
    val_data = pd.concat(
        [val_data, val_data_cluster[[cluster_col_val, cluster_encoder_col_val]]], axis=1
    )

    # 4- Dropping the observation columns to use only the encoded observation data
    ############################################################################################################
    # encode only the observation data
    obs_column = [col for col in train_data.columns if col.startswith(ob_prefix)]
    # drop the observation columns form train and test data
    train_data.drop(columns=obs_column, inplace=True)
    test_data.drop(columns=obs_column, inplace=True)
    val_data.drop(columns=obs_column, inplace=True)

    ############################################################################################################
    # 5- Load the encoder model and encode the observation data
    ############################################################################################################
    load_encoder = LoadEncoder(param_dic)

    print("Encode the observation data using the autoencoder model")
    train_data_encObs = pd.concat(
        [
            train_data,
            load_encoder.return_encod_obs(
                train_path, latent_dim, ob_prefix, param_dic["models_path"]
            ),
        ],
        axis=1,
    )
    test_data_encObs = pd.concat(
        [
            test_data,
            load_encoder.return_encod_obs(
                test_path, latent_dim, ob_prefix, param_dic["models_path"]
            ),
        ],
        axis=1,
    )
    val_data_encObs = pd.concat(
        [
            val_data,
            load_encoder.return_encod_obs(
                val_path, latent_dim, ob_prefix, param_dic["models_path"]
            ),
        ],
        axis=1,
    )

    # 6- Load the episodes based on the encoded, cluster and encoder cluster data
    ############################################################################################################
    print("loadig episodes (train, test and valiadtion) based on encoded data")
    le = LoadEpisodes(param_dic)
    train_episodes_encObs, train_episodes_cluster, train_episodes_encoder_cluster = (
        le.load_episodes(
            train_data_encObs, cluster_col_train, cluster_encoder_col_train
        )
    )
    test_episodes_encObs, test_episodes_cluster, test_episodes_encoder_cluster = (
        le.load_episodes(test_data_encObs, cluster_col_test, cluster_encoder_col_test)
    )  # train_episodes_raw , train_episodes_cluster , train_episodes_encoder_cluster#le.load_episodes('test')
    val_episodes_encObs, val_episodes_cluster, val_episodes_encoder_cluster = (
        le.load_episodes(val_data_encObs, cluster_col_val, cluster_encoder_col_val)
    )  # train_episodes_raw , train_episodes_cluster , train_episodes_encoder_cluster#le.load_episodes('val')

    # 7- Create the d3rlpy datasets based on the episodes from the encoded data
    ################################################################################################
    print("create the d3rlpy datasets based on the episodes from the encoded data")
    dsEpisodes = DatasetFromEpisodes()
    if algo == "BayCQL":
        train_d3r_dataset = dsEpisodes.get_d3rlpy_dataset(
            train_episodes_encObs,
            action_dict_level_1,
            action_dict_level_2,
            action_dict_level_3,
        )
    else:
        train_d3r_dataset = dsEpisodes.get_d3rlpy_dataset_fixed(
            train_episodes_encObs, action_dict_level_2
        )
    behavior_train_episodes = train_episodes_cluster
    behavior_test_episodes = test_episodes_cluster
    behavior_validation_episodes = val_episodes_cluster

    train_episodes = train_episodes_encObs
    test_episodes = test_episodes_encObs
    validation_episodes = val_episodes_encObs

    # put the behavior policy inside the reference episodes

    # Initialize dictionary to store action counts
    train_action_counts = defaultdict(int)
    test_action_counts = defaultdict(int)
    validation_action_counts = defaultdict(int)

    # Iterate over all episodes in the training dataset
    for episode in train_episodes:
        for i in range(
            len(episode.transitions)
        ):  # Iterate over transitions in each episode
            action = episode.transitions[
                i
            ].action_level_2  # Extract action from transition
            train_action_counts[action] += 1  # Increment count for that action

    # Iterate over all episodes in the training dataset
    for episode in test_episodes:
        for i in range(
            len(episode.transitions)
        ):  # Iterate over transitions in each episode
            action = episode.transitions[
                i
            ].action_level_2  # Extract action from transition
            test_action_counts[action] += 1  # Increment count for that action

    for episode in validation_episodes:
        for i in range(len(episode.transitions)):
            action = episode.transitions[i].action_level_2
            validation_action_counts[action] += 1
    # Print final action counts
    print("Action Counts in Trian Dataset:", dict(train_action_counts))
    print("Action Counts in Test Dataset:", dict(test_action_counts))
    print("Action Counts in Validation Dataset:", dict(validation_action_counts))

    # only for the action_dict_level_1 for now
    for dataset, behavior_dataset in [
        (train_episodes, behavior_train_episodes),
        (test_episodes, behavior_test_episodes),
        (validation_episodes, behavior_validation_episodes),
    ]:
        for episode, behavior_episode in zip(dataset, behavior_dataset):
            for transition, behavior_transition in zip(episode, behavior_episode):
                behavior_state = behavior_transition.state
                behavior_state = int(behavior_state[0])
                transition.prediction_probs["best_physician_policy"] = (
                    best_physician_policy[behavior_state, :]
                )

                # single action policies
                for action in action_dict_level_2.keys():
                    transition.prediction_probs[f"single_action_policy_{action}"] = (
                        np.zeros(len(action_dict_level_2))
                    )
                    transition.prediction_probs[f"single_action_policy_{action}"][
                        action
                    ] = 1.0

    # num_layers_list = [2]
    # num_hidden_neurons_list = [32]
    # activation_func_list = ['tanh']
    # learning_rate_list = [0.0001]
    # n_steps_list = [10000]

    # if algo == 'CQL' or algo == 'DCQL':
    #     conservative_alpha_list = [0.001, 0.01, 0.1, 0.5]
    # else:
    #     conservative_alpha_list = [None]

    # for num_layers, num_hidden_neurons, activation_func, learning_rate, n_steps, conservative_alpha in itertools.product(num_layers_list,
    #                                                                                                  num_hidden_neurons_list,
    #                                                                                                 activation_func_list,
    #                                                                                                  learning_rate_list,
    #                                                                                                 n_steps_list,conservative_alpha_list):
    #

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

    # fit the model
    # rl_model = DQmodel.fit_model(train_d3r_dataset, experiment_name)

    rl_algo = DQmodel._get_model(algo=algo)
    policy_eval_results = {}
    evaluation_results = pd.DataFrame()
    # Epoch-wise training with direct access

    for epoch, metrics in rl_algo.fitter(
        dataset=train_d3r_dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        experiment_name=experiment_name,
    ):
        # After each epoch completes, you have direct access to your trained model
        rl_model = rl_algo  # this is your trained algorithm instance at current epoch

        # if algo =='SAC':
        #     print(f"Epoch {epoch}:")
        #     print(f"  SAC1 - Critic Loss: {metrics.get('sac1_critic_loss', 0.0):.2f}, "
        #         f"Actor Loss: {metrics.get('sac1_actor_loss', 0.0):.2f}")
        #     print(f"  SAC2 - Critic Loss: {metrics.get('sac2_critic_loss', 0.0):.2f}, "
        #         f"Actor Loss: {metrics.get('sac2_actor_loss', 0.0):.2f}")

        # else:
        #     print(f"Epoch: {epoch}, Metrics: {metrics}")
        # Now you can do whatever you want after each epoch, for example:
        # - Evaluate your model
        # - Save the current snapshot of the model
        # - Use the model for inference or further experiments
        # DQmodel.save_model(f"rl_model_epoch_{epoch}.pt")

        # save full parameters and configurations in a single file.
        if conservative_alpha is None:
            # for normal DQN
            model_name = f"{num_layers}__{num_hidden_neurons}__{activation_func}__{actor_learning_rate}__{critic_learning_rate}__{n_steps}"
        else:
            # for CQL
            model_name = f"{num_layers}__{num_hidden_neurons}__{activation_func}__{actor_learning_rate}__{critic_learning_rate}__{n_steps}__{conservative_alpha}"

        if ARC == True:
            folder_name = "/work/messier_lab/fuzzy/DQ_resutls/results"
        else:
            folder_name = "results"

        policy_eval_results["epoch"] = epoch
        policy_eval_results["experiment_name"] = experiment_name

        EXPERIMENTS_RESULTS = folder_name + algo
        if not os.path.exists(EXPERIMENTS_RESULTS):
            os.makedirs(EXPERIMENTS_RESULTS, exist_ok=True)

        model_path_ = os.path.join(EXPERIMENTS_RESULTS, experiment_name + ".d3")

        rl_model.save(model_path_)
        print("Model saved at:", model_path_)

        # # # TODO: remove this line
        # model_import_folder = 'models_obstructive_cad5/DQN-all-caths-reward-mace__20240415-155033'
        # model_import_path = os.path.join(EXPERIMENTS_RESULTS, model_name + '.d3')

        # rl_model = d3rlpy.load_learnable(model_path_)

        # evaluate the model
        rl_policy = PolicyResolver(rl_model, list(action_dict_level_2.keys()))
        rl_greedy_policy = PolicyResolver(
            rl_model, list(action_dict_level_2.keys()), greedy=True
        )
        physician_policy = PolicyResolver(
            "best_physician_policy", list(action_dict_level_2.keys())
        )
        physician_policy_greedy = PolicyResolver(
            "best_physician_policy", list(action_dict_level_2.keys()), greedy=True
        )

        for param_name, param_value in model_params.items():
            policy_eval_results[param_name] = param_value

        # # Run all policy evaluations in parallel
        # results = Parallel(n_jobs=n_cores , backend="threading")(
        #     delayed(evaluate_policy)(eval_episodes, eval_name, policy_name, policy, physician_policy)
        #     for eval_episodes, eval_name in zip([test_episodes, validation_episodes, train_episodes], ['test', 'validation', 'train'])
        #     for policy_name, policy in zip(['rl_policy', 'rl_greedy_policy'], [rl_policy, rl_greedy_policy])
        # )

        # # Update results
        # for res in results:
        #     policy_eval_results.update(res)

        # # Best physician policy evaluations (parallelized)
        # best_physician_results = Parallel(n_jobs=n_cores , backend="threading")(
        #     delayed(evaluate_policy)(eval_episodes, eval_name, 'best_physician_policy', physician_policy, physician_policy)
        #     for eval_episodes, eval_name in zip([test_episodes, validation_episodes, train_episodes], ['test', 'validation', 'train'])
        # )

        # for res in best_physician_results:
        #     policy_eval_results.update(res)

        # Best greedy physician policy
        # best_physician_greedy_results = Parallel(n_jobs=n_cores , backend="threading")(
        #     delayed(evaluate_policy)(eval_episodes, eval_name, 'best_physician_policy_greedy', physician_policy_greedy, physician_policy)
        #     for eval_episodes, eval_name in zip([test_episodes, validation_episodes, train_episodes], ['test', 'validation', 'train'])
        # )

        # for res in best_physician_greedy_results:
        #     policy_eval_results.update(res)

        # Single Action Policies
        # single_action_results = Parallel(n_jobs=n_cores , backend="threading")(
        #     delayed(evaluate_policy)(eval_episodes, eval_name, f'single_action_policy_{action_id}',
        #                             PolicyResolver(f'single_action_policy_{action_id}', list(action_dict_level_2.keys())),
        #                             physician_policy)
        #     for eval_episodes, eval_name in zip([test_episodes, validation_episodes, train_episodes], ['test', 'validation', 'train'])
        #     for action_id in action_dict_level_2.keys()
        # )

        # for res in single_action_results:
        #     policy_eval_results.update(res)

        # # evaluate the policies
        # for eval_episodes, eval_name in zip([test_episodes, validation_episodes, train_episodes], ['test', 'validation', 'train']):
        #     for policy_name, policy in tqdm(zip(['rl_policy', 'rl_greedy_policy'], [rl_policy, rl_greedy_policy])):
        #         wis, ci, f1 = WI.weighted_importance_sampling_with_bootstrap(eval_episodes, 0.99, policy, physician_policy, num_bootstrap_samples=1000, N=1000)
        #         policy_eval_results[f'{policy_name}_{eval_name}_wis'] = wis
        #         policy_eval_results[f'{policy_name}_{eval_name}_ci'] = str(ci)
        #         print(f"Weighted importance sampling for the {policy_name} on the {eval_name} data: {wis:.2f}, CI: {ci} -- F1: {f1:.2f}")

        data_dict = {
            "test_rl": (test_episodes, rl_policy),
            "test_rl_greedy": (test_episodes, rl_greedy_policy),
            "val_rl": (validation_episodes, rl_policy),
            "val_rl_greedy": (validation_episodes, rl_greedy_policy),
            "train_rl": (train_episodes, rl_policy),
            "train_rl_greedy": (train_episodes, rl_greedy_policy),
        }
        WI = WeightedImportanceSampling()

        results, df = WI.multi_weighted_importance_sampling_with_bootstrap(
            data_dict=data_dict,
            behavior_policy=physician_policy,  # one shared behavior policy
            gamma=0.99,
            num_bootstrap_samples=1000,
            N=1000,
            confidence_level=95,
            reward_name=None,
        )

        for name, res in results.items():
            wis = res["wis"]
            ci = res["ci"]
            f1 = res["f1"]
            print(f"{name}: WIS={wis}, CI={ci}, F1={f1}")
            policy_eval_results[name + "_wis"] = wis
            policy_eval_results[name + "_ci"] = str(ci)
            policy_eval_results[name + "_f1"] = f1
            with pd.ExcelWriter(
                EXPERIMENTS_RESULTS
                + "/e_"
                + str(epoch)
                + name
                + experiment_name
                + ".xlsx",
                engine="openpyxl",
            ) as writer:
                for subject_id, tb in df.items():
                    tb.to_excel(writer, sheet_name=f"{subject_id}", index=False)

        # # evaluate the best physician policy
        # for eval_episodes, eval_name in zip([test_episodes, validation_episodes, train_episodes], ['test', 'validation', 'train']):
        #     wis, ci , f1= WI.weighted_importance_sampling_with_bootstrap(eval_episodes, 0.99, physician_policy, physician_policy, num_bootstrap_samples=1000, N=1000)
        #     policy_eval_results[f'best_physician_policy_{eval_name}_wis'] = wis
        #     policy_eval_results[f'best_physician_policy_{eval_name}_ci'] = str(ci)
        #     # print(f"Weighted importance sampling for the best physician policy on the {eval_name} data: {wis}, CI: {ci} -- F1: {f1:.2f}")

        #     # evaluate the greedy best physician policy
        #     wis, ci, f1 = WI.weighted_importance_sampling_with_bootstrap(eval_episodes, 0.99, physician_policy_greedy, physician_policy, num_bootstrap_samples=1000, N=1000)
        #     policy_eval_results[f'best_physician_policy_greedy_{eval_name}_wis'] = wis
        #     policy_eval_results[f'best_physician_policy_greedy_{eval_name}_ci'] = str(ci)
        #     # print(f"Weighted importance sampling for the greedy best physician policy on the {eval_name} data: {wis}, CI: {ci} -- F1: {f1:.2f}")

        #     # evaluate the single action policies
        #     for action_id in action_dict_level_2.keys():

        #         single_action_policy = PolicyResolver(f'single_action_policy_{action_id}', list(action_dict_level_2.keys()))
        #         wis, ci, f1 = WI.weighted_importance_sampling_with_bootstrap(eval_episodes, 0.99, single_action_policy, physician_policy, num_bootstrap_samples=1000, N=1000)
        #         policy_eval_results[f'single_action_policy_{action_id}_{action_id}_wis'] = wis
        #         policy_eval_results[f'single_action_policy_{action_id}_{action_id}_ci'] = str(ci)
        #         # print(f"Weighted importance sampling for the single action policy {action_id} on the {eval_name} data: {wis}, CI: {ci} -- F1: {f1:.2f}")

        evaluation_results_ = pd.concat(
            [pd.DataFrame(policy_eval_results, index=[0])], ignore_index=True
        )
        evaluation_results = pd.concat(
            [evaluation_results, evaluation_results_], ignore_index=True, axis=0
        )
        evaluation_results.to_csv(
            os.path.join(EXPERIMENTS_RESULTS, f"{experiment_name}.csv"), index=False
        )
