    
from config.config_param import config_param
from utils import (
    LoadEpisodes,
    PreprocessData,
    PolicyResolver,
    Encoder,
    DatasetFromEpisodes,
    LoadEncoder,
    add_columns_level_1, add_columns_level_2, add_columns_level_3
)
import pandas as pd
import argparse
import os
from tqdm import tqdm
import sys




def load_episodes(args):
  
    # 2- Loading the data and set the actions based on the level of the actions
    ############################################################################################################

    print("Loading the train, test, and validation data")
    config_params = config_param()
    # read the data from config
    param_dic = config_params.get_params_updated()
    encoder = Encoder()

    pred = PreprocessData(param_dic)


    model_path = param_dic["models_path"]


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
    folder_dim = f"{args['cluster_type']}_latend_dim_{args['latent_dim']}/"
    folder_cluster = os.path.join(param_dic["split_data_path"], folder_dim)
    train_cluster_path = os.path.join(folder_cluster, train_path.split("/")[-1])
    test_cluster_path = os.path.join(folder_cluster, test_path.split("/")[-1])
    val_cluster_path = os.path.join(folder_cluster, val_path.split("/")[-1])
    train_data_cluster, test_data_cluster, val_data_cluster = (
        pred.read_data(train_cluster_path),
        pred.read_data(test_cluster_path),
        pred.read_data(val_cluster_path),
    )

    cluster_col_train = f"obs_cluster_{args['n_clusters']}"
    cluster_encoder_col_train = f"obs_encoder_cluster_{args['n_encoder_clusters']}"
    cluster_col_test = f"obs_cluster_{args['n_clusters_test']}"
    cluster_encoder_col_test = f"obs_encoder_cluster_{args['n_encoder_clusters_test']}"
    cluster_col_val = f"obs_cluster_{args['n_clusters_val']}"
    cluster_encoder_col_val = f"obs_encoder_cluster_{args['n_encoder_clusters_val']}"

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
    obs_column = [col for col in train_data.columns if col.startswith(args['ob_prefix'])]
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
                train_path, args['latent_dim'], args['ob_prefix'], param_dic["models_path"]
            ),
        ],
        axis=1,
    )
    test_data_encObs = pd.concat(
        [
            test_data,
            load_encoder.return_encod_obs(
                test_path, args['latent_dim'], args['ob_prefix'], param_dic["models_path"]
            ),
        ],
        axis=1,
    )
    val_data_encObs = pd.concat(
        [
            val_data,
            load_encoder.return_encod_obs(
                val_path, args['latent_dim'], args['ob_prefix'], param_dic["models_path"]
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
    # if algo == "BayCQL":
    #     train_d3r_dataset = dsEpisodes.get_d3rlpy_dataset(train_episodes_encObs, action_dict_level_1, action_dict_level_2, action_dict_level_3)
    # else:
    train_d3r_dataset = dsEpisodes.get_d3rlpy_dataset_fixed(
        train_episodes_encObs, action_dict_level_2
    )
    test_d3r_dataset = dsEpisodes.get_d3rlpy_dataset_fixed(
        test_episodes_encObs, action_dict_level_2
    )

    behavior_train_episodes = train_episodes_cluster
    behavior_test_episodes = test_episodes_cluster
    behavior_validation_episodes = val_episodes_cluster

    train_episodes = train_episodes_encObs
    test_episodes = test_episodes_encObs
    validation_episodes = val_episodes_encObs
    return {
        "train_d3r_dataset": train_d3r_dataset,
        "test_d3r_dataset": test_d3r_dataset,
        "train_episodes": train_episodes,
        "test_episodes": test_episodes,
        "validation_episodes": validation_episodes,
        "behavior_train_episodes": behavior_train_episodes,
        "behavior_test_episodes": behavior_test_episodes,
        "behavior_validation_episodes": behavior_validation_episodes,
        "action_dict_level_2": action_dict_level_2,
        "action_dict_level_1": action_dict_level_1,
        "action_dict_level_3": action_dict_level_3,
        "train_data_encObs": train_data_encObs,
        "test_data_encObs": test_data_encObs,
        "val_data_encObs": val_data_encObs,
    }






if __name__ =="__main__":
    pass
    # args = {


    #     "latent_dim": int(sys.argv[1]),
    #     "ob_prefix": int(sys.argv[2]),
    #     "cluster_type": sys.argv[3],
    #     "n_clusters": int(sys.argv[4]),
    #     "n_encoder_clusters": int(sys.argv[5]),
    #     "n_clusters_test": int(sys.argv[6]),
    #     "n_encoder_clusters_test": int(sys.argv[7]),
    #     "n_clusters_val": int(sys.argv[8]),
    #     "n_encoder_clusters_val": int(sys.argv[9]),

    # }


    # load_episodes(args)

