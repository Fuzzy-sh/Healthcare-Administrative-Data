import argparse
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import config.config_param as config_param
from config.config_param import config_param
from utils import PreprocessData, SplitData, UpdateData
from utils import Encoder, IdentifyBestClusters, ConvertDatasetsToClusters, LoadEncoder

from sklearn.cluster import MiniBatchKMeans
import os
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np  
from sklearn.model_selection import cross_val_score


def add_cluster_knn(features, n_neighbors, labels):
    """
    Perform classification on features using K-Nearest Neighbors (KNN).

    Args:
        features (numpy.ndarray): Feature matrix for classification.
        labels (numpy.ndarray): Labels corresponding to the features for training the KNN model.
        n_neighbors (int): Number of neighbors to consider in KNN.

    Returns:
        numpy.ndarray: Predicted cluster labels for the input features.
    """
    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1, algorithm='auto')
    
    # Train the KNN model on provided features and labels
    knn.fit(features, labels)
    
    # Predict cluster labels for the input features
    clusters = knn.predict(features)
    return clusters

def add_cluster_kmeans(features, n_cluster, labels):
    """
    Perform clustering on observation (o:) and action (a:) features based on raw data.

    Args:
        p_cluster (dict): Parameters for clustering, including data information.
        param_dic (dict): Dictionary containing configuration parameters.

    Returns:
        tuple: Action treatment map and clustered dataset.
    """
  

    kmeans = KMeans(init ='k-means++', n_clusters = n_cluster, n_init = 'auto', max_iter=500, verbose = 0)
    # mbk = MiniBatchKMeans(n_clusters=n_clusters,init="k-means++", n_init='auto', batch_size=100,  max_no_improvement =1  , verbose=1, random_state=42)
    clusters = kmeans.fit_predict(features)
    

    return clusters



def main():
    """
    Main function for clustering observations using specified clustering algorithm.
    This script parses command-line arguments to configure clustering parameters, loads and preprocesses data,
    applies clustering (KMeans or KNN) on both raw and encoded observation features, and saves the resulting cluster assignments.
    Command-line Arguments:
        --file_name (str, required): The name of the file to process.
        --latent_dim_obs (int, optional): Latent space dimensions for observation encoding (default: 64).
        --clustering_trype (str, optional): Clustering algorithm to use ('kmeans' or 'knn', default: 'kmeans').
        --n_cluster (int, optional): Number of clusters to form (default: 2).
    Workflow:
        1. Parses arguments and loads configuration parameters.
        2. Reads and preprocesses the specified data file.
        3. Extracts observation features and their encoded representations.
        4. Applies the selected clustering algorithm to both feature sets.
        5. Saves the cluster assignments to the designated output folder.
    """
    # Argument parsing
    parser = argparse.ArgumentParser(description="Clustering script.")
    
    parser.add_argument("--file_name", type=str, required=True, help="The name of file.")
    parser.add_argument("--latent_dim_obs", type=int, default=64, help="Latent space dimensions for observation.")
    parser.add_argument("--clustering_trype", type=str, default="kmeans", help="Latent space dimensions for observation.")
    parser.add_argument("--n_cluster", type=int, default=2, help="The number of clusters.")
   
    args = parser.parse_args()


    # Load parameters and data
    config_param = config_param()
    param_dic = config_param.get_params_updated()

    # read the file in the processed data folder
    file_path = param_dic['split_files'][args.file_name]
    preprocess = PreprocessData(param_dic)
    data = preprocess.read_data(file_path)
    prefix = param_dic['observation_prefix']
    features = data[[col for col in data.columns if col.startswith(prefix)]]
    load_encoder = LoadEncoder(param_dic)
    data_encoder = load_encoder.return_encod_obs(file_path, args.latent_dim_obs, prefix , param_dic['models_path'])
    featurs_encoder = data_encoder[[col for col in data_encoder.columns if col.startswith(prefix)]]
    cluster_df = data[[col for col in data.columns if not col.startswith(prefix)]].copy()
    labels = data['homeless']


    if args.clustering_trype == 'kmeans':
        add_cluster = add_cluster_kmeans
    elif args.clustering_trype == 'knn':
        add_cluster = add_cluster_knn

    # for n_cluster in tqdm(range(10,11 , 10)):
    cluster_col = f"obs_cluster_{args.n_cluster}"
    cluster_encoder_col = f"obs_encoder_cluster_{args.n_cluster}"
    cluster_df[cluster_col]= add_cluster(features, args.n_cluster, labels)
    cluster_df[cluster_encoder_col]= add_cluster(featurs_encoder,args.n_cluster, labels)


    folder = f"{args.clustering_trype}_latend_dim_{args.latent_dim_obs}/{args.file_name}/"

    folder_cluster = os.path.join(param_dic['split_data_path'], folder)
    if not os.path.exists(folder_cluster):
        os.makedirs(folder_cluster, exist_ok=True)
    # print(cluster_col.head())
    # df_train[["Cluster"]].to_csv("kmeans_clusters.csv", index=False)
    cluster_name = cluster_col+'.h5'
    # cluster_enc_name = cluster_encoder_col+'.h5'
    preprocess.save_data(cluster_df[[cluster_col,cluster_encoder_col]], folder_cluster+cluster_name)
    # preprocess.save_data(cluster_df[cluster_encoder_col], folder_cluster+cluster_enc_name)

if __name__ == "__main__":
    # Main function to execute the clustering process
    # This will be called when the script is run directly
    main()