import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
from tqdm import tqdm
import argparse
import pandas as pd

# from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from config.config_param import config_param
from utils import PreprocessData
import os
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

max_iter = 500
# pbar = tqdm(total=max_iter)


from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin


# class ProgressBarKMeans(MiniBatchKMeans):
#     def fit(self, X, y=None):
#         for _ in range(self.max_iter):
#             super().partial_fit(X)
#             pbar.update(1)
#         pbar.close()
#         return self


class IdentifyBestClusters:
    def __init__(self, data, random_state):
        self.data = data
        self.random_state = random_state

    def cluster_features(self, prefix, n_clusters):
        """
        Perform clustering on features with the given prefix.
        """

        features = self.data[
            [col for col in self.data.columns if col.startswith(prefix)]
        ]

        print("Fitting the features with prefix:", prefix)

        mbk = KMeans(
            init="k-means++",
            n_clusters=n_clusters,
            n_init="auto",
            max_iter=500,
            verbose=1,
        )
        # mbk = MiniBatchKMeans(n_clusters=n_clusters,init="k-means++", n_init='auto', batch_size=100,  max_no_improvement =1  , verbose=1, random_state=42)
        print("Fitting the features with prefix:", prefix, "completed")
        mbk.fit(features)
        # mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis = 0)
        # mbk_means_labels = pairwise_distances_argmin(features, mbk_means_cluster_centers)
        # # print(mbk_means_labels)
        print("Fitting the features with prefix:", prefix, "completed")
        print("Calculating the inertia")
        sse = mbk.inertia_

        # silhouette =  silhouette_score(features, mbk.labels_)
        silhouette = 0

        return sse, silhouette, mbk


class ConvertDatasetsToClusters:
    def __init__(
        self,
        data,
        data_encoder,
        obs_cluster_col,
        act_cluster_col,
        obs_prefix="o:",
        act_prefix="a:",
        rew_prefix="r:",
    ):
        self.data = data
        self.data_encoder = data_encoder
        self.obs_prefix = obs_prefix
        self.act_prefix = act_prefix
        self.rew_prefix = rew_prefix
        self.obs_cluster_col = obs_cluster_col
        self.act_cluster_col = act_cluster_col

    def get_encoder_cluster_action_treatment_map(self, kmeans, treatment_names):

        # Sample action data: rows are patients, columns are treatments
        # actions = np.array([
        #     [0, 1, 1, 0, 0, 0],  # Patient 1 received Treatment B and C
        #     [1, 0, 0, 1, 0, 0],  # Patient 2 received Treatment A and D
        #     [0, 1, 1, 0, 0, 0],  # Patient 3 received Treatment B and C
        #     [0, 0, 0, 1, 1, 0],  # Patient 4 received Treatment D and E
        #     [1, 0, 0, 0, 0, 1],  # Patient 5 received Treatment A and F
        # ])

        # Treatment names for reference
        # treatment_names = ['A', 'B', 'C', 'D', 'E', 'F']

        # Step 1: Apply KMeans clustering

        # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        act_data = self.extract_columns(self.data_encoder, self.act_prefix)
        clusters = kmeans.predict(act_data)

        # Step 2: Analyze cluster centroids
        centroids = kmeans.cluster_centers_
        centroids_df = pd.DataFrame(centroids, columns=treatment_names)

        # Step 3: Map clusters to treatments
        cluster_treatment_map = {}

        for cluster_id, centroid in enumerate(centroids):
            # Identify the treatments with high association for this cluster
            significant_treatments = [
                treatment_names[i] for i, val in enumerate(centroid) if val > 0.5
            ]
            cluster_treatment_map[cluster_id] = significant_treatments
        if len(cluster_treatment_map[0]) == 0:
            cluster_treatment_map[0] = ["no_treatment"]
        # Step 4: Display results
        print("Cluster Assignments for Patients:", clusters)
        print("\nCluster Centroids:\n", centroids_df)
        print("\nCluster to Treatment Mapping:")
        for cluster_id, treatments in cluster_treatment_map.items():
            print(f"Cluster {cluster_id}: {', '.join(treatments)}")

        return cluster_treatment_map

    def get_cluster_action_treatment_map(self, kmeans, treatment_names):

        # Sample action data: rows are patients, columns are treatments
        # actions = np.array([
        #     [0, 1, 1, 0, 0, 0],  # Patient 1 received Treatment B and C
        #     [1, 0, 0, 1, 0, 0],  # Patient 2 received Treatment A and D
        #     [0, 1, 1, 0, 0, 0],  # Patient 3 received Treatment B and C
        #     [0, 0, 0, 1, 1, 0],  # Patient 4 received Treatment D and E
        #     [1, 0, 0, 0, 0, 1],  # Patient 5 received Treatment A and F
        # ])

        # Treatment names for reference
        # treatment_names = ['A', 'B', 'C', 'D', 'E', 'F']

        # Step 1: Apply KMeans clustering

        # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        act_data = self.extract_columns(self.data, self.act_prefix)
        clusters = kmeans.predict(act_data)

        # Step 2: Analyze cluster centroids
        centroids = kmeans.cluster_centers_
        centroids_df = pd.DataFrame(centroids, columns=treatment_names)

        # Step 3: Map clusters to treatments
        cluster_treatment_map = {}

        for cluster_id, centroid in enumerate(centroids):
            # Identify the treatments with high association for this cluster
            significant_treatments = [
                treatment_names[i] for i, val in enumerate(centroid) if val > 0.5
            ]
            cluster_treatment_map[cluster_id] = significant_treatments
        if len(cluster_treatment_map[0]) == 0:
            cluster_treatment_map[0] = ["no_treatment"]
        # Step 4: Display results
        print("Cluster Assignments for Patients:", clusters)
        print("\nCluster Centroids:\n", centroids_df)
        print("\nCluster to Treatment Mapping:")
        for cluster_id, treatments in cluster_treatment_map.items():
            print(f"Cluster {cluster_id}: {', '.join(treatments)}")

        return cluster_treatment_map

    def extract_columns(self, data, prefix):
        """Extract columns starting with a given prefix."""

        return data[[col for col in data.columns if col.startswith(prefix)]]

    def apply_clustering(self, data, kmeans, prefix):
        """
        Apply KMeans clustering and return cluster labels.

        Parameters:
            data (pd.DataFrame): Data to cluster.
            n_clusters (int): Number of clusters.

        Returns:
            np.ndarray: Cluster labels.
            KMeans: Fitted KMeans model.
        """

        clusters = kmeans.predict(data)
        clusters_df = pd.DataFrame(clusters, columns=[f"{prefix}cluster"])

        # clusters_df.to_csv(os.path.join(processed_data_path, f'{dataset_name}_kmeans_clusters_{n_clusters}.csv'), index=False)

        return clusters_df

    def convert_datasets(self, obs_kmeans, act_kmeans):
        """
        Convert train, test, and validation datasets using clustering on observations and actions.

        Parameters:
            train, test, validation (pd.DataFrame): Datasets to convert.
            obs_prefix, act_prefix, rew_prefix (str): Prefixes for observations, actions, and rewards columns.

        Returns:
            dict: Transformed datasets and fitted KMeans models.
        """

        # Extract observation and action columns
        obs_data = self.extract_columns(self.data, self.obs_prefix)
        act_data = self.extract_columns(self.data, self.act_prefix)

        obs_cluster_df = self.apply_clustering(obs_data, obs_kmeans, self.obs_prefix)
        act_cluster_df = self.apply_clustering(act_data, act_kmeans, self.act_prefix)
        transformed_data = self.data.copy()
        transformed_data[self.obs_cluster_col] = obs_cluster_df
        transformed_data[self.act_cluster_col] = act_cluster_df
        return transformed_data

    def convert_encoder_datasets(self, obs_kmeans, act_kmeans):
        """
        Convert train, test, and validation datasets using clustering on observations and actions.

        Parameters:
            train, test, validation (pd.DataFrame): Datasets to convert.
            obs_prefix, act_prefix, rew_prefix (str): Prefixes for observations, actions, and rewards columns.

        Returns:
            dict: Transformed datasets and fitted KMeans models.
        """

        # Extract observation and action columns
        obs_data_encoder = self.extract_columns(self.data_encoder, self.obs_prefix)
        act_data_encoder = self.extract_columns(self.data_encoder, self.act_prefix)

        obs_cluster_df = self.apply_clustering(
            obs_data_encoder, obs_kmeans, self.obs_prefix
        )
        act_cluster_df = self.apply_clustering(
            act_data_encoder, act_kmeans, self.act_prefix
        )
        transformed_data = self.data.copy()
        transformed_data[self.obs_cluster_col] = obs_cluster_df
        transformed_data[self.act_cluster_col] = act_cluster_df
        return transformed_data
