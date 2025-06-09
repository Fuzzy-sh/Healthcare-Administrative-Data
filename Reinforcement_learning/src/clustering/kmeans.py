import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
from tqdm import tqdm

class ConvertDatasetsToClusters:
    def __init__(self, train, test, validation, obs_prefix='o:', act_prefix='a:', rew_prefix='r:', random_state=42):
        self.train = train
        self.test = test
        self.validation = validation
        self.obs_prefix = obs_prefix
        self.act_prefix = act_prefix
        self.rew_prefix = rew_prefix
        self.random_state = random_state
        # self.obs_train = self.extract_columns(self.train, self.obs_prefix)
        # self.act_train = self.extract_columns(self.train, self.act_prefix)
        # self.n_obs_clusters, _, _ = self.determine_best_clusters(self.obs_train)
        # self.n_act_clusters, _, _ = self.determine_best_clusters(self.act_train)
        # self.obs_labels_train, self.obs_kmeans = self.apply_clustering(self.obs_train, self.n_obs_clusters)
        # self.act_labels_train, self.act_kmeans = self.apply_clustering(self.act_train, self.n_act_clusters)
        # self.obs_test = self.extract_columns(self.test, self.obs_prefix)
        # self.act_test = self.extract_columns(self.test, self.act_prefix)
        # self.obs_labels_test = self.obs_kmeans.predict(self.obs_test)
        # self.act_labels_test = self.act_kmeans.predict(self.act_test)
        # self.obs_val = self.extract_columns(self.validation, self.obs_prefix)
        # self.act_val = self.extract_columns(self.validation, self.act_prefix)
        # self.obs_labels_val = self.obs_kmeans.predict(self.obs_val)
        # self.act_labels_val = self.act_kmeans.predict(self.act_val)
        # self.train_transformed = self.train.copy()
        # self.train_transformed['obs_cluster'] = self.obs_labels_train
        # self.train_transformed['act_cluster'] = self.act_labels_train
        # self.test_transformed = self.test.copy()
        # self.test_transformed['obs_cluster'] = self.obs_labels_test
        # self.test_transformed['act_cluster'] = self.act_labels_test
        # self.val_transformed = self.validation.copy()
        # self.val_transformed['obs_cluster'] = self.obs_labels_val
        # self.val_transformed['act_cluster'] = self.act_labels_val
        # self.result = {
        #     "train": self.train_transformed,
        #     "test": self.test_transformed,
        #     "validation": self.val_transformed,
        #     "obs_kmeans": self.obs_kmeans,
        #     "act_kmeans": self.act_kmeans
        # }
    
    def extract_columns(self, data, prefix):
   
        """Extract columns starting with a given prefix."""
        return data[[col for col in data.columns if col.startswith(prefix)]]

    def determine_best_clusters(self, data, max_clusters=50, random_state=42):
        """
        Determine the best number of clusters using the Elbow Method and Silhouette Score.
        
        Parameters:
            data (pd.DataFrame): Data to cluster.
            max_clusters (int): Maximum number of clusters to test.
            random_state (int): Random state for reproducibility.
            
        Returns:
            int: Optimal number of clusters.
        """
        scores = []
        inertias = []

        for n_cluster in tqdm(range(2, max_clusters + 1)):
            kmeans = KMeans(n_clusters=n_cluster, random_state=random_state).fit(data)
            scores.append(silhouette_score(data, kmeans.fit_predict(data)))
            inertias.append(kmeans.inertia_)
        # Step 1: Round each number
        rounded_numbers = [round(num, 0) for num in inertias]

        # Step 2: Find the maximum rounded value
        max_value = min(rounded_numbers)

        # Step 3: Identify indices with the maximum rounded value
        min_indices = [i for i, val in enumerate(rounded_numbers) if val == max_value]

        # Step 4: Choose the index with the smallest original number
        min_original_value = min(inertias[idx] for idx in min_indices)
        best_clusters = [idx for idx in min_indices if inertias[idx] == min_original_value][0]
        best_clusters+=2
        # print("Rounded Numbers:", rounded_numbers)
        # print("Min Rounded Value:", min_value)
        # print("Indices with Min Rounded Value:", min_indices)
        # print("Best Index:", best_clusters)
        # print("Original Number at Best Index:", inertias[best_clusters])

        # best_clusters = np.argmin(inertias) + 2  # +2 because range starts from 2
        return best_clusters, scores, inertias

    def apply_clustering(self, data, n_clusters,prefix, random_state=42, ):
        """
        Apply KMeans clustering and return cluster labels.
        
        Parameters:
            data (pd.DataFrame): Data to cluster.
            n_clusters (int): Number of clusters.
            
        Returns:
            np.ndarray: Cluster labels.
            KMeans: Fitted KMeans model.
        """
        # kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(data)
        max_iter = max(500, 10 * n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, init='k-means++', n_init='auto', max_iter=max_iter).fit(data)
        clusters = kmeans.predict(data)
        clusters_df = pd.DataFrame(clusters, columns=[f'{prefix}cluster'])
        # clusters_df.to_csv(os.path.join(processed_data_path, f'{dataset_name}_kmeans_clusters_{n_clusters}.csv'), index=False)

        return kmeans.labels_, kmeans

    def convert_datasets(self):
        """
        Convert train, test, and validation datasets using clustering on observations and actions.
        
        Parameters:
            train, test, validation (pd.DataFrame): Datasets to convert.
            obs_prefix, act_prefix, rew_prefix (str): Prefixes for observations, actions, and rewards columns.
            
        Returns:
            dict: Transformed datasets and fitted KMeans models.
        """
        # Extract observation and action columns
        obs_train = self.extract_columns(self.train, self.obs_prefix)
        act_train = self.extract_columns(self.train, self.act_prefix)
        
        # Determine optimal clusters for observations and actions
        n_obs_clusters, _, _ = self.determine_best_clusters(obs_train)
        print(n_obs_clusters)
        n_act_clusters, _, _ = self.determine_best_clusters(act_train)
        print(n_act_clusters)
        # Fit KMeans for observations and actions
        obs_labels_train, obs_kmeans = self.apply_clustering(obs_train, n_obs_clusters, self.obs_prefix)
        act_labels_train, act_kmeans = self.apply_clustering(act_train, n_act_clusters, self.act_prefix)
        print(obs_labels_train)
        print(act_labels_train)
        # Apply clustering to test and validation sets
        obs_test = self.extract_columns(self.test, self.obs_prefix)
        act_test = self.extract_columns(self.test, self.act_prefix)
        obs_labels_test = obs_kmeans.predict(obs_test)
        act_labels_test = act_kmeans.predict(act_test)
        # print(obs_labels_test)
        # print(act_labels_test)
        obs_val = self.extract_columns(self.validation, self.obs_prefix)
        act_val = self.extract_columns(self.validation, self.act_prefix)
        obs_labels_val = obs_kmeans.predict(obs_val)
        act_labels_val = act_kmeans.predict(act_val)
        
        # # Replace observations and actions with cluster labels
        # train_transformed = self.train.copy()
        # train_transformed['obs_cluster'] = obs_labels_train
        # train_transformed['act_cluster'] = act_labels_train
        
        # test_transformed = self.test.copy()
        # test_transformed['obs_cluster'] = obs_labels_test
        # test_transformed['act_cluster'] = act_labels_test
        
        # val_transformed = self.validation.copy()
        # val_transformed['obs_cluster'] = obs_labels_val
        # val_transformed['act_cluster'] = act_labels_val
        
        # return  {
        #     "train": train_transformed,
        #     "test": test_transformed,
        #     "validation": val_transformed,
        #     "obs_kmeans": obs_kmeans,
        #     "act_kmeans": act_kmeans
        # }

        # def main ():
        
        
        # for n_clusters in n_clusters_list:
        #     kmeans_filename = f'kmeans_{dataset_name}_{n_clusters}.pkl'
        #     kmeans_filepath = os.path.join(kmeans_folder, kmeans_filename)
        #     if load_existing:
        #         with open(kmeans_filepath, 'rb') as f:
        #             kmeans = pickle.load(f)
        #     else:
        #         max_iter = max(500, 10 * n_clusters)
        #         kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, init='k-means++', n_init='auto', max_iter=max_iter).fit(data_df)
        #         with open(kmeans_filepath, 'wb') as f:
        #             pickle.dump(kmeans, f)

        #     clusters = kmeans.predict(data_df)
        #     clusters_df = pd.DataFrame(clusters, columns=['cluster'])
        #     clusters_df.to_csv(os.path.join(processed_data_path, f'{dataset_name}_kmeans_clusters_{n_clusters}.csv'), index=False)

        #     if verbose:
        #         print(f'K-means clustering for {dataset_name} with {n_clusters} clusters complete and file saved.')


# # Example Usage
# # Assuming train_df, test_df, and val_df are loaded as pandas DataFrames
# train_df = pd.DataFrame({
#     'o:feature1': np.random.rand(100), 'o:feature2': np.random.rand(100),
#     'a:action1': np.random.randint(0, 2, 100), 'a:action2': np.random.randint(0, 2, 100),
#     'r:reward': np.random.rand(100)
# })
# test_df = train_df.copy()
# val_df = train_df.copy()

# result = convert_datasets(train_df, test_df, val_df)

# # Access transformed datasets
# train_transformed = result['train']
# test_transformed = result['test']
# val_transformed = result['validation']

# # Access fitted KMeans models
# obs_kmeans = result['obs_kmeans']
# act_kmeans = result['act_kmeans']

# # Display Results
# print(train_transformed.head())
