"""
Utilities

1- preprocessing_traj.py : add trajectory information to the data from AHS_data to the SC_data
2- preporcessing_split.py : split the data into train, validation and test sets from SC_data to train_val_test_data
3- train the SC network using train_val_test_data and save the model into restuls, run1
4- trian the DR netwrok using the SC netwrok model output as it creats the hidden information as the states and save the encoded data to the run1
5- train the DR network using the encoded data and save the model to the run1

"""

import os
import operator
import gc
import numpy as np
import pandas as pd
import torch
import pyprind
from tqdm import tqdm

import itertools
import torch
import torch.nn as nn
from sklearn.preprocessing import PowerTransformer
from sklearn import preprocessing as sk_preprocessing
import pickle
from joblib import load
import h5py
import sqlite3
import json


from .rl_utils import Transition, Episode


def one_hot(x, num_x, data_type="numpy", device=None):
    if data_type == "numpy":
        res = np.zeros(num_x)
    elif data_type == "torch":
        res = torch.zeros(num_x).to(device)
    res[x] = 1.0
    return res


def add_columns_level_1(df):

    return (
        df[[col for col in df.columns if col.startswith("-a:")]].any(axis=1).astype(int)
    )


def add_columns_level_2(df, action_columns_level_2):
    med_col = [
        col for col in df.columns if "medication" in col and col.startswith("-a:")
    ]
    therapy_col = [
        col for col in df.columns if "therapy" in col and col.startswith("-a:")
    ]
    counseling_col = [
        col
        for col in df.columns
        if col.startswith("-a:") and col not in med_col and col not in therapy_col
    ]
    df[action_columns_level_2[0]] = df[med_col].any(axis=1).astype(int)
    df[action_columns_level_2[1]] = df[therapy_col].any(axis=1).astype(int)
    df[action_columns_level_2[2]] = df[counseling_col].any(axis=1).astype(int)
    return df

    # return df[[col for col in df.columns if col.startswith("-a:")]].any(axis=1).astype(int)


def add_columns_level_3(df):
    med_col = [
        col for col in df.columns if "medication" in col and col.startswith("-a:")
    ]
    # for the columns names in the list of med col for df, change the start of the column from "-a:" to "a3:"
    df.rename(columns={col: col.replace("-a:", "a3:") for col in med_col}, inplace=True)
    return df
    # return df[[col for col in df.columns if col.startswith("-a:")]].any(axis=1).astype(int)


class PreprocessData:
    def __init__(self, param_dic):
        # Initialize parameters from param_dic
        self.data_path = param_dic["data_path"]
        self.data_path_out = param_dic["data_path_out"]
        self.data_name = param_dic["data_name"]
        self.data_type = param_dic["data_type"]
        self.group_colunms = param_dic["group_colunms"]
        self.dummies_columns = param_dic["dummies_columns"]
        self.outcome = param_dic["outcome"]
        self.outcomes = param_dic["outcomes"]
        self.sex_col = param_dic["sex_col"]
        self.colmeta = param_dic["colmeta"]
        self.colbin = param_dic["colbin"]
        self.colamhdiagn = param_dic["colamhdiagn"]
        self.colelxdiag = param_dic["colelxdiag"]
        self.colelxdiag_mental = param_dic["colelxdiag_mental"]
        self.colhsu = param_dic["colhsu"]
        self.colhomeless = param_dic["colhomeless"]
        self.colpolice = param_dic["colpolice"]
        self.coltreatment = param_dic["coltreatment"]
        self.colvisits = param_dic["colvisits"]
        self.collog = param_dic["collog"]
        self.colid = param_dic["colid"]
        self.colstatic = param_dic["colstatic"]
        self.time_interval = param_dic["time_interval"]  # days
        self.time_interval_following = param_dic["time_interval_following"]  # days
        self.all_cols = (
            self.colmeta
            + self.colamhdiagn
            + self.colelxdiag
            + self.colhsu
            + self.coltreatment
            + self.outcomes
        )
        self.dates = param_dic["dates"]
        self.rewards = param_dic["rewards"]
        self.treatment_states = [0, 1]
        # Generate all possible combinations of treatments (actions)
        self.actions = list(
            itertools.product(self.treatment_states, repeat=len(self.coltreatment))
        )
        self.action_dict = {
            action: idx for idx, action in enumerate(self.actions, start=1)
        }
        self.colaction = "action"
        self.coltrj = "traj"
        self.bloc = "bloc"
        self.step = "step"
        self.col_zero = self.colbin
        self.col_age = param_dic["age_index"]
        self.age_cat = param_dic["age_cat"]
        self.col_age_cat = param_dic["col_age_cat"]
        # self.time_buffer = param_dic['time_buffer']  # days

    def read_data(self, file_path, prefix=None):
        """Read data from HDF file and optionally filter by prefix."""
        print(f"Reading data from {file_path}")
        df = pd.read_hdf(file_path, key="df") #.head(100)

        missing_cols = set(self.rewards) - set(df.columns)
        df[list(missing_cols)] = 0
        if prefix is not None:
            features = [col for col in df.columns if col.startswith(prefix)]
            return df[features]
        return df

    def read_data_pickle(self, file_name):
        """Load data from a pickle file."""
        print(f"Reading {file_name}")
        try:
            loaded_data = load(file_name)
            return loaded_data
        except Exception as e:
            print(e)
            return None

    def save_data_pickle(self, df, file_name):
        """Save data as a pickle file."""
        with open(file_name, "wb") as file:
            pickle.dump(df, file)

    def save_data(self, df, file_name):
        """Save DataFrame to HDF file."""
        try:
            df.to_hdf(file_name, key="df", mode="w")
            return True, None
        except Exception as e:
            print(e)
            return False, e

    def save_data_txt(self, results, file_name):
        """Save results to a text file."""
        with open(f"{file_name}", "w") as f:
            f.write(f"{results}")

    def save_data_big_dict(self, episodes, file_name, cluster):
        """Save episodes (list of transitions) to SQLite database."""
        conn = sqlite3.connect(file_name)
        cursor = conn.cursor()
        # Create table for storing transitions
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS dictionary (
            episode_key TEXT PRIMARY KEY,
            transition_key TEXT,
            state TEXT, 
            action TEXT, 
            reward DOUBLE,
            next_state TEXT,
            done TEXT
        )
        """
        )
        key = 0
        for episode in episodes:
            for idx, transition in enumerate(episode):
                if cluster:
                    # If cluster, store raw values
                    cursor.execute(
                        "INSERT OR REPLACE INTO dictionary (episode_key, transition_key, state, action, reward, next_state, done) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            key,
                            idx,
                            transition.state,
                            transition.action,
                            transition.reward,
                            transition.next_state,
                            transition.done,
                        ),
                    )
                else:
                    # Otherwise, store as JSON
                    cursor.execute(
                        "INSERT OR REPLACE INTO dictionary (episode_key, transition_key, state, action, reward, next_state, done) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            key,
                            idx,
                            json.dumps(transition.state),
                            json.dumps(transition.action),
                            transition.reward,
                            json.dumps(transition.next_state),
                            json.dumps(transition.done),
                        ),
                    )
            key += 1
        conn.commit()
        conn.close()

    def read_data_big_dict(self, file_name, cluster):
        """Read episodes from SQLite database."""
        conn = sqlite3.connect(file_name)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT episode_key, transition_key, state, action, reward, next_state, done FROM dictionary GROUP BY episode_key"
        )
        data = cursor.fetchall()
        episodes = []
        if cluster:
            # Not implemented: cluster mode
            pass
        else:
            # Process each row and reconstruct episodes
            for edx, episode in data:
                for row in episode:
                    e = Episode(subject_id=edx)
                    transition_key = row[1]
                    state = json.loads(row[2])
                    action = json.loads(row[3])
                    reward = json.loads(row[4])
                    next_state = json.loads(row[5])
                    done = json.loads(row[6])
                    transition = Transition(state, action, reward, done, next_state)
                    e.add_transition(transition)
                episodes.append(e)
        conn.close()
        return episodes

    def truncate_after_true(self, group):
        """Truncate rows after the first occurrence of True in the homelessness column for each individual."""
        group = group.sort_values("start_date")
        if group[self.colhomeless].sum() > 0:
            true_idx = group[self.colhomeless].idxmax()
            group = group.loc[:true_idx]
        return group

    def max_apply(self, group):
        """Apply max aggregation to a group."""
        return group.apply(lambda x: x.max())

    def create_intervals(self, df_ind):
        """Create date intervals for aggregation."""
        start_date = df_ind["start_date"].min()
        end_date = df_ind["start_date"].max()
        return (
            pd.date_range(
                start=start_date, end=end_date, freq=f"{self.time_interval}D"
            ),
            end_date,
        )

    def add_flags(self, df_ind):
        """Add trigger, police, and homeless flags to the DataFrame."""
        df_ind["Is_Trigger"] = (
            df_ind[self.colamhdiagn + self.colelxdiag_mental] > 0
        ).any(axis=1)
        df_ind["Is_police"] = pd.DataFrame(df_ind[self.colpolice] > 0).any(axis=1)
        df_ind["Is_homeless"] = pd.DataFrame(df_ind[self.colhomeless] > 0).any(axis=1)
        return df_ind

    def create_aggregation_windows(self, df_ind, intervals, end_date_ind):
        """Create aggregation windows and collect special event dates."""
        aggregation_windows = [
            (date, date + pd.Timedelta(days=self.time_interval))
            for date in intervals
            if date + pd.Timedelta(days=self.time_interval) <= end_date_ind
        ]
        trigger_dates = df_ind[df_ind["Is_Trigger"]]["start_date"].sort_values()
        police_dates = df_ind[df_ind["Is_police"]]["start_date"].sort_values()
        homeless_dates = df_ind[df_ind["Is_homeless"]]["start_date"].sort_values()
        for date in trigger_dates:
            aggregation_windows.append(
                (date - pd.Timedelta(days=self.time_interval), date)
            )
        aggregation_windows = list(set(aggregation_windows))  # Remove duplicates
        police_dates = list(set(police_dates))
        homeless_dates = list(set(homeless_dates))
        return aggregation_windows, police_dates, homeless_dates

    def compute_rewards(self, index, end, police_dates, homeless_dates):
        """Compute rewards based on police and homeless events within the window."""
        reward_value = 0
        reward_h = 0
        reward_p = 0
        homeless_date = None
        police_date = None
        for homeless_date in homeless_dates:
            if index <= homeless_date <= end:
                reward_h = (
                    -1 - (homeless_date - index).days / self.time_interval_following
                )
                reward_h = -0.00001 if reward_h == 0 else reward_h
                break
        for police_date in police_dates:
            if index <= police_date <= end:
                reward_p = -(police_date - index).days / self.time_interval_following
                reward_p = -0.00001 if reward_p == 0 else reward_p
                break
        reward_value = reward_h + reward_p
        if not reward_h and not reward_p:
            reward_value = 1
        return reward_value, reward_h, reward_p, homeless_date, police_date

    def aggregate_data(
        self, df_ind, aggregation_windows, police_dates, homeless_dates, end_date_ind
    ):
        """Aggregate data for each window and compute rewards."""
        aggregated_data = []
        for start, index in aggregation_windows:
            end = index + pd.Timedelta(days=self.time_interval_following)
            end = end_date_ind if end > end_date_ind else end
            if end == index:
                continue
            filtered_data = df_ind[
                (df_ind["start_date"] >= start) & (df_ind["start_date"] < index)
            ]
            following_data = df_ind[
                (df_ind["start_date"] >= index) & (df_ind["start_date"] <= end)
            ]
            # treatment_flag = filtered_data['no_treatment'].min()== 0 # Check if any treatment was applied

            reward_value, reward_h, reward_p, homeless_date, police_date = (
                self.compute_rewards(index, end, police_dates, homeless_date)
            )
            if not filtered_data.empty:
                aggregated_row = {
                    "start_date": start,
                    "index_date": index,
                    "end_date": end,
                    "subject_id": filtered_data["subject_id"].values[0],
                    "subject_sex": filtered_data["subject_sex"].values[0],
                }
                for col in self.col_age_cat:
                    aggregated_row[col] = filtered_data[col].values[0]
                for col in self.colamhdiagn + self.colelxdiag + self.colhsu:
                    aggregated_row[col] = filtered_data[col].sum()
                for col in self.coltreatment + self.outcomes:
                    aggregated_row[col] = following_data[col].sum()
                aggregated_row["rewards"] = reward_value
                aggregated_row["rewards_h"] = reward_h
                aggregated_row["rewards_p"] = reward_p
                aggregated_row["rewards_h_date"] = homeless_date
                aggregated_row["rewards_p_date"] = police_date
                aggregated_data.append(aggregated_row)
        return pd.DataFrame(aggregated_data)

    def aggregate_apply(self, df_ind):
        """Apply aggregation to an individual's data."""
        if not df_ind.empty:
            df_ind = df_ind.sort_values("start_date")
            intervals, end_date_ind = self.create_intervals(df_ind)
            df_ind = self.add_flags(df_ind)
            aggregation_windows, police_dates, homeless_dates = (
                self.create_aggregation_windows(df_ind, intervals, end_date_ind)
            )
            return self.aggregate_data(
                df_ind, aggregation_windows, police_dates, homeless_dates, end_date_ind
            )
        return pd.DataFrame()

    def def_age_categorical(self, df_subjects):
        """Categorize age into bins."""
        df_subjects[self.age_cat] = df_subjects[self.col_age[0]].map(
            lambda x: (
                "18-29"
                if ((x >= 18) and (x < 30))
                else (
                    "30-39"
                    if ((x >= 30) and (x < 40))
                    else (
                        "40-49"
                        if ((x >= 40) and (x < 50))
                        else (
                            "50-59"
                            if ((x >= 50) and (x < 60))
                            else ("60+" if x >= 60 else x)
                        )
                    )
                )
            )
        )
        return df_subjects

    def add_cols(self, df):
        """Add missing columns with zero values."""
        missing_cols = set(self.all_cols) - set(df.columns)
        print(f"Adding columns: {missing_cols}")
        df[list(missing_cols)] = 0
        return df

    def convert_boolean_columns_to_binary(self, df):
        """Convert boolean columns to binary (0/1)."""
        boolean_columns = df.select_dtypes(include=["bool"]).columns
        df[boolean_columns] = df[boolean_columns].astype(int)
        return df

    def preprocess(self, df, file):
        """
        Main preprocessing pipeline for a given DataFrame.

        This method performs a series of preprocessing steps on the input DataFrame, including:
        1. Converts 'start_date' and 'end_date' columns to pandas Timestamps.
        2. Computes a 'no_treatment' column indicating rows with no treatments applied.
        3. Prints summary statistics about the dataset and individuals.
        4. Categorizes age and updates the list of columns to be one-hot encoded.
        5. One-hot encodes specified categorical columns.
        6. Converts boolean columns to binary format.
        7. Groups the data by specified columns and aggregates using a custom function.
        8. Converts the sex column to binary (1 for 'M', 0 otherwise).
        9. Truncates records after the first homelessness event for each individual.
        10. Selects relevant columns for further analysis.
        11. Prints statistics about homelessness events and dataset size.
        12. Adds any missing columns required for downstream processing.
        13. Aggregates data for each individual over a specified time interval.
        14. Returns the final preprocessed DataFrame with selected columns.

        Args:
            df (pd.DataFrame): The input DataFrame to preprocess.
            file (str): The filename or identifier for the dataset being processed.

        Returns:
            pd.DataFrame: The preprocessed and aggregated DataFrame ready for analysis.
        """

        df["start_date"] = df["start_date"].map(lambda x: pd.Timestamp(x))
        df["end_date"] = df["end_date"].map(lambda x: pd.Timestamp(x))
        treatment_cols = self.coltreatment[
            1:
        ]  # Exclude the first column which is 'no_treatment'
        df["no_treatment"] = df[treatment_cols].apply(
            lambda x: 1 if x.sum() == 0 else 0, axis=1
        )

        print(f"Processing {file} with {df.shape[0]} records")
        print(
            f"Number of all individuals for {self.data_name}: {df[self.colid].nunique()}"
        )
        df = self.def_age_categorical(df)
        self.dummies_columns = self.dummies_columns + [self.age_cat]
        # One-hot encode categorical columns
        df_encoded = pd.get_dummies(df, columns=self.dummies_columns, drop_first=False)
        df_encoded = self.convert_boolean_columns_to_binary(df_encoded)
        # Group by specified columns and aggregate
        df_group_ = df_encoded.groupby(self.group_colunms)
        df_group = df_group_.progress_apply(self.max_apply).reset_index(drop=True)
        df_group[self.sex_col[0]] = df_group[self.sex_col[0]].map(
            lambda x: 1 if x == "M" else 0
        )
        # Truncate after first homelessness event
        df_truncated_ = df_group.groupby(self.colid).progress_apply(
            self.truncate_after_true
        )
        df_truncated = df_truncated_.reset_index(drop=True)
        # Select relevant columns
        all_cols = (
            self.colmeta
            + self.sex_col
            + self.col_age_cat
            + self.colamhdiagn
            + self.colelxdiag
            + self.coltreatment
            + self.colhsu
            + self.outcomes
        )
        included_col = [col for col in all_cols if col in df_truncated.columns]
        df_updated = df_truncated[included_col]
        print("Number of homelessness individuals")
        print(len(df_updated[df_updated[self.outcome] == 1]))
        print(
            f"Number of all individuals for {file}: {df_updated[self.colid].nunique()}"
        )
        print(f"Number of records for {file}: {df_updated.shape[0]}")
        # Add missing columns
        df_updated = self.add_cols(df_updated)
        df_updated_group_by_ = df_updated.groupby(self.colid)
        print(f"Aggregating data for {self.time_interval} days")
        df_updated_group_by_ = df_updated_group_by_.progress_apply(self.aggregate_apply)
        df_updated_ = df_updated_group_by_.reset_index(drop=True)
        df_updated_ = df_updated_[
            self.colmeta
            + self.dates
            + self.colstatic
            + self.colamhdiagn
            + self.colelxdiag
            + self.colhsu
            + self.coltreatment
            + [self.rewards[0]]
            + self.outcomes
        ]
        print(
            f"Number of records for aggregated {file} for {self.time_interval} days: {df_updated_.shape[0]}"
        )
        return df_updated_


class UpdateData:
    def __init__(self, param_dic) -> None:

        # self.all_files = all_files
        self.colmeta = param_dic["colmeta"]
        self.dates = param_dic["dates"]
        self.colstatic = param_dic["colstatic"]
        self.colamhdiagn = param_dic["colamhdiagn"]
        self.colelxdiag = param_dic["colelxdiag"]
        self.colhsu = param_dic["colhsu"]
        self.coltreatment = param_dic["coltreatment"]
        self.outcomes = param_dic["outcomes"]
        self.rewards = param_dic["rewards"]
        self.group = param_dic["group"]

    def one_hot(self, column):
        try:
            return column.apply(lambda x: 1 if x > 0 else 0)
        except Exception as e:
            # print (column )
            print(e)
            return

    def one_hot_encode_actions(self, data, prefix):

        # Filter columns starting with 'a:'
        action_columns = [col for col in data.columns if col.startswith(prefix)]

        data[action_columns] = data[action_columns].apply(self.one_hot)

        # Apply one-hot encoding to these columns
        return data

    def z_score_standardization(self, column):
        return (column - column.mean()) / column.std()

    def power_transformation(self, column):
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        return transformer.fit_transform(column.values.reshape(-1, 1))

    def log_transform(self, column, epsilon=1e-10):
        return np.log(column + epsilon)

    def normalize(self, data, prefix):
        observation_columns = [col for col in data.columns if col.startswith(prefix)]

        for col in observation_columns:
            data[col] = self.power_transformation(data[col])
        return data

    def change_column_name(self, data):
        # print(data.columns)

        all_cols = (
            self.colmeta
            + self.dates
            + self.colstatic
            + self.colamhdiagn
            + self.colelxdiag
            + self.colhsu
            + self.coltreatment
            + [self.rewards[0]]
            + self.outcomes
        )

        column_name = ["m:" + i for i in self.colmeta + self.dates]

        column_name.extend(
            [
                "o:" + i
                for i in self.colstatic
                + self.colamhdiagn
                + self.colelxdiag
                + self.colhsu
            ]
        )
        column_name.extend(["a:" + i for i in self.coltreatment])
        column_name.extend(["r:" + i for i in [self.rewards[0]]])
        column_name.extend([i for i in self.outcomes])

        data_ = data[all_cols].copy()
        data_.columns = column_name

        return data_

    def column_change(self, data):

        df_updated_columnd = self.change_column_name(data)
        # all_files is the merged files from the preprocessing step

        return df_updated_columnd

    def normalize_data(self, data, prefix="o:"):
        # only normalize observation clumns
        df_normalized = self.normalize(data, prefix)

        return df_normalized

    def one_hot_encoded(self, data, prefix="a:"):

        df_one_hot_encoded = self.one_hot_encode_actions(data, prefix)
        return df_one_hot_encoded

    def get_group_treatments(self, data, param_dic):
        """
        Group the treatments into one-hot encoded columns.
        """
        data_ = data.copy()
        if self.group:

            # group_treatment_cols= param_dic['colgroup_treatment']
            # group_cols_exists = False
            # exist_list = [col for col in group_treatment_cols if col in treatment_data_.columns]
            # print ("Check if the grourp treatment is already in the data")
            # if len(exist_list)>0:
            #     print("Group treatment columns already exist in the data.")
            #     group_cols_exists = True
            #     return treatment_data_[group_treatment_cols], group_cols_exists
            # print("Group treatment columns do not exist in the data.")
            # print ("Grouping the treatments")

            print("Assing treatment column pased on param dic")
            medication_features = param_dic["coltreatment_medication"]
            counseling_features = param_dic["coltreatment_counseling"]
            therapy_features = param_dic["coltreatment_therapy"]
            print("Mdications", medication_features)
            # print("Counseling", counseling_features)
            print("Therapy", therapy_features)
            all_treatment_features = (
                medication_features + counseling_features + therapy_features
            )
            # print (all_treatment_features)

            print("Grouping the treatments")
            # print (treatment_data_[medication_features].apply_progress(lambda x: 1 if x.sum()>0 else 0, axis=1).head())

            data_["a:treatment_medication"] = data_[medication_features].apply(
                lambda x: 1 if x.sum() > 0 else 0, axis=1
            )
            data_["a:treatment_counseling"] = data_[counseling_features].apply(
                lambda x: 1 if x.sum() > 0 else 0, axis=1
            )
            data_["a:treatment_therapy"] = data_[therapy_features].apply(
                lambda x: 1 if x.sum() > 0 else 0, axis=1
            )

            # col_remove = [col for col in self.data.columns if col.startswith('a:') and col not in self.group_treatment_cols]
            data_.columns = [
                col.replace("a:", "-a:") if col in all_treatment_features else col
                for col in data_.columns
            ]

            # treatment_data_['ag:treatment_no_treatment'] = 1 if treatment_data_[all_features].sum(axis=1)<1 else 0
            # group_treatment= treatment_data_[group_treatment_cols]#.apply(lambda x: x.astype(int), axis=1)
            # return treatment_data_[ col for col in treatment_data_.columns if col.startswith('ag:')]
            return data_
        else:

            selected_treatment = param_dic["action_columns"]
            # not_selected_treatment = [col for col in data.columns if col.startswith('a:') and col not in selected_treatment]
            data_.columns = [
                col.replace("a:", "-a:") if col not in selected_treatment else col
                for col in data_.columns
            ]
            return data_


class SplitData:

    def __init__(
        self, data, param_dic
    ):  # id, path,  train_frac=0.8, val_frac=0.1, test_frac=0.2):
        self.data = data
        self.id = param_dic["colid"]
        self.train_frac = param_dic["train_frac"]
        self.val_frac = param_dic["val_frac"]
        self.test_frac = param_dic["test_frac"]
        self.data_path_out = param_dic["data_path_out"]
        self.police_interaction = param_dic["colpolice"][0]
        self.homeless = param_dic["colhomeless"][0]
        # self.split_data_path = param_dic['split_data_path']
        self.file_name_train = param_dic["file_name_train"]
        self.file_name_val = param_dic["file_name_val"]
        self.file_name_test = param_dic["file_name_test"]

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_id = None
        self.val_id = None
        self.test_id = None
        self.path = param_dic["data_path_out"]

    def make_split_by_subj(self):

        # group the data by the subject_id and select the last row for each subject_id

        grouped_by_id = self.data.groupby(self.id)
        last_row_subjects = grouped_by_id.tail(1).reset_index(drop=True)
        # select the rows for each subject_id if the police interaction or homelessness are greater than 0
        print(last_row_subjects)

        for id_no, sub_data in tqdm(grouped_by_id):
            # Check if the id exists in last_row_subjects
            if id_no in last_row_subjects[self.id].values:
                # Update the 'police_interaction' column for the specific id
                last_row_subjects.loc[
                    last_row_subjects[self.id] == id_no, self.police_interaction
                ] = sub_data[self.police_interaction].any()

        print("# Keep only the desired columns")
        result_ddf = last_row_subjects[
            [self.id, self.homeless, self.police_interaction]
        ]

        print("# Split the data into training and testing sets based on subjects")
        test_set = result_ddf.groupby(
            [self.homeless, self.police_interaction], group_keys=False
        ).apply(lambda x: x.sample(frac=self.test_frac))

        train_val_set = result_ddf[~result_ddf[self.id].isin(test_set[self.id])]

        val_set = train_val_set.groupby(
            [self.homeless, self.police_interaction], group_keys=False
        ).apply(lambda x: x.sample(frac=self.val_frac))

        train_set = train_val_set[~train_val_set[self.id].isin(val_set[self.id])]

        self.train_id = list(train_set[self.id])
        self.val_id = list(val_set[self.id])
        self.test_id = list(test_set[self.id])

        self.train_data = self.data[self.data[self.id].isin(self.train_id)].reset_index(
            drop=True
        )
        self.val_data = self.data[self.data[self.id].isin(self.val_id)].reset_index(
            drop=True
        )
        self.test_data = self.data[self.data[self.id].isin(self.test_id)].reset_index(
            drop=True
        )

    def save_split_data(self):
        train_file_name = os.path.join(self.data_path_out, self.file_name_train)
        test_file_name = os.path.join(self.data_path_out, self.file_name_test)
        val_file_name = os.path.join(self.data_path_out, self.file_name_val)
        try:
            self.train_data.to_hdf(train_file_name, key="df", mode="w")
            self.test_data.to_hdf(test_file_name, key="df", mode="w")
            self.val_data.to_hdf(val_file_name, key="df", mode="w")
            return True, None
        except Exception as e:
            return False, e


class DataLoader(object):
    def __init__(
        self,
        encoded_data,
        rng,
        minibatch_size,
        drop_smaller_than_minibatch,
        device,
        str_id="",
    ):
        """
        If encoded_data is str, it is inferred as path-to-file name and will be loaded using pd
        If encoded_data is dict it is inferred as data and will be used directly
        Use: first call make_transition_train_data() once, then call reset() before each epoch of getting all minibatches
        """
        self.rng = rng
        self.device = device
        self.str_id = str_id  # optional: used in logging
        self.minibatch_size = minibatch_size
        self.drop_smaller_than_minibatch = drop_smaller_than_minibatch
        self.ps = None
        self.ns = None
        if isinstance(encoded_data, str):
            self.encoded_data_file = os.path.abspath(encoded_data)
            self.encoded_data = pd.read_csv(self.encoded_data_file)
        elif isinstance(encoded_data, dict):
            self.encoded_data = encoded_data
        else:
            raise ValueError("Unknown encoded data.")
        self.transition_data = {}
        self.transition_indices_pos_last = []
        self.transition_indices_neg_last = []
        self.transition_indices = None
        self.transition_data_size = None
        self.transitions_head = None
        self.transitions_head_pos = None
        self.transitions_head_neg = None
        self.epoch_finished = True  # to enforce reset() before use
        self.epoch_pos_finished = True
        self.epoch_neg_finished = True
        self.num_minibatches_epoch = None

    def reset(self, shuffle, pos_samples_in_minibatch, neg_samples_in_minibatch):
        self.ps = pos_samples_in_minibatch
        self.ns = neg_samples_in_minibatch
        if shuffle:
            self.rng.shuffle(self.transition_indices)
            self.rng.shuffle(self.transition_indices_pos_last)
            self.rng.shuffle(self.transition_indices_neg_last)
        self.transitions_head = 0
        self.transitions_head_pos = 0
        self.transitions_head_neg = 0
        self.epoch_finished = False
        self.epoch_pos_finished = False
        self.epoch_neg_finished = False
        self.num_minibatches_epoch = int(
            np.floor(self.transition_data_size / self.minibatch_size)
        ) + int(1 - self.drop_smaller_than_minibatch)

    def make_transition_data(self, release=False):
        print("DataLoader: making transitions (s,a,r,s') " + self.str_id)
        self.transition_data["s"] = {}
        self.transition_data["actions"] = {}
        self.transition_data["rewards"] = {}
        self.transition_data["next_s"] = {}
        self.transition_data["terminals"] = {}
        indices_pos = []
        indices_neg = []
        counter = 0
        bar = pyprind.ProgBar(len(list(self.encoded_data["traj"].keys())))
        for traj in self.encoded_data["traj"].keys():
            bar.update()
            for t in range(self.encoded_data["traj"][traj]["actions"].shape[0] - 1):
                self.transition_data["s"][counter] = self.encoded_data["traj"][traj][
                    "s"
                ][t, :]
                self.transition_data["next_s"][counter] = self.encoded_data["traj"][
                    traj
                ]["s"][t + 1, :]
                self.transition_data["actions"][counter] = self.encoded_data["traj"][
                    traj
                ]["actions"][t]
                self.transition_data["rewards"][counter] = self.encoded_data["traj"][
                    traj
                ]["rewards"][t]
                self.transition_data["terminals"][counter] = 0
                if traj in self.encoded_data["pos_traj"]:
                    indices_pos.append(counter)
                else:
                    indices_neg.append(counter)
                counter += 1
            # For the last transition in the trajectory
            tlast = self.encoded_data["traj"][traj]["actions"].shape[0] - 1
            self.transition_data["s"][counter] = self.encoded_data["traj"][traj]["s"][
                tlast, :
            ]
            self.transition_data["next_s"][counter] = np.zeros_like(
                self.encoded_data["traj"][traj]["s"][tlast, :]
            )
            self.transition_data["actions"][counter] = self.encoded_data["traj"][traj][
                "actions"
            ][tlast]
            self.transition_data["rewards"][counter] = self.encoded_data["traj"][traj][
                "rewards"
            ][tlast]
            self.transition_data["terminals"][counter] = 1
            if traj in self.encoded_data["pos_traj"]:
                self.transition_indices_pos_last.append(counter)
            else:
                self.transition_indices_neg_last.append(counter)
            counter += 1
        self.transition_data_size = counter
        self.transition_indices = np.arange(self.transition_data_size)
        if release:
            del self.encoded_data
            self.encoded_data = None
            gc.collect()

    def get_next_minibatch(self):
        if self.epoch_finished == True:
            print(
                "Epoch finished, please call reset() method before next call to get_next_minibatch()"
            )
            return None
        # Getting data from dictionaries
        offset = self.ns + self.ps
        minibatch_main_index_list = list(
            self.transition_indices[
                self.transitions_head : self.transitions_head
                + self.minibatch_size
                - offset
            ]
        )
        minibatch_pos_last_index_list = self.transition_indices_pos_last[
            self.transitions_head_pos : self.transitions_head_pos + self.ps
        ]
        minibatch_neg_last_index_list = self.transition_indices_neg_last[
            self.transitions_head_neg : self.transitions_head_neg + self.ns
        ]
        self.transitions_head_pos += self.ps
        self.transitions_head_neg += self.ns
        minibatch_index_list = (
            minibatch_main_index_list
            + minibatch_pos_last_index_list
            + minibatch_neg_last_index_list
        )
        get_from_dict = operator.itemgetter(*minibatch_index_list)
        s_minibatch = get_from_dict(self.transition_data["s"])
        actions_minibatch = get_from_dict(self.transition_data["actions"])
        rewards_minibatch = get_from_dict(self.transition_data["rewards"])
        next_s_minibatch = get_from_dict(self.transition_data["next_s"])
        terminals_minibatch = get_from_dict(self.transition_data["terminals"])
        # Updating current data head
        self.transitions_head += self.minibatch_size
        self.epoch_finished = (
            self.transitions_head
            + self.drop_smaller_than_minibatch * self.minibatch_size
            >= self.transition_data_size
        )
        self.transitions_head_pos = self.transitions_head_pos % len(
            self.transition_indices_pos_last
        )
        self.transitions_head_neg = self.transitions_head_neg % len(
            self.transition_indices_neg_last
        )
        return (
            s_minibatch,
            actions_minibatch,
            rewards_minibatch,
            next_s_minibatch,
            terminals_minibatch,
            self.epoch_finished,
        )
