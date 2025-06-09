import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm


class PreprocessingSplit:

    def __init__(
        self, train_frac=0.8, val_frac=0.05, make_test=True
    ):  # 80% training data + 5% validation data + 15% test data
        self.data_path = "SC_data/"
        self.data_path_out = "train_val_test_data/"
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.make_test = make_test
        self.filetype = ".h5"
        self.filename = "homeless"

    def read_data(self):

        file_list = [
            file
            for file in os.listdir(self.data_path)
            if file.endswith(self.filetype) and self.filename in file
        ]
        file_list.sort()
        print("Reading files from:", self.data_path)
        dfs = []

        for file in tqdm(file_list):
            print(f"Reading file: {file}")
            file_path = os.path.join(self.data_path, file)
            df = pd.read_hdf(file_path)
            print(
                f"Number of all individuals for {self.filename}: {df['traj'].nunique()}"
            )
            dfs.append(df)

        combined_all_df = pd.concat(dfs).reset_index(drop=True)
        print(
            f"Number of all individuals for {self.filename}: {combined_all_df['traj'].nunique()}"
        )
        return combined_all_df

    def make_train_val_test_split(self):
        if self.make_test:
            assert (
                self.train_frac + self.val_frac < 1
            ), "train_frac + val_frac must be less than 1 to leave room for test data."

        # df = pd.read_csv(self.filename)
        df = self.read_data()
        all_traj = df["traj"].unique()
        all_rewards = []
        for traj in tqdm(all_traj):
            r = df[df["traj"] == traj]["r:reward"].sum()
            all_rewards.append(r)
        survivor_traj = [i for i in range(len(all_traj)) if all_rewards[i] == 1.0]
        dead_traj = [i for i in range(len(all_traj)) if all_rewards[i] == -1.0]

        np.random.shuffle(survivor_traj)
        np.random.shuffle(dead_traj)

        train_survivor_end_index = int(
            np.round(self.train_frac * len(survivor_traj), 0)
        )
        val_survivor_end_index = (
            int(np.round(self.val_frac * len(survivor_traj), 0))
            + train_survivor_end_index
        )
        train_dead_end_index = int(np.round(self.train_frac * len(dead_traj), 0))
        val_dead_end_index = (
            int(np.round(self.val_frac * len(dead_traj), 0)) + train_dead_end_index
        )

        train_traj = survivor_traj[:train_survivor_end_index]
        train_traj.extend(dead_traj[:train_dead_end_index])
        val_traj = survivor_traj[train_survivor_end_index:val_survivor_end_index]
        val_traj.extend(dead_traj[train_dead_end_index:val_dead_end_index])

        train_df = df[df["traj"].isin(train_traj)]
        val_df = df[df["traj"].isin(val_traj)]

        train_df.to_hdf(
            self.data_path_out + self.filename[:-4] + "_train.h5",
            key="df",
            mode="w",
            index=False,
        )
        val_df.to_hdf(
            self.data_path_out + self.filename[:-4] + "_validation.h5",
            key="df",
            mode="w",
            index=False,
        )

        if self.make_test:
            test_traj = survivor_traj[val_survivor_end_index:]
            test_traj.extend(dead_traj[val_dead_end_index:])
            test_df = df[df["traj"].isin(test_traj)]
            test_df.to_hdf(
                self.data_path_out + self.filename[:-4] + "_test.h5",
                key="df",
                mode="w",
                index=False,
            )


def main():

    # filename = 'data/rl_data.h5'

    preprocessor = PreprocessingSplit()
    # will save the split data in the same folder as data_file
    print("Processing Splitng the data into train, test, and validation sets ...")
    preprocessor.make_train_val_test_split()
    print("Done.")


if __name__ == "__main__":
    main()
