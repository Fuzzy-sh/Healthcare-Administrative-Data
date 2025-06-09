import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import pyprind

# this class is for reading and writing the data
import config.config_param as config_param
from config.config_param import config_param
import itertools
from scipy import stats

tqdm.pandas()


class preprocess_SC:
    # __slots__ = ['config', 'data', 'data_path', 'data_name', 'data_type' ]
    def __init__(
        self,
        data_path,
        data_path_out,
        data_name,
        data_type,
        group_colunms,
        dommies_columns,
        outcome,
        sex_col,
        colmeta,
        colbin,
        colnorm,
        colamhdiagn,
        colelxdiag,
        coltreatment,
        colvisits,
        collog,
        colid,
    ):

        self.data = None
        self.data_path = data_path
        self.data_path_out = data_path_out
        self.data_name = data_name
        self.data_type = data_type
        self.group_colunms = group_colunms
        self.dommies_columns = dommies_columns
        self.outcome = outcome
        self.sex_col = sex_col

        self.colmeta = colmeta
        self.colbin = colbin
        self.colnorm = colnorm
        self.colamhdiagn = colamhdiagn
        self.colelxdiag = colelxdiag
        self.coltreatment = coltreatment
        self.colvisits = colvisits
        self.collog = collog
        self.data_type = data_type
        self.colid = colid
        # Define the possible states for treatments (0 or 1)

        self.treatment_states = [0, 1]
        # Generate all possible combinations of med1, med2, med3
        self.actions = list(
            itertools.product(self.treatment_states, repeat=len(self.coltreatment))
        )
        # Create a dictionary to map each combination to a unique action number
        self.action_dict = {
            action: idx for idx, action in enumerate(self.actions, start=1)
        }
        self.colaction = "action"
        self.coltrj = "traj"
        self.bloc = "bloc"
        self.step = "step"
        self.col_zero = self.colbin
        # # Create reverse lookup dictionary
        # reverse_lookup = { v : k for k, v in action_dict.items()}

    # Define a function to map treatment values to actions using the reverse lookup dictionary

    def map_treatment_to_action(self, treatment_list):
        treatment_tuple = tuple(1 if value > 0 else value for value in treatment_list)
        return self.action_dict[treatment_tuple]

    # read the data
    def read_data(self, file_path):
        # read the data
        df = pd.read_hdf(file_path, key="df")

        print(
            f"Number of all individuals for {self.data_name}: {df[self.colid].nunique()}"
        )
        return df

    # sum the records for each day
    def sum_apply(self, group):
        return group.sum()

    # to remove the zero values from the data by adding -0.5 to all values
    def remove_zero(self, df, col_lists, added_value=-0.5):
        # Remove zero values by adding -0.5 to all values
        for col_name in col_lists:
            df[col_name] = df[col_name] + added_value
        return df

    def add_trajectories(self, df):
        # Process data into trajectory data for later use
        df_zs = pd.DataFrame(
            df, columns=self.colmeta + self.colbin + self.colnorm + self.collog
        )
        meta_df = pd.DataFrame(df[self.colmeta].values, columns=self.colmeta)
        ob_df = pd.DataFrame(
            df[self.colbin + self.colnorm + self.collog].values,
            columns=self.colbin + self.colnorm + self.collog,
        )
        ac_df = pd.DataFrame(df[self.colaction].values, columns=[self.colaction])
        raw_data_df = df.copy()
        num_actions = len(self.action_dict.keys())
        outcome_key = self.outcome  # Should be homelessness
        meta_cols = meta_df.columns.tolist()
        ob_cols = ob_df.columns.tolist()

        # Add trajectory column to all dataframes
        meta_df[self.coltrj] = df[self.coltrj]
        ob_df[self.coltrj] = df[self.coltrj]
        ac_df[self.coltrj] = df[self.coltrj]
        trajectories = df[self.coltrj].unique()
        data = {}
        data["meta_cols"] = meta_cols
        data["obs_cols"] = ob_cols
        data[self.coltrj] = {}
        print(f"{self.data_name} Cohort -- Making trajectory data")
        # bar = pyprind.ProgBar(len(trajectories))
        # c=0
        for i in tqdm(trajectories):
            data["traj"][i] = {}
            data["traj"][i]["meta"] = meta_df[meta_df["traj"] == i][meta_cols].values.T
            data["traj"][i]["obs"] = ob_df[ob_df["traj"] == i][ob_cols].values.T
            data["traj"][i]["actions"] = ac_df[ac_df["traj"] == i][
                "action"
            ].values.astype(np.int32)
            # print(raw_data_df[raw_data_df['traj'] == i][outcome_key].values[-1])
            data["traj"][i]["outcome"] = raw_data_df[raw_data_df["traj"] == i][
                outcome_key
            ].values[-1]
            data["traj"][i]["rewards"] = np.zeros(len(data["traj"][i]["actions"]))
            data["traj"][i]["rewards"][-1] = 1 - 2 * data["traj"][i]["outcome"]
            # if (data['traj'][i]['rewards'][-1]) < 0:
            #     c+=1
            #     print(c)

            # print(f"the outcome {data['traj'][i]['rewards'][-1]}")
            # bar.update()
            # data[self.coltrj][i] = {}
            # data[self.coltrj][i]['meta'] = meta_df[meta_df[self.coltrj]==i][meta_cols].values.T
            # data[self.coltrj][i]['obs'] = ob_df[ob_df[self.coltrj] == i][ob_cols].values.T
            # data[self.coltrj][i]['actions'] = ac_df[ac_df[self.coltrj] == i][self.colaction].values.astype(np.int32)
            # print(f"length of actions {len(data[self.coltrj][i]['actions'])}")

            # data[self.coltrj][i]['outcome'] = df[df[self.coltrj] == i][outcome_key].values[0]
            # data[self.coltrj][i]['rewards'] = np.zeros(len(data[self.coltrj][i]['actions']))
            # outcome_value = data['traj'][i]['outcome']

            # if isinstance(outcome_value, np.ndarray):
            #     outcome_value = outcome_value.item()  # Extract scalar if array
            # if outcome_value>0:
            #     print("length of rewards", len(data[self.coltrj][i]['actions']))
            #     print(f"The ouctome {outcome_value}")
            # data[self.coltrj][i]['rewards'][-1] = (1 - 2 * outcome_value)
            # # data[self.coltrj][i]['rewards'][-1] = (1 - 2*data['traj'][i]['outcome'])

        print(f"{self.data_name} Cohort -- Making final output file")
        col_names = ["traj", "step"]
        col_names.extend(["m:" + i for i in data["meta_cols"]])
        col_names.extend(["o:" + i for i in data["obs_cols"]])
        col_names.append("a:action")
        col_names.append("r:reward")

        all_data = []
        # bar = pyprind.ProgBar(len(data['traj'].keys()))
        for i in tqdm(data["traj"].keys()):
            # bar.update()
            for ctr in range(data["traj"][i]["actions"].shape[0]):
                all_data.append([])
                all_data[-1].append(i)
                all_data[-1].append(ctr)
                for m_index in range(data["traj"][i]["meta"].shape[0]):
                    all_data[-1].append(data["traj"][i]["meta"][m_index, ctr])
                for o_index in range(data["traj"][i]["obs"].shape[0]):
                    all_data[-1].append(data["traj"][i]["obs"][o_index, ctr])
                all_data[-1].append(data["traj"][i]["actions"][ctr])
                all_data[-1].append(data["traj"][i]["rewards"][ctr])
        df_preprocessed = pd.DataFrame(all_data, columns=col_names)
        return df_preprocessed

    def save_data(self, df, file):
        # save the data
        file_name = os.path.join(self.data_path_out, file)
        try:
            df.to_hdf(file_name, key="df", mode="w")
            return True, None
        except Exception as e:
            return False, e

    # Function to truncate rows after the first occurrence of True for each individual
    def truncate_after_true(self, group):
        # Find the index of the first True value
        true_idx = group[
            self.outcome
        ].idxmax()  # idxmax returns the index of first max value (i.e. True)
        # Truncate if True exists, else keep the whole group
        return group.loc[:true_idx] if group[self.outcome].any() else group

    def preprocess_data(self):

        # only the files with .h5 extension are read and has the outcome in the file name
        print(self.data_path)
        file_list = [
            file
            for file in os.listdir(self.data_path)
            if file.endswith(self.data_type) and self.data_name in file
        ]
        file_list.sort()
        print(file_list)

        print("Processing folder:", self.data_path)
        base_traj = 0
        for file in tqdm(file_list):
            file_path = os.path.join(self.data_path, file)
            df = self.read_data(file_path)

            print(f"Processing {file} with {df.shape[0]} records")
            print(
                f"Number of all individuals for {self.data_name}: {df[self.colid].nunique()}"
            )
            df_encoded = pd.get_dummies(
                df, columns=self.dommies_columns, drop_first=False
            )

            # df_encoded.drop(['visit', 'database'], axis=1, inplace=True)
            df_group_ = df_encoded.groupby(
                self.group_colunms
            )  # group the data by the start date; in some days there are multiple records
            # print(df_group_.index.name)
            df_group = df_group_.progress_apply(self.sum_apply).reset_index(
                drop=False
            )  # .set_index(self.colid) # sum the records for each day
            df_group[self.sex_col] = df_group[self.sex_col].map(
                lambda x: 1 if x == "M" else 0
            )
            # df_group= pd.get_dummies(df_group, columns=self.sex_col, drop_first=False) # one hot encoding for sex and change it to subject_sex_M
            # self.sex_col=df_group.columns[-1]
            self.colbin = self.sex_col
            # df_group = self.remove_zero(df_group, self.col_zero)
            df_group[self.sex_col] = df_group[self.sex_col] - 0.5

            all_cols = (
                self.colmeta
                + self.sex_col
                + self.colnorm
                + self.colamhdiagn
                + self.colelxdiag
                + self.coltreatment
                + self.colvisits
                + [self.outcome]
            )
            included_col = [col for col in all_cols if col in df_group.columns]
            df_group = df_group[included_col]
            print("truncating after true")
            df_group = (
                df_group.groupby(self.colid)
                .progress_apply(self.truncate_after_true)
                .reset_index(drop=True)
            )
            df_group[self.outcome] = df_group[self.outcome].map(lambda x: 1 if x else 0)
            print("Number of homelessness individuals")
            print(len(df_group[df_group[self.outcome] == 1].groupby(self.colid)))

            # df_group = df_group[df_group[self.outcome] == 1]
            # Apply the mapping function to the DataFrame
            df_group[self.colaction] = df_group[self.coltreatment].apply(
                self.map_treatment_to_action, axis=1
            )
            self.collog = [col for col in self.collog if col in df_group.columns]
            df_group[self.collog] = stats.zscore(
                np.log(0.1 + df_group[self.collog].values)
            )

            df_group["bloc"] = df_group.groupby(self.colid).cumcount().values + 1
            df_group["traj"] = (df_group["bloc"] == 1).cumsum().values

            print(
                f"Number of all individuals for {file}: {df_group[self.colid].nunique()}"
            )
            print(f"Number of records for {file}: {df_group.shape[0]}")

            df_preprocessed = self.add_trajectories(df_group)

            df_preprocessed["traj"] = df_preprocessed["traj"] + base_traj

            base_traj = df_preprocessed["traj"].max()

            print(f"Number of all individuals for {file}: {base_traj}")
            print(f"Number of records for {file}: {df_preprocessed.shape[0]}")
            done, e = self.save_data(df_preprocessed, file)
            if done:
                print(f"Data for {file} is saved in {self.data_path_out}")
            else:
                print(
                    f"Data for {file} is not saved in {self.data_path_out} because of {e}"
                )


def main():
    # add config file to read the data and preprocess it
    config_params = config_param()
    # read the data from config
    param_dic = config_params.get_params()

    # create an object from the class preprocess_SC
    preprocess = preprocess_SC(
        param_dic["data_path"],
        param_dic["data_path_out"],
        param_dic["data_name"],
        param_dic["data_type"],
        param_dic["group_colunms"],
        param_dic["dommies_columns"],
        param_dic["outcome"],
        param_dic["sex_col"],
        param_dic["colmeta"],
        param_dic["colbin"],
        param_dic["colnorm"],
        param_dic["colamhdiagn"],
        param_dic["colexdiag"],
        param_dic["coltreatment"],
        param_dic["colvisits"],
        param_dic["collog"],
        param_dic["colid"],
    )
    # preprocess the data
    preprocess.preprocess_data()
    print("The data is preprocessed and saved in the output folder")


if __name__ == "__main__":
    main()
