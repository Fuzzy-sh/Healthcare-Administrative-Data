import pandas as pd
import d3rlpy
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from inference import ModelBasedInference, ClusteringBasedInference
from tqdm import tqdm
from sklearn import preprocessing as sk_preprocessing

# pd.set_option('future.no_silent_downcasting', True)
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import d3rlpy
import numpy as np

# from Qlearning.RLModels_2layer import CustomAlgo
from collections import defaultdict
from sklearn.metrics import f1_score
from mpmath import mp

from scipy import stats

# Set precision to 133 decimal places
mp.dps = 133


# Suppress all RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# what is the best name for this file ?


class Encoder:
    def __init__(self):
        pass

    def action_encoder(self, treatments):
        action_encoder = sk_preprocessing.LabelEncoder()
        action_encoder.fit(np.array(treatments).reshape(-1, 1))
        actions_dict = dict(
            zip(
                action_encoder.classes_,
                action_encoder.transform(action_encoder.classes_),
            )
        )
        # print("Actions dictionary:", actions_dict)
        return actions_dict

    def encode_treatments(self, treatment_data):
        """
        Encodes binary treatment data into a dictionary mapping unique treatment combinations
        to an index.

        Parameters:
        treatment_data (pd.DataFrame): A DataFrame where each column represents a treatment
                                    and each row is a binary array indicating applied treatments.

        Returns:
        dict: A dictionary where keys are unique indices, and values are lists of applied treatments.
        """
        # Get the treatment names from the column headers
        treatment_names = treatment_data.columns
        # print("Treatment names:", treatment_names)

        # Use a dictionary to map unique treatment combinations to an index
        actions_dict = {}
        unique_combinations = {}

        # Iterate through each row in the DataFrame
        for index, row in treatment_data.iterrows():
            # Find the applied treatments for the current row
            applied_treatments_ = tuple(
                treatment_names[i] for i, val in enumerate(row) if val == 1
            )
            applied_treatments = tuple(i for i, val in enumerate(row) if val == 1)

            # Check if this combination already exists in unique_combinations
            if applied_treatments not in unique_combinations:
                unique_combinations[applied_treatments] = len(unique_combinations)

            # Update actions_dict with the corresponding index
            actions_dict[unique_combinations[applied_treatments]] = list(
                applied_treatments
            )

        # Ensure that every key in actions_dict has at least one treatment
        for key, treatments in actions_dict.items():
            if not treatments:
                # actions_dict[key] = [treatment_names[0]]
                actions_dict[key] = [0]

        print("Actions dictionary:", actions_dict)
        print(applied_treatments_)
        return actions_dict

    def action_to_index(self, action_dict, action, treatment_names):
        """
        Maps a treatment combination to an index using the action_dict.


        """
        # print("Action:", action)
        action_index = [i for i, val in enumerate(action) if val == 1]

        if len(action_index) == 0:
            action_index = [0]
        # print("Action index:", action_index)
        # Check if the action is in the action_dict
        for key, value in action_dict.items():

            if set(value) == set(action_index):

                return key

            # Map the index to the applied treatments
            # for treatment in applied_treatments:
            #     # Append the index to the treatment's list in the dictionary
            #     if treatment not in actions_dict:
            #         actions_dict[treatment] = []
            #     actions_dict[treatment].append(index)


class Transition:
    def __init__(
        self,
        state,
        action_level_1,
        action_level_2,
        action_level_3,
        reward,
        done,
        next_state=None,
    ):
        self.state = state
        self.action_level_1 = action_level_1
        self.action_level_2 = action_level_2
        self.action_level_3 = action_level_3
        self.reward = reward
        self.done = done
        self.next_state = next_state

        # the prediction probabilities for the action taken in the state.
        # This is actually the RL model prediction for each action in the state
        # It is used for importance sampling
        # it must be a dictionary of policies with a list of probabilities for each action
        # (e.g., {'policy1': [0.1, 0.2, 0.7], 'policy2': [0.3, 0.3, 0.4]})
        self.prediction_probs = {}

    def __str__(self):
        return "state: {}, actions_level_1: {}, actions_level_2: {}, actions_level_3: {}, reward: {}, done: {}".format(
            self.state,
            self.action_level_1,
            self.action_level_2,
            self.action_level_3,
            self.reward,
            self.done,
        )


class Episode:
    def __init__(self, subject_id=None):
        self.transitions = []
        if subject_id is not None:
            self.subject_id = subject_id
        else:
            self.subject_id = None

    def add_transition(self, transition):
        self.transitions.append(transition)

    def __str__(self):
        return "Episode: {}".format(self.transitions.__str__())

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        return self.transitions[idx]

    def __iter__(self):
        return iter(self.transitions)


# create the class for the get_episodes function
class GetEpisodes:
    def __init__(self, ind_data_, rewards, start_date, subject_id, param_dic):
        self.ind_data_ = ind_data_
        # self.rewards_list = rewards_list
        self.start_date = start_date
        self.subject_id = subject_id
        self.rewards = rewards
        self.observation_columns = None
        # self.action_columns = None
        self.actions_level_1_dict = param_dic["actions_level_1_dict"]
        self.actions_level_2_dict = param_dic["actions_level_2_dict"]
        self.actions_level_3_dict = param_dic["actions_level_3_dict"]

        self.action_columns_level_1 = param_dic["action_columns_level_1"]
        self.action_columns_level_2 = param_dic["action_columns_level_2"]
        self.action_columns_level_3 = param_dic["action_columns_level_3"]

        self.param_dic = param_dic
        self.encoder = Encoder()
        # treatment = ['treatment_medication', 'treatment_therapy']
        # treatment_updated = ['a:'+col for col in treatment]
        # one_hot_combinations = list(product([0, 1], repeat=len(treatment_updated)))
        # self.action_dict= dict([(key,list(value)) for key, value in zip(range(0,len(one_hot_combinations)), one_hot_combinations)])
        # self.action_columns =param_dic ['action_columns'] # ['a:treatment_medication',  'a:treatment_therapy']

    def __call__(self):

        return self.get_episodes(self.ind_data_)

    def get_episodes(self, ind_data_, cluster_col="", cluster_encoder_col=""):
        """
        Create episodes from the ind_data data. It is based on the Episode class and the Transition class defined in this module.

        """
        ind_data = ind_data_.copy().sort_values(by=[self.start_date], ignore_index=True)

        # The action has been already encoded in the preprocessing step, in here we just need to make sure the values are 0 and 1
        # if cluster if True then the action is the cluster column
        # obs_encoder_cluster_11	act_encoder_cluster_3	obs_cluster_8	act_cluster_47
        if cluster_encoder_col != "":
            # self.action_columns = [col for col in ind_data.columns if col.startswith('act_encoder_cluster')]
            self.observation_columns = [
                col for col in ind_data.columns if col.startswith(cluster_encoder_col)
            ]
        elif cluster_col != "":
            # self.action_columns = [col for col in ind_data.columns if col.startswith('act_cluster')]
            self.observation_columns = [
                col for col in ind_data.columns if col.startswith(cluster_col)
            ]

        else:
            # self.action_columns = [col for col in ind_data.columns if col.startswith('a:')]
            self.observation_columns = [
                col for col in ind_data.columns if col.startswith("o:")
            ]

        # create the episodes
        episodes = []
        last_state_idx = ind_data.index[-1]
        # if the next state is the first row of the same subject, then it is the initial state
        start_new_episode_flag = True
        for idx, row in ind_data.iterrows():

            if start_new_episode_flag:

                episode = Episode(subject_id=row[self.subject_id])
                start_new_episode_flag = False

            state = list(row[self.observation_columns].fillna(0).astype(float))

            action_level_1_ = list(row[self.action_columns_level_1])
            action_level_2_ = list(row[self.action_columns_level_2])
            # action_level_3_ = list(row[self.action_columns_level_3])

            action_level_1 = [
                key
                for key, value in self.actions_level_1_dict.items()
                if value == action_level_1_
            ][0]
            action_level_2 = [
                key
                for key, value in self.actions_level_2_dict.items()
                if value == action_level_2_
            ][0]
            # action_level_3 = [key for key, value in self.actions_level_3_dict.items() if value == action_level_3_][0]

            current_state_reward = row[self.rewards]

            done = False

            # if the next state is the last row of the same subject, then it is the terminal state
            if idx == last_state_idx:
                done = True
                next_state = None
                start_new_episode_flag = True

            # if the next states reward is greater than the previous reward then the current one is the terminal and the next state is the initial state
            else:
                next_state_reward = ind_data.loc[idx + 1, self.rewards]
                next_state = list(
                    ind_data.loc[idx + 1, self.observation_columns].fillna(0)
                )
                if next_state_reward > current_state_reward:
                    done = True
                    start_new_episode_flag = True

            # reward = reward_func(reward_dic, followup_time=tc.outcome_followup_time, is_terminal_state=done, action=row['SubsequentTreatment'])
            transition = Transition(
                state,
                action_level_1,
                action_level_2,
                0,
                current_state_reward,
                done,
                next_state,
            )

            episode.add_transition(transition)

            if start_new_episode_flag:
                episodes.append(episode)

        return episodes


# Create load data class to recive the hyperparameters and return the data for the train and test


class LoadEpisodes:
    def __init__(self, param_dic):
        self.param_dic = param_dic

        self.train_data_path = (
            param_dic["split_data_path"] + param_dic["file_name_train"]
        )
        self.test_data_path = param_dic["split_data_path"] + param_dic["file_name_test"]
        self.val_data_path = param_dic["split_data_path"] + param_dic["file_name_val"]
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.rewards_list = param_dic["rewards_list"]
        self.rewards = self.rewards_list[0]
        self.subject_id, self.start_date = (
            param_dic["colmeta"][0],
            param_dic["colmeta"][1],
        )
        self.action_raw = False

    def __call__(self):
        return (
            self.train_data,
            self.test_data,
            self.rewards_list,
            self.start_date,
            self.subject_id,
        )

    def get_data(self):

        return (
            self.train_data,
            self.test_data,
            self.rewards_list,
            self.start_date,
            self.subject_id,
        )

    def _get_episodes(self, data, cluster_col, cluster_encoder_col):
        episodes_raw = []
        episodes_cluster = []
        episodes_encoder = []
        for sub_id, ind_data in tqdm(data):
            ind_data = ind_data.sort_values(by=[self.start_date], ignore_index=True)
            episode = GetEpisodes(
                ind_data, self.rewards, self.start_date, self.subject_id, self.param_dic
            )
            ind_episodes_raw = episode.get_episodes(ind_data)
            ind_episodes_cluster = episode.get_episodes(
                ind_data, cluster_col=cluster_col
            )
            ind_episodes_episode = episode.get_episodes(
                ind_data, cluster_col="", cluster_encoder_col=cluster_encoder_col
            )

            episodes_raw.extend(ind_episodes_raw)
            episodes_cluster.extend(ind_episodes_cluster)
            episodes_encoder.extend(ind_episodes_episode)

        return episodes_raw, episodes_cluster, episodes_encoder

    def load_episodes(self, data, cluster_col, cluster_encoder_col):

        group_data = data.groupby(self.subject_id)
        return self._get_episodes(group_data, cluster_col, cluster_encoder_col)

    # Each episode contains state-action pairs
    def load_states_actions(self, episodes):
        """
        Extract states and actions from episodes for training.
        """
        states = []
        actions = []
        for episode in tqdm(episodes):
            for transition in episode:
                states.append(transition.state)
                actions.append(transition.action)
        return np.array(states), np.array(actions)

        # i = 0
        # for sub_id, ind_data in self.train_data:
        #     ind_data = ind_data.sort_values(by=[self.start_date], ignore_index=True)
        #     episodes = GetEpisodes(ind_data, self.rewards, self.start_date, self.subject_id)
        #     ind_episodes = episodes.get_episodes(ind_data)
        #     train_episodes.extend(ind_episodes)
        #     # print('--------------------------------')
        #     # print(len(train_episodes[i]))

        #     # print('--------------------------------')
        #     # print(train_episodes[i].transitions.__getitem__(-1))
        #     # i += 1
        # for sub_id, ind_data in self.test_data:
        #     ind_data = ind_data.sort_values(by=[self.start_date], ignore_index=True)
        #     episodes = GetEpisodes(ind_data, self.rewards, self.start_date, self.subject_id)
        #     ind_episodes = episodes.get_episodes(ind_data)
        #     test_episodes.extend(ind_episodes)

        # for sub_id, ind_data in self.val_data:
        #     ind_data = ind_data.sort_values(by=[self.start_date], ignore_index=True)
        #     episodes = GetEpisodes(ind_data, self.rewards, self.start_date, self.subject_id)
        #     ind_episodes = episodes.get_episodes(ind_data)
        #     val_episodes.extend(ind_episodes)


class PolicyResolver:
    def __init__(self, model, action_space, greedy=False):

        if isinstance(model, dict):
            self.model_type = dict
        elif isinstance(model, np.ndarray):
            self.model_type = np.ndarray
        elif isinstance(model, np.matrix):
            self.model_type = np.matrix

        elif isinstance(model, torch.nn.Module):
            self.model_type = torch.nn.Module
            model.eval()

        elif isinstance(model, ModelBasedInference):
            self.model_type = ModelBasedInference

        elif isinstance(model, ClusteringBasedInference):
            self.model_type = ClusteringBasedInference

        elif isinstance(model, str):
            self.model_type = str

        # elif isinstance(model, CustomAlgo):
        #     self.model_type = "d3rlpy.algos"

        elif "d3rlSAC" in model.__class__.__module__:
            self.model_type = "d3rlpy.algos"

        elif "d3rlCQL" in model.__class__.__module__:
            self.model_type = "d3rlpy.algos"

        # elif isinstance(model, d3rlpy.algos.dqn.DQN):
        #     self.model_type = d3rlpy.algos.dqn.DQN
        elif "d3rlpy.algos" in model.__class__.__module__:
            self.model_type = "d3rlpy.algos"

        else:
            raise ValueError(
                f"Unsupported model type: {type(model.__class__)} model: {model.__class__.__module__}"
            )
        self.model = model
        self.action_space = action_space
        self.greedy = greedy

        # elif self.policy_name == "random":
        #     return 1 / len(self.action_space)
        # elif self.policy_name == "behavior_physician_policy":
        #     return 0.8 if action in self.action_space else 0.2 / (len(self.action_space) - 1)
        # return epsilon

    def predict_prob(self, transition, action):
        state = transition.state

        # if self.model_type == int:
        #     print (transition.prediction_probs[self.model][action])
        #     # probs = [1.0 if int(self.model[a]) == action else 0 for a in self.action_space]
        #     return transition.prediction_probs[self.model][action]

        if self.model_type == dict:
            probs = [self.model[state][a] for a in self.action_space]
            if self.greedy:
                max_action = np.argmax(probs)
                probs = np.zeros(len(self.action_space))
                probs[max_action] = 1
            # else:
            #     # Normalize to ensure it's a valid probability distribution if not already
            #     probs = probs / np.sum(probs)
            return probs[action]

        elif self.model_type == str:

            probs = transition.prediction_probs[self.model]
            # Normalize to ensure it's a valid probability distribution if not already
            # probs = probs / np.sum(probs)

            if self.greedy:
                max_action = np.argmax(probs)
                probs = np.zeros(len(self.action_space))
                probs[max_action] = 1.0

            # else:
            #     # Normalize to ensure it's a valid probability distribution if not already
            #     probs = probs / np.sum(probs)
            return probs[action]

        elif self.model_type == np.ndarray:
            # if state in an a list or array, then just use the first element
            if isinstance(state, (list, np.ndarray)):
                state = int(state[0])
            probs = self.model[state, :]

            if self.greedy:
                max_action = np.argmax(probs)

                # probs = np.zeros(len(self.action_space))
                probs[max_action] = 1
            # else:
            #     # Normalize to ensure it's a valid probability distribution if not already
            #     probs = probs / np.sum(probs)
            return probs[action]

        # elif self.model_type == np.matrix:
        #     probs = self.model[state, :]
        #     if self.greedy:
        #         probs = np.zeros(len(self.action_space))
        #         probs[np.argmax(probs)] = 1
        #     return probs[0, action]

        elif self.model_type == torch.nn.Module:
            with torch.no_grad():
                sample = torch.tensor(state.astype(np.float32).reshape(1, -1))
                logits = self.model(sample)
                probs = torch.nn.functional.softmax(logits, dim=1).numpy()
                probs = probs.reshape(-1)

            if self.greedy:
                max_action = np.argmax(probs)
                probs = np.zeros(len(self.action_space))
                probs[max_action] = 1
            # else:
            #     # Normalize to ensure it's a valid probability distribution if not already
            #     probs = probs / np.sum(probs)

            return probs[action]

        elif self.model_type == ModelBasedInference:
            samples = []
            for i in range(len(self.action_space)):
                action_dummies = np.zeros(len(self.action_space))
                action_dummies[i] = 1
                state_action = np.concatenate([state, action_dummies])
                samples.append(state_action)
            samples = np.array(samples)
            rewards = self.model.predict(samples)
            if (
                not self.model.maximize_outcome
            ):  # if the model is trained to minimize the outcome, then reverse the rewards
                rewards = -rewards
            rewards = rewards.reshape(-1)
            probs = torch.nn.functional.softmax(torch.tensor(rewards), dim=0).numpy()

            if self.greedy:
                max_action = np.argmax(probs)
                probs = np.zeros(len(self.action_space))
                probs[max_action] = 1

            return probs[action]

        elif self.model_type == ClusteringBasedInference:
            # get the probs from the model
            probs = self.model.get_policy(state.reshape(1, -1))

            if self.greedy:
                probs = np.zeros(len(self.action_space))
                probs[np.argmax(probs)] = 1
            # else:
            #     # Normalize to ensure it's a valid probability distribution if not already
            #     probs = probs / np.sum(probs)

            return probs[action]

        # elif self.model_type == d3rlpy.algos.dqn.DQN:
        elif self.model_type == "d3rlpy.algos":
            state = np.array(state)
            # print("State shape:", state.shape)
            observation = np.tile(state.astype(np.float32), (len(self.action_space), 1))
            # print ("Observation shape:", observation.shape)
            logits = self.model.predict_value(observation, np.array(self.action_space))
            # action = self.model.predict(observation)
            # q_values = self.model._module

            # probs = np.array(logits)
            # normalize the logits to their summation to get the probabilities
            # probs  = logits / np.sum(logits)

            probs = F.softmax(torch.tensor(logits), dim=-1).numpy()

            if self.greedy:
                max_action = np.argmax(probs)
                probs = np.zeros(len(self.action_space))
                probs[max_action] = 1.0

            # else:
            #     # Normalize to ensure it's a valid probability distribution if not already
            #     probs = probs / np.sum(probs)

            return probs[action], logits
    def _predict_prob(self, transition, action):
        state = transition.state

        # if self.model_type == int:
        #     print (transition.prediction_probs[self.model][action])
        #     # probs = [1.0 if int(self.model[a]) == action else 0 for a in self.action_space]
        #     return transition.prediction_probs[self.model][action]

        if self.model_type == dict:
            probs = [self.model[state][a] for a in self.action_space]
            if self.greedy:
                max_action = np.argmax(probs)
                probs = np.zeros(len(self.action_space))
                probs[max_action] = 1
            # else:
            #     # Normalize to ensure it's a valid probability distribution if not already
            #     probs = probs / np.sum(probs)
            return probs[action]

        elif self.model_type == str:

            probs = transition.prediction_probs[self.model]
            # Normalize to ensure it's a valid probability distribution if not already
            # probs = probs / np.sum(probs)

            if self.greedy:
                max_action = np.argmax(probs)
                probs = np.zeros(len(self.action_space))
                probs[max_action] = 1.0

            # else:
            #     # Normalize to ensure it's a valid probability distribution if not already
            #     probs = probs / np.sum(probs)
            return probs[action]

        elif self.model_type == np.ndarray:
            # if state in an a list or array, then just use the first element
            if isinstance(state, (list, np.ndarray)):
                state = int(state[0])
            probs = self.model[state, :]

            if self.greedy:
                max_action = np.argmax(probs)

                # probs = np.zeros(len(self.action_space))
                probs[max_action] = 1
            # else:
            #     # Normalize to ensure it's a valid probability distribution if not already
            #     probs = probs / np.sum(probs)
            return probs[action]

        # elif self.model_type == np.matrix:
        #     probs = self.model[state, :]
        #     if self.greedy:
        #         probs = np.zeros(len(self.action_space))
        #         probs[np.argmax(probs)] = 1
        #     return probs[0, action]

        elif self.model_type == torch.nn.Module:
            with torch.no_grad():
                sample = torch.tensor(state.astype(np.float32).reshape(1, -1))
                logits = self.model(sample)
                probs = torch.nn.functional.softmax(logits, dim=1).numpy()
                probs = probs.reshape(-1)

            if self.greedy:
                max_action = np.argmax(probs)
                probs = np.zeros(len(self.action_space))
                probs[max_action] = 1
            # else:
            #     # Normalize to ensure it's a valid probability distribution if not already
            #     probs = probs / np.sum(probs)

            return probs[action]

        elif self.model_type == ModelBasedInference:
            samples = []
            for i in range(len(self.action_space)):
                action_dummies = np.zeros(len(self.action_space))
                action_dummies[i] = 1
                state_action = np.concatenate([state, action_dummies])
                samples.append(state_action)
            samples = np.array(samples)
            rewards = self.model.predict(samples)
            if (
                not self.model.maximize_outcome
            ):  # if the model is trained to minimize the outcome, then reverse the rewards
                rewards = -rewards
            rewards = rewards.reshape(-1)
            probs = torch.nn.functional.softmax(torch.tensor(rewards), dim=0).numpy()

            if self.greedy:
                max_action = np.argmax(probs)
                probs = np.zeros(len(self.action_space))
                probs[max_action] = 1

            return probs[action]

        elif self.model_type == ClusteringBasedInference:
            # get the probs from the model
            probs = self.model.get_policy(state.reshape(1, -1))

            if self.greedy:
                probs = np.zeros(len(self.action_space))
                probs[np.argmax(probs)] = 1
            # else:
            #     # Normalize to ensure it's a valid probability distribution if not already
            #     probs = probs / np.sum(probs)

            return probs[action]

        # elif self.model_type == d3rlpy.algos.dqn.DQN:
        elif self.model_type == "d3rlpy.algos":
            state = np.array(state)
            # print("State shape:", state.shape)
            observation = np.tile(state.astype(np.float32), (len(self.action_space), 1))
            # print ("Observation shape:", observation.shape)
            logits = self.model.predict_value(observation, np.array(self.action_space))
            # action = self.model.predict(observation)
            # q_values = self.model._module

            # probs = np.array(logits)
            # normalize the logits to their summation to get the probabilities
            # probs  = logits / np.sum(logits)

            probs = F.softmax(torch.tensor(logits), dim=-1).numpy()

            if self.greedy:
                max_action = np.argmax(probs)
                probs = np.zeros(len(self.action_space))
                probs[max_action] = 1.0

            # else:
            #     # Normalize to ensure it's a valid probability distribution if not already
            #     probs = probs / np.sum(probs)

            return probs[action]


class DatasetFromEpisodes:
    def __init__(self):
        pass
        # self.episodes = episodes
        # self.actions_dict = actions_dict

    # def __call__(self):
    #     return self.get_d3rlpy_dataset(episodes, self.actions_dict)

    def get_d3rlpy_dataset(
        self, episodes, action_dict_level_1, action_dict_level_2, action_dict_level_3
    ):
        """
        Create a d3rlpy dataset from the episodes
        """

        observations = []
        actions_level_1 = []
        actions_level_2 = []
        actions_level_3 = []
        rewards = []
        terminals = []
        for episode in episodes:
            for transition in episode:
                observations.append(transition.state)
                actions_level_1.append(transition.action_level_1)
                actions_level_2.append(transition.action_level_2)
                actions_level_3.append(transition.action_level_3)
                rewards.append(transition.reward)
                terminals.append(transition.done)

        observations = np.array(observations)
        # actions_level_1 = np.array(actions_level_1)
        # action_dict_level_2 = np.array(actions_level_2)
        # action_dict_level_3 = np.array(actions_level_3)

        rewards = np.array(rewards)
        terminals = np.array(terminals)
        # print (actions)
        # Convert hierarchical actions into a single tensor
        actions = np.array(
            [
                np.concatenate(
                    [
                        np.atleast_1d(
                            level_1_action
                        ),  # Ensures it's at least a 1D array
                        np.atleast_1d(level_2_action),
                        np.atleast_1d(level_3_action),
                    ]
                )
                for level_1_action, level_2_action, level_3_action in zip(
                    actions_level_1, actions_level_2, actions_level_3
                )
            ]
        )
        dataset = d3rlpy.dataset.MDPDataset(
            observations,
            actions,
            rewards,
            terminals,
            action_space=d3rlpy.ActionSpace.DISCRETE,
            action_size=sum(
                [
                    len(action_dict_level_1),
                    len(action_dict_level_2),
                    len(action_dict_level_3),
                ]
            ),
        )  # Combined action size)
        return dataset

    def get_d3rlpy_dataset_fixed(self, episodes, action_dict_level_2):
        """
        Create a d3rlpy dataset from the episodes
        """

        observations = []
        actions = []
        rewards = []
        terminals = []
        for episode in episodes:
            for transition in episode:
                observations.append(transition.state)
                actions.append(transition.action_level_2)
                rewards.append(transition.reward)
                terminals.append(transition.done)

        observations = np.array(observations)
        # actions_level_1 = np.array(actions_level_1)
        # action_dict_level_2 = np.array(actions_level_2)
        # action_dict_level_3 = np.array(actions_level_3)

        rewards = np.array(rewards)
        terminals = np.array(terminals)
        actions = np.array(actions)

        dataset = d3rlpy.dataset.MDPDataset(
            observations,
            actions,
            rewards,
            terminals,
            action_space=d3rlpy.ActionSpace.DISCRETE,
            action_size=len(action_dict_level_2),
        )
        return dataset


class WeightedImportanceSampling:
    def __init__(self):
        pass

    def multi_weighted_importance_sampling_with_bootstrap(
        self,
        data_dict,
        # data_dict = {
        #   "test_rl": (test_episodes, target_policy_rl),
        #   "test_rl_greedy": (test_episodes, target_policy_greedy),
        #   "val_rl": (val_episodes, target_policy_rl), ...
        # }
        behavior_policy,
        gamma=0.99,
        num_bootstrap_samples=1000,
        N=1000,
        confidence_level=95,
        reward_name=None,
    ):
        """
        Handles multiple (episodes, target_policy) pairs in one call.
        Caches behavior-policy probabilities once, then for each entry:
          1) Compute target-policy weights
          2) Perform bootstrap
          3) Return WIS estimate, CI, F1
        data_dict keys (str) are just labels (e.g. 'test_rl', 'train_greedy').
        Each value is (episodes, target_policy).
        All share the same `behavior_policy`, discount factor `gamma`.
        """

        results = {}

        # 1) Gather *all unique episodes* from data_dict into a single list to cache
        #    If the same episodes appear multiple times, you can unify them to avoid duplication.
        all_episodes = set()
        for episodes, _ in data_dict.values():
            for e in episodes:
                all_episodes.add(e)
        all_episodes = list(all_episodes)

        # 2) Precompute behavior-policy probabilities (one pass)
        #    We store for each episode: a list of behavior probs, actions, rewards
        #    so we don't have to recalc behavior probabilities for every target policy
        #    This is your "partial caching".
        behavior_cache = self._precompute_behavior_probs(
            all_episodes, behavior_policy, reward_name
        )

        # 3) Now handle each dataset–policy pair
        for name, (episodes, target_policy) in data_dict.items():
            # a) Build per-episode target weights & discounted rewards
            episode_weights, episode_rewards, episode_dr_returns, f1, df = (
                self._compute_target_weights_rewards(
                    episodes, gamma, target_policy, behavior_cache, reward_name
                )
            )

            # with pd.ExcelWriter(name+".xlsx", engine='openpyxl') as writer:
            #     for subject_id, tb in df.items():
            #         tb.to_excel(writer, sheet_name=f"{subject_id}", index=False)

            # b) Do bootstrap
            bootstrap_wis_estimates = []
            bootstrap_dr_estimates = []
            num_eps = len(episodes)
            effective_N = min(N, num_eps)

            for _ in range(num_bootstrap_samples):
                sampled_indices = np.random.choice(num_eps, effective_N, replace=True)
                wis_estimate = self.compute_wis_for_sample(
                    sampled_indices, episode_weights, episode_rewards
                )
                bootstrap_wis_estimates.append(wis_estimate)
                
          

            wis_mean = np.mean(bootstrap_wis_estimates)
            wis_sem = stats.sem(bootstrap_wis_estimates)
            overall_dr_estimate = np.mean(episode_dr_returns)
            # Optionally: standard error or confidence interval
            stderr = np.std(episode_dr_returns) / np.sqrt(len(episode_dr_returns))
            ci_95 = (overall_dr_estimate - 1.96 * stderr, overall_dr_estimate + 1.96 * stderr)
        

            conf_interval = stats.norm.interval(
                confidence_level / 100, loc=wis_mean, scale=wis_sem
            )
            lower_bound, upper_bound = conf_interval

            dr_mean= round(overall_dr_estimate,2)
            print (dr_mean)
            results[name] = {
                "wis": round(wis_mean, 2),
                "ci": (round(lower_bound, 2), round(upper_bound, 2)),
                "f1": round(f1, 2),
                "dr": dr_mean,
                "dr_ci": (round(ci_95[0], 2), round(ci_95[1], 2)),
            }

        return results, df

    def _precompute_behavior_probs(self, episodes, behavior_policy, reward_name):
        """
        Caches the behavior-policy probabilities for each step in each episode
        so we only compute them once.

        Returns a dict keyed by episode object with
          'actions' = [action_1, action_2, ...]
          'behavior_probs' = [p(a_1|s_1), p(a_2|s_2), ...]
          'rewards' = [r_1, r_2, ...]
        (You can store transitions or anything else needed.)
        """
        epsilon = 1e-6
        cache = {}

        for ep in episodes:
            actions = []
            behavior_probs = []
            rewards = []
            for t in ep.transitions:
                a = t.action_level_2
                actions.append(a)

                # Behavior probability for chosen action
                p_beh = behavior_policy.predict_prob(t, a) + epsilon
                behavior_probs.append(p_beh)

                # Possibly handle different reward_name logic
                if reward_name is None:
                    rewards.append(t.reward)
                else:
                    # e.g. if reward is a dict
                    r = t.reward.get(reward_name, 0.0)
                    rewards.append(r)

            cache[ep] = {
                "actions": actions,
                "behavior_probs": behavior_probs,
                "rewards": rewards,
            }
        return cache

    def _compute_target_weights_rewards(
        self, episodes, gamma, target_policy, behavior_cache, reward_name
    ):
        """
        For a given target_policy, we compute w_i and discounted total reward
        for each episode. We reuse the cached behavior probabilities to avoid
        recomputing them.
        """
        epsilon = 1e-6
        episode_weights = []
        episode_rewards = []
        episode_dr_returns = []


        y_true = []
        y_pred = []
        ep_dic = {
            "subject_id": [],
            "actions": [],
            "behavior_probs": [],
            "rewards": [],
            "w_i": [],
            "total_reward": [],
            "gamma": [],
            "best_action": [],
            "true_action": [],
            "rl_prob": [],
            "dr_return": [],
        }

        df = {}
        # with pd.ExcelWriter('behavior_cache.xlsx', engine='openpyxl', mode='w') as writer:
        for ep in episodes:
            ep_ind = pd.DataFrame(ep_dic)
            cache_info = behavior_cache[ep]
            # e.g. cache_info['actions'], cache_info['behavior_probs'], cache_info['rewards']
            actions = cache_info["actions"]
            beh_probs = cache_info["behavior_probs"]
            base_rewards = cache_info["rewards"]

            w_i = 1.0
            total_reward = 0.0
            p_targs = []
            best_actions = []
            true_actions = []
            dr_return = []
            G = 0
            for j, (a, p_beh, r) in enumerate(zip(actions, beh_probs, base_rewards)):
                # Retrieve target policy prob for action 'a'
                p_targ, q_val = target_policy.predict_prob(ep.transitions[j], a)
                p_targ = max(p_targ, epsilon)
                v_val = q_val[a]

                dr_reward = w_i * (r - q_val) + v_val
                G = dr_reward + gamma * G
                dr_return.insert(0, G)

                # p_targ, q_val = max(target_policy.predict_prob(ep.transitions[j], a), epsilon)
                # predict_value = target_policy.predict_value(ep.transitions[j], torch.tensor(action for action in range(8)))
 
                # Weighted ratio in log-space for stability
                log_ratio = np.log(p_targ) - np.log(p_beh)
                log_w_i = np.log(max(w_i, 1e-6)) + log_ratio
                w_i = np.exp(log_w_i)
  

                dr_reward = w_i * (r - q_val) + v_val
                G = dr_reward + gamma * G
                dr_return.insert(0, G)
                # t += 1
                # dr_return += (gamma**t) * w_i * (r + gamma * v_val - q_val)
                # dr_return /= t

                # Check for overflow
                if np.isinf(w_i) or np.isnan(w_i):
                    w_i = np.clip(w_i, epsilon, 1 / epsilon)

                total_reward += r * (gamma**j)

                # For F1 classification metrics
                y_true.append(a)
                # pick the argmax action from the target policy
                best_a = np.argmax(
                    [target_policy._predict_prob(ep.transitions[j], a_cand) for a_cand in range(8)]
                )
                y_pred.append(best_a)
                p_targs.append(p_targ)
                best_actions.append(best_a)
                true_actions.append(a)

            ep_ind["actions"] = actions
            ep_ind["behavior_probs"] = beh_probs
            ep_ind["rewards"] = base_rewards
            ep_ind["w_i"] = w_i
            ep_ind["total_reward"] = total_reward
            ep_ind["gamma"] = gamma
            ep_ind["best_action"] = best_actions
            ep_ind["true_action"] = true_actions
            ep_ind["rl_prob"] = p_targs
            ep_ind["subject_id"] = str(ep.subject_id)
           

            df[str(ep.subject_id)] = ep_ind

            #

            # df.to_excel(writer, sheet_name=str(ep.subject_id), index=False)

            episode_weights.append(w_i)
            episode_rewards.append(total_reward)
            normalized_dr_return = self.normalize_to_0_100(dr_return)
            episode_dr_returns.append(np.mean(normalized_dr_return))  # Use the first element as the return
         

        f1 = f1_score(
            np.array(y_true), np.array(y_pred), average="weighted", zero_division=0
        )

        return episode_weights, episode_rewards, episode_dr_returns, f1, df

    def normalize_to_0_100(self, x):
        x = np.array(x)
        min_val = x.min()
        max_val = x.max()
        if max_val - min_val == 0:
            return np.zeros_like(x)  # or fill with 50, depending on your needs
        return 100 * (x - min_val) / (max_val - min_val)
    def compute_wis_for_sample(self, sample_indices, episode_weights, episode_rewards):
        episode_weights = np.array(episode_weights)
        episode_rewards = np.array(episode_rewards)

        sampled_weights = episode_weights[sample_indices]
        sampled_rewards = episode_rewards[sample_indices]

        total_weight = np.sum(sampled_weights)
        wis = np.sum(sampled_weights * sampled_rewards) / max(total_weight, 1e-6)
        return wis


class WeightedImportanceSampling_:
    def __init__(self):
        pass
        # self.episodes = episodes
        # self.gamma = gamma
        # self.policy = policy
        # self.behavior_policy = behavior_policy
        # self.num_bootstrap_samples = num_bootstrap_samples
        # self.N = N
        # self.confidence_level = confidence_level
        # self.reward_name = reward_name

    # def __call__(self):
    #     return self.weighted_importance_sampling_with_bootstrap(self.episodes, self.gamma, self.policy, self.behavior_policy, self.num_bootstrap_samples, self.N, self.confidence_level, self.reward_name)

    def weighted_importance_sampling_with_bootstrap(
        self,
        episodes,
        gamma,
        policy,
        behavior_policy,
        num_bootstrap_samples,
        N,
        confidence_level=95,
        reward_name=None,
    ):
        """
        Computes weighted importance sampling estimate with bootstrap confidence intervals
        idea from https://github.com/IntelLabs/coach/blob/master/rl_coach/off_policy_evaluators/rl/weighted_importance_sampling.py

        Args:
        :param episodes: list of episodes
        :param gamma: discount factor
        :param policy: target policy
        :param behavior_policy: behavior policy
        :param num_bootstrap_samples: number of bootstrap samples to use
        :param N: number of episodes to sample for each bootstrap sample
        :param confidence_level: confidence level for confidence intervals
        :param reward_name: name of reward to use (if None, reward is assumed to be a number, and not a dictionary)

        Returns:
        :return: weighted importance sampling estimate, bootstrap confidence interval
        """
        # Precompute weights and rewards for all episodes
        episode_weights, episode_rewards, episode_dr_returns, f1 = self.precompute_weights_rewards(
            episodes, gamma, policy, behavior_policy, reward_name
        )

        bootstrap_wis_estimates = []
        N = min(N, len(episodes))
        for _ in range(num_bootstrap_samples):
            sampled_indices = np.random.choice(len(episodes), N, replace=True)
            wis_estimate = self.compute_wis_for_sample(
                sampled_indices, episode_weights, episode_rewards
            )
            bootstrap_wis_estimates.append(wis_estimate)

        print(bootstrap_wis_estimates)

        # lower_bound = np.percentile(bootstrap_wis_estimates, (100 - confidence_level) / 2)
        # upper_bound = np.percentile(bootstrap_wis_estimates, 100 - (100 - confidence_level) / 2)

        wis_mean = np.mean(bootstrap_wis_estimates)
        wis_sem = stats.sem(bootstrap_wis_estimates)

        conf_interval = stats.norm.interval(
            confidence_level / 100, loc=wis_mean, scale=wis_sem
        )
        lower_bound, upper_bound = conf_interval
        dr_mean = np.mean(episode_dr_returns)

        return (
            round(wis_mean, 2),
            (round(lower_bound, 2), round(upper_bound, 2)),
            round(f1, 2),
            round(dr_mean, 2),
            # (round(np.percentile(episode_dr_returns, 25), 2), round(np.percentile(episode_dr_returns, 75), 2)),
        )

    def precompute_weights_rewards(
        self, episodes, gamma, policy, behavior_policy, reward_name
    ):
        """
        Precomputes weights and rewards for all episodes

        Args:
        :param episodes: list of episodes
        :param gamma: discount factor
        :param policy: target policy
        :param behavior_policy: behavior policy
        :param reward_name: name of reward to use

        Returns:
        :return: episode weights and episode rewards
        """
        epsilon = 1e-6
        episode_weights = []
        episode_rewards = []
        # define a dictionary to store the number of times each action is selected
        # set the default value to 0

        action_selected = defaultdict(int)
        y_true = []
        y_pred = []
        y_pred_prob = []
        y_true_prob = []
        for episode in episodes:
            w_i = 1
            total_reward = 0

            for j, transition in enumerate(episode.transitions):

                a = transition.action_level_2
                action_selected[a] = action_selected[a] + 1

                prob_behavior = behavior_policy.predict_prob(transition, a) + epsilon

                prob_policy = max(policy.predict_prob(transition, a), epsilon)

                # ratio = prob_policy / prob_behavior
                # w_ = w_i
                # w_i = w_i * ratio

                # Compute Importance Sampling Ratio
                log_ratio = np.log(max(prob_policy, 1e-6)) - np.log(
                    max(prob_behavior, 1e-6)
                )

                # Use Log-Space for Stability
                log_w_i = np.log(max(w_i, 1e-6)) + log_ratio
                w_i = np.exp(log_w_i)  # Convert back to normal scale

                # Check if w_i is inf and replace it with 1/epsilon
                if np.isinf(w_i) or np.isnan(w_i):
                    w_i = np.clip(w_i, epsilon, 1 / epsilon)

                total_reward += transition.reward * (gamma**j)
                # Store values for classification comparison
                y_true.append(a)  # Ground truth action as label
                y_pred.append(
                    np.argmax([policy.predict_prob(transition, a) for a in range(8)])
                )  # RL predicted action
                # y_pred_prob.append(policy.predict_action_probs(transition, a))  # RL predicted probabilities
                # y_true_prob.append(behavior_policy.predict_action_probs(transition))  # Ground truth action probabilities

            episode_weights.append(w_i)
            episode_rewards.append(total_reward)
            # f1 = f1_score( np.array(y_true), np.array(y_pred), average="weighted", zero_division=0)

            # print ("Total reward:", total_reward)
            # print ("Episode weight:", w_i)

        f1 = f1_score(
            np.array(y_true), np.array(y_pred), average="weighted", zero_division=0
        )

        # print (y_pred_prob)
        # print (y_true_prob)
        return episode_weights, episode_rewards, f1

    def compute_wis_for_sample(self, sample_indices, episode_weights, episode_rewards):
        """
        Computes weighted importance sampling estimate for a sample of episodes

        Args:
        :param sample_indices: indices of episodes to sample
        :param episode_weights: list of weights for each episode
        :param episode_rewards: list of rewards for each episode

        Returns:
        :return: weighted importance sampling estimate
        """
        # print ("Sample indices:", sample_indices)
        # print ("Episode weights:", episode_weights)
        episode_weights = np.array(episode_weights)
        sampled_weights = [episode_weights[i] for i in sample_indices]
        sampled_rewards = [episode_rewards[i] for i in sample_indices]

        total_weight = np.sum(sampled_weights)
        # if total_weight == 0 or total_weight == np.nan:
        #     return 0

        wis = np.sum(np.array(sampled_weights) * np.array(sampled_rewards)) / max(
            total_weight, 1e-6
        )
        # ml_values = np.array(sampled_weights) * np.array(sampled_rewards)
        # sum without np.sum() to avoid nan values
        # sum_values = fsum(w * r for w, r in zip(sampled_weights, sampled_rewards))
        # wis = sum_values/ total_weight
        return wis


class QLearning:
    def __init__(
        self,
        transition_matrix,
        reward_matrix,
        discount=0.99,
        n_iter=100000,
        learning_rate=0.1,
        epsilon=0.1,
    ):
        """
        Initialize the QLearning algorithm.

        Args:
            transition_matrix (numpy.ndarray): 3D array [n_actions, n_states, n_states]
                Transition probabilities for each action.
            reward_matrix (numpy.ndarray): 3D array [n_actions, n_states, n_states]
                Rewards for each action and state transition.
            discount (float): Discount factor for future rewards (gamma).
            n_iter (int): Number of iterations for Q-learning.
            learning_rate (float): Learning rate (alpha).
            epsilon (float): Probability of exploration (epsilon-greedy).
        """
        self.transition_matrix = transition_matrix
        self.reward_matrix = reward_matrix
        self.n_states = transition_matrix.shape[1]
        self.n_actions = transition_matrix.shape[0]
        self.discount = discount
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q_matrix = np.zeros((self.n_states, self.n_actions))

    def train(self):
        """
        Train the Q-learning model to compute the optimal Q-matrix.
        """
        for _ in range(self.n_iter):
            # Randomly select a state to start from
            current_state = np.random.randint(0, self.n_states)

            # Select an action (epsilon-greedy policy)
            if np.random.rand() < self.epsilon:
                # Exploration: choose a random action
                action = np.random.randint(0, self.n_actions)
            else:
                # Exploitation: choose the best known action
                action = np.argmax(self.q_matrix[current_state])

            # Simulate the environment: sample the next state based on the transition probabilities
            next_state = np.random.choice(
                range(self.n_states), p=self.transition_matrix[action, current_state]
            )

            # Get the immediate reward for this transition
            reward = self.reward_matrix[action, current_state, next_state]

            # Update the Q-value using the Bellman equation
            best_next_action = np.max(self.q_matrix[next_state])  # max_a' Q(s', a')
            # Q[(s,a)] += alpha * (r + gamma * Q[(s_,a_)]-Q[(s,a)])
            self.q_matrix[current_state, action] += self.learning_rate * (
                reward
                + self.discount * best_next_action
                - self.q_matrix[current_state, action]
            )

    def get_optimal_policy(self):
        """
        Derive the optimal policy from the Q-matrix.

        Returns:
            numpy.ndarray: Array of optimal actions for each state.
        """
        return np.argmax(self.q_matrix, axis=1)

    def get_q_matrix(self):
        """
        Get the Q-matrix.

        Returns:
            numpy.ndarray: Q-matrix.
        """
        return self.q_matrix


# if you customize a Q-function of actor-critic algorithm (e.g. SAC), you need to prepare an action-conditioned model.
class CustomEncoderWithAction(nn.Module):
    def __init__(self, observation_shape, action_size, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.fc1 = nn.Linear(observation_shape[0] + action_size, feature_size)
        self.fc2 = nn.Linear(feature_size, feature_size)

    def forward(self, x, action):
        h = torch.cat([x, action], dim=1)
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        return h


# Customize Neural Network
class CustomEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.fc1 = nn.Linear(observation_shape[0], feature_size)
        self.fc2 = nn.Linear(feature_size, feature_size)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return h
