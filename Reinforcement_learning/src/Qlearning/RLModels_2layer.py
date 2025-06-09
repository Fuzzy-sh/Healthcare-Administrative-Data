import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dataclasses
from typing import Sequence, cast
from collections import deque
import copy
import d3rlpy
from d3rlpy.torch_utility import hard_sync, soft_sync
import pandas as pd
import datetime
from collections import defaultdict

# from config.config_param import config_param
import random
from tqdm import tqdm

# Display DataFrame
import math
from abc import abstractmethod
from collections import defaultdict

from d3rlSAC import HierarchicalDiscreteSACConfig, EncoderFactory
from d3rlCQL import DiscreteCQLConfig_H

from typing import (
    Callable,
    Generator,
    Generic,
    Optional,
    Sequence,
    TypeVar,
)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm, trange
from typing_extensions import Self

from d3rlpy.algos.qlearning.base import (
    ImplBase,
    LearnableBase,
    LearnableConfig,
    save_config,
)
from d3rlpy.constants import (
    IMPL_NOT_INITIALIZED_ERROR,
    ActionSpace,
    LoggingStrategy,
)
from d3rlpy.dataset import (
    ReplayBufferBase,
    TransitionMiniBatch,
    check_non_1d_array,
    create_fifo_replay_buffer,
    is_tuple_shape,
)
from d3rlpy.logging import (
    LOG,
    D3RLPyLogger,
    FileAdapterFactory,
    LoggerAdapterFactory,
)
from d3rlpy.metrics import EvaluatorProtocol, evaluate_qlearning_with_environment

# from d3rlpy.models.torch import Policy
# from d3rlpy.torch_utility import (
#     TorchMiniBatch,
#     convert_to_torch,
#     convert_to_torch_recursively,
#     eval_api,
#     hard_sync,
#     sync_optimizer_state,
#     train_api,
# )
# from d3rlpy.types import GymEnv, NDArray, Observation, TorchObservation
from d3rlpy.algos.utility import (
    assert_action_space_with_dataset,
    assert_action_space_with_env,
    build_scalers_with_env,
    build_scalers_with_transition_picker,
)


class QNetwork(nn.Module):
    """Q-Network: Estimates action-values for discrete actions"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # Outputs Q-values for each action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # No activation, raw Q-values


class PolicyNetwork(nn.Module):
    """Policy Network: Outputs a categorical distribution over actions"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # Logits for action probabilities

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)  # Convert to probability distribution

    def sample_action(self, state):
        """Sample action from the policy"""
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class QNetworkEncoderFactory(EncoderFactory):
    """Wraps QNetwork to be used as an Encoder Factory."""

    def __init__(self, hidden_dim=256):
        self.hidden_dim = hidden_dim

    def create(self, observation_shape: int, action_dim: int) -> nn.Module:
        return QNetwork(observation_shape, action_dim, hidden_dim=self.hidden_dim)


class DQNModel:

    def __init__(self, algo, model_params, device) -> None:
        assert algo in [
            "BQL",
            "SAC",
            "DQN",
            "CQL",
            "DDQN",
            "UA-CQL",
            "UHA-CQL",
            "UHA-SAC",
            "UA-SAC",
        ], "Invalid algorithm name"
        self.set_seed(42)
        self.model_params = model_params
        self.device = device
        self.num_layers = model_params["num_layers"]
        self.num_hidden_neurons = model_params["num_hidden_neurons"]
        self.activation_func = model_params["activation_func"]
        self.n_steps = model_params["n_steps"]
        self.conservative_alpha = model_params["conservative_alpha"]
        self.batch_size = model_params["batch_size"]
        self.target_update_interval = model_params["target_update_interval"]
        self.gamma = model_params["gamma"]
        self.n_epochs = model_params["n_epochs"]
        self.n_steps_per_epoch = model_params["n_steps_per_epoch"]
        self.critic_learning_rate = model_params["critic_learning_rate"]
        self.actor_learning_rate = model_params["actor_learning_rate"]

        # UHA_SAC leavel 1 parameters
        self.use_attention_1 = model_params["use_attention_1"]
        self.num_attention_heads_1 = model_params["num_attention_heads_1"]
        self.use_uncertainty_actor_loss_1 = model_params["use_uncertainty_actor_loss_1"]
        self.use_uncertainty_critic_loss_1 = model_params[
            "use_uncertainty_critic_loss_1"
        ]
        self.uncertainty_critic_weight_1 = model_params["uncertainty_critic_weight_1"]
        self.uncertainty_actor_weight_1 = model_params["uncertainty_actor_weight_1"]

        # UHA_SAC leavel 2 parameters
        self.use_attention_2 = model_params["use_attention_2"]
        self.num_attention_heads_2 = model_params["num_attention_heads_2"]
        self.use_uncertainty_actor_loss_2 = model_params["use_uncertainty_actor_loss_2"]
        self.use_uncertainty_critic_loss_2 = model_params[
            "use_uncertainty_critic_loss_2"
        ]
        self.uncertainty_critic_weight_2 = model_params["uncertainty_critic_weight_2"]
        self.uncertainty_actor_weight_2 = model_params["uncertainty_actor_weight_2"]

        # encoder factory
        self.encoder_factory = d3rlpy.models.VectorEncoderFactory(
            hidden_units=[self.num_hidden_neurons] * self.num_layers,
            activation=self.activation_func,
        )

        self.encoder_actor_factory = d3rlpy.models.VectorEncoderFactory(
            hidden_units=[self.num_hidden_neurons] * self.num_layers,
            activation=self.activation_func,
        )
        self.rl_model = self._get_model(algo)

        # 1) Build a dict for the UHA-SAC_1 parameters

        self.model_params_UHA_SAC_1 = {
            # fixed parameters for the first level
            "sac1_config": None,
            "use_sac1_filtering": False,
            "sac1_action_size": 2,
            "sac2_action_size": 8,
            "critic_encoder_factory": self.encoder_factory,
            "actor_encoder_factory": self.encoder_actor_factory,
            "use_uncertainty_actor_loss": self.use_uncertainty_actor_loss_1,
            "use_uncertainty_critic_loss": self.use_uncertainty_critic_loss_1,
            "uncertainty_critic_weight": self.uncertainty_critic_weight_1,
            "uncertainty_actor_weight": self.uncertainty_actor_weight_1,
            "use_attention": self.use_attention_1,
            "num_attention_heads": self.num_attention_heads_1,
        }

    def _get_model(self, algo):
        if algo == "BQL":
            rl_model = d3rlpy.algos.DiscreteBCQConfig(
                encoder_factory=self.encoder_factory,
                learning_rate=self.critic_learning_rate,
                batch_size=self.batch_size,
            ).create(device=self.device)   

        elif algo == "SAC":
            rl_model = d3rlpy.algos.DiscreteSACConfig(
                critic_encoder_factory=self.encoder_factory,
                actor_encoder_factory=self.encoder_actor_factory,
                critic_learning_rate=self.critic_learning_rate,
                actor_learning_rate=self.actor_learning_rate,
                batch_size=self.batch_size,
            ).create(device=self.device)

        elif algo == "DQN":
            rl_model = d3rlpy.algos.DQNConfig(
                encoder_factory=self.encoder_factory,
                learning_rate=self.critic_learning_rate,
                batch_size=self.batch_size,
            ).create(device=self.device)

        elif algo == "CQL":
            rl_model = d3rlpy.algos.DiscreteCQLConfig(
                encoder_factory=self.encoder_factory,
                learning_rate=self.critic_learning_rate,
                batch_size=self.batch_size,
            ).create(device=self.device)

        elif algo == "DDQN":
            # q_func = d3rlpy.models.QRQFunctionFactory(n_quantiles=32)
            rl_model = d3rlpy.algos.DoubleDQNConfig(
                encoder_factory=self.encoder_factory,
                learning_rate=self.critic_learning_rate,
                batch_size=self.batch_size,
            ).create(device=self.device)


        elif algo == "UA-CQL":
            rl_model = DiscreteCQLConfig_H(
                parent_config=None,
                encoder_factory=self.encoder_factory,
                learning_rate=self.critic_learning_rate,
                batch_size=self.batch_size,
                use_attention_cql=self.use_attention_2,
                num_attention_heads_cql=self.num_attention_heads_2,
                use_uncertainty_loss_cql=self.use_uncertainty_critic_loss_2,
                uncertainty_weight_cql=self.uncertainty_critic_weight_2,
            ).create(device=self.device)

        elif algo == "UHA-CQL":

            parent_config_obj = DiscreteCQLConfig_H(
                use_attention_parent=self.use_attention_1,
                num_attention_heads_parent=self.num_attention_heads_1,
                batch_size=self.batch_size,
                learning_rate=self.critic_learning_rate,
                encoder_factory=self.encoder_factory,
                parent_config=None,
                use_uncertainty_loss_parent=self.use_uncertainty_critic_loss_1,
                uncertainty_weight_parent=self.uncertainty_critic_weight_1,
            )
            rl_model = DiscreteCQLConfig_H(
                parent_config=parent_config_obj,
                use_attention_cql=self.use_attention_2,
                num_attention_heads_cql=self.num_attention_heads_2,
                encoder_factory=self.encoder_factory,
                learning_rate=self.critic_learning_rate,
                batch_size=self.batch_size,
                use_uncertainty_loss_cql=self.use_uncertainty_critic_loss_2,
                uncertainty_weight_cql=self.uncertainty_critic_weight_2,
            ).create(device=self.device)



        elif algo == "UA-SAC":
            # Only for the first level
            # 2) Create the actual sac1_config object

            rl_model = HierarchicalDiscreteSACConfig(
                # The level 1 configuration is not being used
                parent_sac_config=None,
                use_parent_sac_filtering=False,
                parent_sac_action_size=2,
                action_size=8,
                critic_encoder_factory=self.encoder_factory,  # Use wrapped factory
                actor_encoder_factory=self.encoder_actor_factory,  # Use wrapped factory
                use_attention_sac=self.use_attention_2,
                num_attention_heads_sac=self.num_attention_heads_2,
                use_uncertainty_actor_loss=self.use_uncertainty_actor_loss_2,
                use_uncertainty_critic_loss=self.use_uncertainty_critic_loss_2,
                uncertainty_critic_weight=self.uncertainty_critic_weight_2,
                uncertainty_actor_weight=self.uncertainty_actor_weight_2,
                initial_temperature=1.0,
            ).create(device=self.device)

        elif algo == "UHA-SAC":
            # 1) Create the parent sac1_config object
            parent_sac_config_obj = HierarchicalDiscreteSACConfig(
                parent_sac_config=None,
                use_parent_sac_filtering=False,
                parent_sac_action_size=2,
                action_size=8,
                critic_encoder_factory=self.encoder_factory,  # Use wrapped factory
                actor_encoder_factory=self.encoder_actor_factory,  # Use wrapped factory
                use_attention_parent=self.use_attention_1,
                num_attention_heads_parent=self.num_attention_heads_1,
                use_uncertainty_actor_loss=self.use_uncertainty_actor_loss_1,
                use_uncertainty_critic_loss=self.use_uncertainty_critic_loss_1,
                uncertainty_critic_weight=self.uncertainty_critic_weight_1,
                uncertainty_actor_weight=self.uncertainty_actor_weight_1,
            )

            rl_model = HierarchicalDiscreteSACConfig(
                # for the hierarchical model, we use the parent_sac_config_obj
                parent_sac_config=parent_sac_config_obj,
                use_parent_sac_filtering=True,
                parent_sac_action_size=2,  # Action size for the first level
                action_size=8,
                critic_encoder_factory=self.encoder_factory,  # Use wrapped factory
                actor_encoder_factory=self.encoder_actor_factory,  # Use wrapped factory
                use_attention_sac=self.use_attention_2,
                num_attention_heads_sac=self.num_attention_heads_2,
                use_uncertainty_actor_loss=self.use_uncertainty_actor_loss_2,
                use_uncertainty_critic_loss=self.use_uncertainty_critic_loss_2,
                uncertainty_critic_weight=self.uncertainty_critic_weight_2,
                uncertainty_actor_weight=self.uncertainty_actor_weight_2,
            ).create(device=self.device)

            # rl_model = DiscreteSACConfig_H(use_sac1_filtering=True, sac1_config= sac_1_config).create(device=self.device)

        # elif algo == 'DCQL':
        #     q_func = d3rlpy.models.QRQFunctionFactory(n_quantiles=32)
        #     rl_model = d3rlpy.algos.DiscreteCQLConfig(alpha=self.conservative_alpha, q_func_factory=q_func, encoder_factory=self.encoder_factory, learning_rate=self.learning_rate).create(device=self.device)

        # elif  algo == 'BayDQN':

        #     rl_model = CustomAlgoConfig( batch_size= self.batch_size, learning_rate= self.learning_rate, target_update_interval= self.target_update_interval, gamma= self.gamma, n_steps = self.n_steps ).create()
        elif algo == "BayCQL":
            rl_model = CustomAlgoConfig(
                batch_size=self.batch_size,
                critic_learning_rate=self.critic_learning_rate,
                actor_learning_rate=self.actor_learning_rate,
                target_update_interval=self.target_update_interval,
                gamma=self.gamma,
                n_steps=self.n_steps,
                epochs=self.n_epochs,
            ).create()

        return rl_model

    def fit_model(self, train_dataset, experiment_name):
        # build the model

        self.rl_model.build_with_dataset(train_dataset)

        # td_error_evaluator = d3rlpy.metrics.TDErrorEvaluator(episodes=train_dataset.episodes)

        # buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=1000000, env=train_dataset)
        # train the RL model

        self.rl_model.fit(
            train_dataset,
            n_steps=self.n_steps,
            experiment_name=experiment_name,
            n_steps_per_epoch=self.n_steps_per_epoch,
            # save_interval=self.n_steps,
            # eval_episodes=eval_episodes,
            # eval_interval=eval_interval,
            # n_steps_per_epoch=n_steps_per_epoch,
            # n_steps_per_cycle=n_steps_per_cycle,
            # update_interval=update_interval,
            # batch_size=batch_size,
            # n_epochs=n_epochs,
            # ent_coef=ent_coef,
            # lam=lam,
            # n_critics=n_critics,
            # explorer=explorer,
        )

        return self.rl_model

    def set_seed(self, seed=42):
        """Set all relevant seeds for reproducibility"""
        random.seed(seed)  # Python's built-in random module
        np.random.seed(seed)  # NumPy random seed
        torch.manual_seed(seed)  # PyTorch random seed

        # Ensures deterministic behavior on GPUs (if available)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Set seed for d3rlpy (if applicable)
        d3rlpy.seed(seed)

    # Call the function at the beginning of your script


class Actor(nn.Module):
    """
    Actor network for policy learning in Actor-Critic.
    Outputs action probabilities for discrete actions.
    """

    def __init__(self, observation_shape: Sequence[int], action_size: int):
        super().__init__()
        self.activation = nn.ReLU()
        self.n_nurons = 64
        self.fc1 = nn.Linear(observation_shape[0], self.n_nurons)
        self.fc2 = nn.Linear(self.n_nurons, self.n_nurons)
        self.action_probs = nn.Linear(
            self.n_nurons, action_size
        )  # Softmax layer for discrete actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return torch.softmax(
            self.action_probs(x), dim=1
        )  # Probability distribution over actions


#######################################################################
# Self_Attention
#######################################################################


class Self_Attention(nn.Module):
    def __init__(self, in_dim, num_heads=4):
        super(Self_Attention, self).__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.query = nn.Linear(in_dim, in_dim // num_heads)
        self.key = nn.Linear(in_dim, in_dim // num_heads)
        self.value = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len)
        Returns:
            out: Self-attention output of shape (batch_size, seq_len)
            attention: Attention weights of shape (batch_size, seq_len, seq_len)
        """
        B, S = x.shape  # B is batch_size (may vary), S is feature dim (in_dim)

        # Compute query, key, value projections
        query = self.query(x).unsqueeze(1)  # (B, 1, in_dim/num_heads)
        key = self.key(x).unsqueeze(2)  # (B, in_dim/num_heads, 1)
        value = self.value(x).unsqueeze(1)  # (B, 1, in_dim)

        # Compute attention scores
        energy = torch.bmm(query, key)  # (B, 1, 1) since key is transposed
        attention = self.softmax(energy)  # (B, 1, 1)

        # Compute weighted values
        out = torch.bmm(attention, value).squeeze(1)  # (B, in_dim)

        # Residual connection
        out = self.gamma * out + x

        return out, attention.squeeze(1)  # Squeeze last dim for correct shape


class QFunction(nn.Module):  # type: ignore
    def __init__(self, observation_shape: Sequence[int], action_size: int):
        super().__init__()
        self.n_nurons = 64
        self.activation = nn.ReLU()

        self._fc1 = nn.Linear(observation_shape[0], self.n_nurons)
        self._fc2 = nn.Linear(self.n_nurons, self.n_nurons)
        # self._fc3 = nn.Linear(self.n_nurons, action_size)

        # Attention mechanism
        # self.attn_weights = nn.Linear(self.n_nurons, self.n_nurons)  # Generates attention scores
        self.attn1 = Self_Attention(observation_shape[0])  # Attention class
        self.attn2 = Self_Attention(self.n_nurons)  # Attention class

        # Bayesian Q-value estimation
        self.q_mean_layer = nn.Linear(self.n_nurons, action_size)  # Mean Q-values
        self.q_logvar_layer = nn.Linear(
            self.n_nurons, action_size
        )  # Log-variance (Uncertainty)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        attn_out_1, attn_scores_1 = self.attn1(x)
        h = self.activation(self._fc1(x))

        attn_out_2, attn_scores_2 = self.attn2(h)
        h = self.activation(self._fc2(h))

        # Ensure consistent shapes before merging attention outputs
        if attn_out_1.shape == attn_out_2.shape:
            attn_out = attn_out_1 + attn_out_2  # Element-wise addition (if same shape)
        else:
            attn_out = torch.cat(
                (attn_out_1, attn_out_2), dim=-1
            )  # Concatenate along feature dimension

        h = h + attn_out  # Residual connection
        # Ensure attention scores are consistent
        if attn_scores_1.shape == attn_scores_2.shape:
            attn_scores = attn_scores_1 + attn_scores_2
        else:
            attn_scores = torch.cat((attn_scores_1, attn_scores_2), dim=-1)

        # Compute Bayesian Q-values (Mean and Standard Deviation)
        q_mean = self.q_mean_layer(h)  # Mean of Q-values

        q_logvar = torch.clamp(
            self.q_logvar_layer(h), min=-1, max=1
        )  # Prevent extreme variance
        # q_exp = torch.exp(0.5 * q_logvar)  # Gradual uncertainty decay
        q_std = torch.std(
            q_logvar, dim=1, keepdim=True
        )  # Standard deviation of Q-values
        # penalty_q = torch.exp(-q_std)  # Penalize high uncertainty
        return q_mean, q_std, attn_out  # Return Q-values and attention scores

        # compute attention scores
        # attn_output, attn_scores = self.attn_class(h)  # Apply attention mechanism

        # attn_scores = F.softmax(self.attn_weights(h), dim=1)  # Importance of each feature
        # h = h * attn_scores  # Apply attention to the input features
        # h = h + h * attn_out_2 + attn_out_1  # Residual attention
        # q_values = self._fc3(h)  # Compute Q-values


class DiscreteSAC:
    """Discrete Soft Actor-Critic (SAC) implementation"""

    def __init__(
        self, state_dim, action_dim, gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4
    ):
        self.gamma = gamma  # Discount factor
        self.tau = tau  # Target update rate
        self.alpha = alpha  # Temperature parameter for entropy regularization

        # Networks
        self.q_net = QNetwork(state_dim, action_dim)
        self.q_target = QNetwork(state_dim, action_dim)
        self.policy_net = PolicyNetwork(state_dim, action_dim)

        # Copy target Q-network parameters
        self.q_target.load_state_dict(self.q_net.state_dict())

        # Optimizers
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.alpha_optimizer = optim.Adam(
            [torch.tensor(self.alpha, requires_grad=True)], lr=lr
        )

    def compute_target_value(self, next_state):
        """Computes target state-value function V(s)"""
        with torch.no_grad():
            action_probs = self.policy_net(next_state)
            q_values = self.q_target(next_state)

            entropy_term = -self.alpha * torch.sum(
                action_probs * torch.log(action_probs + 1e-10), dim=-1
            )
            target_v = torch.sum(
                action_probs
                * (q_values - self.alpha * torch.log(action_probs + 1e-10)),
                dim=-1,
            )
            return target_v + entropy_term

    def update_q_network(self, state, action, reward, next_state, done):
        """Q-network update step"""
        target_v = self.compute_target_value(next_state)
        target_q = reward + self.gamma * (1 - done) * target_v

        q_values = self.q_net(state)
        q_value = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)

        loss_q = F.mse_loss(q_value, target_q)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()
        return loss_q.item()

    def update_policy(self, state):
        """Policy update step"""
        action_probs = self.policy_net(state)
        q_values = self.q_net(state)

        policy_loss = torch.sum(
            action_probs * (self.alpha * torch.log(action_probs + 1e-10) - q_values),
            dim=-1,
        ).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        return policy_loss.item()

    def update_alpha(self, state):
        """Temperature (alpha) update step"""
        action_probs = self.policy_net(state)
        entropy = -torch.sum(
            action_probs * torch.log(action_probs + 1e-10), dim=-1
        ).mean()

        alpha_loss = -(self.alpha * (entropy + 1.0)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        return alpha_loss.item()

    def soft_update_target(self):
        """Soft update of target Q-network"""
        for param, target_param in zip(
            self.q_net.parameters(), self.q_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def train_step(self, batch):
        """Perform one training step"""
        state, action, reward, next_state, done = batch

        # Convert data to tensors
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        loss_q = self.update_q_network(state, action, reward, next_state, done)
        loss_policy = self.update_policy(state)
        loss_alpha = self.update_alpha(state)

        self.soft_update_target()

        return {"Q Loss": loss_q, "Policy Loss": loss_policy, "Alpha Loss": loss_alpha}

    def select_action(self, state):
        """Select an action using the learned policy"""
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action, _ = self.policy_net.sample_action(state)
            return action.item()


# Defining algorithm components
# A custom module container is defined using datac
# lasses: CustomAlgoModules.
# frozen=True ensures immutability (safe for multi-threaded execution).


@dataclasses.dataclass(frozen=True)
class CustomAlgoModules(d3rlpy.Modules):
    q_func_1: QFunction
    targ_q_func_1: QFunction
    q_func_2: QFunction
    targ_q_func_2: QFunction

    actor_1: Actor
    actor_2: Actor

    critic_optim_1: torch.optim.Optimizer
    critic_optim_2: torch.optim.Optimizer

    actor_optim_1: torch.optim.Optimizer
    actor_optim_2: torch.optim.Optimizer


# Inherits from QLearningAlgoImplBase, the base class for Q-learning algorithms in d3rlpy.
class CustomAlgoImpl(d3rlpy.algos.QLearningAlgoImplBase):
    _modules: CustomAlgoModules  # stors neural networks and optimizer

    def __init__(
        self,
        observation_shape: d3rlpy.types.Shape,
        action_size: int,
        modules: CustomAlgoModules,
        target_update_interval: int,
        gamma: float,
        device: str,
        n_steps: int,
        epochs: int,
        attention_log_dict_1={},
        attention_log_dict_2={},
        uncertainty_log_dict_1={},
        uncertainty_log_dict_2={},
        selected_rewards_log_dict_2={},
        sampled_q_log_dict_1=defaultdict(lambda: defaultdict(list)),
        sampled_q_log_dict_2=defaultdict(lambda: defaultdict(list)),
        sampled_q_values_log_dict_2=defaultdict(lambda: defaultdict(list)),
        sampled_action_log_dict_2=defaultdict(lambda: defaultdict(list)),
        sampled_q_penaltiy_log_dict_2=defaultdict(lambda: defaultdict(list)),
        sampled_q_normalized_log_dict_2=defaultdict(lambda: defaultdict(list)),
        sampled_q_target_log_dict_2=defaultdict(lambda: defaultdict(list)),
        loss_log_dict_1=defaultdict(lambda: defaultdict(list)),
        loss_log_dict_2=defaultdict(lambda: defaultdict(list)),
        level_2_steps=0,
    ):

        super().__init__(observation_shape, action_size, modules, device)
        self._target_update_interval = target_update_interval
        self._gamma = gamma
        self.n_steps = n_steps  # Number of steps to train
        self.attention_log_dict_1 = attention_log_dict_1
        self.uncertainty_log_dict_1 = uncertainty_log_dict_1
        self.sampled_q_log_dict_1 = sampled_q_log_dict_1
        self.loss_log_dict_1 = loss_log_dict_1

        self.attention_log_dict_2 = attention_log_dict_2
        self.uncertainty_log_dict_2 = uncertainty_log_dict_2
        self.selected_rewards_log_dict_2 = selected_rewards_log_dict_2

        self.sampled_q_log_dict_2 = sampled_q_log_dict_2
        self.sampled_q_values_log_dict_2 = sampled_q_values_log_dict_2
        self.sampled_action_log_dict_2 = sampled_action_log_dict_2
        self.sampled_q_penaltiy_log_dict_2 = sampled_q_penaltiy_log_dict_2
        self.sampled_q_normalized_log_dict_2 = sampled_q_normalized_log_dict_2
        self.sampled_q_target_log_dict_2 = sampled_q_target_log_dict_2
        self.loss_log_dict_2 = loss_log_dict_2
        self.level_2_steps = level_2_steps
        self.epoch = epochs

        self.action_dict_level_1 = {0: [0], 1: [1]}
        self.action_dict_level_2 = {
            0: [0, 0, 0],
            1: [1, 0, 0],
            2: [0, 1, 0],
            3: [0, 0, 1],
            4: [1, 1, 0],
            5: [1, 0, 1],
            6: [0, 1, 1],
            7: [1, 1, 1],
        }
        self.action_dict_level_3 = {
            0: [0, 0, 0, 0],
            1: [1, 0, 0, 0],
            2: [0, 1, 0, 0],
            3: [0, 0, 1, 0],
            4: [0, 0, 0, 1],
            5: [1, 1, 0, 0],
            6: [1, 0, 1, 0],
            7: [1, 0, 0, 1],
            8: [0, 1, 1, 0],
            9: [0, 1, 0, 1],
            10: [0, 0, 1, 1],
            11: [1, 1, 1, 0],
            12: [1, 1, 0, 1],
            13: [1, 0, 1, 1],
            14: [0, 1, 1, 1],
            15: [1, 1, 1, 1],
        }

    def cql_loss(self, action_probs, actions, conservative_weight):
        """
        Computes the Conservative Q-Learning (CQL) loss.
        - Penalizes overestimation of Q-values for OOD actions.
        """
        logsumexp_q = torch.logsumexp(action_probs, dim=-1, keepdim=True)
        action_mask = F.one_hot(
            actions.view(-1).long(), num_classes=action_probs.shape[1]
        )

        q_taken = (action_mask * action_probs).sum(dim=-1, keepdims=True)

        conservative_loss = (logsumexp_q - q_taken).mean()

        return conservative_weight * conservative_loss

    def update(
        self, batch: d3rlpy.TorchMiniBatch, grad_step: int, epoch: int
    ) -> dict[str, float]:
        return self.inner_update(batch, grad_step, epoch)

    # The main function for training the RL agent.
    def inner_update(
        self, batch: d3rlpy.TorchMiniBatch, grad_step: int, epoch: int
    ) -> dict[str, float]:

        self._modules.critic_optim_1.zero_grad()
        self._modules.critic_optim_2.zero_grad()
        self._modules.actor_optim_1.zero_grad()
        self._modules.actor_optim_2.zero_grad()

        self.grad_step = grad_step

        q_values_1, q_std_1, attn_scores_1 = self._modules.q_func_1(batch.observations)

        #
        penalty_q_1 = torch.exp(-q_std_1)  # Penalize high uncertainty
        q_penalized_1 = q_values_1 * penalty_q_1
        q_normalized_1 = (q_penalized_1 - q_penalized_1.mean()) / (
            q_penalized_1.std() + 1e-6
        )  # Normalize Q-values

        # q_values_3 = (q_values_3 - q_values_3.mean()) / (q_values_3.std() + 1e-6) # Normalize Q-values

        actions_1 = batch.actions[:, 0]
        actions_2 = batch.actions[:, 1]
        action_probs_1 = self._modules.actor_1(batch.observations)

        predicted_action_1 = torch.argmax(action_probs_1, axis=1)

        # for unsupervised learning, we use the predicted action to calculate the q_values
        action_mask_1 = F.one_hot(
            predicted_action_1, num_classes=q_normalized_1.shape[-1]
        )

        # for supervised learning, we use the actual action to calculate the q_values
        # action_mask_1 = F.one_hot(batch.actions.view(-1).long(), num_classes=self._action_size)

        q_1 = (action_mask_1 * q_normalized_1).sum(dim=1, keepdims=True)
        # q_2 = (action_mask_2 * q_values_2).sum(dim=1, keepdims=True)
        # q_3 = (action_mask_3 * q_values_3).sum(dim=1, keepdims=True)
        # q = (action_mask * self._modules.q_func(batch.observations)).sum(dim=1, keepdims=True)
        # Compute Q-values from critics

        with torch.no_grad():
            # (N, 1)
            # Get the next Q-value estimates from the target Q-function.
            targ_q_1, _, _ = self._modules.targ_q_func_1(
                batch.next_observations
            )  # .max(dim=1, keepdims=True).values)
            targ_q_1 = targ_q_1.max(dim=1, keepdims=True).values

            actions_1_expanded = actions_1.unsqueeze(1)  # Convert (32,) → (32, 1)

            modified_rewards = batch.rewards
            modified_rewards = torch.where(
                (actions_1_expanded == 1) & (modified_rewards < 1),
                modified_rewards,
                modified_rewards,
            )
            # modified_rewards = modified_rewards_ - torch.abs(penalty_q_1)*0.001  # Penalize high uncertainty
            # Normalize rewards if they have high variance
            # if modified_rewards.std() > 10:
            #     modified_rewards = (modified_rewards - modified_rewards.mean()) / (modified_rewards.std() + 1e-6)

            # Compute target Q-values
            y_1 = modified_rewards + self._gamma * targ_q_1 * (1 - batch.terminals)

        # Compute TD-loss (MSE loss)
        td_loss_1 = ((torch.abs(q_1 - y_1)) ** 2).mean()

        # compute the loss for the actor, instead of using Q_value, we use the penalty_q_1
        # Actor Loss (Policy Gradient)
        cql_penalty_1 = self.cql_loss(action_probs_1, actions_1, conservative_weight=1)
        log_action_prob1 = torch.log(action_probs_1 + 1e-6)
        actor_loss_1 = -(log_action_prob1 * q_normalized_1.detach()).sum(dim=1).mean()
        actor_loss_1 = actor_loss_1 + cql_penalty_1  # Detach Q-values
        loss_1 = td_loss_1

        # loss backward and step for the critic and actor
        loss_1.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(
            self._modules.q_func_1.parameters(), max_norm=1.0
        )  # Clipping
        self._modules.critic_optim_1.step()

        actor_loss_1.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(
            self._modules.actor_1.parameters(), max_norm=1.0
        )  # Clipping
        self._modules.actor_optim_1.step()

        #######################################################################################################################
        # # selected_log_probs_1 = torch.log(action_probs_1.gather(1, actions_1.long().unsqueeze(1)) + 1e-6)
        # selected_log_probs_1 = F.log_softmax(action_probs_1 , dim=-1).gather(1, actions_1.long().unsqueeze(1))
        # # actor_loss_1 = -(log_action_1 * q_values_1.detach()).sum(dim=1).mean()  # Detach Q-values
        # actor_loss_1 = -(selected_log_probs_1 * q_values_1.detach()).sum(dim=1).mean()  # Detach Q-values

        # error = torch.nn.functional.cross_entropy(action_probs_1, action_mask_1.float())

        # # Compute TD-loss (MSE loss)
        # # td_loss = ((q - y) ** 2).mean()
        # q_1_prob = F.softmax(q_1, dim=-1)  # Convert Q-values to probabilities
        # kl_loss_1 = torch.nn.functional.kl_div(F.log_softmax(action_probs_1, dim=-1), q_1_prob, reduction="batchmean")

        # loss_1 =  td_loss_1 + kl_loss_1*0.1
        # actor_loss_1= error + actor_loss_1 + cql_penalty_1
        # q_values = (q_values - q_values.min()) / (q_values.max() - q_values.min() + 1e-6)  # Scales Q-values between 0 and 1
        # compute TD loss| Compute Q-values for the selected actions.
        #######################################################################################################################

        # Hierarchical Level 2 Training
        # treatment_probs = torch.nn.functional.gumbel_softmax(log_action_1, tau=0.9, hard=False)[:, 1]

        # treatment_probs = F.gumbel_softmax(action_probs_1)

        # high_treatment_mask = actions_1==1  # High confidence in treatment

        # action_counts = torch.bincount(action_probs_1, minlength=8).float()
        # inverse_weights = 1.0 / (action_counts + 1)  # Give rare actions higher weight
        # treatment_probs = treatment_probs * inverse_weights[action_probs_1]
        # Calculate the probabiliyt of the treatment based on Q-values

        treatment_ = q_penalized_1[:, 1]
        # Selectively train action_2 for confident decisions
        # treatment_probs = F.softmax(action_probs_1, dim=-1)
        treatment_prob = torch.exp(treatment_) / torch.exp(treatment_).sum()
        mean_probability = treatment_prob.mean()

        # Compute action counts

        high_treatment_mask = (
            treatment_prob >= mean_probability
        )  # High confidence in treatment

        if high_treatment_mask.any():

            self.level_2_steps += 1
            # Convert defaultdict to a normal dict

            selected_obs = batch.observations[high_treatment_mask]
            selected_next_obs = batch.next_observations[high_treatment_mask]
            selected_terminal = batch.terminals[high_treatment_mask]

            # using the modified reward as used in the level 1 training
            selected_rewards = modified_rewards[high_treatment_mask]

            selected_actions_2 = actions_2[high_treatment_mask]

            action_probs_2 = self._modules.actor_2(selected_obs)

            q_values_2, q_std_2, attn_scores_2 = self._modules.q_func_2(selected_obs)
            penalty_q_2 = torch.exp(-q_std_2)  # Penalize high uncertainty
            q_penalized_2 = q_values_2 * penalty_q_2  # Penalize high uncertainty
            q_normalized_2 = (q_penalized_2 - q_penalized_2.mean()) / (
                q_penalized_2.std() + 1e-6
            )  # Normalize Q-values

            predicted_action_2 = torch.argmax(action_probs_2, axis=1)

            # calculate the q taken for the selected actions
            action_mask_2 = F.one_hot(
                predicted_action_2, num_classes=q_normalized_2.shape[-1]
            )
            q_2 = (action_mask_2 * q_normalized_2).sum(dim=1, keepdims=True)

            with torch.no_grad():
                targ_q_2, _, _ = self._modules.targ_q_func_2(
                    selected_next_obs
                )  # .max(dim=1, keepdims=True).values)
                targ_q_2 = targ_q_2.max(dim=1, keepdims=True).values
                y_2 = selected_rewards + self._gamma * targ_q_2 * (
                    1 - selected_terminal
                )

            # Compute TD-loss (MSE loss)
            td_loss_2 = ((torch.abs(q_2 - y_2)) ** 2).mean()
            cql_penalty_2 = self.cql_loss(
                action_probs_2, selected_actions_2, conservative_weight=1
            )
            loss_2 = td_loss_2

            # Actor Loss (Policy Gradient)
            action_probs_2 = self._modules.actor_2(selected_obs)
            log_action_prob2 = torch.log(action_probs_2 + 1e-6)

            actor_loss_2 = (
                -(log_action_prob2 * q_normalized_2.detach()).sum(dim=1).mean()
            )  # Detach Q-values
            actor_loss_2 = actor_loss_2 + cql_penalty_2

            loss_2.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                self._modules.actor_2.parameters(), max_norm=1.0
            )  # Clipping
            self._modules.critic_optim_2.step()

            actor_loss_2.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                self._modules.q_func_2.parameters(), max_norm=1.0
            )  # Clipping
            self._modules.actor_optim_2.step()

            #######################################################################################################################
            # q_2_prob = F.softmax(q_2, dim=-1)  # Convert Q-values to probabilities
            # # F.kl_div() expects the first argument to be in log-space and the second to be probability
            # kl_loss = torch.nn.functional.kl_div(F.log_softmax(action_probs_2, dim=-1), q_2_prob, reduction="batchmean")

            # loss_2 = td_loss_2 + kl_loss*0.01
            # # start
            # # actor_loss_2 = -(log_action_2 * torch.softmax(q_values_2, dim=-1).detach()).sum(dim=1).mean()  # Detach Q-values

            # selected_log_probs_2 = F.log_softmax(action_probs_2 , dim=-1).gather(1, selected_actions_2.long().unsqueeze(1))

            # # Compute policy gradient loss

            # actor_loss_2 = -(selected_log_probs_2 * q_values_2.detach() ).mean()

            # actor_loss_2 = actor_loss_2 #+ cql_penalty_2

            #######################################################################################################################

            # keep the record of the loss, attention and uncertainty for the first level

            avg_attention_1 = attn_scores_1.mean(dim=0).detach().cpu().numpy()
            avg_uncertainty_1 = q_std_1.mean().detach().cpu().numpy()
            avg_loss_1 = loss_1.detach().cpu().numpy()
            avg_actor_loss_1 = actor_loss_1.detach().cpu().numpy()
            avg_td_loss_1 = td_loss_1.detach().cpu().numpy()
            avg_cql_penalty_1 = cql_penalty_1.detach().cpu().numpy()
            self.loss_log_dict_1[grad_step] = {
                "Loss": avg_loss_1,
                "TD Loss": avg_td_loss_1,
                "CQL Loss": avg_cql_penalty_1,
                "Actor Loss": avg_actor_loss_1,
            }

            self.attention_log_dict_1[grad_step] = avg_attention_1.T
            self.uncertainty_log_dict_1[grad_step] = avg_uncertainty_1.T

            for loss_name, loss_value in zip(
                ["Loss", "TD Loss", "CQL Loss", "Actor Loss"],
                [avg_loss_1, avg_td_loss_1, avg_cql_penalty_1, avg_actor_loss_1],
            ):
                self.loss_log_dict_1[grad_step][loss_name] = loss_value

            # Log sampled Q-values for visualization
            for action in range(len(self.action_dict_level_1)):
                self.sampled_q_log_dict_1[grad_step][action].append(
                    action_probs_1[:, action].mean().detach().cpu().numpy()
                )

            avg_attention_2 = attn_scores_2.mean(dim=0).detach().cpu().numpy()
            avg_uncertainty_2 = q_std_2.mean().detach().cpu().numpy()
            avg_reward_2 = selected_rewards.mean().detach().cpu().numpy()
            avg_loss_2 = loss_2.detach().cpu().numpy()
            avg_actor_loss_2 = actor_loss_2.detach().cpu().numpy()
            avg_td_loss_2 = td_loss_2.detach().cpu().numpy()
            avg_cql_penalty_2 = cql_penalty_2.detach().cpu().numpy()
            for loss_name, loss_value in zip(
                ["Loss", "TD Loss", "CQL Loss", "Actor Loss"],
                [avg_loss_2, avg_td_loss_2, avg_cql_penalty_2, avg_actor_loss_2],
            ):
                self.loss_log_dict_2[grad_step][loss_name] = loss_value
            # self.loss_log_dict_2[grad_step] = {"Loss": avg_loss_2, "TD Loss": avg_td_loss_2, "CQL Loss": avg_cql_penalty_2, "Actor Loss": avg_actor_loss_2}

            self.attention_log_dict_2[grad_step] = avg_attention_2.T
            self.uncertainty_log_dict_2[grad_step] = avg_uncertainty_2.T
            self.selected_rewards_log_dict_2[grad_step] = avg_reward_2.T

            # Log sampled Q-values for visualization
            for action in range(len(self.action_dict_level_2)):
                self.sampled_action_log_dict_2[grad_step][action].append(
                    action_probs_2[:, action].mean().detach().cpu().numpy()
                )
                # self.sampled_q_log_dict_2[grad_step][action].append(q_2[:, action].mean().detach().cpu().numpy())
                self.sampled_q_penaltiy_log_dict_2[grad_step][action].append(
                    q_penalized_2[:, action].mean().detach().cpu().numpy()
                )
                self.sampled_q_normalized_log_dict_2[grad_step][action].append(
                    q_normalized_2[:, action].mean().detach().cpu().numpy()
                )
                # self.sampled_q_target_log_dict_2[grad_step][action].append(targ_q_2[:, action].mean().detach().cpu().numpy())
                self.sampled_q_values_log_dict_2[grad_step][action].append(
                    q_values_2[:, action].mean().detach().cpu().numpy()
                )
                # self.reward_log_dict_2[grad_step][action].append(selected_rewards[:, action].mean().detach().cpu().numpy())

            return_value = {
                "setp": grad_step,
                "loss": float(loss_2.detach().numpy()),
                "TD Loss": float(td_loss_2.detach().numpy()),
                "CQL Loss": float(cql_penalty_2.detach().numpy()),
                "Actor Loss": float(actor_loss_2.detach().numpy()),
            }

        else:
            return_value = {
                "setp": grad_step,
                "loss": float(loss_1.detach().numpy()),
                "TD Loss": float(td_loss_1.detach().numpy()),
                "CQL Loss": float(cql_penalty_1.detach().numpy()),
                "Actor Loss": float(actor_loss_1.detach().numpy()),
            }

        # Soft update target network
        if grad_step % self._target_update_interval == 0:
            hard_sync(
                self._modules.targ_q_func_1, self._modules.q_func_1
            )  # , tau=0.005)

        # Soft update target network
        if self.level_2_steps % self._target_update_interval == 0:
            hard_sync(
                self._modules.targ_q_func_2, self._modules.q_func_2
            )  # , tau=0.005)

        if grad_step % 500 == 0:

            try:
                # generate the time for now to be added to the name
                time_now = datetime.datetime.now()
                # create a dataframe from the attention log dictionary
                # attention_log_df_1 = pd.DataFrame(self.attention_log_dict_1)
                attention_log_df_2 = pd.DataFrame(self.attention_log_dict_2)
                # save the attention log to a hdf5 file
                folder_name = f"attention_results"
                # attention_log_df_1.to_hdf(f"{folder_name}/attention_log_level_1_{epoch}.h5", key="df")
                attention_log_df_2.to_hdf(
                    f"{folder_name}/attention_log_level_2_{epoch}.h5", key="df"
                )
                # create a dataframe from the uncertainty log dictionary

                # uncertainty_log_df_1 = pd.DataFrame.from_dict(self.uncertainty_log_dict_1, orient="index")
                uncertainty_log_df_2 = pd.DataFrame.from_dict(
                    self.uncertainty_log_dict_2, orient="index"
                )
                selected_rewards_df = pd.DataFrame.from_dict(
                    self.selected_rewards_log_dict_2, orient="index"
                )
                # save the uncertainty log to a hdf5 file
                # uncertainty_log_df_1.to_hdf(f"{folder_name}/uncertainty_log_level_1_{epoch}.h5", key="df")
                uncertainty_log_df_2.to_hdf(
                    f"{folder_name}/uncertainty_log_level_2_{epoch}.h5", key="df"
                )
                # print the attention log dictionary
                selected_rewards_df.to_hdf(
                    f"{folder_name}/selected_rewards_{epoch}.h5", key="df"
                )

                #######################

                # sampled_q_log_df_1 = pd.DataFrame (self.sampled_q_log_dict_1).map(lambda x: x[0])
                # sampled_q_log_df_2 = pd.DataFrame(self.sampled_q_log_dict_2 ).map(lambda x: x[0])
                # sampled_action_log_df_2 = pd.DataFrame(self.sampled_action_log_dict_2).map(lambda x: x[0])
                # sampled_q_penaltiy_log_df_2 = pd.DataFrame(self.sampled_q_penaltiy_log_dict_2).map(lambda x: x[0])
                sampled_q_normalized_log_df_2 = pd.DataFrame(
                    self.sampled_q_normalized_log_dict_2
                ).map(lambda x: x[0])
                # sampled_q_target_log_df_2 = pd.DataFrame(self.sampled_q_target_log_dict_2).map(lambda x: x[0])
                # sampled_q_values_log_df_2 = pd.DataFrame(self.sampled_q_values_log_dict_2).map(lambda x: x[0])

                # loss_log_df_1 = pd.DataFrame(self.loss_log_dict_1).T.apply(lambda x: x.astype(float),axis=1)
                loss_log_df_2 = pd.DataFrame(self.loss_log_dict_2).T.apply(
                    lambda x: x.astype(float), axis=1
                )

                # sampled_q_log_df_1.reset_index(inplace=True, drop=True)
                # sampled_q_log_df_2.reset_index(inplace=True, drop=True)

                # loss_log_df_1.reset_index(inplace=True, drop=True)
                loss_log_df_2.reset_index(inplace=True, drop=True)

                # sampled_q_log_df_1.rename(columns={'index': 'grad_step'}, inplace=True)
                # sampled_q_log_df_2.rename(columns={'index': 'grad_step'}, inplace=True)
                # loss_log_df_1.rename(columns={'index': 'grad_step'}, inplace=True)
                # loss_log_df_2.rename(columns={'index': 'grad_step'}, inplace=True)

                # # Ensure all columns are numeric
                # sampled_q_log_df_1 = sampled_q_log_df_1.apply(pd.to_numeric, errors='coerce')
                # sampled_q_log_df_2 = sampled_q_log_df_2.apply(pd.to_numeric, errors='coerce')

                # # Ensure index is integer
                # sampled_q_log_df_1.index = sampled_q_log_df_1.index.astype(int)
                # sampled_q_log_df_2.index = sampled_q_log_df_2.index.astype(int)

                # sampled_q_log_df_1.to_parquet("attention_results/sample_q_log_level_1.parquet")
                # sampled_q_log_df_2.to_parquet("attention_results/sample_q_log_level_2.parquet")

                # sampled_q_log_df_1.to_hdf(f"{folder_name}/sample_q_log_level_1_{epoch}.h5", key="df", mode="w")
                # sampled_q_log_df_2.to_hdf("attention_results/sample_q_log_level_2.h5", key="df", mode="w")
                # sampled_action_log_df_2.to_hdf(f"{folder_name}/sample_action_log_level_2_{epoch}.h5", key="df", mode="w")
                # sampled_q_penaltiy_log_df_2.to_hdf(f"{folder_name}/sample_q_penaltiy_log_level_2_{epoch}.h5", key="df", mode="w")
                sampled_q_normalized_log_df_2.to_hdf(
                    f"{folder_name}/sample_q_normalized_log_level_2_{epoch}.h5",
                    key="df",
                    mode="w",
                )
                # sampled_q_target_log_df_2.to_hdf("attention_results/sample_q_target_log_level_2.h5", key="df", mode="w")
                # sampled_q_values_log_df_2.to_hdf(f"{folder_name}/sample_q_values_log_level_2_{epoch}.h5", key="df", mode="w")

                # loss_log_df_1.to_parquet("attention_results/loss_log_level_1.parquet")
                # loss_log_df_2.to_parquet("attention_results/loss_log_level_2.parquet")

                # # save the uncertainty log to a hdf5 file
                # loss_log_df_1.to_hdf(f"{folder_name}/loss_log_level_1_{epoch}.h5", key="df", mode="w")
                loss_log_df_2.to_hdf(
                    f"{folder_name}/loss_log_level_2_{epoch}.h5", key="df", mode="w"
                )
                # # print the attention log dictionary
            except Exception as e:
                pass

        return return_value

        # # Soft update target network
        # if grad_step % self._target_update_interval == 0:
        #     for param, target_param in zip(self._modules.q_func.parameters(), self._modules.targ_q_func.parameters()):
        #         target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # # Update target network  (targ_q_func) periodically using hard_sync.

    # The agent selects the best action based on the learned Q-values. [Uses greedy action selection: select the action with the highest Q-value]
    def inner_predict_best_action(
        self, x: d3rlpy.types.TorchObservation
    ) -> torch.Tensor:
        action_prob = self._modules.actor_2(x)  # Get Q-values and attention
        # print (log_action_prob)
        # print (log_action_prob.shape)
        best_action = action_prob.argmax(dim=-1)
        print(best_action)

        # q_mean = (q_mean - q_mean.mean(dim=1, keepdim=True)) / (q_mean.std(dim=1, keepdim=True) + 1e-6)

        # Sampled Q-values (Thompson Sampling)
        # sampled_q = q_mean + q_std * torch.randn_like(q_std)
        # sampled_q = q_mean + (q_std * torch.randn_like(q_std) * 0.5)  # Scale uncertainty
        # exploration_decay = max(0.1, 1.0 - (self.grad_step / self.n_steps))  # Gradually reduce noise
        # decay_rate = 1/self.n_steps  # Adjust based on dataset size
        # exploration_decay = max(0.1, np.exp(-decay_rate * self.grad_step))
        # sampled_q = q_mean + (q_std * torch.randn_like(q_std) * exploration_decay)

        # best_action = sampled_q.argmax(dim=1)  # Select the action with the highest Q-value
        # # Log attention for interpretability
        # print(f"Action: {best_action.item()}, Attention Scores: {attn_scores.detach().cpu().numpy()}")
        return best_action  # , q_mean #, attn_scores, sampled_q

    def inner_sample_action(self, x: d3rlpy.types.TorchObservation) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def inner_predict_value(
        self, x: d3rlpy.types.TorchObservation, action_2: torch.Tensor
    ) -> torch.Tensor:

        q_value_2, q_std_2, _ = self._modules.q_func_2(x)
        # q2 penalized
        penalty_q_2 = torch.exp(-q_std_2)
        q_penalized_2 = q_value_2 * penalty_q_2
        q_normalized_2 = (q_penalized_2 - q_penalized_2.mean(dim=-1, keepdim=True)) / (
            q_penalized_2.std(dim=-1, keepdim=True) + 1e-6
        )

        # q_2= (q_2 - q_2.mean(dim=1, keepdim=True)) / (q_2.std(dim=1, keepdim=True) + 1e-6)
        q_penalized_exp = torch.exp(q_penalized_2)
        q_penalized_prob = q_penalized_exp / q_penalized_exp.sum(dim=-1, keepdim=True)
        # # q = torch.clamp(q, min=-1, max=1)  # Prevent extreme Q-values
        # return q_2[torch.arange(0, q_2.size(0)), flat_action].reshape(-1)
        # Get Q-values and standard deviation (for uncertainty estimation)

        # Compute Q-values using the target network (for stability)
        q_target, q_target_std, _ = self._modules.targ_q_func_2(x)
        # Penalize high uncertainty
        # penalty = torch.exp(-q_target_std)  # Penalize high uncertainty
        # q_target_penalized_2 = q_target * penalty

        # convert Q_values to policy probabilities
        q_target_exp = torch.exp(q_target)
        q_target_prob = q_target_exp / q_target_exp.sum(dim=-1, keepdim=True)

        # Compute Behavioral Policy Probabilities (Actor Network)
        action_probs = self._modules.actor_2(x)

        # Compute importance sampling ratios
        # Compute Importance Sampling Ratios for all actions
        importance_sampling_action = torch.abs(
            action_probs / torch.clamp(q_target, min=1e-6)
        )  # Avoid division by zero
        importance_sampling_critict = q_value_2 / torch.clamp(
            q_target, min=1e-6
        )  # Avoid division by zero

        importance_sampling = importance_sampling_action + importance_sampling_critict
        # Step 9: Handle Zero Importance Weights (Avoid NaN)
        importance_sum = importance_sampling.sum(dim=-1, keepdim=True)
        importance_sum = torch.where(
            importance_sum > 0,
            importance_sum,
            torch.tensor(1e-6, device=importance_sum.device),
        )
        normalized_importance = importance_sampling / torch.clamp(
            importance_sum, min=1e-6
        )

        # # Step 4: Compute Expected Probability for All Actions
        # expected_probabilities = q_penalized_prob * importance_sampling
        # expected_probabilities /= torch.clamp(expected_probabilities.sum(dim=-1, keepdim=True), min=1e-6)

        flat_action = action_2.reshape(-1)
        combination = q_normalized_2
        # prob = q_target_prob[torch.arange(0, q_target_prob.size(0)), flat_action]
        prob = combination[torch.arange(0, combination.size(0)), flat_action]

        return prob


# This dataclass stores hyperparameters and creates the algorithm instance.
# Stores learning rate, batch size, target update interval, and discount factor.
@dataclasses.dataclass()
class CustomAlgoConfig(d3rlpy.base.LearnableConfig):
    def __init__(
        self,
        batch_size: int,
        critic_learning_rate: float,
        actor_learning_rate: float,
        target_update_interval: int,
        gamma: float,
        n_steps: int,
        epochs: int,
    ):

        self.batch_size = batch_size
        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.target_update_interval = target_update_interval
        self.gamma = gamma
        self.n_steps = n_steps
        self.epochs = epochs

    # Defines create(), which initializes a CustomAlgo instance.
    # Ensures compatibility with d3rlpy's algorithm creation process.
    def create(
        self, device: d3rlpy.base.DeviceArg = False, enable_ddp: bool = False
    ) -> "CustomAlgo":
        return CustomAlgo(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "custom"


# This class wraps CustomAlgoImpl and defines how the algorithm is used.
class CustomAlgo(d3rlpy.algos.QLearningAlgoBase[CustomAlgoImpl, CustomAlgoConfig]):

    def inner_create_impl(
        self, observation_shape: d3rlpy.types.Shape, action_size: int
    ) -> None:
        # create Q-functions
        q_func_1 = QFunction(cast(Sequence[int], observation_shape), action_size=2)
        targ_q_func_1 = copy.deepcopy(q_func_1)
        q_func_2 = QFunction(cast(Sequence[int], observation_shape), action_size=8)
        targ_q_func_2 = copy.deepcopy(q_func_2)

        # Create Actor
        actor_1 = Actor(observation_shape, action_size=2)
        actor_2 = Actor(observation_shape, action_size=8)

        # move to device
        q_func_1.to(self._device)
        targ_q_func_1.to(self._device)
        q_func_2.to(self._device)
        targ_q_func_2.to(self._device)

        actor_1.to(self._device)
        actor_2.to(self._device)

        # Create optimizers
        critic_optim_1 = torch.optim.Adam(
            q_func_1.parameters(), lr=self._config.critic_learning_rate
        )
        critic_optim_2 = torch.optim.Adam(
            q_func_2.parameters(), lr=self._config.critic_learning_rate
        )

        actor_optim_1 = torch.optim.Adam(
            actor_1.parameters(), lr=self._config.actor_learning_rate
        )
        actor_optim_2 = torch.optim.Adam(
            actor_2.parameters(), lr=self._config.actor_learning_rate
        )

        # prepare Modules object
        modules = CustomAlgoModules(
            q_func_1=q_func_1,
            targ_q_func_1=targ_q_func_1,
            q_func_2=q_func_2,
            targ_q_func_2=targ_q_func_2,
            actor_1=actor_1,
            actor_2=actor_2,
            critic_optim_1=critic_optim_1,
            critic_optim_2=critic_optim_2,
            actor_optim_1=actor_optim_1,
            actor_optim_2=actor_optim_2,
        )

        # create implementation of CustomAlgoImpl object
        self._impl = CustomAlgoImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            target_update_interval=self._config.target_update_interval,
            gamma=self._config.gamma,
            device=self._device,
            n_steps=self._config.n_steps,
            epochs=self._config.epochs,
        )

    def update(self, batch: d3rlpy.TorchMiniBatch, epoch: int) -> dict[str, float]:
        """Update parameters with mini-batch of data.

        Args:
            batch: Mini-batch data.

        Returns:
            Dictionary of metrics.
        """
        assert self._impl, d3rlpy.IMPL_NOT_INITIALIZED_ERROR

        torch_batch = d3rlpy.TorchMiniBatch.from_batch(
            batch=batch,
            gamma=self._config.gamma,
            compute_returns_to_go=self.need_returns_to_go,
            device=self._device,
            observation_scaler=self._config.observation_scaler,
            action_scaler=self._config.action_scaler,
            reward_scaler=self._config.reward_scaler,
        )
        self._epoch = epoch
        loss = self._impl.update(torch_batch, self._grad_step, self._epoch)
        self._grad_step += 1
        return loss

    def fitter(
        self,
        dataset: ReplayBufferBase,
        n_steps: int,
        n_steps_per_epoch: int = 10000,
        logging_steps: int = 500,
        logging_strategy: d3rlpy.LoggingStrategy = d3rlpy.LoggingStrategy.EPOCH,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logger_adapter: LoggerAdapterFactory = FileAdapterFactory(),
        show_progress: bool = True,
        save_interval: int = 1,
        evaluators: Optional[dict[str, EvaluatorProtocol]] = None,
        callback: Optional[Callable[[Self, int, int], None]] = None,
        epoch_callback: Optional[Callable[[Self, int, int], None]] = None,
    ) -> Generator[tuple[int, dict[str, float]], None, None]:
        """Iterate over epochs steps to train with the given dataset. At each
        iteration algo methods and properties can be changed or queried.

        .. code-block:: python

            for epoch, metrics in algo.fitter(episodes):
                my_plot(metrics)
                algo.save_model(my_path)

        Args:
            dataset: Offline dataset to train.
            n_steps: Number of steps to train.
            n_steps_per_epoch: Number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            experiment_name: Experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: Flag to add timestamp string to the last of
                directory name.
            logging_steps: Number of steps to log metrics. This will be ignored
                if logging_strategy is EPOCH.
            logging_strategy: Logging strategy to use.
            logger_adapter: LoggerAdapterFactory object.
            show_progress: Flag to show progress bar for iterations.
            save_interval: Interval to save parameters.
            evaluators: List of evaluators.
            callback: Callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
            epoch_callback: Callable function that takes
                ``(algo, epoch, total_step)``, which is called at the end of
                every epoch.

        Returns:
            Iterator yielding current epoch and metrics dict.
        """
        LOG.info("dataset info", dataset_info=dataset.dataset_info)

        # check action space
        assert_action_space_with_dataset(self, dataset.dataset_info)

        # initialize scalers
        build_scalers_with_transition_picker(self, dataset)

        # instantiate implementation
        if self._impl is None:
            LOG.debug("Building models...")
            action_size = dataset.dataset_info.action_size
            observation_shape = dataset.sample_transition().observation_signature.shape
            if len(observation_shape) == 1:
                observation_shape = observation_shape[0]  # type: ignore
            self.create_impl(observation_shape, action_size)
            LOG.debug("Models have been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        # setup logger
        if experiment_name is None:
            experiment_name = self.__class__.__name__
        logger = D3RLPyLogger(
            algo=self,
            adapter_factory=logger_adapter,
            experiment_name=experiment_name,
            n_steps_per_epoch=n_steps_per_epoch,
            with_timestamp=with_timestamp,
        )

        # save hyperparameters
        save_config(self, logger)

        # training loop
        n_epochs = n_steps // n_steps_per_epoch
        total_step = 0
        for epoch in range(1, n_epochs + 1):
            # dict to add incremental mean losses to epoch
            epoch_loss = defaultdict(list)

            range_gen = tqdm(
                range(n_steps_per_epoch),
                disable=not show_progress,
                desc=f"Epoch {int(epoch)}/{n_epochs}",
            )

            for itr in range_gen:
                with logger.measure_time("step"):
                    # pick transitions
                    with logger.measure_time("sample_batch"):
                        batch = dataset.sample_transition_batch(self._config.batch_size)

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = self.update(batch, epoch)

                    # record metrics
                    for name, val in loss.items():
                        logger.add_metric(name, val)
                        epoch_loss[name].append(val)

                    # update progress postfix with losses
                    if itr % 10 == 0:
                        mean_loss = {k: np.mean(v) for k, v in epoch_loss.items()}
                        range_gen.set_postfix(mean_loss)

                total_step += 1

                if (
                    logging_strategy == d3rlpy.LoggingStrategy.STEPS
                    and total_step % logging_steps == 0
                ):
                    metrics = logger.commit(epoch, total_step)

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            # call epoch_callback if given
            if epoch_callback:
                epoch_callback(self, epoch, total_step)

            if evaluators:
                for name, evaluator in evaluators.items():
                    test_score = evaluator(self, dataset)
                    logger.add_metric(name, test_score)

            # save metrics
            if logging_strategy == LoggingStrategy.EPOCH:
                metrics = logger.commit(epoch, total_step)

            # save model parameters
            if epoch % save_interval == 0:
                logger.save_model(total_step, self)

            yield epoch, metrics

        logger.close()

    def get_action_type(self) -> d3rlpy.ActionSpace:
        return d3rlpy.ActionSpace.DISCRETE
