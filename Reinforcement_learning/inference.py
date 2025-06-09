import torch
import torch.nn as nn
import numpy as np


class RewardEstimator(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_size, activation_fn):
        super(RewardEstimator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn

        layers = [nn.Linear(input_size, hidden_size), activation_fn]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation_fn)
        layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ModelBasedInference:
    def __init__(self, model_path, maximize_outcome=True):
        checkpoint = torch.load(model_path)
        input_size = checkpoint["parameters"]["input_size"]
        output_size = checkpoint["parameters"]["output_size"]
        num_layers = checkpoint["parameters"]["num_layers"]
        hidden_size = checkpoint["parameters"]["hidden_size"]
        activation_fn = checkpoint["parameters"]["activation_fn"]

        self.model = RewardEstimator(
            input_size, output_size, num_layers, hidden_size, activation_fn
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.maximize_outcome = maximize_outcome

    def predict(self, x):
        with torch.no_grad():
            x = torch.tensor(x.astype(np.float32))
            outputs = self.model(x)
            return outputs.numpy()


class ClusteringBasedInference:
    def __init__(self, model, n_states, policy_probs) -> None:
        """
        A class to perform inference for a clustering-based MDP model
        Args:
        model: the clustering model (e.g., KMeans)
        n_states: the number of states (clusters)
        policy: the optimal policy for the MDP
        """
        self.model = model
        self.n_states = n_states
        self.optimal_policy = policy_probs

    def get_state(self, X):
        """
        Get the state of the input data
        Args:
        X: the input data
        """
        return self.model.predict(X)

    def get_policy(self, X):
        """
        Get the policy for the input data
        Args:
        X: the input data
        """
        state = self.get_state(X)
        return self.optimal_policy[state, :]
