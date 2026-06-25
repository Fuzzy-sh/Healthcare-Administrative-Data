"""Offline dataset container + a synthetic generator for testing.

The library trains from a fixed batch of logged transitions (offline RL): no
environment interaction. ``OfflineDataset`` is a thin, dependency-light container;
``from_d3rlpy`` adapts a d3rlpy ``MDPDataset`` if you already use one; and
``make_synthetic_dataset`` produces a learnable toy problem so the whole pipeline
runs end to end with no private data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class OfflineDataset:
    observations: np.ndarray        # (N, state_dim) float
    actions: np.ndarray             # (N,) int  in [0, n_actions)
    rewards: np.ndarray             # (N,) float
    next_observations: np.ndarray   # (N, state_dim) float
    terminals: np.ndarray           # (N,) float {0,1}

    def __post_init__(self):
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.next_observations = np.asarray(self.next_observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.int64).reshape(-1)
        self.rewards = np.asarray(self.rewards, dtype=np.float32).reshape(-1)
        self.terminals = np.asarray(self.terminals, dtype=np.float32).reshape(-1)
        n = len(self.actions)
        assert self.observations.shape[0] == n and self.next_observations.shape[0] == n
        assert self.rewards.shape[0] == n and self.terminals.shape[0] == n

    @property
    def state_dim(self) -> int:
        return self.observations.shape[1]

    @property
    def n_actions(self) -> int:
        return int(self.actions.max()) + 1

    def __len__(self) -> int:
        return len(self.actions)

    def sample(self, batch_size: int, device: str = "cpu", rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, len(self), size=batch_size)
        t = lambda a: torch.as_tensor(a[idx]).to(device)
        return {
            "observations": t(self.observations),
            "actions": t(self.actions),
            "rewards": t(self.rewards),
            "next_observations": t(self.next_observations),
            "terminals": t(self.terminals),
        }


def from_d3rlpy(mdp_dataset) -> OfflineDataset:
    """Convert a d3rlpy MDPDataset into an OfflineDataset (best-effort across versions)."""
    obs, act, rew, nxt, term = [], [], [], [], []
    for ep in mdp_dataset.episodes:
        o = np.asarray(ep.observations)
        a = np.asarray(ep.actions).reshape(-1)
        r = np.asarray(ep.rewards).reshape(-1)
        n = len(a)
        for i in range(n):
            obs.append(o[i])
            act.append(a[i])
            rew.append(r[i])
            nxt.append(o[i + 1] if i + 1 < len(o) else o[i])
            term.append(1.0 if i == n - 1 else 0.0)
    return OfflineDataset(np.array(obs), np.array(act), np.array(rew), np.array(nxt), np.array(term))


def make_synthetic_dataset(
    n_transitions: int = 5000,
    state_dim: int = 16,
    n_actions: int = 8,
    seed: int = 0,
    task_seed: int = 1234,
) -> OfflineDataset:
    """A learnable toy offline problem.

    The optimal action is the one whose fixed weight vector best aligns with the
    state, so reward carries real signal and a trained policy should beat random.

    ``task_seed`` fixes the reward model (the action weights), while ``seed`` varies
    the states/behaviour. Train and test sets that share ``task_seed`` are therefore
    the SAME task — essential for a valid generalization check (a different
    ``task_seed`` is a different problem, and a policy can't transfer across them).
    """
    rng = np.random.default_rng(seed)
    task_rng = np.random.default_rng(task_seed)
    obs = rng.normal(size=(n_transitions, state_dim)).astype(np.float32)
    action_weights = task_rng.normal(size=(n_actions, state_dim)).astype(np.float32)

    scores = obs @ action_weights.T                       # (N, n_actions)
    best = scores.argmax(axis=1)
    # Behaviour policy: mostly-good but noisy (so the data isn't already optimal).
    actions = np.where(rng.random(n_transitions) < 0.7, best, rng.integers(0, n_actions, n_transitions))
    rewards = scores[np.arange(n_transitions), actions]
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    next_obs = (0.9 * obs + 0.1 * rng.normal(size=obs.shape)).astype(np.float32)
    # Single-step (contextual-bandit) toy task: reward depends only on (state, action),
    # so every transition terminates. This gives a clean, stable learning signal for
    # the offline value learning (no bootstrap blow-up). The algorithm itself fully
    # supports multi-step sequential data (gamma, bootstrapping) — see README.
    terminals = np.ones(n_transitions, dtype=np.float32)
    ds = OfflineDataset(obs, actions, rewards, next_obs, terminals)
    # Ground-truth optimum, exposed so examples/tests can score the learned policy.
    ds.optimal_actions = best.astype(np.int64)
    ds.action_weights = action_weights
    return ds
