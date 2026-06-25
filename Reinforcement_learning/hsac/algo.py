"""Hierarchical Discrete Soft Actor-Critic for offline RL.

A two-level controller for sequential decision problems with a structured discrete
action space (the motivating case: choosing combinations of treatments over time):

  - LEVEL 1 (manager)  picks a coarse *option* from a small set.
  - LEVEL 2 (worker)   picks the concrete action, conditioned on the manager's option.

Both levels are discrete Soft Actor-Critic learners trained **offline** from logged
transitions, each with:
  - an attention (or MLP) state encoder,
  - an ensemble critic whose disagreement is an epistemic-uncertainty signal,
  - a Conservative Q-Learning (CQL) penalty so out-of-distribution actions are not
    over-valued — essential when you can only learn from a fixed dataset.

This is a clean re-implementation of the design in the original research code
(two-layer hierarchical SAC + attention + CQL + uncertainty), built on PyTorch with
a d3rlpy-style ``Config.create().fit(...)`` API.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import OfflineDataset
from .networks import CategoricalPolicy, EnsembleQ, make_encoder


@dataclasses.dataclass
class HSACConfig:
    """Hyper-parameters. Call :meth:`create` to build the algorithm."""

    n_options: int = 2                  # size of the high-level (manager) action set
    encoder: str = "mlp"                # "mlp" (robust default) or "attention"
                                        # ("attention" suits rich feature-interaction
                                        #  inputs and exposes attention maps; it needs
                                        #  more data/tuning than "mlp" on small tabular tasks)
    embed_dim: int = 256
    num_heads: int = 4
    hidden_dim: int = 256
    n_critics: int = 2                  # ensemble size (uncertainty + conservative min)
    gamma: float = 0.99
    tau: float = 0.005                  # soft target-update rate
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    conservative_weight: float = 0.5    # CQL penalty strength (offline conservatism)
    target_entropy_ratio: float = 0.98
    batch_size: int = 256

    def create(self, device: str = "cpu") -> "HierarchicalDiscreteSAC":
        return HierarchicalDiscreteSAC(self, device)


def _soft_update(targ: nn.Module, src: nn.Module, tau: float) -> None:
    for tp, sp in zip(targ.parameters(), src.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * sp.data)


class _Level(nn.Module):
    """One discrete-SAC level: encoder + categorical policy + ensemble critic."""

    def __init__(self, state_dim: int, n_act: int, cfg: HSACConfig, context_dim: int = 0):
        super().__init__()
        self.encoder = make_encoder(cfg.encoder, state_dim, cfg.embed_dim, cfg.num_heads)
        in_dim = self.encoder.out_dim + context_dim
        self.policy = CategoricalPolicy(in_dim, n_act, cfg.hidden_dim)
        self.q = EnsembleQ(in_dim, n_act, cfg.n_critics, cfg.hidden_dim)
        self.targ_q = copy.deepcopy(self.q)
        for p in self.targ_q.parameters():
            p.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.target_entropy = cfg.target_entropy_ratio * math.log(n_act)

    def feat(self, obs: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.encoder(obs)
        if context is not None:
            h = torch.cat([h, context], dim=-1)
        return h


class HierarchicalDiscreteSAC:
    def __init__(self, config: HSACConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self._manager: Optional[_Level] = None
        self._worker: Optional[_Level] = None
        self._opt = {}
        self._n_actions = None
        self._rng = np.random.default_rng()

    # --- setup -------------------------------------------------------------- #
    def _build(self, state_dim: int, n_actions: int) -> None:
        cfg = self.config
        if n_actions % cfg.n_options != 0:
            raise ValueError(
                f"n_actions ({n_actions}) must be divisible by n_options ({cfg.n_options}); "
                "options partition the action set into equal groups.")
        self._n_actions = n_actions
        self._group_size = n_actions // cfg.n_options
        # Manager chooses an option; worker chooses a LOCAL action within that option's
        # group, conditioned on the option one-hot. Local action space => the worker can
        # only ever pick actions it was trained on.
        self._manager = _Level(state_dim, cfg.n_options, cfg).to(self.device)
        self._worker = _Level(state_dim, n_actions, cfg).to(self.device)
        for name, lvl in (("manager", self._manager), ("worker", self._worker)):
            self._opt[name] = {
                "critic": torch.optim.Adam(
                    list(lvl.encoder.parameters()) + list(lvl.q.parameters()), lr=cfg.critic_lr),
                "actor": torch.optim.Adam(lvl.policy.parameters(), lr=cfg.actor_lr),
                "alpha": torch.optim.Adam([lvl.log_alpha], lr=cfg.alpha_lr),
            }

    # --- training ----------------------------------------------------------- #
    def _update_level(self, name, lvl, obs, next_obs, act, rew, term, ctx, next_ctx) -> dict:
        cfg = self.config
        alpha = lvl.log_alpha.exp().detach()

        # ----- critic (TD + CQL) -----
        with torch.no_grad():
            nf = lvl.feat(next_obs, next_ctx)
            next_probs, next_logp = lvl.policy.probs(nf)
            targ_next = lvl.targ_q.min_q(nf)
            v_next = (next_probs * (targ_next - alpha * next_logp)).sum(dim=-1)
            # torch.where (not (1 - term) * v_next): a terminal must take target == rew
            # exactly. Multiplying by a zero mask would propagate any non-finite v_next
            # as 0 * inf = nan and silently poison the critic.
            target = torch.where(term > 0.5, rew, rew + cfg.gamma * v_next)

        feat = lvl.feat(obs, ctx)
        q_all = lvl.q(feat)                                       # (K, B, A)
        a_idx = act.view(1, -1, 1).expand(q_all.size(0), -1, 1)
        q_taken = q_all.gather(2, a_idx).squeeze(-1)             # (K, B)
        td_loss = ((q_taken - target.unsqueeze(0)) ** 2).mean()
        cql = (torch.logsumexp(q_all, dim=-1) - q_taken).mean()  # conservative penalty
        critic_loss = td_loss + cfg.conservative_weight * cql

        self._opt[name]["critic"].zero_grad()
        critic_loss.backward()
        self._opt[name]["critic"].step()

        # ----- actor (features detached so the actor doesn't drag the encoder) -----
        feat_d = lvl.feat(obs, ctx).detach()
        probs, logp = lvl.policy.probs(feat_d)
        with torch.no_grad():
            q_pi = lvl.q.min_q(feat_d)
        actor_loss = (probs * (alpha * logp - q_pi)).sum(dim=-1).mean()

        self._opt[name]["actor"].zero_grad()
        actor_loss.backward()
        self._opt[name]["actor"].step()

        # ----- temperature (auto-tuned entropy) -----
        alpha_loss = (probs.detach() * (-lvl.log_alpha * (logp.detach() + lvl.target_entropy))).sum(dim=-1).mean()
        self._opt[name]["alpha"].zero_grad()
        alpha_loss.backward()
        self._opt[name]["alpha"].step()

        _soft_update(lvl.targ_q, lvl.q, cfg.tau)

        with torch.no_grad():
            uncertainty = lvl.q.uncertainty(feat_d).mean().item()
        return {
            f"{name}/td": float(td_loss), f"{name}/cql": float(cql),
            f"{name}/actor": float(actor_loss), f"{name}/alpha": float(alpha),
            f"{name}/uncertainty": uncertainty,
        }

    def _update_manager(self, obs: torch.Tensor) -> dict:
        """Hierarchical value decomposition.

        The value of an option is the best value the worker can achieve under it.
        We compute that for *every* option (a dense, clean target) and regress the
        manager's option-Q onto it, so ``argmax`` over options recovers the option
        containing the globally best action.
        """
        cfg = self.config
        gs = self._group_size
        with torch.no_grad():
            qW = self._worker.q.min_q(self._worker.feat(obs, None))         # (B, n_actions)
            vals = [qW[:, g * gs:(g + 1) * gs].max(dim=-1).values for g in range(cfg.n_options)]
            target = torch.stack(vals, dim=1)                               # (B, n_options)
        qm = self._manager.q(self._manager.feat(obs, None))                 # (K, B, n_options)
        loss = ((qm - target.unsqueeze(0)) ** 2).mean()
        self._opt["manager"]["critic"].zero_grad()
        loss.backward()
        self._opt["manager"]["critic"].step()
        _soft_update(self._manager.targ_q, self._manager.q, cfg.tau)
        return {"manager/distill": float(loss)}

    def fit(self, dataset: OfflineDataset, n_steps: int = 2000,
            log_interval: int = 200, verbose: bool = True) -> list[dict]:
        if self._manager is None:
            self._build(dataset.state_dim, dataset.n_actions)
        cfg = self.config
        history = []
        for step in range(1, n_steps + 1):
            b = dataset.sample(cfg.batch_size, self.device, self._rng)
            obs, next_obs = b["observations"], b["next_observations"]
            act, rew, term = b["actions"], b["rewards"], b["terminals"]

            # Worker: discrete SAC + CQL over the full action space (proven Q-regression).
            # Manager: option-value distilled from the worker's best-in-group value (below).
            logs = {}
            logs.update(self._update_level("worker", self._worker, obs, next_obs, act, rew, term, None, None))
            logs.update(self._update_manager(obs))

            if step % log_interval == 0 or step == n_steps:
                logs["step"] = step
                history.append(logs)
                if verbose:
                    print(f"step {step:5d} | worker td {logs['worker/td']:.3f} "
                          f"cql {logs['worker/cql']:.3f} unc {logs['worker/uncertainty']:.3f} "
                          f"| manager distill {logs['manager/distill']:.3f}")
        return history

    # --- inference ---------------------------------------------------------- #
    @torch.no_grad()
    def predict(self, observations: np.ndarray, return_info: bool = False):
        """Greedy hierarchical action: manager picks an option, worker picks the action."""
        obs = torch.as_tensor(np.atleast_2d(observations).astype(np.float32)).to(self.device)
        fW = self._worker.feat(obs, None)
        qW = self._worker.q.min_q(fW)                                   # (B, n_actions)
        option = self._manager.q.min_q(self._manager.feat(obs, None)).argmax(dim=-1)
        # Restrict the worker to the manager's chosen option group, then pick the best action.
        action_group = torch.arange(self._n_actions, device=self.device) // self._group_size
        valid = action_group.unsqueeze(0) == option.unsqueeze(1)        # (B, n_actions)
        action = qW.masked_fill(~valid, float("-inf")).argmax(dim=-1)
        if return_info:
            info = {
                "option": option.cpu().numpy(),
                "worker_uncertainty": self._worker.q.uncertainty(fW).cpu().numpy(),
                "attention": self._worker.encoder.last_attention,
            }
            return action.cpu().numpy(), info
        return action.cpu().numpy()

    def save(self, path: str) -> None:
        torch.save({"manager": self._manager.state_dict(),
                    "worker": self._worker.state_dict(),
                    "config": dataclasses.asdict(self.config),
                    "n_actions": self._n_actions}, path)

    def load(self, path: str, state_dim: int, n_actions: int) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self._build(state_dim, n_actions)
        self._manager.load_state_dict(ckpt["manager"])
        self._worker.load_state_dict(ckpt["worker"])
