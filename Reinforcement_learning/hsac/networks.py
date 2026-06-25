"""Neural network building blocks for Hierarchical Discrete SAC.

Three pieces, kept small and independently testable:

- ``AttentionEncoder`` — a self-attention state encoder. Each scalar feature is
  lifted to a token and passed through multi-head self-attention, so the model can
  learn which clinical features matter for the current decision. The attention map
  is retained for interpretability/uncertainty inspection.
- ``CategoricalPolicy`` — a discrete (softmax) policy head.
- ``EnsembleQ`` — an ensemble of discrete Q-heads. The spread across heads is the
  epistemic-uncertainty signal that makes the method "uncertainty-guided"; the min
  across heads is used for conservative target values.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """Plain MLP state encoder (fast default)."""

    def __init__(self, state_dim: int, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
        )
        self.out_dim = embed_dim
        self.last_attention = None  # API parity with AttentionEncoder

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class AttentionEncoder(nn.Module):
    """Self-attention over state features.

    The state vector ``(B, state_dim)`` is treated as a length-``state_dim`` sequence
    of scalar tokens; multi-head self-attention mixes them into a pooled embedding.
    ``last_attention`` holds the most recent attention map for inspection.
    """

    def __init__(self, state_dim: int, embed_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.token = nn.Linear(1, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim
        self.last_attention = None

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = self.token(s.unsqueeze(-1))                # (B, state_dim, embed_dim)
        h, w = self.attn(x, x, x, need_weights=True, average_attn_weights=True)
        self.last_attention = w.detach()               # (B, state_dim, state_dim)
        h = self.norm(h + x)
        return h.mean(dim=1)                            # (B, embed_dim)


def make_encoder(kind: str, state_dim: int, embed_dim: int, num_heads: int) -> nn.Module:
    if kind == "attention":
        return AttentionEncoder(state_dim, embed_dim, num_heads)
    if kind == "mlp":
        return MLPEncoder(state_dim, embed_dim)
    raise ValueError(f"unknown encoder kind: {kind!r} (use 'attention' or 'mlp')")


class CategoricalPolicy(nn.Module):
    """Discrete softmax policy head over ``n_actions``."""

    def __init__(self, in_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)  # logits

    def probs(self, h: torch.Tensor):
        logits = self.forward(h)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.exp(), log_probs


class EnsembleQ(nn.Module):
    """Ensemble of discrete Q-heads; spread across heads = epistemic uncertainty."""

    def __init__(self, in_dim: int, n_actions: int, n_critics: int = 2, hidden: int = 256):
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, n_actions),
            )
            for _ in range(n_critics)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(h) for head in self.heads], dim=0)  # (K, B, A)

    def min_q(self, h: torch.Tensor) -> torch.Tensor:
        return self.forward(h).min(dim=0).values                     # (B, A)

    def uncertainty(self, h: torch.Tensor) -> torch.Tensor:
        return self.forward(h).std(dim=0).mean(dim=-1)               # (B,)
