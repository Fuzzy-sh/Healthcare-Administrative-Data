# hsac — Hierarchical Discrete Soft Actor-Critic for offline RL

A small, installable library for **offline reinforcement learning with a two-level
hierarchical controller**, built for sequential decision problems with a structured
discrete action space — the motivating case being **optimal treatment identification**
(choosing combinations of treatments over time from logged patient trajectories).

This is a clean, tested re-implementation of the hierarchical RL method from the
original research code, packaged like `d3rlpy`: `pip install`, a
`Config.create().fit(...)` API, and a runnable example that learns end to end with
no private data.

## The method

```
            ┌─────────────── manager (high level) ───────────────┐
   state ──▶│  picks an OPTION (a region of the action space)     │
            └───────────────────────┬─────────────────────────────┘
                                    │  option
            ┌───────────────────────▼─────────────────────────────┐
   state ──▶│  worker (low level): picks the ACTION within the     │
            │  chosen option's group                               │
            └──────────────────────────────────────────────────────┘
```

- **Two-level hierarchy.** A *manager* selects a coarse option; a *worker* selects
  the concrete action within that option's group. The manager's option-value is
  distilled from the worker's best achievable value (a hierarchical value
  decomposition), so the two levels stay consistent.
- **Offline + conservative (CQL).** Both levels learn from a fixed dataset with a
  Conservative Q-Learning penalty, so out-of-distribution actions are not
  over-valued — essential when you can only learn from logged data.
- **Ensemble uncertainty.** Each critic is an ensemble; the spread across members is
  an epistemic-uncertainty signal exposed at prediction time ("uncertainty-guided").
- **Attention encoder (optional).** A self-attention state encoder over features is
  available (and exposes its attention map for interpretability); an MLP encoder is
  the robust default.

## Install

```bash
pip install -e .            # core: numpy + torch
pip install -e ".[d3rlpy]"  # optional: convert a d3rlpy MDPDataset
pip install -e ".[dev]"     # pytest
```

## Quickstart

```python
import hsac

# A learnable offline toy task (train/test share the reward model -> valid generalization test)
train = hsac.make_synthetic_dataset(n_transitions=10000, n_actions=8, seed=0)
test  = hsac.make_synthetic_dataset(n_transitions=1500,  n_actions=8, seed=99)

algo = hsac.HSACConfig(n_options=2, encoder="mlp").create()
algo.fit(train, n_steps=3000)

action, info = algo.predict(test.observations[:5], return_info=True)
print(action)                    # chosen actions
print(info["option"])            # manager's high-level options
print(info["worker_uncertainty"])# epistemic uncertainty per decision
```

Run the bundled example:

```bash
python examples/quickstart.py
```

which trains and reports generalization on a held-out test set:

```
step  3000 | worker td 0.75 cql 2.00 unc 0.02 | manager distill 0.02
Greedy policy match-with-optimal: 0.868  (random baseline 0.125)
```

i.e. the learned hierarchical policy recovers the optimal action **~87%** of the
time on unseen states, versus 12.5% for random — and the full hierarchical predict
matches or beats the worker alone, confirming the manager routes to the right option.

## Bring your own data

`hsac` trains from any fixed batch of transitions:

```python
from hsac import OfflineDataset
data = OfflineDataset(observations, actions, rewards, next_observations, terminals)
```

or convert an existing d3rlpy dataset with `hsac.from_d3rlpy(mdp_dataset)`. The
algorithm supports full multi-step sequential data (discounting + bootstrapping);
the bundled toy task is single-step for a fast, clean learning signal.

## Package layout

```
hsac/
├── __init__.py     # public API
├── networks.py     # AttentionEncoder / MLPEncoder, CategoricalPolicy, EnsembleQ
├── algo.py         # HSACConfig + HierarchicalDiscreteSAC (manager + worker, CQL, uncertainty)
├── dataset.py      # OfflineDataset, synthetic generator, d3rlpy adapter
└── version.py
examples/quickstart.py
tests/test_smoke.py
```

## Notes

- **Encoder choice.** `encoder="mlp"` is the robust default. `encoder="attention"`
  targets rich feature-interaction inputs and exposes attention maps, but needs more
  data/tuning than MLP on small tabular tasks.
- **Conservatism.** `conservative_weight` trades safety (penalizing OOD actions)
  against fit; `0.5` is a balanced default. Heavier values are more conservative but
  can hurt generalization on near-on-policy data.
- This re-implementation preserves the *design* of the original two-layer
  hierarchical SAC (attention + CQL + ensemble uncertainty); it is built on PyTorch
  and the maintained d3rlpy, rather than the original vendored copy of d3rlpy internals.
