"""hsac — Hierarchical Discrete Soft Actor-Critic for offline RL.

Quickstart::

    import hsac
    data = hsac.make_synthetic_dataset(n_actions=8)
    algo = hsac.HSACConfig(n_options=2, encoder="attention").create()
    algo.fit(data, n_steps=2000)
    actions = algo.predict(data.observations[:5])
"""

from .algo import HierarchicalDiscreteSAC, HSACConfig
from .dataset import OfflineDataset, from_d3rlpy, make_synthetic_dataset
from .networks import AttentionEncoder, CategoricalPolicy, EnsembleQ, MLPEncoder
from .version import __version__

__all__ = [
    "HSACConfig",
    "HierarchicalDiscreteSAC",
    "OfflineDataset",
    "make_synthetic_dataset",
    "from_d3rlpy",
    "AttentionEncoder",
    "MLPEncoder",
    "CategoricalPolicy",
    "EnsembleQ",
    "__version__",
]
