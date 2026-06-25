"""Smoke tests — the package imports, trains a few steps, and predicts without error."""

import numpy as np

import hsac


def test_import_and_version():
    assert isinstance(hsac.__version__, str)


def test_synthetic_dataset_shapes():
    d = hsac.make_synthetic_dataset(n_transitions=200, state_dim=8, n_actions=4, seed=1)
    assert len(d) == 200
    assert d.state_dim == 8
    assert d.n_actions == 4


def test_train_and_predict_attention():
    d = hsac.make_synthetic_dataset(n_transitions=400, state_dim=8, n_actions=4, seed=2)
    algo = hsac.HSACConfig(n_options=2, encoder="attention", batch_size=64,
                           embed_dim=32, hidden_dim=32, num_heads=2).create()
    hist = algo.fit(d, n_steps=50, log_interval=25, verbose=False)
    assert len(hist) >= 1
    pred = algo.predict(d.observations[:10])
    assert pred.shape == (10,)
    assert pred.min() >= 0 and pred.max() < d.n_actions


def test_train_mlp_encoder_and_info():
    d = hsac.make_synthetic_dataset(n_transitions=300, state_dim=6, n_actions=6, seed=3)
    algo = hsac.HSACConfig(n_options=3, encoder="mlp", batch_size=64,
                           embed_dim=32, hidden_dim=32).create()
    algo.fit(d, n_steps=30, verbose=False)
    action, info = algo.predict(d.observations[:5], return_info=True)
    assert action.shape == (5,)
    assert info["option"].shape == (5,)
    assert info["worker_uncertainty"].shape == (5,)


def test_learning_beats_random():
    # On the learnable synthetic task, training should beat the random baseline.
    train = hsac.make_synthetic_dataset(n_transitions=4000, state_dim=12, n_actions=6, seed=0)
    algo = hsac.HSACConfig(n_options=2, encoder="mlp", batch_size=256).create()
    algo.fit(train, n_steps=1500, verbose=False)
    acc = float((algo.predict(train.observations) == train.optimal_actions).mean())
    assert acc > 1.0 / 6  # better than random
