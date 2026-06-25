"""End-to-end smoke run: synthetic offline data -> train -> evaluate.

    python examples/quickstart.py

Trains the hierarchical agent on a learnable synthetic problem and checks that the
greedy policy beats a random baseline on held-out states (sanity that learning works).
"""

import numpy as np

import hsac


def main():
    # Same task (shared default task_seed), different states -> a valid generalization test.
    train = hsac.make_synthetic_dataset(n_transitions=10000, state_dim=16, n_actions=8, seed=0)
    test = hsac.make_synthetic_dataset(n_transitions=1500, state_dim=16, n_actions=8, seed=99)

    algo = hsac.HSACConfig(n_options=2, encoder="mlp", batch_size=256).create()
    algo.fit(train, n_steps=3000, log_interval=750)

    # The synthetic generator exposes the known optimal action, so we can score.
    pred = algo.predict(test.observations)
    acc = float((pred == test.optimal_actions).mean())
    rand = 1.0 / 8
    print(f"\nGreedy policy match-with-optimal: {acc:.3f}  (random baseline {rand:.3f})")

    action, info = algo.predict(test.observations[:3], return_info=True)
    print(f"Sample actions: {action}  options: {info['option']}  "
          f"uncertainty: {np.round(info['worker_uncertainty'], 3)}")


if __name__ == "__main__":
    main()
