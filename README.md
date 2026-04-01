# TD-MPC2 Reproduction

Repository for reproducing TD-MPC2.

# This code is inspired by 
https://github.com/nicklashansen/tdmpc2

## Training

Single run from the repo root:

```bash
python train.py task=mujoco-walker seed=1
```

Hydra multirun from the repo root:

```bash
python train.py -m task=mujoco-walker seed=1,2,3
```

This goes through the normal Hydra multirun path.

Optuna-driven Slurm study with pruning:

```bash
python train.py -optuna task=mujoco-walker exp_name=optuna-search
```

This uses the custom Optuna runner plus Slurm workers launched through `submitit`. Each worker owns real Optuna trials backed by shared study storage, reports intermediate evaluation metrics during training, and can prune weak trials early.

By default the Optuna study uses file-backed `JournalStorage` in the sweep directory, which is a better fit than SQLite for distributed runs over a shared filesystem.

Useful overrides:

```bash
python train.py -optuna \
  task=mujoco-walker \
  exp_name=optuna-search \
  optuna.n_trials=64 \
  optuna.n_jobs=16 \
  optimize_metric=episode_reward
```
