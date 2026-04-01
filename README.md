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

By default the launcher submits the Slurm workers and exits immediately. This avoids making the login-node shell a single point of failure: once the jobs are submitted, the login-node process can die without stopping the already-running workers.

By default the Optuna study uses file-backed `JournalStorage` in a stable path under `logs/optuna/<task>/<exp_name>/<study_name>/optuna_journal.log`, which is a better fit than SQLite for distributed runs over a shared filesystem and lets later launches reattach to the same study by default.

Useful overrides:

```bash
python train.py -optuna \
  task=mujoco-walker \
  exp_name=optuna-search \
  optuna.n_trials=64 \
  optuna.n_jobs=16 \
  optimize_metric=episode_reward
```

If you want the launching process to stay attached and wait for completion, override:

```bash
python train.py -optuna task=mujoco-walker exp_name=optuna-search optuna.wait_for_completion=true
```

Each launch also writes a manifest with the submitted Slurm job ids to:

```text
logs/optuna/<task>/<exp_name>/<study_name>/latest_launch.json
```

To sync offline Weights & Biases runs from a login node while Optuna workers are still writing them:

```bash
bash scripts/sync_wandb_offline.sh
```

By default this watches `logs/hydra/multirun`, polls every 60 seconds, and re-syncs a run only when its files have changed since the previous successful sync. You can stop it with `Ctrl+C`.
