# TD-MPC2 Reproduction

Repository for reproducing TD-MPC2.

# This code is inspired by 
https://github.com/nicklashansen/tdmpc2

## Training

Single run from the repo root:

```bash
python train.py task=mujoco-walker seed=1
```

Optuna-driven Slurm multirun from the repo root:

```bash
python train.py -m task=mujoco-walker exp_name=optuna-search
```

The multirun path uses Hydra's Optuna sweeper plus the existing `submitit_slurm` launcher, so each Optuna trial is submitted as its own Slurm job. The training function now returns the best evaluation objective from the run, which Optuna uses to rank trials.

Useful overrides:

```bash
python train.py -m \
  task=mujoco-walker \
  exp_name=optuna-search \
  hydra.sweeper.n_trials=64 \
  hydra.sweeper.n_jobs=16 \
  optimize_metric=episode_reward \
  hydra.sweeper.params.lr='tag(log, interval(1e-5, 1e-3))' \
  hydra.sweeper.params.batch_size='choice(128,256,512)'
```
