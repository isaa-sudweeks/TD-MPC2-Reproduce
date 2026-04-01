import datetime
import time
from pathlib import Path

import optuna
import submitit
from omegaconf import OmegaConf
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from train import run_training


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
TERMINAL_STATES = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)


def _load_cfg(overrides):
    cfg = OmegaConf.load(CONFIG_PATH)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    timestamp = datetime.datetime.now()
    runtime_cwd = Path.cwd()
    cfg.hydra.run.dir = str(
        runtime_cwd / "logs" / "hydra" / "run" / str(cfg.task) / str(cfg.exp_name)
        / timestamp.strftime("%Y-%m-%d") / timestamp.strftime("%H-%M-%S")
    )
    cfg.hydra.sweep.dir = str(
        runtime_cwd / "logs" / "hydra" / "multirun" / str(cfg.task) / str(cfg.exp_name)
        / timestamp.strftime("%Y-%m-%d") / timestamp.strftime("%H-%M-%S")
    )
    cfg.hydra.sweep.subdir = "${hydra.job.num}"
    return cfg


def _study_storage(cfg):
    if cfg.optuna.storage not in {None, "null"}:
        return cfg.optuna.storage
    storage_path = Path(cfg.hydra.sweep.dir) / "optuna_study.db"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{storage_path}"


def _make_sampler(cfg):
    sampler_cfg = cfg.optuna.sampler
    sampler_type = str(sampler_cfg.type).lower()
    if sampler_type == "tpe":
        return optuna.samplers.TPESampler(seed=sampler_cfg.seed)
    raise ValueError(f"Unsupported Optuna sampler type: {sampler_cfg.type}")


def _make_pruner(cfg):
    pruner_cfg = cfg.optuna.pruner
    pruner_type = str(pruner_cfg.type).lower()
    if pruner_type == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=pruner_cfg.n_startup_trials,
            n_warmup_steps=pruner_cfg.n_warmup_steps,
            interval_steps=pruner_cfg.interval_steps,
        )
    if pruner_type == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=pruner_cfg.get("min_resource", "auto"),
            reduction_factor=pruner_cfg.get("reduction_factor", 4),
            min_early_stopping_rate=pruner_cfg.get("min_early_stopping_rate", 0),
        )
    raise ValueError(f"Unsupported Optuna pruner type: {pruner_cfg.type}")


def _suggest_params(trial, cfg):
    params = {}
    for name, spec in cfg.optuna.search_space.items():
        spec_type = str(spec.type).lower()
        if spec_type == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec.low),
                float(spec.high),
                log=bool(spec.get("log", False)),
                step=spec.get("step", None),
            )
        elif spec_type == "int":
            params[name] = trial.suggest_int(
                name,
                int(spec.low),
                int(spec.high),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        elif spec_type == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec.choices))
        else:
            raise ValueError(f"Unsupported search-space type '{spec.type}' for '{name}'")
    return params


def _trial_cfg(base_cfg, params, trial):
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    for key, value in params.items():
        OmegaConf.update(cfg, key, value, merge=False)
    cfg.work_dir = str(Path(base_cfg.hydra.sweep.dir) / f"trial-{trial.number:04d}")
    cfg.optuna_study_name = str(base_cfg.optuna.study_name)
    cfg.optuna_trial_number = int(trial.number)
    cfg.optuna_trial_params = dict(params)
    cfg.optuna_trial_state = "running"
    return cfg


def _objective(base_cfg, trial):
    params = _suggest_params(trial, base_cfg)
    cfg = _trial_cfg(base_cfg, params, trial)
    return run_training(cfg, trial=trial)


class OptunaWorker:
    def __init__(self, base_cfg):
        self.base_cfg = OmegaConf.to_container(base_cfg, resolve=True)

    def __call__(self):
        cfg = OmegaConf.create(self.base_cfg)
        storage = _study_storage(cfg)
        study = optuna.load_study(study_name=cfg.optuna.study_name, storage=storage)
        callbacks = [
            MaxTrialsCallback(
                cfg.optuna.n_trials,
                states=(TrialState.COMPLETE, TrialState.PRUNED),
            )
        ]
        completed = 0
        while True:
            before = len(study.get_trials(deepcopy=False, states=TERMINAL_STATES))
            if before >= cfg.optuna.n_trials:
                return
            study.optimize(
                lambda trial: _objective(cfg, trial),
                n_trials=1,
                callbacks=callbacks,
                catch=(RuntimeError,),
            )
            completed += 1
            if cfg.optuna.max_worker_trials and completed >= cfg.optuna.max_worker_trials:
                return


def run_multirun(overrides):
    cfg = _load_cfg(overrides)
    storage = _study_storage(cfg)
    study = optuna.create_study(
        study_name=cfg.optuna.study_name,
        storage=storage,
        direction=cfg.optimize_direction,
        sampler=_make_sampler(cfg),
        pruner=_make_pruner(cfg),
        load_if_exists=True,
    )

    submitit_dir = Path(cfg.hydra.sweep.dir) / ".submitit"
    submitit_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(submitit_dir))
    executor.update_parameters(
        timeout_min=cfg.hydra.launcher.timeout_min,
        mem_gb=cfg.hydra.launcher.mem_gb,
        slurm_additional_parameters=OmegaConf.to_container(
            cfg.hydra.launcher.additional_parameters,
            resolve=True,
        ),
        name=f"{cfg.task}-{cfg.exp_name}-optuna",
    )

    futures = [executor.submit(OptunaWorker(cfg)) for _ in range(cfg.optuna.n_jobs)]
    print(
        f"Launched {len(futures)} Slurm Optuna workers for study '{study.study_name}'. "
        f"Target trials: {cfg.optuna.n_trials}. Storage: {storage}"
    )

    while True:
        terminal_trials = study.get_trials(deepcopy=False, states=TERMINAL_STATES)
        if len(terminal_trials) >= cfg.optuna.n_trials:
            break
        if all(job.done() for job in futures):
            break
        time.sleep(cfg.optuna.poll_interval_sec)

    for job in futures:
        job.result()

    completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    pruned_trials = study.get_trials(deepcopy=False, states=(TrialState.PRUNED,))
    failed_trials = study.get_trials(deepcopy=False, states=(TrialState.FAIL,))
    print(
        f"Study finished with {len(completed_trials)} complete, "
        f"{len(pruned_trials)} pruned, {len(failed_trials)} failed trials."
    )
    if not completed_trials:
        raise RuntimeError("Optuna study finished without any completed trials.")
    best_trial = study.best_trial
    print(f"Best trial: #{best_trial.number}")
    print(f"Best value: {best_trial.value}")
    print(f"Best params: {best_trial.params}")
