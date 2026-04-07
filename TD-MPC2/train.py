from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch
import optuna

import hydra
from termcolor import colored
from omegaconf import OmegaConf

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from trainer.online_multitask_trainer import OnlineMultitaskTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True 
torch.set_float32_matmul_precision('high')

def run_training(cfg, trial=None):
    """
    Execute one training run and return the best objective value seen.
    """
    assert torch.cuda.is_available(), "CUDA not available, please run on a GPU"
    assert cfg.steps > 0, "Number of steps must be positive"
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

    if cfg.multitask:
        trainer_clf = OfflineTrainer if cfg.task in {'mt30', 'mt80'} else OnlineMultitaskTrainer
    else:
        trainer_clf = OnlineTrainer
    env = make_env(cfg)
    trainer = trainer_clf(
        cfg = cfg,
        env = env, # I need to make this 
        agent=TDMPC2(cfg),
        logger=Logger(cfg),
        buffer=Buffer(cfg),
        trial=trial,
    )

    try:
        trainer.train()
        objective_value, objective_metric = trainer.best_objective()
        if trial is not None:
            cfg.optuna_trial_state = "complete"
        print(
            colored("Optimization objective:", "cyan", attrs=["bold"]),
            f"{objective_metric}={objective_value:.6f}",
        )
        print(colored('Training completed!', 'green', attrs=['bold']))
        return objective_value
    except optuna.TrialPruned:
        cfg.optuna_trial_state = "pruned"
        trainer.logger.finish()
        raise
    except Exception:
        if trial is not None:
            cfg.optuna_trial_state = "failed"
        trainer.logger.finish()
        raise
    finally:
        env.close()


@hydra.main(config_name='config', config_path='.')
def train(cfg):
    """
    Script for training TD-MPC2 agents.

    Most relevant args:
    	`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)
    
    Example usage:
    	python train.py task=mt80 model_size=5 steps=10M seed=1
    
    """
    return run_training(cfg)


if __name__ == '__main__':
    train()
