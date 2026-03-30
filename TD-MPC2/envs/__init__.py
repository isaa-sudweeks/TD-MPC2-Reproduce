from copy import deepcopy 
import warnings 

import gymnasium as gym 

# TODO: Add support for multitask envs 

# try:
# 	from envs.dmcontrol import make_env as make_dm_control_env
# except:
# 	make_dm_control_env = missing_dependencies
# try:
# 	from envs.maniskill import make_env as make_maniskill_env
# except:
# 	make_maniskill_env = missing_dependencies
# try:
# 	from envs.metaworld import make_env as make_metaworld_env
# except:
# 	make_metaworld_env = missing_dependencies
# try:
# 	from envs.myosuite import make_env as make_myosuite_env
# except:
# 	make_myosuite_env = missing_dependencies
try:
	from envs.mujoco import make_env as make_mujoco_env
except:
	make_mujoco_env = missing_dependencies

warnings.filterwarnings('ignore', category=DeprecationWarning)

def make_env(cfg):
    """
    Make an enviroment for TD-MPC2 experiments.
    """
    gym.logger.set_level(40)
    if cfg.multitask:
        raise NotImplementedError("Multitask envs not implemented yet")
    else:
        env = None 
        for fn in [make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env, make_mujoco_env]:
            try:
                env = fn(cfg)
            except ValueError:
                pass 
        if env is None:
            raise ValueError(f'Failed to make enviroment "{cfg.task}": please verify that dependecies are installed and that the task exists.')
        env = TensorWrapper(env)
        try: # Dict
            cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
        except: #Box 
            cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
        cfg.action_dim = env.action_space.shape[0]
        cfg.episode_length = env.max_episode_stems
        cfg.seed_steps = max(1000, 5*cfg.episode_length)
        # TODO: Add support for wrappers
        return env 

def missing_dependencies():
    raise ValueError("Missing dependencies for this environment")