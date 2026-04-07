from copy import deepcopy 
import warnings 

import gymnasium as gym
from envs.wrappers.tensor import TensorWrapper

# TODO: Add support for multitask envs 

def missing_dependencies(task):
	raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

try:
	from envs.dmcontrol import make_env as make_dm_control_env
except:
	make_dm_control_env = missing_dependencies
try:
	from envs.maniskill import make_env as make_maniskill_env
except:
	make_maniskill_env = missing_dependencies
try:
	from envs.metaworld import make_env as make_metaworld_env
except:
	make_metaworld_env = missing_dependencies
try:
	from envs.myosuite import make_env as make_myosuite_env
except:
	make_myosuite_env = missing_dependencies
try:
	from envs.mujoco import make_env as make_mujoco_env
except:
	make_mujoco_env = missing_dependencies

warnings.filterwarnings('ignore', category=DeprecationWarning)

def _max_episode_steps(env):
    """
    Resolve max episode steps across wrapper stacks.
    """
    current = env
    while current is not None:
        steps = getattr(current, 'max_episode_steps', None)
        if steps is not None:
            return steps
        current = getattr(current, 'env', None)
    spec = getattr(env.unwrapped, 'spec', None)
    if spec is not None and getattr(spec, 'max_episode_steps', None) is not None:
        return spec.max_episode_steps
    raise AttributeError('Environment does not define max_episode_steps')

def make_env(cfg):
    """
    Make an environment for TD-MPC2 experiments.
    """
    if hasattr(gym.logger, "set_level"):
        gym.logger.set_level(40)
    else:
        gym.logger.min_level = 40
    env = None 
    if cfg.multitask:
        from envs.wrappers.multitask import MultitaskWrapper
        env = MultitaskWrapper(cfg, [make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env, make_mujoco_env])
        for i in range(len(env.envs)):
            env.envs[i] = TensorWrapper(env.envs[i])
        cfg.obs_shapes = []
        cfg.action_dims = []
        cfg.episode_lengths = []
        for e in env.envs:
            try:
                cfg.obs_shapes.append({k: v.shape for k, v in e.observation_space.spaces.items()})
            except:
                cfg.obs_shapes.append({cfg.get('obs', 'state'): e.observation_space.shape})
            cfg.action_dims.append(e.action_space.shape[0])
            cfg.episode_lengths.append(_max_episode_steps(e))
        cfg.action_dim = max(cfg.action_dims)
        cfg.episode_length = max(cfg.episode_lengths)
        cfg.seed_steps = max(1000, 5*cfg.episode_length)
        cfg.obs_shape = {}
        for shape_dict in cfg.obs_shapes:
            for k, v in shape_dict.items():
                if k not in cfg.obs_shape:
                    cfg.obs_shape[k] = list(v)
                else:
                    cfg.obs_shape[k] = [max(a, b) for a, b in zip(cfg.obs_shape[k], v)]
        for k in cfg.obs_shape:
            cfg.obs_shape[k] = tuple(cfg.obs_shape[k])
        return env
    else:
        for fn in [make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env, make_mujoco_env]:
            try:
                env = fn(cfg)
            except ValueError:
                pass 
        if env is None:
            raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')
        episode_length = _max_episode_steps(env)
        env = TensorWrapper(env)
        try: # Dict
            cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
        except: #Box 
            cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
        cfg.action_dim = env.action_space.shape[0]
        cfg.episode_length = episode_length
        cfg.seed_steps = max(1000, 5*cfg.episode_length)
        # TODO: Add support for wrappers
        return env 

def missing_dependencies():
    raise ValueError("Missing dependencies for this environment")
