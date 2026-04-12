import mujoco
import numpy as np
from gymnasium import spaces

from envs.truss.relative_observation_env import MujocoRelativeObsEnv
from gymnasium.envs.registration import register

register(
    id="MujocoVelocityCommandEnvRight-v0",
    entry_point="envs.truss.velocity_command_env:MujocoVelocityCommandEnv",
)


class MujocoVelocityCommandEnv(MujocoRelativeObsEnv):
    """
    MuJoCo truss environment with direct velocity commands.
    """

    def __init__(self, config, render_mode=None, rank=0):
        super().__init__(config, render_mode, rank)
        self.action_space = spaces.Box(low=-self.config.speed, high=self.config.speed, shape=(self.mj_model.model.nu,), dtype=np.float32)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.mj_model.data.ctrl[:] = action

        for _ in range(self.nsubsteps):
            mujoco.mj_step(self.mj_model.model, self.mj_model.data)
            if self.viewer is not None:
                self.viewer.sync()

        self.steps += 1

        truncate = self.steps >= self.max_steps
        reward, reward_dict, terminate = self._compute_reward(action)
        obs = self._get_obs()

        return obs, reward, terminate, truncate, reward_dict
