#!/usr/bin/env python3
"""
Train SAC behavior policies and save TD-MPC2-compatible offline dataset shards.

For a multitask TD-MPC2 task set, this script trains one single-task SAC policy per
selected subtask and stores the collected episodes with the corresponding task id.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "TD-MPC2"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_default_mujoco_gl = "glfw" if platform.system() == "Darwin" else "egl"
os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", _default_mujoco_gl)
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "tdmpc2_matplotlib"))

from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from common.parser import parse_cfg  # noqa: E402
from common.seed import set_seed  # noqa: E402
from envs.mujoco import make_env as make_mujoco_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAC and save its online replay as TD-MPC2 offline shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", required=True, help="Directory for dataset shards, SAC models, and manifest.")
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="SAC training steps per selected task.")
    parser.add_argument("--shard-episodes", type=int, default=25, help="Number of episodes per saved shard.")
    parser.add_argument(
        "--flush-every-episodes",
        type=int,
        default=None,
        help="Flush completed episodes to disk every N episodes per length bucket. Defaults to --shard-episodes.",
    )
    parser.add_argument("--task-indices", default=None, help="Comma-separated subtask indices. Defaults to all subtasks.")
    parser.add_argument("--save-model", action="store_true", help="Save each trained SAC policy.")
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--progress-bar", action="store_true", help="Show SB3 training progress bar.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf overrides, e.g. task=truss-velocity-command max_steps=1000 save_video=false",
    )
    return parser.parse_args()


def is_missing(cfg, key: str) -> bool:
    return OmegaConf.is_missing(cfg, key) or cfg.get(key, None) == "???"


def load_cfg(overrides: list[str], seed: int | None):
    base_cfg = OmegaConf.load(PROJECT_ROOT / "config.yaml")
    cli_cfg = OmegaConf.from_dotlist(overrides)
    cfg = OmegaConf.merge(base_cfg, cli_cfg)
    if seed is not None:
        cfg.seed = seed
    if is_missing(cfg, "save_video"):
        cfg.save_video = False
    if is_missing(cfg, "checkpoint"):
        cfg.checkpoint = None
    if is_missing(cfg, "data_dir"):
        cfg.data_dir = None
    if is_missing(cfg, "work_dir"):
        cfg.work_dir = str(REPO_ROOT / "logs" / "collect_sac_dataset")
    return parse_cfg(cfg)


def selected_task_indices(cfg, raw_indices: str | None) -> list[int]:
    if raw_indices is None:
        return list(range(len(cfg.tasks)))
    indices = [int(x.strip()) for x in raw_indices.split(",") if x.strip()]
    invalid = [i for i in indices if i < 0 or i >= len(cfg.tasks)]
    if invalid:
        raise ValueError(f"Invalid task indices {invalid}; configured tasks are 0..{len(cfg.tasks) - 1}.")
    return indices


def single_task_cfg(cfg, task_name: str):
    task_cfg = copy.copy(cfg)
    task_cfg.task = task_name
    task_cfg.task_title = task_name.replace("-", " ").title()
    task_cfg.multitask = False
    task_cfg.tasks = [task_name]
    return task_cfg


def td_initial(obs: np.ndarray, action_shape: tuple[int, ...], task_idx: int) -> TensorDict:
    return TensorDict(
        {
            "obs": torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
            "action": torch.full(action_shape, float("nan"), dtype=torch.float32).unsqueeze(0),
            "reward": torch.tensor([float("nan")], dtype=torch.float32),
            "terminated": torch.tensor([float("nan")], dtype=torch.float32),
            "task": torch.tensor([task_idx], dtype=torch.long),
        },
        batch_size=(1,),
    )


def td_step(obs: np.ndarray, action: np.ndarray, reward: float, terminated: bool, task_idx: int) -> TensorDict:
    return TensorDict(
        {
            "obs": torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
            "action": torch.as_tensor(action, dtype=torch.float32).unsqueeze(0),
            "reward": torch.tensor([reward], dtype=torch.float32),
            "terminated": torch.tensor([float(terminated)], dtype=torch.float32),
            "task": torch.tensor([task_idx], dtype=torch.long),
        },
        batch_size=(1,),
    )


class GymnasiumCompatWrapper(gym.Wrapper):
    """Adapt the repo's old-style env wrappers to Gymnasium's reset/step API."""

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            try:
                self.env.action_space.seed(seed)
                self.env.observation_space.seed(seed)
            except Exception:
                pass
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = bool(info.get("terminated", done))
        truncated = bool(done and not terminated)
        return obs, float(reward), terminated, truncated, info


class DatasetRecorderWrapper(gym.Wrapper):
    """Record completed SB3 environment episodes as TensorDict trajectories."""

    def __init__(
        self,
        env,
        output_dir: Path,
        task_idx: int,
        task_name: str,
        shard_episodes: int,
        flush_every_episodes: int,
    ):
        super().__init__(env)
        self.output_dir = output_dir
        self.task_idx = task_idx
        self.task_name = task_name
        self.shard_episodes = shard_episodes
        self.flush_every_episodes = flush_every_episodes
        self.pending_by_length = {}
        self.shard_idx_by_length = {}
        self.episode = []
        self.num_episodes = 0
        self.num_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.shards = []
        self._last_obs = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.episode = [td_initial(obs, self.action_space.shape, self.task_idx)]
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode.append(td_step(obs, action, reward, terminated, self.task_idx))
        self.num_steps += 1
        if terminated or truncated:
            episode = torch.cat(self.episode)
            episode_length = int(episode.shape[0])
            pending = self.pending_by_length.setdefault(episode_length, [])
            pending.append(episode)
            self.num_episodes += 1
            self.episode_rewards.append(float(sum(episode["reward"][1:].tolist())))
            self.episode_lengths.append(episode_length)
            self.episode_successes.append(float(info.get("success", 0.0)))
            if len(pending) >= self.flush_every_episodes:
                self.flush(episode_length)
        return obs, reward, terminated, truncated, info

    def flush(self, episode_length: int):
        pending = self.pending_by_length.get(episode_length, [])
        if not pending:
            return None
        shard_idx = self.shard_idx_by_length.get(episode_length, 0)
        shard = torch.stack(pending, dim=0)
        path = self.output_dir / f"shard_task{self.task_idx:02d}_len{episode_length:05d}_{shard_idx:05d}.pt"
        torch.save(shard, path)
        self.pending_by_length[episode_length] = []
        self.shard_idx_by_length[episode_length] = shard_idx + 1
        metadata = {
            "file": path.name,
            "task_idx": self.task_idx,
            "task": self.task_name,
            "episodes": int(shard.shape[0]),
            "episode_length": int(shard.shape[1]),
        }
        self.shards.append(metadata)
        print(
            f"Saved {metadata['episodes']} episode(s) for task {self.task_idx} "
            f"length {episode_length} -> {path.name}",
            flush=True,
        )
        return metadata

    def flush_all(self):
        for episode_length in list(self.pending_by_length.keys()):
            self.flush(episode_length)

    def stats(self):
        return {
            "task_idx": self.task_idx,
            "saved": self.num_episodes,
            "steps": self.num_steps,
            "mean_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else None,
            "mean_length": float(np.mean(self.episode_lengths)) if self.episode_lengths else None,
            "mean_success": float(np.mean(self.episode_successes)) if self.episode_successes else None,
        }


class FlushRecorderCallback(BaseCallback):
    def __init__(self, recorder: DatasetRecorderWrapper):
        super().__init__()
        self.recorder = recorder

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        self.recorder.flush_all()


def make_sac_env(
    cfg,
    task_idx: int,
    task_name: str,
    output_dir: Path,
    shard_episodes: int,
    flush_every_episodes: int,
):
    task_cfg = single_task_cfg(cfg, task_name)
    env = make_mujoco_env(task_cfg)
    env = GymnasiumCompatWrapper(env)
    env = DatasetRecorderWrapper(env, output_dir, task_idx, task_name, shard_episodes, flush_every_episodes)
    env = Monitor(env)
    return env


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    flush_every_episodes = args.flush_every_episodes or args.shard_episodes
    if flush_every_episodes <= 0:
        raise ValueError("--flush-every-episodes must be positive.")

    cfg = load_cfg(args.overrides, args.seed)
    set_seed(cfg.seed)
    task_indices = selected_task_indices(cfg, args.task_indices)

    manifest = {
        "task": cfg.task,
        "tasks": list(cfg.tasks),
        "selected_task_indices": task_indices,
        "sac": {
            "total_timesteps": args.total_timesteps,
            "learning_starts": args.learning_starts,
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "shard_episodes": args.shard_episodes,
            "flush_every_episodes": flush_every_episodes,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "tau": args.tau,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
        },
        "shards": [],
        "per_task": {},
    }

    for task_idx in task_indices:
        task_name = cfg.tasks[task_idx]
        print(f"Training SAC behavior policy for task {task_idx}: {task_name}")
        env = make_sac_env(cfg, task_idx, task_name, output_dir, args.shard_episodes, flush_every_episodes)
        recorder = env.env
        callback = FlushRecorderCallback(recorder)
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            verbose=1,
            seed=int(cfg.seed) + task_idx,
            device="auto",
        )
        model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=args.progress_bar)
        if args.save_model:
            model.save(output_dir / f"sac_task{task_idx:02d}_{task_name}")
        manifest["shards"].extend(recorder.shards)
        manifest["per_task"][task_name] = recorder.stats()
        env.close()

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved SAC dataset to {output_dir}")


if __name__ == "__main__":
    main()
