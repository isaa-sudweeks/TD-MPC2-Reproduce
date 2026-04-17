# Copied from https://github.com/edbeeching/td_mpc2/blob/main/common/logger.py

# TODO: Consider finding out what this is and why it is important.
import dataclasses
import os
import datetime
import re

import numpy as np
import pandas as pd
from termcolor import colored

from common import TASK_SET


CONSOLE_FORMAT = [
	("iteration", "I", "int"),
	("episode", "E", "int"),
	("step", "I", "int"),
	("episode_reward", "R", "float"),
	("episode_success", "S", "float"),
	("elapsed_time", "T", "time"),
]

CAT_TO_COLOR = {
	"pretrain": "yellow",
	"train": "blue",
	"eval": "green",
}


def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def print_run(cfg):
	"""
	Pretty-printing of current run information.
	Logger calls this method at initialization.
	"""
	prefix, color, attrs = "  ", "green", ["bold"]

	def _limstr(s, maxlen=36):
		return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

	def _pprint(k, v):
		print(
			prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs), _limstr(v)
		)

	observations  = ", ".join([str(v) for v in cfg.obs_shape.values()])
	kvs = [
		("task", cfg.task_title),
		("steps", f"{int(cfg.steps):,}"),
		("observations", observations),
		("actions", cfg.action_dim),
		("experiment", cfg.exp_name),
	]
	w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
	div = "-" * w
	print(div)
	for k, v in kvs:
		_pprint(k, v)
	print(div)


def cfg_to_group(cfg, return_list=False):
	"""
	Return a wandb-safe group name for logging.
	Optionally returns group name as list.
	"""
	lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
	return lst if return_list else "-".join(lst)


def wandb_run_name(cfg):
	"""
	Construct a wandb run name for both standard runs and Optuna trials.
	"""
	if cfg.get("optuna_trial_number", None) is None:
		return str(cfg.seed)
	return f"trial-{int(cfg.optuna_trial_number):04d}-seed-{cfg.seed}"


def wandb_tags(cfg):
	"""
	Construct wandb tags, including Optuna metadata when present.
	"""
	tags = cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"]
	study_name = cfg.get("optuna_study_name", None)
	trial_number = cfg.get("optuna_trial_number", None)
	if study_name:
		tags.append("optuna")
		tags.append(f"study:{study_name}")
	if trial_number is not None:
		tags.append(f"trial:{int(trial_number)}")
	return tags


class VideoRecorder:
	"""Utility class for logging evaluation videos."""

	def __init__(self, cfg, wandb, fps=15):
		self.cfg = cfg
		self._save_dir = make_dir(cfg.work_dir / 'eval_video')
		self._wandb = wandb
		self.fps = fps
		self.frames = []
		self.enabled = False

	def init(self, env, enabled=True):
		self.frames = []
		self.enabled = self._save_dir and self._wandb and enabled
		self.record(env)

	def record(self, env):
		if self.enabled:
			self.frames.append(env.render())

	def save(self, step, key='videos/eval_video'):
		if self.enabled and len(self.frames) > 0:
			frames = np.stack(self.frames)
			return self._wandb.log(
				{key: self._wandb.Video(frames.transpose(0, 3, 1, 2), fps=self.fps, format='mp4')}, step=step
			)


class Logger:
	"""Primary logging object. Logs either locally or using wandb."""

	def __init__(self, cfg):
		self.cfg = cfg
		self._log_dir = make_dir(cfg.work_dir)
		self._model_dir = make_dir(self._log_dir / "models")
		self._save_csv = cfg.save_csv
		self._save_agent = cfg.save_agent
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		self._eval = []
		print_run(cfg)
		self.project = cfg.get("wandb_project", "none")
		self.entity = cfg.get("wandb_entity", "none")
		if not cfg.enable_wandb or self.project == "none" or self.entity == "none":
			print(colored("Wandb disabled.", "blue", attrs=["bold"]))
			cfg.save_agent = False
			cfg.save_video = False
			self._wandb = None
			self._video = None
			return
		os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
		wandb_mode = cfg.get("wandb_mode", "offline")
		import wandb

		wandb.init(
			project=self.project,
			entity=self.entity,
			mode=wandb_mode,
			name=wandb_run_name(cfg),
			group=self._group,
			tags=wandb_tags(cfg),
			dir=self._log_dir,
			config=dataclasses.asdict(cfg),
		)
		print(colored(f"Wandb initialized in {wandb_mode} mode.", "blue", attrs=["bold"]))
		if wandb_mode == "offline":
			print(colored(f"Sync with: wandb sync {self._log_dir / 'wandb' / 'offline-run-*'}", "blue"))
		self._wandb = wandb
		self._video = (
			VideoRecorder(cfg, self._wandb)
			if self._wandb and cfg.save_video
			else None
		)

	@property
	def video(self):
		return self._video

	@property
	def model_dir(self):
		return self._model_dir

	def save_agent(self, agent=None, identifier='final'):
		if self._save_agent and agent:
			fp = self._model_dir / f'{str(identifier)}.pt'
			agent.save(fp)
			if self._wandb:
				artifact = self._wandb.Artifact(
					self._group + '-' + str(self._seed) + '-' + str(identifier),
					type='model',
				)
				artifact.add_file(fp)
				self._wandb.log_artifact(artifact)

	def finish(self, agent=None):
		try:
			self.save_agent(agent)
		except Exception as e:
			print(colored(f"Failed to save model: {e}", "red"))
		if self._wandb:
			self._wandb.config.update(
				{
					"optuna_trial_state": self.cfg.get("optuna_trial_state", None),
					"optuna_trial_params": self.cfg.get("optuna_trial_params", None),
				},
				allow_val_change=True,
			)
			self._wandb.finish()

	def _format(self, key, value, ty):
		if ty == "int":
			return f'{colored(key+":", "blue")} {int(value):,}'
		elif ty == "float":
			return f'{colored(key+":", "blue")} {value:.01f}'
		elif ty == "time":
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "blue")} {value}'
		else:
			raise f"invalid log format type: {ty}"

	def _print(self, d, category):
		category = colored(category, CAT_TO_COLOR[category])
		pieces = [f" {category:<14}"]
		for k, disp_k, ty in CONSOLE_FORMAT:
			if k in d:
				pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
		print("   ".join(pieces))

	def pprint_multitask(self, d, cfg):
		"""Pretty-print evaluation metrics for multi-task training."""
		print(colored(f'Evaluated agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
		dmcontrol_reward = []
		metaworld_reward = []
		metaworld_success = []
		for k, v in d.items():
			if '+' not in k:
				continue
			task = k.split('+')[1]
			if k.startswith('episode_reward'):
				print(colored(f'  {task:<32}\tR: {v:.01f}', 'yellow'))
			elif k.startswith('episode_success'):
				print(colored(f'  {task:<32}\tS: {v:.02f}', 'yellow'))
		
		# DMControl and Meta-World specific aggregation:
		for k, v in d.items():
			if '+' not in k:
				continue
			task = k.split('+')[1]
			if task in TASK_SET.get('mt30', []) and k.startswith('episode_reward'):
				dmcontrol_reward.append(v)
			elif task in TASK_SET.get('mt80', []) and task not in TASK_SET.get('mt30', []):
				if k.startswith('episode_reward'):
					metaworld_reward.append(v)
				elif k.startswith('episode_success'):
					metaworld_success.append(v)
		
		if dmcontrol_reward:
			dmcontrol_reward = np.nanmean(dmcontrol_reward)
			d['episode_reward+avg_dmcontrol'] = dmcontrol_reward
			print(colored(f'  {"dmcontrol":<32}\tR: {dmcontrol_reward:.01f}', 'yellow', attrs=['bold']))
		if metaworld_reward:
			metaworld_reward = np.nanmean(metaworld_reward)
			metaworld_success = np.nanmean(metaworld_success)
			d['episode_reward+avg_metaworld'] = metaworld_reward
			d['episode_success+avg_metaworld'] = metaworld_success
			print(colored(f'  {"metaworld":<32}\tR: {metaworld_reward:.01f}', 'yellow', attrs=['bold']))
			print(colored(f'  {"metaworld":<32}\tS: {metaworld_success:.02f}', 'yellow', attrs=['bold']))

	def log(self, d, category="train"):
		assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
		if self._wandb:
			if category in {"train", "eval"}:
				xkey = "step"
			elif category == "pretrain":
				xkey = "iteration"
			_d = dict()
			for k, v in d.items():
				_d[category + "/" + k] = v
			self._wandb.log(_d, step=d[xkey])
		if category == "eval" and self._save_csv:
			keys = ["step", "episode_reward"]
			self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
			pd.DataFrame(np.array(self._eval)).to_csv(
				self._log_dir / "eval.csv", header=keys, index=None
			)
		self._print(d, category)
