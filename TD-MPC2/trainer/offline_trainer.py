import os
from copy import deepcopy
from time import time 
from pathlib import Path 
from glob import glob #TODO I don't know what this lib does

import numpy as np 
import torch 
from tqdm import tqdm 

from common.buffer import Buffer 
from trainer.base import Trainer 

class OfflineTrainer(Trainer):
    """
    Trainer class for multi-task offline TD-MPC2 training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_time = time()

    def eval(self):
        """
        Evaluate a TD-MPC2 agent 
        """
        results = dict()
        for task_idx in tqdm(range(len(self.cfg.tasks)), desc="Evaluating"):
            ep_rewards, ep_successes = [], []
            for _ in range(self.cfg.eval_episodes):
                obs, done, ep_reward, t = self.env.reset(task_idx), False, 0, 0
                while not done:
                    if getattr(self.cfg, 'device', 'cuda') == 'cuda':
                        torch.compiler.cudagraph_mark_step_begin()
                    action = self.agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
                    obs, reward, done, info = self.env.step(action)
                    ep_reward += reward
                    t += 1
                ep_rewards.append(ep_reward)
                ep_successes.append(info['success'])

            results.update({
                f'episode_reward+{self.cfg.tasks[task_idx]}' : np.nanmean(ep_rewards),
                f'episode_success+{self.cfg.tasks[task_idx]}' : np.nanmean(ep_successes),
            })
        if len(self.cfg.tasks) > 0:
            results['episode_reward'] = np.nanmean([
                results[f'episode_reward+{task}'] for task in self.cfg.tasks
            ])
            results['episode_success'] = np.nanmean([
                results[f'episode_success+{task}'] for task in self.cfg.tasks
            ])
        return results

    def _load_dataset(self):
        """
        Load offline dataset from disk for offline training.
        """
        fp = Path(os.path.join(self.cfg.data_dir, '*.pt'))
        fps = sorted(glob(str(fp)))
        assert len(fps) > 0, f'No data found at {fp}'
        print(f'Found {len(fps)} files at {fp}')
        if self.cfg.task in {'mt30', 'mt80'} and len(fps) < (20 if self.cfg.task == 'mt80' else 4):
            print(f'WARNING: expected 20 files for mt80 task set, 4 files for mt30 task set, found {len(fps)} files.')

        first_td = torch.load(fps[0], weights_only=False)

        # Create buffer for sampling TODO: Are they not just hard setting these in this code here?
        _cfg = deepcopy(self.cfg)
        if self.cfg.task == 'mt80':
            _cfg.episode_length = 101
            _cfg.buffer_size = 550_450_000
        elif self.cfg.task == 'mt30':
            _cfg.episode_length = 501
            _cfg.buffer_size = 345_690_000
        else:
            _cfg.episode_length = first_td.shape[1]
            _cfg.buffer_size = 0
            for fp in fps:
                td = torch.load(fp, weights_only=False)
                _cfg.buffer_size += td.shape[0] * td.shape[1]
        _cfg.steps = _cfg.buffer_size
        self.buffer = Buffer(_cfg)
        for fp in tqdm(fps, desc='Loading dataset'):
            td = torch.load(fp, weights_only=False)
            if self.cfg.task in {'mt30', 'mt80'}:
                assert td.shape[1] == _cfg.episode_length, \
                    f'Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, ' \
                        f' please double check your config.'

            self.buffer.load(td)
        if self.cfg.task in {'mt30', 'mt80'}:
            expected_episodes = _cfg.buffer_size // _cfg.episode_length
            if self.buffer.num_eps != expected_episodes:
                print(f'WARNING: expected {expected_episodes} episodes, found {self.buffer.num_eps} episodes.')
        
    def train(self):
        """
        Train the TD-MPC2 agent.
        """

        assert self.cfg.multitask, f'Task {self.cfg.task} must be multitask for offline training.'
        self._load_dataset()
        print(f'Training for {self.cfg.steps} steps.')
        metrics = {}
        for i in range(self.cfg.steps):
            # Update agent 
            train_metrics = self.agent.update(self.buffer)

            # Evaluate agent periodically 
            if i % self.cfg.eval_freq == 0 or i % 10_000 == 0:
                metrics = {
                    'iteration': i,
                    'elapsed_time' : time() - self._start_time,
                }
                metrics.update(train_metrics)
                if i % self.cfg.eval_freq == 0:
                    metrics.update(self.eval())
                    self.logger.pprint_multitask(metrics, self.cfg)
                    self.report_eval_metrics(metrics, i)
                    if i > 0:
                        self.logger.save_agent(self.agent, identifier=f'{i}')
                    self.logger.log(metrics, 'pretrain')
        self.logger.finish(self.agent)
        return self._best_eval_metrics
        
