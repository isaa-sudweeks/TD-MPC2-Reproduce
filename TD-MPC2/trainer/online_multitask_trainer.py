from time import time 
import numpy as np 
import torch 
from tqdm import tqdm
from tensordict.tensordict import TensorDict 
from trainer.base import Trainer 

class OnlineMultitaskTrainer(Trainer):
    """
    Trainer class for multi-task online TD-MPC2 training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0 
        self._ep_idx = 0 
        self._start_time = time() 

    def common_metrics(self):
        elapsed_time = time() - self._start_time 
        return dict(
            step= self._step,
            episode= self._ep_idx,
            elapsed_time= elapsed_time,
            steps_per_sec= self._step / elapsed_time if elapsed_time > 0 else 0,
        )
    
    def eval(self):
        results = dict()
        for task_idx in tqdm(range(len(self.cfg.tasks)), desc="Evaluating"):
            ep_rewards, ep_successes, ep_lengths = [], [], []
            for i in range(self.cfg.eval_episodes):
                obs, done, ep_reward, t = self.env.reset(task_idx=task_idx), False, 0, 0
                if self.cfg.save_video:
                    self.logger.video.init(self.env, enabled=(i==0))
                while not done:
                    if getattr(self.cfg, 'device', 'cuda') == 'cuda':
                        torch.compiler.cudagraph_mark_step_begin()
                    action = self.agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
                    obs, reward, done, info = self.env.step(action)
                    ep_reward += reward
                    t += 1
                    if self.cfg.save_video:
                        self.logger.video.record(self.env)
                ep_rewards.append(ep_reward)
                ep_successes.append(info['success'])
                ep_lengths.append(t)
                if self.cfg.save_video:
                    self.logger.video.save(f'{self._step}_{self.cfg.tasks[task_idx]}')
            results.update({
                f'episode_reward+{self.cfg.tasks[task_idx]}' : np.nanmean(ep_rewards),
                f'episode_success+{self.cfg.tasks[task_idx]}' : np.nanmean(ep_successes),
            })
        return results
    
    def to_td(self, obs, action=None, reward=None, terminated=None, task_idx=None):
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device='cpu').unsqueeze(0)
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act(), float('nan'))
        if reward is None:
            reward = torch.tensor(float('nan'))
        if terminated is None:
            terminated = torch.tensor(float('nan'))
        td = TensorDict(
            obs=obs,
            action = action.unsqueeze(0),
            reward = reward.unsqueeze(0),
            terminated = terminated.unsqueeze(0),
            batch_size=(1,),
        )
        if task_idx is not None:
            td['task'] = torch.tensor([task_idx])
        return td

    def train(self):
        train_metrics, done, eval_next = {}, True, False 
        task_idx = 0
        while self._step <= self.cfg.steps:
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True 
            
            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, 'eval')
                    self.report_eval_metrics(eval_metrics, self._step)
                    eval_next = False
                if self._step > 0:
                    train_metrics.update(
                        episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
                        episode_success = info['success'],
                        episode_length = len(self._tds),
                        episode_terminated=info['terminated'])
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, 'train')
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))

                task_idx = np.random.randint(len(self.cfg.tasks))
                obs = self.env.reset(task_idx=task_idx)
                self._tds = [self.to_td(obs, task_idx=task_idx)]
            
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds)==1, task=task_idx)
            else:
                action = self.env.rand_act()
            obs, reward, done, info = self.env.step(action)
            terminated = info['terminated'] if self.cfg.episodic else torch.tensor(0.0)
            self._tds.append(self.to_td(obs, action, reward, terminated, task_idx))

            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print('Pretraining agent on seed data...')
                else:
                    num_updates = 1
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

            self._step += 1
        self.logger.finish(self.agent)
        return self._best_eval_metrics
