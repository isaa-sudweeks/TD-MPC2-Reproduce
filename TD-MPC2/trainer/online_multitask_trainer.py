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
        import concurrent.futures
        results = dict()
        num_tasks = len(self.cfg.tasks)
         
        # Run sequentially if video is enabled because VideoRecorder only tracks 1 env at a time
        if self.cfg.save_video:
            for task_idx in tqdm(range(len(self.cfg.tasks)), desc="Evaluating (Sequential for Video)"):
                ep_rewards, ep_successes, ep_lengths = [], [], []
                for i in range(int(self.cfg.eval_episodes)):
                    obs, done, ep_reward, t = self.env.reset(task_idx=task_idx), False, 0, 0
                    self.logger.video.init(self.env, enabled=(i==0))
                    while not done:
                        if getattr(self.cfg, 'device', 'cuda') == 'cuda':
                            torch.compiler.cudagraph_mark_step_begin()
                        action = self.agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
                        obs, reward, done, info = self.env.step(action)
                        ep_reward += reward
                        t += 1
                        self.logger.video.record(self.env)
                    ep_rewards.append(ep_reward)
                    ep_successes.append(info['success'])
                    ep_lengths.append(t)
                    self.logger.video.save(self._step, key=f'videos/eval_video_{self.cfg.tasks[task_idx]}')
                results.update({
                    f'episode_reward+{self.cfg.tasks[task_idx]}' : np.nanmean(ep_rewards),
                    f'episode_success+{self.cfg.tasks[task_idx]}' : np.nanmean(ep_successes),
                })
        else:
            ep_rewards = [[] for _ in range(num_tasks)]
            ep_successes = [[] for _ in range(num_tasks)]
            print("Evaluating 4 tasks in parallel...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_tasks) as executor:
                for _ in range(int(self.cfg.eval_episodes)):
                    obs = [self.env.envs[task_idx].reset() for task_idx in range(num_tasks)]
                    dones = [False] * num_tasks
                    ep_reward = [0.0] * num_tasks
                    t = [0] * num_tasks
                    infos = [None] * num_tasks
                    prev_means = [torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.agent.device) for _ in range(num_tasks)]
                    
                    while not all(dones):
                        actions = []
                        for task_idx in range(num_tasks):
                            if dones[task_idx]:
                                actions.append(None)
                                continue
                            
                            if getattr(self.cfg, 'device', 'cuda') == 'cuda':
                                torch.compiler.cudagraph_mark_step_begin()
                                
                            self.agent._prev_mean.copy_(prev_means[task_idx])
                            action = self.agent.act(obs[task_idx], t0=t[task_idx]==0, eval_mode=True, task=task_idx)
                            prev_means[task_idx].copy_(self.agent._prev_mean)
                            actions.append(action)
                        
                        futures = []
                        for task_idx in range(num_tasks):
                            if not dones[task_idx]:
                                futures.append(executor.submit(self.env.envs[task_idx].step, actions[task_idx]))
                            else:
                                futures.append(None)
                                
                        for task_idx in range(num_tasks):
                            if futures[task_idx] is not None:
                                _obs, _reward, _done, _info = futures[task_idx].result()
                                obs[task_idx] = _obs
                                ep_reward[task_idx] += _reward
                                t[task_idx] += 1
                                infos[task_idx] = _info
                                dones[task_idx] = _done
                                
                    for task_idx in range(num_tasks):
                        ep_rewards[task_idx].append(ep_reward[task_idx])
                        ep_successes[task_idx].append(infos[task_idx]['success'])
            
            for task_idx in range(num_tasks):
                results.update({
                    f'episode_reward+{self.cfg.tasks[task_idx]}' : np.nanmean(ep_rewards[task_idx]),
                    f'episode_success+{self.cfg.tasks[task_idx]}' : np.nanmean(ep_successes[task_idx]),
                })
        
        if len(self.cfg.tasks) > 0:
            results['episode_reward'] = np.nanmean([results[f'episode_reward+{self.cfg.tasks[i]}'] for i in range(len(self.cfg.tasks))])
            results['episode_success'] = np.nanmean([results[f'episode_success+{self.cfg.tasks[i]}'] for i in range(len(self.cfg.tasks))])
        
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
        import concurrent.futures
        train_metrics, eval_next = {}, False 
        
        num_tasks = len(self.cfg.tasks)
        dones = [True] * num_tasks
        obs_list = [None] * num_tasks
        tds_list = [[] for _ in range(num_tasks)]
        infos_list = [None] * num_tasks
        prev_means = [torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.agent.device) for _ in range(num_tasks)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_tasks) as executor:
            while self._step <= self.cfg.steps:
                if self._step > 0 and self._step % self.cfg.eval_freq < num_tasks:
                    eval_next = True 
                
                # Handle reset and evaluation at the start of any new episode
                for i in range(num_tasks):
                    if dones[i]:
                        if eval_next and i == 0:
                            eval_metrics = self.eval()
                            eval_metrics.update(self.common_metrics())
                            self.logger.log(eval_metrics, 'eval')
                            self.report_eval_metrics(eval_metrics, self._step)
                            eval_next = False
                        if len(tds_list[i]) > 0:
                            if self._step > 0:
                                log_metrics = train_metrics.copy()
                                ep_reward = torch.tensor([td['reward'] for td in tds_list[i][1:]]).sum()
                                log_metrics.update({
                                    f'episode_reward+{self.cfg.tasks[i]}': ep_reward,
                                    f'episode_success+{self.cfg.tasks[i]}': infos_list[i]['success'],
                                    f'episode_length+{self.cfg.tasks[i]}': len(tds_list[i]),
                                    f'episode_terminated+{self.cfg.tasks[i]}': infos_list[i]['terminated'],
                                    'episode_reward': ep_reward,
                                    'episode_success': infos_list[i]['success'],
                                })
                                log_metrics.update(self.common_metrics())
                                self.logger.log(log_metrics, 'train')
                                train_metrics = {}
                                self._ep_idx = self.buffer.add(torch.cat(tds_list[i]))
                        
                        obs_list[i] = self.env.envs[i].reset()
                        tds_list[i] = [self.to_td(obs_list[i], task_idx=i)]
                        prev_means[i].zero_()
                        dones[i] = False
                
                # Act simultaneously
                actions = []
                for i in range(num_tasks):
                    if self._step > self.cfg.seed_steps:
                        self.agent._prev_mean.copy_(prev_means[i])
                        action = self.agent.act(obs_list[i], t0=(len(tds_list[i])==1), task=i)
                        prev_means[i].copy_(self.agent._prev_mean)
                    else:
                        self.env.active_env_idx = i
                        action = self.env.rand_act()
                    actions.append(action)
                
                # Step environments concurrently via ThreadPoolExecutor
                futures = [executor.submit(self.env.envs[i].step, actions[i]) for i in range(num_tasks)]
                
                for i in range(num_tasks):
                    obs, reward, done, info = futures[i].result()
                    obs_list[i], dones[i], infos_list[i] = obs, done, info
                    terminated = info['terminated'] if self.cfg.episodic else torch.tensor(0.0)
                    tds_list[i].append(self.to_td(obs, actions[i], reward, terminated, i))
    
                if self._step >= self.cfg.seed_steps:
                    if self._step - num_tasks < self.cfg.seed_steps:
                        num_updates = self.cfg.seed_steps
                        print('Pretraining agent on seed data...')
                    else:
                        num_updates = num_tasks  # maintain 1 update per env step collected
                    for _ in range(int(num_updates)):
                        _train_metrics = self.agent.update(self.buffer)
                    train_metrics.update(_train_metrics)
    
                self._step += num_tasks
                
        self.logger.finish(self.agent)
        return self._best_eval_metrics
