from copy import deepcopy 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from common import layers, math, init 
from tensordict import TensorDict 
from tensordict.nn import TensorDictParams 

class WorldModel(nn.Module):
    """
    TD-MPC2 implicit world model.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.multitask:
            max_norm = None if torch.device(getattr(cfg, 'device', 'cuda')).type == 'mps' else 1
            self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=max_norm) # TODO: Figure out what the nn.Embedding does '
            self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim)) # TODO: Figure out what this does
            for i in range(len(cfg.tasks)):
                self._action_masks[i, :cfg.action_dims[i]] = 1.
            
            
        # Initialize all the required networks 
        self._encoder = layers.enc(cfg) # This could be made a GNN for our situation maybe
        self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim +cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act = layers.SimNorm(cfg))
        self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins,1))
        self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
        self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
        self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(int(cfg.num_q))])
            
        # Custom Weight Initialization
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer("log_std_def", torch.tensor(cfg.log_std_max) - self.log_std_min)
        self.init()

    def _task_indices(self, task, device):
        if torch.device(device).type == 'mps':
            device = 'cpu'
        if isinstance(task, int):
            task = torch.tensor([task], device=device, dtype=torch.long)
        else:
            task = task.to(device=device, dtype=torch.long)
        if (task < 0).any() or (task >= len(self.cfg.tasks)).any():
            raise ValueError(f"Invalid task ids: {task.tolist()}")
        return task

    def _task_one_hot(self, task, device, dtype):
        task = self._task_indices(task, device)
        return F.one_hot(task, num_classes=len(self.cfg.tasks)).to(device=device, dtype=dtype)

    def action_mask(self, task, device, dtype):
        if torch.device(device).type == 'mps':
            return self._task_one_hot(task, device, dtype) @ self._action_masks.to(dtype=dtype)
        task = self._task_indices(task, device)
        return self._action_masks[task].to(dtype=dtype)
    
    def init(self):
        # Create params
        self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True) # TODO I have no idea what this is or why it is needed
        self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

        # Create modules 
        with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
            self._detach_Qs = deepcopy(self._Qs)
            self._target_Qs = deepcopy(self._Qs)
        
        # Assign params to modules 
        # We do this strange assignment to avoid having duplicated tensors in the state-dict
        delattr(self._detach_Qs, "params")
        self._detach_Qs.__dict__["params"] = self._detach_Qs_params
        delattr(self._target_Qs, "params")
        self._target_Qs.__dict__["params"] = self._target_Qs_params

    def __repr__(self):
        repr = "TD-MPC2 World Model\n"
        modules = ['Encoder', 'Dynamics', 'Reward', 'Termination', 'Policy', 'Critics']
        for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._termination, self._pi, self._Qs]):
            if m == self._termination and not self.cfg.episodic:
                continue 
            repr += f"{modules[i]}: {m}\n"
        repr += "Learnable parameters: {:,}".format(self.total_params)
        return repr 

    @property 
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.init()
        return self

    def train(self, mode=True):
        """
        Overriding 'train' method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self 

    def soft_update_target_Q(self):
        """
        Soft update target Q-networks using Polyak averaging.
        """
        self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

    def task_emb(self, x, task):
        """Continuous task embedding for multi-task experiments.
        Retrieves the task embedding for a given task ID 'task'
        and concatenates it to the input 'x'.
        """
        # TODO: I don't really think I need this because I am just doing one task for now 
        if torch.device(x.device).type == 'mps':
            emb = self._task_one_hot(task, x.device, self._task_emb.weight.dtype) @ F.normalize(self._task_emb.weight, p=2, dim=1)
        else:
            task = self._task_indices(task, x.device)
            emb = self._task_emb(task)
        if x.ndim ==3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs, task):
        """
        Encodes an observation into its laten representation.
        This implementation assumes a single state-based observation
        """
        if self.cfg.multitask:
            obs = self.task_emb(obs, task)
        if self.cfg.obs == 'rgb' and obs.ndim == 5:
            return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])

        return self._encoder[self.cfg.obs](obs)

    def next(self, z, a, task):
        """
        Predicts the next latent state given the current latent state and action.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._dynamics(z)
    
    def reward(self, z, a, task):
        """
        Predicts single step reward.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def termination(self, z, task, unnormalized=False):
        """
        Predicts termination signal.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        if unnormalized:
            return self._termination(z)
        return torch.sigmoid(self._termination(z))

    def pi(self, z, task):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with mean and (log) std predicted by a NN.
        """

        # TODO: Figure this out why does it predict the Gaussian distribution rather than predict it directly
        if self.cfg.multitask:
            task = self._task_indices(task, z.device)
            z = self.task_emb(z, task)

        # Gaussian policy prior 
        mean, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_def)
        eps = torch.randn_like(mean)

        if self.cfg.multitask: # mask out unused action dimensions 
            action_mask = self.action_mask(task, mean.device, mean.dtype)
            mean = mean * action_mask # This masking stuff I have no idea what is happening
            log_std = log_std * action_mask
            action_dims = action_mask.sum(-1, keepdim=True)
        else: # No masking 
            action_dims = None 

        log_prob = math.gaussian_logprob(eps, log_std)

        # Scale log probability by action dimensions 
        size = eps.shape[-1] if action_dims is None else action_dims 
        scaled_log_prob = log_prob * size

        # Reparameterization trick 
        action = mean + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = TensorDict({
            "mean": mean,
            "log_std" : log_std,
            "action_prob": 1.,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        })
        return action, info

    def Q(self, z, a, task, return_type="min", target=False, detach=False):
        """
        Predict state-action value.
        'return_type' can be one of the following: ['min', 'avg', 'all']:
            'min': returns the minimum Q-value across all Q-networks
            'avg': returns the average Q-value across all Q-networks
            'all': returns all Q-values
        'target' determines whether to use the target Q-networks
        'detach' determines whether to detach the Q-networks
        """
        assert return_type in {'min', 'avg', 'all'}

        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        
        out = qnet(z)

        if return_type == 'all':
            return out
        
        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q = math.two_hot_inv(out[qidx], self.cfg)
        if return_type == 'min':
            return Q.min(0).values 
        return Q.sum(0) / 2
        
            

    
