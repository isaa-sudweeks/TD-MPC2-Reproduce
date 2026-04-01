from copy import deepcopy


class Trainer:
    """
    Base trainer class for TD-MPC2.
    """
    def __init__(self, cfg, env, agent, buffer, logger):
        self.cfg = cfg 
        self.env = env 
        self.agent = agent 
        self.buffer = buffer 
        self.logger = logger 
        self._best_eval_metrics = None
        print('Architecture:' , self.agent.model)

    def _objective_metric_keys(self):
        if self.cfg.optimize_metric != "auto":
            return [self.cfg.optimize_metric]
        if self.cfg.multitask:
            if self.cfg.task == "mt80":
                return [
                    "episode_success+avg_metaworld",
                    "episode_reward+avg_metaworld",
                    "episode_reward+avg_dmcontrol",
                ]
            return ["episode_reward+avg_dmcontrol", "episode_reward"]
        return ["episode_reward", "episode_success"]

    def objective_value(self, metrics):
        """
        Extract the scalar objective value used for hyperparameter optimization.
        """
        for key in self._objective_metric_keys():
            value = metrics.get(key)
            if value is not None:
                return float(value), key
        raise KeyError(
            f"Unable to find optimization metric in evaluation metrics. "
            f"Tried {self._objective_metric_keys()}, got {sorted(metrics.keys())}."
        )

    def update_best_eval_metrics(self, metrics):
        """
        Track the best evaluation metrics seen during training.
        """
        candidate_value, _ = self.objective_value(metrics)
        if self._best_eval_metrics is None:
            self._best_eval_metrics = deepcopy(metrics)
            return

        best_value, _ = self.objective_value(self._best_eval_metrics)
        if self.cfg.optimize_direction == "minimize":
            is_better = candidate_value < best_value
        else:
            is_better = candidate_value > best_value
        if is_better:
            self._best_eval_metrics = deepcopy(metrics)

    def best_objective(self):
        """
        Return the best objective value and metric key seen during training.
        """
        if self._best_eval_metrics is None:
            raise RuntimeError("Training finished without producing any evaluation metrics.")
        return self.objective_value(self._best_eval_metrics)

    def eval(self):
        """
        Evaluate a TD-MPC2 agent.
        """
        raise NotImplementedError

    def train(self):
        """
        Train a TD-MPC2 agent.
        """
        raise NotImplementedError

        
