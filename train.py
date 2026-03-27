import gymnasium as gym
from stable_baselines3 import SAC
from dm_control import suite
from shimmy.dm_control_compatibility import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.callbacks import BaseCallback

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from wandb.integration.sb3 import WandbCallback
import os

class RenderCallback(BaseCallback):
    def _on_step(self):
        # Render the environment at each step
        self.training_env.render()
        return True

@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(cfg: DictConfig):
    # 1. Initialize wandb
    print("Initializing Weights & Biases...")
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=True,  # Automatically syncs Stable Baselines 3 tensorboard logs to wandb
        monitor_gym=True,       # Auto-upload videos of agents
        save_code=True,         # Save the training code to wandb
    )

    domain_name = cfg.env.domain_name
    task_name = cfg.env.task_name
    render_mode = cfg.env.render_mode if cfg.env.render_mode != "null" else None

    # 2. Load the underlying dm_control environment natively
    print(f"Loading dm_control environment: {domain_name} - {task_name}")
    dm_env = suite.load(domain_name=domain_name, task_name=task_name)

    # 3. Wrap the dm_control environment so it conforms to the Gymnasium API
    env = DmControlCompatibilityV0(dm_env, render_mode=render_mode)
    env = FlattenObservation(env)

    # 4. Initialize the Stable Baselines 3 Soft Actor-Critic (SAC) algorithm.
    print("Initializing SAC model...")
    # Add a tensorboard_log directory so the tensorboard metrics can be piped to WandB
    log_dir = f"runs/{run.id}"
    os.makedirs(log_dir, exist_ok=True)
    model = SAC("MlpPolicy", env, verbose=cfg.training.verbose, tensorboard_log=log_dir)

    # 5. Set up callbacks
    callbacks = [
        WandbCallback(
            gradient_save_freq=1000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        )
    ]
    if render_mode == "human":
        print("Rendering mechanism active via RenderCallback!")
        callbacks.append(RenderCallback())

    # 6. Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=cfg.training.total_timesteps, 
        log_interval=cfg.training.log_interval,
        callback=callbacks
    )

    # 7. Save the trained model
    save_path = f"sac_{domain_name}_{task_name}"
    model.save(save_path)
    print(f"Training finished! Model saved to {save_path}.zip")

    # Clean up environment and wandb connection
    env.close()
    run.finish()

if __name__ == "__main__":
    main()
