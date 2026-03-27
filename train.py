import gymnasium as gym
from stable_baselines3 import SAC
from dm_control import suite

# We use the shimmy package which provides official Gymnasium compatibility for DeepMind Control environments.
# To install: pip install shimmy[dm-control] gymnasium stable-baselines3
from shimmy.dm_control_compatibility import DmControlCompatibilityV0

def main():
    # 1. Load the underlying dm_control environment natively
    domain_name = "walker"
    task_name = "walk"
    print(f"Loading dm_control environment: {domain_name} - {task_name}")
    dm_env = suite.load(domain_name=domain_name, task_name=task_name)

    # 2. Wrap the dm_control environment so it conforms to the Gymnasium API
    #    This handles converting observation dictionaries to generic Box spaces,
    #    and maps the DeepMind continuous actions to Gymnasium Box spaces.
    from gymnasium.wrappers import FlattenObservation
    env = DmControlCompatibilityV0(dm_env, render_mode="human")
    env = FlattenObservation(env)

    # 3. Initialize the Stable Baselines 3 Soft Actor-Critic (SAC) algorithm.
    print("Initializing SAC model...")
    model = SAC("MlpPolicy", env, verbose=1)

    # 4. Train the model with a custom rendering callback
    from stable_baselines3.common.callbacks import BaseCallback
    class RenderCallback(BaseCallback):
        def _on_step(self):
            # Render the environment at each step
            self.training_env.render()
            return True

    print("Starting training...")
    model.learn(total_timesteps=10_000, log_interval=4, callback=RenderCallback())

    # 5. Save the trained model
    save_path = f"sac_{domain_name}_{task_name}"
    model.save(save_path)
    print(f"Training finished! Model saved to {save_path}.zip")

    # Clean up environment
    env.close()

if __name__ == "__main__":
    main()
