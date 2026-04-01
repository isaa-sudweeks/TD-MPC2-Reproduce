# TD-MPC2 Learning Notes

This document explains the parts of the codebase that were marked with learning comments, and it also gives an overview of how the whole repository fits together.

## What This Repository Is

This repository is a reproduction of TD-MPC2, a model-based reinforcement learning algorithm.

At a high level:

1. The agent collects transitions from an environment.
2. It encodes observations into a latent state.
3. It learns a world model that predicts:
   - the next latent state
   - the reward
   - optionally whether the episode terminates
4. It learns Q-functions that estimate long-term value.
5. It learns a policy that proposes good actions in latent space.
6. At action time, it can either:
   - use the policy directly, or
   - do MPC planning with the world model and Q-function

The key idea is: instead of planning in raw observation space, TD-MPC2 plans in a learned latent space.

## End-to-End Flow

### Training entry point

`TD-MPC2/train.py` is the main entry point.

- It parses config with Hydra.
- It seeds randomness.
- It creates the environment with `envs.make_env`.
- It creates:
  - `TDMPC2(cfg)` as the agent
  - `Logger(cfg)` for metrics/checkpoints
  - `Buffer(cfg)` for replay data
- It chooses the trainer:
  - `OnlineTrainer` for single-task online learning
  - `OfflineTrainer` for multi-task offline learning

### Environment side

`envs/__init__.py` chooses an environment backend.

In your current repo, the only implemented backend is `envs/mujoco.py`, which maps friendly names like `mujoco-walker` to Gymnasium environments.

The environment is then wrapped by:

- `MuJoCoWrapper`: normalizes Gymnasium outputs into the format the trainer expects.
- `Timeout`: forces a max episode length.
- `TensorWrapper`: converts numpy observations/actions to PyTorch tensors.

### Data flow side

`common/buffer.py` stores episodes and samples contiguous subsequences of length `horizon + 1`.

That shape is important:

- observations: `o_t, o_{t+1}, ..., o_{t+h}`
- actions/rewards/terminated: aligned to transitions between those observations

This lets the model train on short imagined rollouts rather than isolated one-step transitions.

### Model side

`common/world_model.py` builds the learnable components:

- `_encoder`: observation -> latent state `z`
- `_dynamics`: `(z_t, a_t)` -> `z_{t+1}`
- `_reward`: `(z_t, a_t)` -> reward prediction
- `_termination`: `z_{t+1}` -> done probability, if episodic
- `_pi`: latent state -> action distribution
- `_Qs`: ensemble of Q-networks

### Agent side

`tdmpc2.py` combines the world model with:

- planning (`_plan`)
- acting (`act`)
- policy update (`update_pi`)
- TD target construction (`_td_target`)
- full training step (`_update`)

### Logging side

`common/logger.py` prints training/eval metrics, saves models, and optionally logs to Weights & Biases.

## How TD-MPC2 Learns

Each update in `tdmpc2.py` does four main things:

1. Encode future observations into latent targets.
2. Roll the dynamics model forward from the current latent state using sampled actions.
3. Train predictions:
   - consistency loss: predicted latent vs encoded latent
   - reward loss: predicted reward vs real reward
   - value loss: predicted Q vs TD target
   - termination loss: optional done prediction
4. Update the policy to choose actions that get high Q-values while retaining some entropy.

Then it softly updates a target critic network, which stabilizes TD learning.

## Explanations For Your Learning Comments

Below, each subsection references the comment location and explains what that code is doing.

### `TD-MPC2/common/parser.py:8`

This file is about Hydra/OmegaConf config handling.

What Hydra/OmegaConf are doing here:

- Hydra loads `config.yaml` plus command-line overrides.
- OmegaConf stores that config in a flexible config object.
- `parse_cfg` normalizes and augments the config before training starts.

The important helper is `cfg_to_dataclass`.

Why convert the config to a dataclass:

- `torch.compile` dislikes highly dynamic Python objects.
- OmegaConf containers are dynamic and can trigger graph breaks.
- A plain dataclass is more static and predictable, so compiled code is less likely to deopt.

So this file is mostly a compatibility and convenience layer around config.

### `TD-MPC2/trainer/offline_trainer.py:5`

`from glob import glob` imports Python’s filename pattern matcher.

`glob("path/*.pt")` means:

- look in that directory
- return all filenames ending in `.pt`

In `_load_dataset`, it is used to find all offline dataset shards on disk before loading them.

### `TD-MPC2/trainer/offline_trainer.py:58`

You asked whether they are “hard setting these in this code here.”

Yes, they are.

This block:

- copies the config to `_cfg`
- overwrites `episode_length`
- overwrites `buffer_size`
- sets `steps = buffer_size`

Why:

- Offline training uses a fixed dataset layout.
- The buffer must match the dataset’s episode length and total number of stored transitions.
- They are forcing the replay buffer to be sized for the known mt30/mt80 datasets.

So this is not a learned parameter or inferred value. It is a practical dataset-specific override.

### `TD-MPC2/common/scale.py:4`

`RunningScale` is a running normalizer for Q-values used during policy optimization.

What it stores:

- `value`: one running scale number
- `_percentiles`: `[5, 95]`

What it does:

1. Take a batch of Q-values.
2. Estimate the 5th and 95th percentiles.
3. Compute the trimmed range: `p95 - p5`.
4. Smoothly update `self.value` with `lerp_(..., tau)`.
5. Divide later Q-values by this scale.

Why this exists:

- Raw Q magnitudes can drift during training.
- Policy loss mixes Q-values and entropy.
- If Q-values get very large, they can dominate the loss.
- Scaling them keeps the actor update numerically stable.

Why percentiles instead of min/max:

- Percentiles are less sensitive to outliers.
- One weird Q-value will not explode the scale.

Why `Buffer(...)` is used:

- In this PyTorch version, `torch.nn.Buffer` marks tensors as module buffers.
- Buffers move with `.to(device)` and appear in state dicts, but are not optimized like parameters.

### `TD-MPC2/common/layers.py:8`

`Ensemble` is a vectorized wrapper around several copies of the same network.

Why TD-MPC2 uses an ensemble:

- It has multiple Q-networks, not just one.
- Using more than one critic reduces overestimation and improves robustness.

What this code is doing:

- `from_modules(*modules, as_module=True)` packs multiple module parameter sets into a TensorDict structure.
- `deepcopy(modules[0])` creates one representative network structure.
- `torch.vmap` runs that same structure across all parameter sets in parallel.

Conceptually:

- instead of a Python loop over `num_q` networks
- it batches them into one vectorized call

So `Ensemble.forward(...)` returns Q predictions from all critic heads at once.

### `TD-MPC2/common/layers.py:36`

`SimNorm` means simplicial normalization.

What it does:

- It reshapes the feature dimension into groups of size `simnorm_dim`.
- It applies `softmax` inside each group.
- It reshapes back.

Effect:

- Each small feature group becomes nonnegative and sums to 1.

Why this can help:

- It constrains latent activations.
- It makes the latent dynamics smoother and more structured.
- It can stabilize model learning similarly to a specialized activation/normalization combo.

This is used as the activation on the encoder output and dynamics output, so the latent state lives in a more controlled space.

### `TD-MPC2/common/layers.py:64`

`Mish` is an activation function.

Formula:

- `mish(x) = x * tanh(softplus(x))`

Practical meaning:

- It is a smooth nonlinear activation, like ReLU/GELU/SiLU family.
- It preserves small negative values instead of hard-zeroing them like ReLU.
- It often gives smoother optimization in MLPs.

In this repo, `NormedLinear` means:

1. linear layer
2. layer norm
3. activation (`Mish` by default)
4. optional dropout

### `TD-MPC2/common/layers.py:94`

`enc(cfg)` builds the encoder module dictionary.

Why it returns a `ModuleDict`:

- The original TD-MPC2 supports different observation types such as `state` and `rgb`.
- Each observation type can have its own encoder.

In your current repo only `state` is implemented.

For `state`, the encoder is just an MLP:

- input size = observation dimension + task embedding dimension
- hidden layers = `enc_dim`
- output size = `latent_dim`
- final activation = `SimNorm`

So this is not doing anything exotic right now. It is just building the state encoder in a format that could later support multiple modalities.

### `TD-MPC2/common/init.py:4`

This file defines custom parameter initialization.

Why initialization matters:

- RL is very sensitive to unstable early gradients.
- Bad initialization can lead to exploding values or very slow learning.

What it does:

- `nn.Linear`: truncated normal weights with std `0.02`, zero bias
- `nn.Embedding`: uniform weights in `[-0.02, 0.02]`
- `zero_(...)`: explicitly zeroes selected parameters

Why zero some output layers:

- It makes initial reward and value predictions conservative.
- Early in training, before the model knows anything, predicting near-zero outputs is safer than large random values.

### `TD-MPC2/common/world_model.py:16`

`nn.Embedding(num_tasks, task_dim)` is a learnable lookup table.

Input:

- an integer task id like `0`, `1`, `2`

Output:

- a dense learned vector of size `task_dim`

Why multi-task TD-MPC2 needs this:

- different tasks may share observations/actions but mean different things
- the embedding tells the shared model which task it is currently solving

In single-task mode, none of this matters because `cfg.multitask` is false.

### `TD-MPC2/common/world_model.py:17`

`register_buffer("_action_masks", ...)` stores a non-trainable tensor on the module.

What `_action_masks` is for:

- In multi-task training, different tasks may have different action dimensions.
- The shared policy/Q/dynamics still use one maximum action dimension.
- The mask marks which dimensions are valid for each task.

Example:

- if one task has 4 action dims and another has 7
- the shared model may allocate 7 dims
- for the 4-dim task, the last 3 dims are masked to zero

So this is bookkeeping for variable action spaces in the multi-task setting.

### `TD-MPC2/common/world_model.py:40`

`TensorDictParams(self._Qs.params.data, no_convert=True)` creates a parameter wrapper around the critic ensemble parameters.

Why there are two special copies:

- `_detach_Qs`: used when updating the policy
- `_target_Qs`: used for TD targets

Why detached critics are needed:

- In `update_pi`, the actor should optimize against critic values without also backpropagating through the critic parameters.
- That gives gradients to the actor only.

Why target critics are needed:

- TD targets should come from a slowly moving network.
- This makes bootstrap targets less noisy and reduces instability.

Why the code looks strange:

- It is building lightweight alternate parameter sets for the same network structure without duplicating all tensors in the state dict.
- That is mostly a TensorDict performance/memory trick.

Conceptually, you can think of it as:

- online critic parameters
- detached view of critic parameters
- slowly updated target critic parameters

### `TD-MPC2/common/world_model.py:93`

Your comment is right for your current setup.

If you are only doing one task:

- `cfg.multitask` is false
- `task_emb(...)` is never used

It only matters when the same model is trained across many tasks.

### `TD-MPC2/common/world_model.py:150`

Why the policy predicts a Gaussian distribution instead of a single action directly:

Because the actor needs stochasticity during training.

If it predicted only one deterministic action:

- exploration would be weaker
- entropy regularization would not make sense
- the planner would get less diverse policy trajectories

So the policy predicts:

- `mean`
- `log_std`

Then it samples:

- `action = mean + eps * std`

Benefits:

- exploration
- entropy bonus
- differentiable sampling via the reparameterization trick

At evaluation time, the code can use the mean action instead.

### `TD-MPC2/common/math.py:5`

`soft_ce` means soft cross-entropy.

Normal cross-entropy assumes the target is one exact class.

Here the target is a soft two-hot distribution over value bins, not a single hard class.

So this function:

1. log-softmaxes the prediction logits
2. converts the scalar target into a soft target distribution with `two_hot(...)`
3. computes cross-entropy between them

Why use this for reward and value prediction:

- TD-MPC2 predicts a distribution over bins instead of a raw scalar
- that can be more stable than direct scalar regression

### `TD-MPC2/common/buffer.py:6`

This file implements the replay buffer, which is central to training.

What a replay buffer is:

- a memory of previous experience
- training samples are drawn from it repeatedly
- that breaks strong temporal correlation and improves sample efficiency

What `SliceSampler` is doing:

- It samples contiguous trajectory slices, not random independent steps.
- Each slice has length `horizon + 1`.

Why contiguous slices are necessary:

- The model learns latent rollouts over multiple steps.
- It needs `o_t, a_t, r_t, ..., o_{t+h}` in sequence.

What `LazyTensorStorage` is doing:

- It stores the replay data efficiently without eagerly materializing everything in Python structures.

Why `_init` estimates bytes per step:

- The buffer can be huge.
- It decides whether to store replay on GPU or CPU based on free GPU memory.

### `TD-MPC2/common/logger.py:3`

This file is not algorithmically important, but it is operationally important.

It is responsible for:

- printing run metadata
- formatting train/eval metrics
- saving checkpoints
- optionally sending metrics and videos to Weights & Biases

Why this matters:

- RL runs are long and noisy
- without consistent logging, it is hard to tell whether training is improving, diverging, or stalling

So this is experiment management infrastructure, not learning logic.

### `TD-MPC2/tdmpc2.py:23`

This optimizer setup uses parameter groups.

Why parameter groups exist:

- some submodules can use different learning rates
- here the encoder uses `enc_lr_scale * lr`
- everything else uses the base `lr`

Why the policy has a separate optimizer:

- the model components and the actor are updated with different objectives
- world model / critics use `total_loss`
- actor uses `pi_loss`

Keeping separate optimizers makes that separation clean:

- different backward passes
- different gradient clipping
- easy to update actor after critic/model

### `TD-MPC2/tdmpc2.py:35`

`self.cfg.iterations += 2*int(cfg.action_dim >= 20)` means:

- if action space has fewer than 20 dims: add 0
- if action space has 20 or more dims: add 2 planning iterations

Why:

- higher-dimensional action spaces are harder for sampling-based planners
- MPPI needs more refinement passes to find good elites

So this is just a heuristic that says “harder action search -> spend a bit more planning compute.”

### `TD-MPC2/tdmpc2.py:45`

The `plan` property is lazy caching.

What it does:

- first access: store `self._plan` in `self._plan_val`
- later accesses: return the cached bound method

In practice here, it mostly serves as a small indirection layer so planning stays eager and is not accidentally replaced by a compiled variant.

This is not core RL logic. It is mostly execution-control plumbing.

### `TD-MPC2/tdmpc2.py:162`

`_z = self.model.next(_z, pi_actions[t], task)` rolls the latent state forward one imagined step.

Meaning:

- current imagined latent state: `_z`
- current imagined action: `pi_actions[t]`
- predicted next latent state: `_z_next`

Why it is inside planning:

- the planner seeds some candidate trajectories from the policy itself
- to get the policy’s action at time `t+1`, it first needs the latent state at time `t+1`

So this is “imagine one step ahead using the learned dynamics model.”

### `TD-MPC2/tdmpc2.py:231`

This policy loss is the actor objective.

The code:

- samples actions from the current policy
- evaluates them with the critic
- adds an entropy bonus
- averages across batch/time dimensions
- discounts later rollout steps by `rho^t`
- negates the result because optimizers minimize losses

Interpreting it:

- higher Q should make the loss smaller
- higher entropy should also help, weighted by `entropy_coef`
- nearer-term latent states get larger weight than farther ones

So the policy is trained to choose actions that are:

- high value according to the critic
- still somewhat stochastic

### `TD-MPC2/tdmpc2.py:233`

`clip_grad_norm_` rescales gradients if their total norm exceeds a threshold.

Why it exists:

- RL losses can spike
- unbounded gradients can destabilize training

This does not eliminate gradients. It caps their overall size.

So it is a standard stability measure, especially useful for world-model training and actor-critic setups.

### `TD-MPC2/tdmpc2.py:237`

The entropy terms come from the stochastic Gaussian policy.

Interpretation:

- high entropy means the policy is spread out and exploratory
- low entropy means the policy is concentrated and nearly deterministic

Why add entropy to the objective:

- prevents premature collapse to a narrow policy
- encourages exploration
- often improves optimization early in training

In this code:

- `entropy = -log_prob`
- `scaled_entropy` adjusts that value relative to action dimension

The actor objective is basically:

- maximize Q
- plus a small bonus for maintaining stochasticity

### `TD-MPC2/tdmpc2.py:264`

TD means temporal difference.

The TD target here is:

`reward + discount * next_state_value`

More concretely in this code:

- sample next action from current policy
- evaluate it with the target critic
- use the minimum of two randomly selected Q heads
- zero out future value if terminated

Why this exists:

- you usually do not know the full return exactly
- TD bootstraps from a value estimate of the next state

This is standard Bellman backup logic in actor-critic RL.

### `TD-MPC2/tdmpc2.py:303`

What Q-networks are:

- critics that estimate expected long-term return for `(state, action)`

In this repo they operate on latent state `z`, not raw observation.

Why there are multiple Q-networks:

- ensembles reduce optimistic bias
- sampling two heads and taking the min is a common anti-overestimation trick

What this loss line is doing:

- for each rollout step `t`
- for each critic head
- compare predicted value distribution against TD target
- use soft cross-entropy because values are represented with two-hot bins
- discount later rollout steps by `rho^t`

So this is the core critic training loop.

### `TD-MPC2/envs/mujoco.py:72`

`cfg.rho = 0.7` is overriding the temporal weighting used in multi-step losses.

Where `rho` appears:

- policy loss weighting over latent rollout steps
- reward loss weighting
- consistency loss weighting
- value loss weighting

Effect:

- larger `rho` means farther imagined steps matter more
- smaller `rho` means the update focuses more on near-term steps

The comment “increase this for tasks that are episodic” means:

- when episode structure matters more, you may want longer-horizon imagined loss terms to count more heavily

### `TD-MPC2/evaluate.py:14`

The comment says `make_env` was not implemented, but in this repo it is implemented in `envs/__init__.py`.

So evaluation does have an environment factory available for single-task environments.

The real limitation is that `envs.make_env` still raises `NotImplementedError` for multitask environments.

### `TD-MPC2/train.py:51`

The trainer expects an environment object with:

- `reset(...)`
- `step(action)`
- `close()`
- `render()` if video is enabled
- `action_space`
- `observation_space`
- `rand_act()` after wrapping with `TensorWrapper`

You already create that by calling `env = make_env(cfg)`.

So the “I need to make this” note is outdated for single-task environments.

## Pieces That Are Easy To Lose Track Of

### Why latent consistency loss exists

The dynamics model predicts the next latent state.

But you also have an encoder that can directly encode the real next observation into a latent state.

Consistency loss forces:

`predicted next latent ~= encoded real next latent`

That ties imagination to reality.

### Why rewards and values use bins instead of raw scalars

This code uses the two-hot representation for distributional regression.

Why:

- it can make optimization smoother
- it handles large value ranges better
- it is a standard TD-MPC2 design choice

### Why planning and policy both exist

The policy gives cheap action proposals.

The planner improves action selection online by evaluating many imagined trajectories through the world model.

TD-MPC2 combines both:

- policy helps seed good candidate trajectories
- planner refines them using model-based lookahead

## Short Mental Model For Each Core File

- `train.py`: wire everything together and start training
- `evaluate.py`: load a checkpoint and run evaluation episodes
- `tdmpc2.py`: agent logic, planning, actor update, critic/world-model update
- `common/world_model.py`: all learnable networks
- `common/layers.py`: building blocks for the networks
- `common/math.py`: RL math utilities and distributional value helpers
- `common/buffer.py`: replay memory and sequence sampling
- `envs/*`: environment construction and wrappers
- `trainer/*`: run loops for online or offline training
- `common/logger.py`: experiment logging/checkpointing
- `common/parser.py`: config normalization

## Things I Noticed While Reading

These are not part of your learning comments, but they are worth knowing because they may affect runs:

- `TD-MPC2/trainer/offline_trainer.py:39` uses `infor['success']`, which looks like a typo for `info['success']`.
- `TD-MPC2/common/buffer.py:57` uses `x.nume1()` with the digit `1`, which looks like a typo for `x.numel()`.
- `TD-MPC2/common/logger.py` imports `pandas` and uses `dataclasses.asdict(cfg)`, so config really does need to be a dataclass by the time logging starts.
- `envs/__init__.py` still has no multitask environment implementation, so the offline multitask path is structurally incomplete unless more env files are added.

## Best Order To Learn This Codebase

If you want to understand the repository with minimal confusion, read in this order:

1. `TD-MPC2/train.py`
2. `TD-MPC2/trainer/online_trainer.py`
3. `TD-MPC2/common/buffer.py`
4. `TD-MPC2/tdmpc2.py`
5. `TD-MPC2/common/world_model.py`
6. `TD-MPC2/common/layers.py`
7. `TD-MPC2/common/math.py`
8. `TD-MPC2/envs/__init__.py` and `TD-MPC2/envs/mujoco.py`
9. `TD-MPC2/common/parser.py`
10. `TD-MPC2/common/logger.py`

That order follows the actual runtime path.
