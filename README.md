# Franka Panda Reinforcement Learning with SAC + HER

Goal-conditioned robotic manipulation using **Soft Actor-Critic (SAC)** and **Hindsight Experience Replay (HER)** in MuJoCo.

This project documents my reinforcement learning experiments with a Franka Panda robotic arm, trained to push a cube toward randomly sampled target positions under sparse rewards.

**Key result: 88% success rate after 400,000 environment timesteps**, evaluated over 100 episodes with fixed random seeds.

## Demo

A demonstration GIF of the trained policy will be added here.

## Results

| Training Stage | Environment Timesteps | Success Rate |
|---|---:|---:|
| Initial training | 50,000 | 7% |
| Continued training | 200,000 | 43% |
| Selected checkpoint | 400,000 | **88%** |

The 50k and 200k figures are historical experiment results. The published 400k checkpoint has been independently re-evaluated from this repository.

### Verified 400k Evaluation

| Metric | Result |
|---|---|
| Environment | `FrankaPushSparse-v0` |
| Evaluation episodes | 100 |
| Fixed seeds | 1000–1099 |
| Policy | Deterministic |
| Successful episodes | 88 / 100 |
| Success rate | **88.0%** |
| Average episode length | 11.85 steps |
| Episode limit | 50 steps |

An episode is counted as successful if the environment reports `is_success=True` at least once during that episode.

The full recorded evaluation output is available in `results/evaluation_400k.txt`.

The checkpoint is stored in `models/sac_her_local_v1_500k/`. The directory name reflects the original continuation experiment's planned 500k-step target; **the published checkpoint was saved at 400k steps**.

## Methodology

### Soft Actor-Critic (SAC)

SAC is an off-policy reinforcement learning algorithm designed for continuous action spaces.

It uses a stochastic Actor to generate actions and two Critic networks to estimate action values. The policy is optimized using both expected action value and an entropy term that encourages exploration.

### Hindsight Experience Replay (HER)

Sparse rewards make goal-conditioned robotic manipulation difficult because the agent initially fails to reach its target most of the time.

HER addresses this by relabeling previously collected experiences with alternative goals achieved later in the same episode and recomputing their rewards.

In this project, HER is integrated into SAC through Stable-Baselines3's `HerReplayBuffer`.

### Main Training Configuration

| Parameter | Value |
|---|---|
| Policy | `MultiInputPolicy` |
| Actor/Critic architecture | `[256, 256]` |
| Replay buffer | `HerReplayBuffer` |
| HER goal strategy | `future` |
| HER sampled goals | 4 |
| Replay buffer capacity | 200,000 |
| Batch size | 256 |
| Learning starts | 1,000 steps |
| Learning rate | 0.0003 |
| Discount factor | 0.95 |
| Soft-update coefficient | 0.05 |
| Entropy coefficient | `auto` |
| Training frequency | 1 |
| Gradient steps | 1 |
| Training device | CPU |

## My Contributions

I used an existing Franka Panda MuJoCo environment and built an experimental SAC + HER workflow around it.

My work includes:

- Configuring SAC with goal-conditioned observations and HER replay
- Implementing local training, periodic evaluation, and checkpoint saving
- Implementing continued training with restoration of the model and HER replay buffer
- Evaluating checkpoints with a reproducible, fixed-seed protocol
- Creating a visualization script for the trained policy
- Tracking training progress and selecting a 400k checkpoint
- Applying a small environment compatibility fix

The Franka Panda simulation environment itself was developed by the upstream author and is not claimed as my original implementation.

## Repository Structure

```text
franka-panda-sac-her/
├── assets/
├── models/
│   └── sac_her_local_v1_500k/
│       └── sac_her_checkpoint_400000_steps.zip
├── results/
│   └── evaluation_400k.txt
├── scripts/
│   ├── train_local.py
│   ├── continue_to_200k.py
│   ├── continue_to_500k.py
│   ├── evaluate_400k.py
│   └── visualize_400k.py
├── environment_bool_fix.patch
├── requirements.txt
└── README.md
```

## Installation (Windows / Anaconda Prompt)

The following setup is designed for Windows with Conda and Git installed.

The simulation environment is hosted separately because this repository focuses on the reinforcement learning experiments.

### 1. Clone both repositories

```bat
cd /d D:\Projects
git clone https://github.com/xintian00/franka-panda-sac-her.git
git clone https://github.com/zichunxx/panda_mujoco_gym.git
```

This creates two sibling directories inside `D:\Projects`.

### 2. Check out the upstream environment version

```bat
cd /d D:\Projects\panda_mujoco_gym
git checkout e9881d993234875ee1d55d717176bf7999bdb583
```

### 3. Apply the compatibility patch

```bat
git apply ..\franka-panda-sac-her\environment_bool_fix.patch
```

This patch converts the environment's `terminated` value to a native Python Boolean.

### 4. Set up Python dependencies

```bat
conda create -n franka_sac_her python=3.9 -y
conda activate franka_sac_her
cd /d D:\Projects\franka-panda-sac-her
python -m pip install -r requirements.txt
```

### 5. Make the upstream environment importable

The specified upstream commit does not include `setup.py` or `pyproject.toml`, so it cannot be installed directly with `pip install -e .`.

Instead, add its parent repository directory to the Python module search path for the current Anaconda Prompt session:

```bat
set "PYTHONPATH=D:\Projects\panda_mujoco_gym;%PYTHONPATH%"
```

Verify the environment:

```bat
python -c "import panda_mujoco_gym; import gymnasium as gym; env=gym.make('FrankaPushSparse-v0'); print('ENVIRONMENT OK'); env.close()"
```

The expected final line is `ENVIRONMENT OK`.

The `PYTHONPATH` setting must be repeated in a new terminal session, unless the upstream directory has been registered permanently in the Python environment.

## Evaluate the Published Checkpoint

From the repository root, run:

```bat
python scripts\evaluate_400k.py
```

The script evaluates available models for 100 episodes each with fixed seeds from 1000 through 1099.

Only the 400k checkpoint is distributed in this repository. References to the optional 200k and Best Model checkpoints are retained for comparison with the original experiments; missing checkpoints are skipped automatically.

## Visualize the Trained Policy

Run:

```bat
python scripts\visualize_400k.py
```

This loads the 400k checkpoint and opens the MuJoCo viewer to visualize 10 episodes of robotic pushing.

A graphical desktop environment is required.

## Training from Scratch

Run these scripts in sequence from the repository root:

**Stage 1: Initial 50k training**

```bat
python scripts\train_local.py
```

**Stage 2: Continue from 50k to 200k**

```bat
python scripts\continue_to_200k.py
```

**Stage 3: Continue from 200k toward 500k**

```bat
python scripts\continue_to_500k.py
```

The third script has a configured target of 500k total timesteps. The project's reported result comes from its 400k checkpoint.

Continuation requires both the saved SAC model and the corresponding HER replay buffer from the previous stage. These intermediate training artifacts are not included in the repository.

The scripts preserve the original experiment workflow. Evaluation and visualization of the supplied 400k checkpoint have been validated; a completely fresh end-to-end training run using these published instructions has not yet been independently verified.

## Software Versions

- Python 3.9
- MuJoCo 2.3.3
- Gymnasium 0.29.1
- Gymnasium Robotics 1.2.2
- Stable-Baselines3 2.2.1
- NumPy 1.23.5

## Acknowledgments

The robotic simulation environment is based on [panda_mujoco_gym](https://github.com/zichunxx/panda_mujoco_gym), developed by **Zichun Xu**, and distributed under the MIT License.

The environment is reproduced from upstream commit `e9881d993234875ee1d55d717176bf7999bdb583`, with a small local compatibility patch.

SAC and HER are implemented using the open-source Stable-Baselines3 ecosystem.

This repository documents my training experiments, evaluation methodology, saved policy checkpoint, and visualization workflow, rather than claiming authorship of the original simulation environment or reinforcement learning algorithms.