
import os
os.environ["MUJOCO_GL"] = "glfw"

import gymnasium as gym
import panda_mujoco_gym

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)

# ==========================================
# Paths
# ==========================================

BASE_DIR = "./models/sac_her_local_v1"

MODEL_PATH = os.path.join(BASE_DIR, "final_model.zip")
BUFFER_PATH = os.path.join(BASE_DIR, "final_replay_buffer.pkl")

OUTPUT_DIR = "./models/sac_her_local_v1_200k"
LOG_DIR = "./logs/sac_her_local_v1_200k"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================
# Environments
# ==========================================

train_env = Monitor(gym.make("FrankaPushSparse-v0"))
eval_env = Monitor(gym.make("FrankaPushSparse-v0"))

# ==========================================
# Load existing model and HER replay buffer
# ==========================================

print("=" * 50)
print("Loading existing 50k model...")
print("=" * 50)

model = SAC.load(
    MODEL_PATH,
    env=train_env,
    device="cpu",
)

print("Existing timesteps:", model.num_timesteps)

if model.num_timesteps != 50000:
    raise ValueError(
        f"Expected 50000 timesteps, got {model.num_timesteps}. "
        "Please check the model file."
    )

print("\nLoading HER replay buffer...")
model.load_replay_buffer(BUFFER_PATH)

print("Replay buffer loaded successfully.")

# ==========================================
# Callbacks
# ==========================================

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=OUTPUT_DIR,
    log_path=LOG_DIR,
    eval_freq=5000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
)

checkpoint_callback = CheckpointCallback(
    save_freq=25000,
    save_path=OUTPUT_DIR,
    name_prefix="checkpoint",
    save_replay_buffer=True,
)

callback = CallbackList([
    eval_callback,
    checkpoint_callback,
])

# ==========================================
# Continue training: 50k -> 200k
# ==========================================

print("\n" + "=" * 50)
print("Starting continued training")
print("Current timesteps: 50,000")
print("Additional timesteps: 150,000")
print("Target timesteps: 200,000")
print("=" * 50)

model.learn(
    total_timesteps=150000,
    callback=callback,
    reset_num_timesteps=False,
    progress_bar=False,
)

# ==========================================
# Save final model and replay buffer
# ==========================================

model.save(
    os.path.join(OUTPUT_DIR, "final_model_200k")
)

model.save_replay_buffer(
    os.path.join(OUTPUT_DIR, "final_replay_buffer_200k.pkl")
)

print("\n" + "=" * 50)
print("Training completed!")
print("Final timesteps:", model.num_timesteps)
print("Model saved:", OUTPUT_DIR)
print("=" * 50)

train_env.close()
eval_env.close()
