
import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)

import panda_mujoco_gym


# ==========================================
# Configuration
# ==========================================

SOURCE_DIR = "./models/sac_her_local_v1_200k"
OUTPUT_DIR = "./models/sac_her_local_v1_500k"
LOG_DIR = "./logs/sac_her_local_v1_500k"

MODEL_PATH = os.path.join(
    SOURCE_DIR, "final_model_200k.zip"
)

BUFFER_PATH = os.path.join(
    SOURCE_DIR, "final_replay_buffer_200k.pkl"
)

START_STEPS = 200_000
TARGET_STEPS = 500_000
ADDITIONAL_STEPS = TARGET_STEPS - START_STEPS

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ==========================================
# Environments
# ==========================================

train_env = Monitor(
    gym.make("FrankaPushSparse-v0")
)

eval_env = Monitor(
    gym.make("FrankaPushSparse-v0")
)


# ==========================================
# Load model
# ==========================================

print("=" * 55)
print("Continue SAC + HER Training")
print("Target: 200k -> 500k")
print("=" * 55)

print("\nLoading model...")

model = SAC.load(
    MODEL_PATH,
    env=train_env,
    device="cpu",
    tensorboard_log=LOG_DIR,
)

print("Existing timesteps:", model.num_timesteps)

if model.num_timesteps != START_STEPS:
    raise ValueError(
        f"Expected {START_STEPS} steps, "
        f"but found {model.num_timesteps}"
    )


# ==========================================
# Load HER replay buffer
# ==========================================

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
    verbose=1,
)

checkpoint_callback = CheckpointCallback(
    save_freq=25000,
    save_path=OUTPUT_DIR,
    name_prefix="sac_her_checkpoint",
    save_replay_buffer=True,
    save_vecnormalize=False,
    verbose=1,
)

callbacks = CallbackList([
    eval_callback,
    checkpoint_callback,
])


# ==========================================
# Continue training
# ==========================================

print("\nStarting training...")
print("Additional steps:", ADDITIONAL_STEPS)
print("Target total steps:", TARGET_STEPS)

try:
    model.learn(
        total_timesteps=ADDITIONAL_STEPS,
        reset_num_timesteps=False,
        callback=callbacks,
        progress_bar=False,
    )

    # ======================================
    # Save final model and replay buffer
    # ======================================

    final_model_path = os.path.join(
        OUTPUT_DIR, "final_model_500k"
    )

    final_buffer_path = os.path.join(
        OUTPUT_DIR, "final_replay_buffer_500k.pkl"
    )

    model.save(final_model_path)
    model.save_replay_buffer(final_buffer_path)

    print("\n" + "=" * 55)
    print("Training completed!")
    print("Final timesteps:", model.num_timesteps)
    print("Model saved:", final_model_path)
    print("Replay buffer saved:", final_buffer_path)
    print("=" * 55)

finally:
    train_env.close()
    eval_env.close()
