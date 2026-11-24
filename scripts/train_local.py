
import os
import gymnasium as gym
import panda_mujoco_gym

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor


# ==========================================
# 1. Directories
# ==========================================

log_dir = "./logs/sac_her_local_v1/"
model_dir = "./models/sac_her_local_v1/"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


# ==========================================
# 2. Create environments
# ==========================================

env = Monitor(gym.make("FrankaPushSparse-v0"))
eval_env = Monitor(gym.make("FrankaPushSparse-v0"))


# ==========================================
# 3. Evaluation callback
# ==========================================

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=model_dir,
    log_path=log_dir,
    eval_freq=5000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
    verbose=1,
)


# ==========================================
# 4. Checkpoint callback
# ==========================================

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=model_dir,
    name_prefix="checkpoint",
    save_replay_buffer=True,
    verbose=1,
)

callbacks = CallbackList([
    eval_callback,
    checkpoint_callback,
])


# ==========================================
# 5. SAC + HER
# ==========================================

model = SAC(
    policy="MultiInputPolicy",
    env=env,

    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs={
        "n_sampled_goal": 4,
        "goal_selection_strategy": "future",
    },

    buffer_size=200000,
    learning_starts=1000,
    batch_size=256,

    gamma=0.95,
    tau=0.05,
    learning_rate=3e-4,

    ent_coef="auto",

    train_freq=1,
    gradient_steps=1,

    policy_kwargs={
        "net_arch": [256, 256],
    },

    tensorboard_log=log_dir,
    verbose=1,
    device="cpu",
)


# ==========================================
# 6. Start training
# ==========================================

print("\nStarting SAC + HER training...")
print("Environment: FrankaPushSparse-v0")
print("Training steps: 50,000")
print("Device: CPU")
print("Replay buffer: HER")
print("=" * 50)

model.learn(
    total_timesteps=50000,
    callback=callbacks,
    progress_bar=False,
)


# ==========================================
# 7. Save final model and replay buffer
# ==========================================

model.save(os.path.join(model_dir, "final_model"))

model.save_replay_buffer(
    os.path.join(model_dir, "final_replay_buffer.pkl")
)

print("\nTraining finished!")
print("Model saved in:", model_dir)

env.close()
eval_env.close()
