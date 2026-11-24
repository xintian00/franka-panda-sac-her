
import time

import gymnasium as gym
from stable_baselines3 import SAC

import panda_mujoco_gym


# ==========================================
# Configuration
# ==========================================

MODEL_PATH = (
    "./models/sac_her_local_v1_500k/"
    "sac_her_checkpoint_400000_steps.zip"
)

NUM_EPISODES = 10
START_SEED = 1000

# Delay between episodes for recording
PAUSE_BETWEEN_EPISODES = 2


# ==========================================
# Create environment
# ==========================================

env = gym.make(
    "FrankaPushSparse-v0",
    render_mode="human",
)

print("Loading 400k model...")

model = SAC.load(
    MODEL_PATH,
    env=env,
    device="cpu",
)

print("Model loaded successfully!")
print("Starting visualization...")


# ==========================================
# Run trained policy
# ==========================================

successes = 0

try:
    for episode in range(NUM_EPISODES):

        obs, info = env.reset(
            seed=START_SEED + episode
        )

        print("\n" + "=" * 45)
        print(f"Episode {episode + 1}/{NUM_EPISODES}")
        print("=" * 45)

        success = False

        for step in range(50):

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            env.render()

            if info.get("is_success", False):
                success = True

            if terminated or truncated:
                break

        if success:
            successes += 1
            print("SUCCESS!")
        else:
            print("FAILED")

        print("Steps:", step + 1)

        time.sleep(PAUSE_BETWEEN_EPISODES)

finally:
    env.close()


print("\n" + "=" * 45)
print("Visualization finished!")
print(f"Successful episodes: {successes}/{NUM_EPISODES}")
print("=" * 45)
