
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

import panda_mujoco_gym


MODELS = {
    "200k Final": (
        "./models/sac_her_local_v1_200k/"
        "final_model_200k.zip"
    ),
    "400k Checkpoint": (
        "./models/sac_her_local_v1_500k/"
        "sac_her_checkpoint_400000_steps.zip"
    ),
    "Best Model": (
        "./models/sac_her_local_v1_500k/"
        "best_model.zip"
    ),
}

NUM_EPISODES = 100
START_SEED = 1000


def evaluate(name, path):
    print("\n" + "=" * 50)
    print("Testing:", name)
    print("=" * 50)

    if not os.path.exists(path):
        print("Model not found:", path)
        return None

    env = gym.make("FrankaPushSparse-v0")
    model = SAC.load(path, env=env, device="cpu")

    successes = 0
    lengths = []

    for episode in range(NUM_EPISODES):
        obs, info = env.reset(
            seed=START_SEED + episode
        )

        success = False
        steps = 0

        for step in range(50):
            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            steps = step + 1

            if info.get("is_success", False):
                success = True

            if terminated or truncated:
                break

        if success:
            successes += 1

        lengths.append(steps)

        if (episode + 1) % 20 == 0:
            print(
                f"Completed {episode + 1}/"
                f"{NUM_EPISODES}"
            )

    env.close()

    rate = successes / NUM_EPISODES * 100
    avg_length = np.mean(lengths)

    print("\nResults:")
    print("Successful episodes:", successes)
    print(f"Success rate: {rate:.1f}%")
    print(f"Average episode length: {avg_length:.2f}")

    return rate


def main():
    print("=" * 50)
    print("SAC + HER Model Comparison")
    print("100 episodes per model")
    print("Seeds: 1000 - 1099")
    print("=" * 50)

    results = {}

    for name, path in MODELS.items():
        rate = evaluate(name, path)

        if rate is not None:
            results[name] = rate

    print("\n" + "=" * 50)
    print("FINAL COMPARISON")
    print("=" * 50)

    for name, rate in results.items():
        print(f"{name}: {rate:.1f}%")

    print("\nAll tests finished.")


if __name__ == "__main__":
    main()
