import csv
import os
from typing import List, Tuple

import gymnasium as gym
import numpy as np

from lam_sac_env import SimpleLAM, TwoWallsGap10x10LAMEnv, RewardCfg

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv


GOALS: List[Tuple[int, int]] = [(9, 1), (9, 3), (9, 5), (9, 7), (9, 9)]


def evaluate_model(model: SAC, goal, lam, reward_cfg, n_episodes=100):
    env = TwoWallsGap10x10LAMEnv(goal=goal, reward_cfg=reward_cfg, lam=lam)
    successes = 0
    lengths = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            if done:
                if info.get("is_success"):
                    successes += 1
                lengths.append(step)
    env.close()
    success_rate = successes / n_episodes
    avg_len = float(np.mean(lengths)) if lengths else float('nan')
    return success_rate, avg_len


def main():
    os.makedirs("models", exist_ok=True)
    log_path = "training_log.csv"
    lam = SimpleLAM()
    model = None
    prev_success_rate = None

    # Prepare CSV
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "goal", "success_rate", "avg_length"])

    for i, goal in enumerate(GOALS):
        print(f"\n=== Stage {i+1}/{len(GOALS)}: goal={goal} ===")
        # Ask LAM for any reward tweaks
        reward_cfg = lam.suggest_reward_cfg(prev_success_rate)
        env = TwoWallsGap10x10LAMEnv(goal=goal, reward_cfg=reward_cfg, lam=lam)

        # SB3 prefers vectorized envs
        venv = DummyVecEnv([lambda: env])

        if model is None:
            model = SAC("MlpPolicy", venv, verbose=1)
        else:
            model.set_env(venv)

        timesteps = 20000
        print(f"Training for {timesteps} timesteps...")
        model.learn(total_timesteps=timesteps)

        # Save intermediate model
        model_path = f"models/sac_lam_stage_{i+1}.zip"
        model.save(model_path)
        print(f"Saved model to {model_path}")

        # Evaluate
        success_rate, avg_len = evaluate_model(model, goal, lam, reward_cfg, n_episodes=100)
        print(f"Stage {i+1} success_rate={success_rate:.3f}, avg_len={avg_len:.1f}")

        # Log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([i + 1, goal, success_rate, avg_len])

        prev_success_rate = success_rate

    # final save
    if model is not None:
        final_path = "models/sac_lam_grid.zip"
        model.save(final_path)
        print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()
