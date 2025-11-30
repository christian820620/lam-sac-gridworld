"""Train across multiple start/goal trials (4 trials by default).

This script reuses one SAC model and continues training across each trial (start,goal).
It logs per-trial performance and saves the final model.
"""
from typing import List, Tuple
import csv
import os

from lam_sac_env import SimpleLAM, TwoWallsGap10x10LAMEnv, RewardCfg
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

# Define 4 different trials: (start, goal)
TRIALS: List[Tuple[Tuple[int,int], Tuple[int,int]]] = [
    ((0, 0), (9, 9)),  # top-left -> bottom-right
    ((9, 9), (0, 0)),  # bottom-right -> top-left
    ((0, 9), (9, 0)),  # bottom-left -> top-right
    ((9, 0), (0, 9)),  # top-right -> bottom-left
]


def evaluate_model(model: SAC, start, goal, lam, reward_cfg, n_episodes=50):
    env = TwoWallsGap10x10LAMEnv(goal=goal, start=start, reward_cfg=reward_cfg, lam=lam)
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
    log_path = "training_trials_log.csv"
    lam = SimpleLAM()
    model = None
    prev_success_rate = None

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_idx", "start", "goal", "success_rate", "avg_length"])

    for i, (start, goal) in enumerate(TRIALS):
        print(f"\n=== Trial {i+1}/{len(TRIALS)}: start={start} goal={goal} ===")
        reward_cfg = lam.suggest_reward_cfg(prev_success_rate)
        env = TwoWallsGap10x10LAMEnv(goal=goal, start=start, reward_cfg=reward_cfg, lam=lam)
        venv = DummyVecEnv([lambda: env])

        if model is None:
            model = SAC("MlpPolicy", venv, verbose=1)
        else:
            model.set_env(venv)

        timesteps = 20000
        model.learn(total_timesteps=timesteps)

        model_path = f"models/sac_lam_trial_{i+1}.zip"
        model.save(model_path)
        print(f"Saved model to {model_path}")

        success_rate, avg_len = evaluate_model(model, start, goal, lam, reward_cfg, n_episodes=50)
        print(f"Trial {i+1} success_rate={success_rate:.3f}, avg_len={avg_len:.1f}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([i + 1, start, goal, success_rate, avg_len])

        prev_success_rate = success_rate

    if model is not None:
        final_path = "models/sac_lam_trials_final.zip"
        model.save(final_path)
        print(f"Final model saved to {final_path}")


if __name__ == '__main__':
    main()
