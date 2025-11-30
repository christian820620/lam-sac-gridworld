import argparse
import time
import ast

import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from lam_sac_env import TwoWallsGap10x10LAMEnv


def _parse_trials(trial_strings):
    """Parse trial specs of the form 'sx,sy:gx,gy' into [(start),(goal), ...]."""
    trials = []
    for s in trial_strings:
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid trial spec: {s}. Expected 'sx,sy:gx,gy'")
        start = tuple(int(x) for x in parts[0].split(","))
        goal = tuple(int(x) for x in parts[1].split(","))
        trials.append((start, goal))
    return trials


def watch(model_path: str = "models/sac_lam_grid.zip", trials=None, episodes_per_trial: int = 3, pause: float = 0.001):
    model = SAC.load(model_path)

    plt.ion()
    fig, ax = plt.subplots(figsize=(4, 4))
    img = None

    # For each trial, run episodes_per_trial and report which episodes succeeded
    for t_idx, (start, goal) in enumerate(trials, start=1):
        env = TwoWallsGap10x10LAMEnv(goal=goal, start=start)
        print(f"\n--- Trial {t_idx}: start={start} goal={goal} ---")
        for ep in range(episodes_per_trial):
            obs, _ = env.reset()
            done = False
            steps = 0
            success = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                frame = env.render()
                if img is None:
                    img = ax.imshow(frame)
                    ax.axis('off')
                else:
                    img.set_data(frame)
                plt.pause(pause)
                done = terminated or truncated
                steps += 1
            success = bool(info.get('is_success'))
            print(f"Trial {t_idx} Episode {ep+1}: steps={steps}, success={success}")
        env.close()

    plt.ioff()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/sac_lam_grid.zip')
    parser.add_argument('--trial', type=str, action='append', help="Trial spec 'sx,sy:gx,gy'. Can be repeated.")
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--pause', type=float, default=0.001)
    args = parser.parse_args()

    if not args.trial:
        # default set of 4 trials when none provided
        trials = [((0, 0), (9, 9)), ((9, 9), (0, 0)), ((0, 9), (9, 0)), ((9, 0), (0, 9))]
    else:
        trials = _parse_trials(args.trial)

    watch(model_path=args.model, trials=trials, episodes_per_trial=args.episodes, pause=args.pause)
