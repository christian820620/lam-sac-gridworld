# LAM + SAC Gridworld

This project implements a small 10x10 Gridworld with two walls and a gap and trains a single Stable-Baselines3 SAC agent across a curriculum of goals using a simple LAM (high-level planner) that supplies directional hints.

Files:
- `lam_sac_env.py`: environment, `RewardCfg`, `SimpleLAM`, and `TwoWallsGap10x10LAMEnv`.
- `train_curriculum.py`: trains a single SAC model across GOALS and logs results.
- `watch_policy.py`: loads a trained model and visualizes episodes at high speed.
- `requirements.txt`: Python dependencies.

Quickstart (Windows PowerShell):

1. Create and activate a virtualenv (optional but recommended):

```powershell
python3 -m venv venv
& .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train across the curriculum (this will save models to `models/`):

```powershell
python3 train_curriculum.py
```

3. Watch the trained policy (default loads `models/sac_lam_grid.zip`):

```powershell
python3 watch_policy.py --model models/sac_lam_grid.zip --goal 9 9 --episodes 5
```

Notes:
- The environment provides only distance-based progress rewards, a per-step penalty, bump penalty for hitting walls, and a goal bonus. There are no path/rail rewards.
- `SimpleLAM` only suggests directions and optional reward tuning; it does not define any rewarded path.
-simpleLAM is now commented out and replaced with a more optimal algorithm
- the terminal should not depend on the train_trials file, it should be the opposite. The training file should depend on the commander_terminal, given the terminal is what is processing the prompts for training. 

Qt driver installed for faster loading