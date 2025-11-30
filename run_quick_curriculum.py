"""Quick curriculum runner: trains the same SAC model across GOALS with fewer timesteps.

This is for fast experimentation. It uses the same logic as `train_curriculum.py`
but with shorter per-stage timesteps and fewer evaluation episodes.
"""
from lam_sac_env import SimpleLAM, TwoWallsGap10x10LAMEnv, RewardCfg
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

GOALS = [(9, 1), (9, 3), (9, 5), (9, 7), (9, 9)]


def quick_run(per_stage_timesteps=2000, eval_episodes=20):
    lam = SimpleLAM()
    model = None
    prev_success = None

    for i, goal in enumerate(GOALS):
        print(f"Stage {i+1}/{len(GOALS)} goal={goal}")
        reward_cfg = lam.suggest_reward_cfg(prev_success)
        env = TwoWallsGap10x10LAMEnv(goal=goal, reward_cfg=reward_cfg, lam=lam)
        venv = DummyVecEnv([lambda: env])

        if model is None:
            model = SAC("MlpPolicy", venv, verbose=0)
        else:
            model.set_env(venv)

        model.learn(total_timesteps=per_stage_timesteps)

        # quick eval
        successes = 0
        lengths = []
        for ep in range(eval_episodes):
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done and steps < env.max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            if info.get('is_success'):
                successes += 1
            lengths.append(steps)
        success_rate = successes / eval_episodes
        print(f" Eval success_rate={success_rate:.2f}, avg_len={np.mean(lengths):.1f}")
        prev_success = success_rate

    model.save('models/sac_lam_quick.zip')
    print('Saved quick model to models/sac_lam_quick.zip')


if __name__ == '__main__':
    quick_run()
