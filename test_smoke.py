from lam_sac_env import TwoWallsGap10x10LAMEnv, SimpleLAM, RewardCfg


def run_smoke_test(episodes=5, max_steps=50):
    lam = SimpleLAM()
    cfg = RewardCfg()
    goal = (9, 9)
    env = TwoWallsGap10x10LAMEnv(goal=goal, reward_cfg=cfg, lam=lam)

    successes = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        print(f"Episode {ep+1} start obs: {obs}")
        done = False
        steps = 0
        total_reward = 0.0
        while not done and steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        print(f"Episode {ep+1} finished: steps={steps}, total_reward={total_reward:.3f}, success={info.get('is_success')}")
        if info.get('is_success'):
            successes += 1

    print(f"Smoke test completed: {successes}/{episodes} successes")


if __name__ == '__main__':
    run_smoke_test()
