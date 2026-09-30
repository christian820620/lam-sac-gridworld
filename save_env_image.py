import argparse
from PIL import Image

from lam_sac_env import TwoWallsGap10x10LAMEnv


def save_env_image(start, goal, out_path: str, scale: int = 1):
    env = TwoWallsGap10x10LAMEnv(goal=tuple(goal), start=tuple(start))
    obs, _ = env.reset()
    frame = env.render()
    img = Image.fromarray(frame)
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    img.save(out_path)
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, nargs=2, default=[0, 0], help='Start x y')
    parser.add_argument('--goal', type=int, nargs=2, default=[9, 9], help='Goal x y')
    parser.add_argument('--out', type=str, default='env_snapshot.png')
    parser.add_argument('--scale', type=int, default=1)
    args = parser.parse_args()
    save_env_image(args.start, args.goal, args.out, scale=args.scale)
    print(f"Saved environment image to {args.out}")
