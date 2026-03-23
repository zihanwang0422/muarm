"""RL/IL training entry for Panda tasks."""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv

from learning.panda_rl.envs.panda_env import FrankaPandaEnv
from learning.panda_rl.il.bc import rollout_bc, train_bc


def make_env(task, render_mode=None):
    return FrankaPandaEnv(task_name=task, render_mode=render_mode)


def train_rl(task: str, algo: str, timesteps: int, save_dir: Path, render: bool):
    env = DummyVecEnv([lambda: make_env(task, render_mode="human" if render else None)])

    algo = algo.lower()
    if algo == "ppo":
        model = PPO("MultiInputPolicy", env, verbose=1)
    elif algo == "sac":
        model = SAC("MultiInputPolicy", env, verbose=1)
    elif algo == "td3":
        model = TD3("MultiInputPolicy", env, verbose=1)
    elif algo == "ddpg":
        model = DDPG("MultiInputPolicy", env, verbose=1)
    else:
        raise ValueError(f"Unsupported RL algo: {algo}")

    model.learn(total_timesteps=timesteps)
    out = save_dir / f"{task}_{algo}.zip"
    model.save(str(out))
    env.close()
    print(f"Saved RL model to {out}")


def train_il(task: str, save_dir: Path, render: bool):
    env = make_env(task, render_mode="human" if render else None)
    out = save_dir / f"{task}_bc.pt"
    train_bc(env, str(out))
    env.close()
    print(f"Saved BC model to {out}")


def eval_model(task: str, algo: str, model_path: Path, episodes: int):
    algo = algo.lower()
    env = make_env(task, render_mode="human")

    if algo == "bc":
        rollout_bc(env, str(model_path), episodes=episodes)
        env.close()
        return

    cls_map = {"ppo": PPO, "sac": SAC, "td3": TD3, "ddpg": DDPG}
    if algo not in cls_map:
        raise ValueError(f"Unsupported algo: {algo}")
    model = cls_map[algo].load(str(model_path))

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--family", choices=["rl", "il"], default="rl")
    parser.add_argument("--task", choices=["reach", "push", "pick_place"], default="reach")
    parser.add_argument("--algo", choices=["ppo", "sac", "td3", "ddpg", "bc"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--save-dir", type=str, default="learning/panda_rl/runs")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        if args.family == "rl":
            train_rl(args.task, args.algo, args.timesteps, save_dir, args.render)
        else:
            train_il(args.task, save_dir, args.render)
    else:
        if not args.model_path:
            raise ValueError("--model-path is required in eval mode")
        eval_model(args.task, args.algo, Path(args.model_path), args.episodes)


if __name__ == "__main__":
    main()
