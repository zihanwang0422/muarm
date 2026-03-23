"""RL / IL training entry point.

Examples
--------
# RL – headless (fast)
python learning/train.py --robot panda --task reach --algo sac --timesteps 200000

# RL – with live MuJoCo viewer (1 env only)
python learning/train.py --robot panda --task reach --algo sac --timesteps 200000 --render

# IL – Behavior Cloning
python learning/train.py --robot panda --task reach --algo bc --family il

# Multi-robot
python learning/train.py --robot so_arm --task reach --algo sac --timesteps 100000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.envs.registry import make_env
from learning.algos.rl_trainer import train_rl
from learning.algos.il.bc import train_bc


def _env_fn(robot: str, task: str, render: bool):
    """Return a zero-arg callable that creates an env (optionally with viewer)."""
    def _factory():
        return make_env(robot, task, render_mode="human" if render else None)
    return _factory


def main():
    parser = argparse.ArgumentParser(description="Train RL/IL policy for robot manipulation.")
    parser.add_argument("--robot",      type=str, default="panda",
                        help="Robot type: panda | so_arm")
    parser.add_argument("--task",       type=str, default="reach",
                        choices=["reach", "push", "pick_place"],
                        help="Task name")
    parser.add_argument("--algo",       type=str, default="sac",
                        choices=["ppo", "sac", "td3", "ddpg", "bc"],
                        help="Algorithm: ppo/sac/td3/ddpg (RL) or bc (IL)")
    parser.add_argument("--family",     type=str, default="rl", choices=["rl", "il"],
                        help="Algorithm family (rl or il). Auto-detected for 'bc'.")
    parser.add_argument("--timesteps",  type=int, default=200_000,
                        help="Total env steps for RL training")
    parser.add_argument("--n-envs",     type=int, default=1,
                        help="Number of parallel envs (>1 → SubprocVecEnv, headless only)")
    parser.add_argument("--render",     action="store_true",
                        help="Open MuJoCo viewer during training (forces n-envs=1)")
    parser.add_argument("--save-dir",   type=str, default="learning/runs",
                        help="Directory to save trained model checkpoints")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging under save-dir/tb/")
    args = parser.parse_args()

    # BC always belongs to IL family
    if args.algo == "bc":
        args.family = "il"

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_stem = save_dir / f"{args.robot}_{args.task}_{args.algo}"

    if args.render and args.n_envs > 1:
        print("[warn] --render forces --n-envs 1")
        args.n_envs = 1

    if args.family == "il" or args.algo == "bc":
        # IL: single env, collect demos then train
        env = make_env(args.robot, args.task,
                       render_mode="human" if args.render else None)
        train_bc(env, str(save_stem))
        env.close()

    else:
        tb_log = str(save_dir / "tb") if args.tensorboard else None
        train_rl(
            env_fn=_env_fn(args.robot, args.task, render=args.render),
            algo=args.algo,
            total_timesteps=args.timesteps,
            save_path=str(save_stem),
            n_envs=args.n_envs,
            tensorboard_log=tb_log,
        )

    print(f"\nDone. Model saved to: {save_stem}(.zip or .pt)")


if __name__ == "__main__":
    main()
