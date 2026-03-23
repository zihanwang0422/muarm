"""Play – load a trained model and run it in the MuJoCo viewer at real-time speed.

The viewer renders at the simulation's own timestep so the motion looks
natural (neither too fast nor too slow).

Examples
--------
# Play RL model
python learning/play.py --robot panda --task reach --algo sac \
    --model learning/runs/panda_reach_sac.zip

# Play BC model
python learning/play.py --robot panda --task reach --algo bc \
    --model learning/runs/panda_reach_bc.pt

# 10 episodes
python learning/play.py --robot panda --task push --algo sac \
    --model learning/runs/panda_push_sac.zip --episodes 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.envs.registry import make_env
from learning.algos.rl_trainer import load_rl_model
from learning.algos.il.bc import BCPolicy


# ---------------------------------------------------------------------------
# Real-time step helper
# ---------------------------------------------------------------------------

def _rt_step(env, action: np.ndarray) -> tuple:
    """Step env and sleep to match wall-clock simulation time."""
    t0 = time.perf_counter()
    result = env.step(action)
    # env.dt = model.opt.timestep * n_substeps  (set in MuJocoRobotEnv)
    dt = getattr(env, "dt", 0.02)
    elapsed = time.perf_counter() - t0
    sleep_time = dt - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)
    return result


# ---------------------------------------------------------------------------
# Policy wrappers
# ---------------------------------------------------------------------------

def _load_policy(algo: str, model_path: str):
    """Return a callable: obs_dict → action_array."""
    algo_key = algo.lower()

    if algo_key == "bc":
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        policy_net = BCPolicy(ckpt["obs_dim"], ckpt["act_dim"], hidden=tuple(ckpt["hidden"]))
        policy_net.load_state_dict(ckpt["state_dict"])
        policy_net.eval()

        def _bc_predict(obs_dict):
            x = torch.from_numpy(obs_dict["observation"]).unsqueeze(0)
            with torch.no_grad():
                return policy_net(x).squeeze(0).numpy()

        return _bc_predict

    else:
        sb3_model = load_rl_model(algo_key, model_path)

        def _sb3_predict(obs_dict):
            action, _ = sb3_model.predict(obs_dict, deterministic=True)
            return action

        return _sb3_predict


# ---------------------------------------------------------------------------
# Play loop
# ---------------------------------------------------------------------------

def play(robot: str, task: str, algo: str, model_path: str, episodes: int) -> None:
    env = make_env(robot, task, render_mode="human")
    predict = _load_policy(algo, model_path)

    successes = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0.0

        while not done:
            action = predict(obs)
            obs, reward, terminated, truncated, info = _rt_step(env, action)
            total_reward += float(reward)
            done = terminated or truncated
            steps += 1

        success = bool(info.get("is_success", terminated))
        successes += int(success)
        status = "SUCCESS ✓" if success else "fail ✗"
        print(f"  ep {ep+1:3d}/{episodes}  {status}  steps={steps:4d}  reward={total_reward:.3f}")

    env.close()
    sr = successes / episodes
    print(f"\nFinal success rate: {sr:.2f}  ({successes}/{episodes})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Play a trained robot policy with MuJoCo viewer.")
    parser.add_argument("--robot",    type=str, default="panda",
                        help="Robot type: panda | so_arm")
    parser.add_argument("--task",     type=str, default="reach",
                        choices=["reach", "push", "pick_place"],
                        help="Task name")
    parser.add_argument("--algo",     type=str, default="sac",
                        choices=["ppo", "sac", "td3", "ddpg", "bc"],
                        help="Algorithm used to train the model")
    parser.add_argument("--model",    type=str, required=True,
                        help="Path to the saved model (.zip for SB3 or .pt for BC)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    args = parser.parse_args()

    play(
        robot=args.robot,
        task=args.task,
        algo=args.algo,
        model_path=args.model,
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()
