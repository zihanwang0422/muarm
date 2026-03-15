"""Example: PPO training for Panda end-effector reaching.

Demonstrates:
- Creating a PandaReachEnv Gymnasium environment
- Training a PPO agent using Stable-Baselines3
- Testing the trained agent

Usage:
    python examples/demo_ppo_reach.py train    # Train the agent
    python examples/demo_ppo_reach.py test     # Test a trained agent
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from learning import PandaReachEnv, train_ppo, test_ppo

MODEL_PATH = "panda_ppo_reach_demo"

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    if mode == "train":
        train_ppo(
            env_class=PandaReachEnv,
            n_envs=8,
            total_timesteps=1_000_000,
            model_save_path=MODEL_PATH,
            env_kwargs={"scene_xml": "models/franka_emika_panda/scene.xml"},
        )
    elif mode == "test":
        test_ppo(
            env_class=PandaReachEnv,
            model_path=MODEL_PATH,
            total_episodes=5,
            env_kwargs={"scene_xml": "models/franka_emika_panda/scene.xml"},
        )
    else:
        print(f"Usage: python {sys.argv[0]} [train|test]")
