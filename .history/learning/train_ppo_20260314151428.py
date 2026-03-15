"""PPO training and testing utilities for Panda RL environments.

Provides functions to train PPO agents on PandaReachEnv and PandaObstacleAvoidEnv
using Stable-Baselines3 with parallel environments.
"""
import os
import numpy as np
from typing import Optional


def train_ppo(
    env_class,
    n_envs=16,
    total_timesteps=10_000_000,
    model_save_path="panda_ppo_model",
    resume_from=None,
    policy_kwargs=None,
    learning_rate=2e-4,
    env_kwargs=None,
):
    """Train a PPO agent on a Panda Gymnasium environment.

    Args:
        env_class: The environment class (e.g., PandaReachEnv)
        n_envs: Number of parallel environments
        total_timesteps: Total training timesteps
        model_save_path: Path to save the trained model
        resume_from: Path to load a pre-trained model for continued training
        policy_kwargs: Custom policy network kwargs
        learning_rate: Learning rate (scalar or schedule function)
        env_kwargs: Additional keyword arguments for the environment

    Example::

        from learning import PandaReachEnv, train_ppo
        train_ppo(PandaReachEnv, n_envs=16, total_timesteps=5_000_000)
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError(
            "Training requires: pip install stable-baselines3 torch"
        )

    env_kwargs = env_kwargs or {}

    env = make_vec_env(
        env_id=lambda: env_class(**env_kwargs),
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"},
    )

    if resume_from is not None:
        model = PPO.load(resume_from, env=env)
    else:
        if policy_kwargs is None:
            policy_kwargs = dict(
                activation_fn=nn.ReLU,
                net_arch=[dict(pi=[256, 128], vf=[256, 128])],
            )
        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=2048,
            batch_size=2048,
            n_epochs=10,
            gamma=0.99,
            learning_rate=learning_rate,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log=f"./tensorboard/{env_class.__name__}/",
        )

    print(f"Training {env_class.__name__} | envs={n_envs} | steps={total_timesteps}")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(model_save_path)
    env.close()
    print(f"Model saved to: {model_save_path}")
    return model


def test_ppo(
    env_class,
    model_path,
    total_episodes=5,
    env_kwargs=None,
):
    """Test a trained PPO agent on a Panda environment.

    Args:
        env_class: The environment class
        model_path: Path to the trained PPO model
        total_episodes: Number of test episodes
        env_kwargs: Additional environment kwargs

    Example::

        from learning import PandaReachEnv, test_ppo
        test_ppo(PandaReachEnv, "panda_ppo_reach", total_episodes=10)
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        raise ImportError("Testing requires: pip install stable-baselines3")

    env_kwargs = env_kwargs or {}
    env_kwargs["render_mode"] = "human"
    env = env_class(**env_kwargs)
    model = PPO.load(model_path, env=env)

    success_count = 0
    print(f"Testing {env_class.__name__} | episodes={total_episodes}")

    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        is_success = info.get('is_success', False)
        if is_success:
            success_count += 1
        status = "SUCCESS" if is_success else "FAIL"
        print(f"  Episode {ep+1:3d} | Reward: {episode_reward:8.2f} | {status}")

    rate = (success_count / total_episodes) * 100
    print(f"Success rate: {rate:.1f}% ({success_count}/{total_episodes})")
    env.close()


if __name__ == "__main__":
    from panda_reach_env import PandaReachEnv
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode == "train":
        train_ppo(PandaReachEnv, n_envs=16, total_timesteps=5_000_000,
                  model_save_path="panda_ppo_reach")
    else:
        test_ppo(PandaReachEnv, "panda_ppo_reach", total_episodes=5)
