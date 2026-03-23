"""RL trainer using Stable-Baselines3.

Supports PPO, SAC, TD3, DDPG with optional MuJoCo headless training
(render=False) or live visualization (render=True).

SB3 callback for periodic success-rate logging is also included.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

_ALGO_MAP = {
    "ppo":  PPO,
    "sac":  SAC,
    "td3":  TD3,
    "ddpg": DDPG,
}

# Default hyperparams tuned for manipulation tasks
_DEFAULT_KWARGS: dict[str, dict] = {
    "ppo":  {"n_steps": 2048, "batch_size": 256, "learning_rate": 3e-4, "ent_coef": 0.01},
    "sac":  {"batch_size": 256, "learning_rate": 3e-4, "buffer_size": 300_000,
              "learning_starts": 1000, "train_freq": 1},
    "td3":  {"batch_size": 256, "learning_rate": 3e-4, "buffer_size": 300_000,
              "learning_starts": 1000, "train_freq": 1},
    "ddpg": {"batch_size": 256, "learning_rate": 3e-4, "buffer_size": 300_000,
              "learning_starts": 1000, "train_freq": 1},
}


class SuccessRateCallback(BaseCallback):
    """Log mean success rate every `check_freq` steps."""

    def __init__(self, check_freq: int = 2000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self._successes: list[bool] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "is_success" in info:
                self._successes.append(bool(info["is_success"]))

        if self.n_calls % self.check_freq == 0 and self._successes:
            sr = float(np.mean(self._successes[-500:]))
            if self.verbose:
                print(f"[SuccessRate @ {self.num_timesteps}] {sr:.3f}")
            self.logger.record("train/success_rate", sr)
            self._successes = self._successes[-500:]
        return True


def train_rl(
    env_fn: Callable,
    algo: str,
    total_timesteps: int,
    save_path: str,
    *,
    n_envs: int = 1,
    tensorboard_log: str | None = None,
    extra_kwargs: dict | None = None,
) -> None:
    """Train an SB3 RL agent.

    Args:
        env_fn:          Zero-arg callable returning a gymnasium env (headless).
        algo:            One of 'ppo', 'sac', 'td3', 'ddpg'.
        total_timesteps: Total environment steps.
        save_path:       Path to save the final model (without .zip).
        n_envs:          Number of parallel envs (>1 uses SubprocVecEnv).
        tensorboard_log: Directory for TensorBoard logs (None = disabled).
        extra_kwargs:    Override default SB3 hyperparameters.
    """
    algo_key = algo.lower()
    if algo_key not in _ALGO_MAP:
        raise ValueError(f"Unknown algo '{algo}'. Choose from: {list(_ALGO_MAP)}")

    # Wrap with Monitor so SB3 can track episode stats
    monitored = lambda: Monitor(env_fn())  # noqa: E731

    if n_envs > 1:
        vec_env = SubprocVecEnv([monitored] * n_envs)
    else:
        vec_env = DummyVecEnv([monitored])

    algo_cls = _ALGO_MAP[algo_key]
    kwargs = {**_DEFAULT_KWARGS[algo_key], **(extra_kwargs or {})}
    if tensorboard_log:
        kwargs["tensorboard_log"] = tensorboard_log

    model = algo_cls("MultiInputPolicy", vec_env, verbose=1, **kwargs)

    callbacks = [SuccessRateCallback(check_freq=2000, verbose=1)]

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(save_path)
    vec_env.close()
    print(f"[RL] Saved model → {save_path}.zip")


def load_rl_model(algo: str, model_path: str):
    """Load a saved SB3 model."""
    algo_key = algo.lower()
    if algo_key not in _ALGO_MAP:
        raise ValueError(f"Unknown algo '{algo}'.")
    return _ALGO_MAP[algo_key].load(model_path)
