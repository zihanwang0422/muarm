"""Behavior Cloning (IL) for Panda tasks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class BCConfig:
    episodes: int = 20
    horizon: int = 120
    batch_size: int = 256
    epochs: int = 30
    lr: float = 3e-4


def _heuristic_action(obs_dict):
    achieved = obs_dict["achieved_goal"]
    desired = obs_dict["desired_goal"]
    delta = desired - achieved
    act_xyz = np.clip(delta / 0.08, -1.0, 1.0)
    grip = np.array([0.5], dtype=np.float32)
    return np.concatenate([act_xyz, grip]).astype(np.float32)


def collect_demos(env, cfg: BCConfig):
    obs_buf, act_buf = [], []
    for _ in range(cfg.episodes):
        obs, _ = env.reset()
        for _ in range(cfg.horizon):
            act = _heuristic_action(obs)
            obs_buf.append(obs["observation"].copy())
            act_buf.append(act.copy())
            obs, _, terminated, truncated, _ = env.step(act)
            if terminated or truncated:
                break
    return np.asarray(obs_buf, dtype=np.float32), np.asarray(act_buf, dtype=np.float32)


def train_bc(env, save_path: str, cfg: BCConfig | None = None):
    cfg = cfg or BCConfig()
    obs, act = collect_demos(env, cfg)
    if len(obs) == 0:
        raise RuntimeError("No demos collected for BC.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = BCPolicy(obs_dim=obs.shape[1], act_dim=act.shape[1]).to(device)

    x = torch.from_numpy(obs).to(device)
    y = torch.from_numpy(act).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    for _ in range(cfg.epochs):
        perm = torch.randperm(x.shape[0], device=device)
        x_shuf = x[perm]
        y_shuf = y[perm]
        for i in range(0, x.shape[0], cfg.batch_size):
            xb = x_shuf[i:i + cfg.batch_size]
            yb = y_shuf[i:i + cfg.batch_size]
            pred = policy(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(
        {
            "state_dict": policy.state_dict(),
            "obs_dim": obs.shape[1],
            "act_dim": act.shape[1],
        },
        save_path,
    )
    return save_path


def rollout_bc(env, model_path: str, episodes=5):
    ckpt = torch.load(model_path, map_location="cpu")
    policy = BCPolicy(ckpt["obs_dim"], ckpt["act_dim"])
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                x = torch.from_numpy(obs["observation"]).unsqueeze(0)
                action = policy(x).squeeze(0).numpy()
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
