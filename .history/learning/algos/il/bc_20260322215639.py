"""Behavior Cloning (Imitation Learning).

Pipeline:
  1. collect_demos()  – run heuristic policy to gather (obs, action) pairs
  2. train_bc()       – train a MLP policy on the demos with supervised learning
  3. rollout_bc()     – run the learned policy (used in play.py)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class BCPolicy(nn.Module):
    """MLP policy: obs → action."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, act_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BCConfig:
    episodes: int = 30          # demo episodes
    horizon: int = 150          # max steps per episode
    batch_size: int = 256
    epochs: int = 50
    lr: float = 3e-4
    hidden: tuple[int, ...] = field(default_factory=lambda: (256, 256))
    device: str = "auto"        # "auto" | "cpu" | "cuda"


# ---------------------------------------------------------------------------
# Heuristic demo collection
# ---------------------------------------------------------------------------

def _heuristic_action(obs_dict: dict, action_dim: int = 4) -> np.ndarray:
    """Simple P-controller: push EE toward goal."""
    achieved = obs_dict["achieved_goal"]
    desired = obs_dict["desired_goal"]
    delta = desired - achieved
    act_xyz = np.clip(delta / 0.08, -1.0, 1.0)
    if action_dim == 4:
        grip = np.array([0.5], dtype=np.float32)
        return np.concatenate([act_xyz, grip]).astype(np.float32)
    elif action_dim == 3:
        return act_xyz.astype(np.float32)
    else:
        # For n-DOF joint-space envs, fall back to zero padding
        act = np.zeros(action_dim, dtype=np.float32)
        act[:3] = act_xyz.astype(np.float32)
        return act


def collect_demos(env, cfg: BCConfig) -> tuple[np.ndarray, np.ndarray]:
    """Rollout heuristic policy to collect demo transitions."""
    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    act_dim = env.action_space.shape[0]

    for ep in range(cfg.episodes):
        obs, _ = env.reset()
        for _ in range(cfg.horizon):
            act = _heuristic_action(obs, act_dim)
            obs_buf.append(obs["observation"].copy())
            act_buf.append(act.copy())
            obs, _, terminated, truncated, _ = env.step(act)
            if terminated or truncated:
                break

    if len(obs_buf) == 0:
        raise RuntimeError("No demo transitions collected — check heuristic policy.")

    print(f"[BC] Collected {len(obs_buf)} transitions from {cfg.episodes} episodes.")
    return np.asarray(obs_buf, dtype=np.float32), np.asarray(act_buf, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_bc(env, save_path: str, cfg: BCConfig | None = None) -> str:
    """Train BC policy and save to `save_path` (.pt)."""
    cfg = cfg or BCConfig()

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    obs, act = collect_demos(env, cfg)
    obs_t = torch.from_numpy(obs).to(device)
    act_t = torch.from_numpy(act).to(device)

    policy = BCPolicy(obs_t.shape[1], act_t.shape[1], hidden=cfg.hidden).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(obs_t, act_t)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    print(f"[BC] Training on {device}: obs={obs_t.shape[1]}, act={act_t.shape[1]}, "
          f"epochs={cfg.epochs}, samples={len(obs)}")

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            pred = policy(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}/{cfg.epochs}  loss={epoch_loss/len(obs):.6f}")

    save_path = str(save_path)
    if not save_path.endswith(".pt"):
        save_path += ".pt"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": policy.state_dict(),
        "obs_dim": obs_t.shape[1],
        "act_dim": act_t.shape[1],
        "hidden": list(cfg.hidden),
    }, save_path)
    print(f"[BC] Saved model → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Rollout (used by play.py)
# ---------------------------------------------------------------------------

def rollout_bc(env, model_path: str, episodes: int = 5) -> float:
    """Run BC policy and return mean success rate."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    policy = BCPolicy(ckpt["obs_dim"], ckpt["act_dim"], hidden=tuple(ckpt["hidden"]))
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    successes = 0
    with torch.no_grad():
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done:
                x = torch.from_numpy(obs["observation"]).unsqueeze(0)
                action = policy(x).squeeze(0).numpy()
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            success = bool(info.get("is_success", terminated))
            successes += int(success)
            print(f"  ep {ep+1}/{episodes}: {'SUCCESS' if success else 'fail'}  steps={steps}")

    sr = successes / episodes
    print(f"[BC] Mean success rate: {sr:.2f}  ({successes}/{episodes})")
    return sr
