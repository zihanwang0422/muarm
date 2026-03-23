"""Task protocol – defines the interface every task must implement."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseTask(ABC):
    """All task objects must implement this interface.

    The env calls these methods; no robot-specific code should live here.
    """

    @abstractmethod
    def reset(self, model, data, np_random) -> None:
        """Randomize episode-level state (object pose, etc.)."""

    @abstractmethod
    def sample_goal(self, np_random) -> np.ndarray:
        """Return a (3,) goal position for this episode."""

    @abstractmethod
    def get_obs(self, model, data, env) -> np.ndarray:
        """Return task-specific observation vector."""

    @abstractmethod
    def achieved_goal(self, model, data, env) -> np.ndarray:
        """Return the currently achieved goal (3,)."""

    @abstractmethod
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> float:
        """Return scalar reward."""

    @abstractmethod
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        """Return True when the episode is considered solved."""
