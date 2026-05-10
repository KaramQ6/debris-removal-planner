"""Custom Stable-Baselines3 callbacks for debris-removal training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeMetricsCallback(BaseCallback):
    """Log per-episode debris-removal metrics during PPO training.

    Tracks cleared count, total delta-v, fuel remaining, and reward.
    Saves a JSON history file that can be plotted later.
    """

    def __init__(
        self,
        output_path: str | Path = "results/training_history.json",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._output_path = Path(output_path)
        
        # Load existing history if file exists
        if self._output_path.exists():
            try:
                with open(self._output_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                self._episode_rewards = history.get("episode_rewards", [])
                self._episode_lengths = history.get("episode_lengths", [])
                self._episode_delta_vs = history.get("episode_delta_vs", [])
                self._episode_cleared = history.get("episode_cleared", [])
                if verbose:
                    print(f"Loaded existing history from {self._output_path} ({len(self._episode_rewards)} episodes)")
            except (json.JSONDecodeError, IOError):
                if verbose:
                    print(f"Warning: Could not load existing history from {self._output_path}. Starting fresh.")
                self._episode_rewards = []
                self._episode_lengths = []
                self._episode_delta_vs = []
                self._episode_cleared = []
        else:
            self._episode_rewards = []
            self._episode_lengths = []
            self._episode_delta_vs = []
            self._episode_cleared = []
            
        self._current_rewards: dict[int, float] = {}

    def _on_step(self) -> bool:
        # Check for episode completion via info dicts
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        if infos and dones is not None:
            for i, (info, done) in enumerate(zip(infos, dones)):
                if done:
                    self._episode_delta_vs.append(
                        float(info.get("total_delta_v", 0.0))
                    )
                    self._episode_cleared.append(int(info.get("cleared", 0)))
                    if "episode" in info:
                        self._episode_rewards.append(
                            float(info["episode"].get("r", 0.0))
                        )
                        self._episode_lengths.append(
                            int(info["episode"].get("l", 0))
                        )

        return True

    def _on_training_end(self) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        history = {
            "episode_rewards": self._episode_rewards,
            "episode_lengths": self._episode_lengths,
            "episode_delta_vs": self._episode_delta_vs,
            "episode_cleared": self._episode_cleared,
        }
        self._output_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if self.verbose:
            print(f"Training history saved to {self._output_path}")

    @property
    def summary(self) -> dict[str, Any]:
        """Return a summary dict of training performance."""
        if not self._episode_rewards:
            return {}
        return {
            "total_episodes": len(self._episode_rewards),
            "mean_reward": float(np.mean(self._episode_rewards)),
            "mean_delta_v": float(np.mean(self._episode_delta_vs))
            if self._episode_delta_vs
            else 0.0,
            "mean_cleared": float(np.mean(self._episode_cleared))
            if self._episode_cleared
            else 0.0,
        }
