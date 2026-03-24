"""Tests for AI evaluation utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.ai_eval import run_ai_vs_ai, run_ai_vs_minimax
from src.agents import RandomAgent


class RandomWrapper:
    """Adapter that matches the AI evaluation protocol."""

    def __init__(self, seed: int) -> None:
        self.agent = RandomAgent(seed=seed)

    def select_action(self, game):
        return self.agent.select_action(game)

    def reset_search_tree(self) -> None:
        return None


class TestAIEval(unittest.TestCase):
    """Check evaluation summaries and plot output."""

    def test_run_ai_vs_ai_returns_stats_and_plot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_path = Path(temp_dir) / "winrate.png"
            summary = run_ai_vs_ai(
                num_games=4,
                plot_path=plot_path,
                ai1_factory=lambda: RandomWrapper(seed=1),
                ai2_factory=lambda: RandomWrapper(seed=2),
            )

            self.assertEqual(summary.total_games, 4)
            self.assertEqual(summary.ai1_wins + summary.ai2_wins + summary.draws, 4)
            self.assertTrue(plot_path.exists())

    def test_run_ai_vs_minimax_returns_stats(self) -> None:
        summary = run_ai_vs_minimax(
            num_games=2,
            ai_factory=lambda: RandomWrapper(seed=3),
        )

        self.assertEqual(summary.total_games, 2)
        self.assertEqual(summary.ai1_wins + summary.ai2_wins + summary.draws, 2)


if __name__ == "__main__":
    unittest.main()
