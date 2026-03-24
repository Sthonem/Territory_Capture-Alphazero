"""Tests for AlphaZero-style self-play data generation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.agents import RandomAgent
from src.self_play import generate_self_play_games


class DummySearchAgent:
    """Simple search-agent adapter with one-hot policies."""

    def __init__(self) -> None:
        self.agent = RandomAgent(seed=5)

    def reset_search_tree(self) -> None:
        return None

    def select_action_with_policy(
        self,
        game,
        temperature: float = 0.0,
        add_exploration_noise: bool = False,
    ):
        action = self.agent.select_action(game)
        action_index = action[0] * 5 + action[1]
        policy = [0.0] * 25
        policy[action_index] = 1.0
        return action, policy


class TestSelfPlay(unittest.TestCase):
    """Validate self-play export format."""

    def test_generate_self_play_games_writes_expected_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "self_play_data.json"
            summary = generate_self_play_games(
                num_games=2,
                output_path=output_path,
                agent=DummySearchAgent(),
            )

            data = json.loads(output_path.read_text())

        self.assertEqual(summary.total_games, 2)
        self.assertGreater(summary.total_samples, 0)
        self.assertIn("state", data[0])
        self.assertIn("policy", data[0])
        self.assertIn("value", data[0])
        self.assertEqual(len(data[0]["state"]), 2)
        self.assertEqual(len(data[0]["policy"]), 25)


if __name__ == "__main__":
    unittest.main()
