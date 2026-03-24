"""Tests for the dataset generation script helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.generate_dataset import (
    build_generation_tasks,
    build_timestamped_output_path,
    generate_dataset,
    get_matchup_code,
)


class TestGenerateDataset(unittest.TestCase):
    """Validate naming and task-splitting helpers for dataset generation."""

    def test_matchup_code_is_generated_correctly(self) -> None:
        self.assertEqual(get_matchup_code("minimax", "heuristic"), "mh")
        self.assertEqual(get_matchup_code("heuristic", "heuristic"), "hh")
        self.assertEqual(get_matchup_code("heuristic", "random"), "hr")
        self.assertEqual(get_matchup_code("random", "random"), "rr")

    def test_timestamped_output_path_keeps_base_name(self) -> None:
        path = build_timestamped_output_path("mh_40k.json")

        self.assertTrue(path.name.startswith("mh_40k_"))
        self.assertTrue(path.name.endswith(".json"))

    def test_task_builder_preserves_exact_game_count(self) -> None:
        tasks = build_generation_tasks(
            num_games=250,
            x_agent_name="heuristic",
            o_agent_name="heuristic",
            seed=10,
            encoding_name="turn-plane",
        )

        self.assertEqual(sum(task[0] for task in tasks), 250)

    def test_generate_dataset_returns_summary(self) -> None:
        summary = generate_dataset(
            num_games=2,
            x_agent_name="heuristic",
            o_agent_name="random",
            output_path="tiny_dataset.json",
            workers=1,
            progress_every=0,
        )

        self.assertEqual(summary.total_games, 2)
        self.assertEqual(summary.matchup_code, "hr")
        self.assertTrue(Path(summary.output_path).name.startswith("tiny_dataset_"))


if __name__ == "__main__":
    unittest.main()
