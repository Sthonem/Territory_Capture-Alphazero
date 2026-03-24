"""Tests for checkpoint arena evaluation."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from src.arena import evaluate_model_against_best


class TestArena(unittest.TestCase):
    """Check arena result structure for candidate evaluation."""

    def test_evaluate_model_against_best_returns_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.pth"
            incumbent = Path(temp_dir) / "incumbent.pth"
            shutil.copyfile("src/model.pth", candidate)
            shutil.copyfile("src/model.pth", incumbent)

            result = evaluate_model_against_best(
                candidate_path=candidate,
                incumbent_path=incumbent,
                num_games=2,
                acceptance_threshold=0.0,
                promote_on_win=False,
            )

            self.assertEqual(result.summary.total_games, 2)
            self.assertEqual(
                result.summary.ai1_wins + result.summary.ai2_wins + result.summary.draws,
                2,
            )


if __name__ == "__main__":
    unittest.main()
