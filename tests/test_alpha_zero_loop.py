"""Tests for the top-level AlphaZero-style iteration script."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.alpha_zero_loop import run_alpha_zero_iteration
from src.model import PolicyValueNet
import torch


class TestAlphaZeroLoop(unittest.TestCase):
    """Validate one end-to-end iteration on tiny settings."""

    def test_run_alpha_zero_iteration_returns_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            incumbent_path = temp_root / "incumbent_model.pth"
            torch.save(PolicyValueNet().state_dict(), incumbent_path)
            summary = run_alpha_zero_iteration(
                self_play_games=1,
                arena_games=2,
                epochs=1,
                self_play_path=temp_root / "self_play.json",
                replay_buffer_path=temp_root / "replay_buffer.json",
                candidate_model_path=temp_root / "latest_model.pth",
                incumbent_model_path=incumbent_path,
                acceptance_threshold=0.0,
            )

            self.assertEqual(summary.self_play.total_games, 1)
            self.assertTrue(summary.training.output_path.exists())
            self.assertEqual(summary.arena.summary.total_games, 2)
            self.assertTrue((temp_root / "replay_buffer.json").exists())
            self.assertGreater(summary.replay_buffer_size, 0)
            self.assertTrue(summary.metadata_path.exists())


if __name__ == "__main__":
    unittest.main()
