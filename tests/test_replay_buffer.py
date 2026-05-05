"""Tests for the replay buffer used by iterative training."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    """Validate bounded accumulation and JSON persistence."""

    def test_extend_trims_oldest_records_when_over_capacity(self) -> None:
        buffer = ReplayBuffer(max_samples=3)
        buffer.extend([{"id": 1}, {"id": 2}])
        buffer.extend([{"id": 3}, {"id": 4}])

        self.assertEqual(buffer.records, [{"id": 2}, {"id": 3}, {"id": 4}])

    def test_save_and_load_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "buffer.json"
            saved_buffer = ReplayBuffer(max_samples=5, records=[{"id": 1}, {"id": 2}])
            saved_buffer.save_json(path)

            loaded_buffer = ReplayBuffer(max_samples=5)
            loaded_buffer.load_json(path)

        self.assertEqual(loaded_buffer.records, [{"id": 1}, {"id": 2}])


if __name__ == "__main__":
    unittest.main()
