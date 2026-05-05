"""Tests for the model training pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.encoding import ACTION_SPACE_SIZE, BOARD_SIZE
from src.train import load_self_play_records, train_policy_value_model


class TestTrain(unittest.TestCase):
    """Validate self-play loading and checkpoint writing."""

    def test_train_policy_value_model_saves_checkpoint(self) -> None:
        records = [
            {
                "state": [[[0] * BOARD_SIZE for _ in range(BOARD_SIZE)] for _ in range(2)],
                "policy": [1.0] + [0.0] * (ACTION_SPACE_SIZE - 1),
                "value": 1,
            }
            for _ in range(10)
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.json"
            output_path = Path(temp_dir) / "model.pth"
            data_path.write_text(json.dumps(records), encoding="utf-8")

            summary = train_policy_value_model(
                data_path=data_path,
                output_path=output_path,
                epochs=1,
                batch_size=2,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(summary.output_path, output_path.resolve())
            self.assertTrue(summary.metadata_path.exists())
            self.assertEqual(len(summary.history), 1)
            self.assertEqual(summary.records_used, 10)

    def test_load_self_play_records_reads_json(self) -> None:
        records = [{"state": [], "policy": [], "value": 0}]

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.json"
            data_path.write_text(json.dumps(records), encoding="utf-8")
            loaded = load_self_play_records(data_path)

        self.assertEqual(loaded, records)

    def test_train_policy_value_model_accepts_in_memory_records(self) -> None:
        records = [
            {
                "state": [[[0] * BOARD_SIZE for _ in range(BOARD_SIZE)] for _ in range(2)],
                "policy": [1.0] + [0.0] * (ACTION_SPACE_SIZE - 1),
                "value": 0,
            }
            for _ in range(10)
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.pth"
            summary = train_policy_value_model(
                output_path=output_path,
                epochs=1,
                batch_size=2,
                records=records,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(summary.output_path, output_path.resolve())
            self.assertTrue(summary.metadata_path.exists())


if __name__ == "__main__":
    unittest.main()
