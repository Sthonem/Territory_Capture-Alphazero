"""Tests for self-play dataset generation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.agents import HeuristicAgent, RandomAgent
from src.dataset import (
    TrainingSample,
    create_one_hot_policy_target,
    load_samples_from_json,
    outcome_value_for_player,
    play_self_play_episode,
    save_samples_to_json,
)
from src.encoding import ACTION_SPACE_SIZE, action_to_index


class TestDatasetPipeline(unittest.TestCase):
    """Validate self-play sample generation and export."""

    def test_one_hot_policy_target_has_single_active_index(self) -> None:
        action_index = action_to_index((2, 3))

        policy_target = create_one_hot_policy_target(action_index)

        self.assertEqual(len(policy_target), ACTION_SPACE_SIZE)
        self.assertEqual(sum(policy_target), 1)
        self.assertEqual(policy_target[action_index], 1)

    def test_value_target_propagation_from_player_perspective(self) -> None:
        self.assertEqual(outcome_value_for_player("X", "X"), 1)
        self.assertEqual(outcome_value_for_player("X", "O"), -1)
        self.assertEqual(outcome_value_for_player(None, "X"), 0)

    def test_self_play_episode_produces_valid_training_samples(self) -> None:
        episode = play_self_play_episode(
            x_agent=HeuristicAgent(),
            o_agent=RandomAgent(seed=5),
        )

        self.assertEqual(episode.move_count, 16)
        self.assertEqual(len(episode.samples), 16)
        self.assertTrue(all(isinstance(sample, TrainingSample) for sample in episode.samples))

    def test_sample_structure_contains_aligned_state_policy_and_mask(self) -> None:
        episode = play_self_play_episode(
            x_agent=RandomAgent(seed=1),
            o_agent=RandomAgent(seed=2),
        )
        sample = episode.samples[0]

        self.assertEqual(len(sample.encoded_state), 2)
        self.assertEqual(len(sample.policy_target), ACTION_SPACE_SIZE)
        self.assertEqual(len(sample.legal_action_mask), ACTION_SPACE_SIZE)
        self.assertEqual(sum(sample.policy_target), 1)
        self.assertEqual(sample.policy_target[sample.selected_action_index], 1)
        self.assertEqual(sample.legal_action_mask[sample.selected_action_index], 1)

    def test_json_export_round_trip_preserves_samples(self) -> None:
        episode = play_self_play_episode(
            x_agent=RandomAgent(seed=3),
            o_agent=HeuristicAgent(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "samples.json"
            save_samples_to_json(episode.samples, output_path)
            loaded_samples = load_samples_from_json(output_path)

        self.assertEqual(loaded_samples, episode.samples)


if __name__ == "__main__":
    unittest.main()
