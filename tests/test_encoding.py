"""Tests for the state and action representation layer."""

from __future__ import annotations

import unittest

from src.encoding import (
    ACTION_SPACE_SIZE,
    action_to_index,
    encode_state,
    encode_state_with_turn_plane,
    get_legal_action_mask,
    index_to_action,
)
from src.game import TerritoryCaptureGame


class TestEncoding(unittest.TestCase):
    """Validate fixed-size state and action representations."""

    def test_encode_state_has_expected_shape(self) -> None:
        game = TerritoryCaptureGame()

        encoded = encode_state(game)

        self.assertEqual(len(encoded), 2)
        self.assertEqual(len(encoded[0]), 5)
        self.assertEqual(len(encoded[0][0]), 5)

    def test_encode_state_uses_current_player_perspective_for_x(self) -> None:
        game = TerritoryCaptureGame()
        game.board = [
            ["X", ".", ".", ".", "."],
            [".", "O", ".", ".", "."],
            [".", ".", "X", ".", "."],
            [".", ".", ".", "O", "."],
            [".", ".", ".", ".", "."],
        ]
        game.current_player = "X"

        encoded = encode_state(game)

        self.assertEqual(encoded[0][0][0], 1)
        self.assertEqual(encoded[0][2][2], 1)
        self.assertEqual(encoded[1][1][1], 1)
        self.assertEqual(encoded[1][3][3], 1)
        self.assertEqual(encoded[0][1][1], 0)

    def test_encode_state_uses_current_player_perspective_for_o(self) -> None:
        game = TerritoryCaptureGame()
        game.board = [
            ["X", ".", ".", ".", "."],
            [".", "O", ".", ".", "."],
            [".", ".", "X", ".", "."],
            [".", ".", ".", "O", "."],
            [".", ".", ".", ".", "."],
        ]
        game.current_player = "O"

        encoded = encode_state(game)

        self.assertEqual(encoded[0][1][1], 1)
        self.assertEqual(encoded[0][3][3], 1)
        self.assertEqual(encoded[1][0][0], 1)
        self.assertEqual(encoded[1][2][2], 1)
        self.assertEqual(encoded[0][0][0], 0)

    def test_action_index_round_trip(self) -> None:
        for index in range(ACTION_SPACE_SIZE):
            action = index_to_action(index)
            self.assertEqual(action_to_index(action), index)

    def test_turn_plane_encoding_has_expected_shape_and_turn_channel(self) -> None:
        game = TerritoryCaptureGame()
        game.board = [
            ["X", ".", ".", ".", "."],
            [".", "O", ".", ".", "."],
            [".", ".", ".", ".", "."],
            [".", ".", ".", ".", "."],
            [".", ".", ".", ".", "."],
        ]
        game.current_player = "X"

        encoded = encode_state_with_turn_plane(game)

        self.assertEqual(len(encoded), 3)
        self.assertEqual(encoded[0][0][0], 1)
        self.assertEqual(encoded[1][1][1], 1)
        self.assertTrue(all(value == 1 for row in encoded[2] for value in row))

    def test_legal_action_mask_marks_legal_moves(self) -> None:
        game = TerritoryCaptureGame()
        game.board = [
            ["X", ".", ".", ".", "."],
            [".", "O", ".", ".", "."],
            [".", ".", "X", ".", "."],
            [".", ".", ".", "O", "."],
            [".", ".", ".", ".", "."],
        ]

        mask = get_legal_action_mask(game)

        self.assertEqual(mask[action_to_index((0, 0))], 0)
        self.assertEqual(mask[action_to_index((1, 1))], 0)
        self.assertEqual(mask[action_to_index((0, 1))], 1)
        self.assertEqual(len(mask), ACTION_SPACE_SIZE)

    def test_legal_action_mask_matches_engine_legal_moves(self) -> None:
        game = TerritoryCaptureGame()
        game.apply_action((2, 2))
        game.apply_action((1, 1))
        game.apply_action((0, 0))

        mask = get_legal_action_mask(game)
        legal_moves = set(game.get_legal_actions())
        mask_moves = {
            index_to_action(index)
            for index, value in enumerate(mask)
            if value == 1
        }

        self.assertEqual(mask_moves, legal_moves)


if __name__ == "__main__":
    unittest.main()
