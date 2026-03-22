"""Tests for the main Territory Capture game environment."""

from __future__ import annotations

import unittest

from src.game import TerritoryCaptureGame
from src.rules import EMPTY, PLAYER_O, PLAYER_X


class TestTerritoryCaptureGame(unittest.TestCase):
    """Validate move handling, turns, and terminal game flow."""

    def setUp(self) -> None:
        self.game = TerritoryCaptureGame()

    def test_initial_legal_moves_cover_entire_board(self) -> None:
        self.assertEqual(len(self.game.get_legal_moves()), 25)

    def test_move_placement_updates_board(self) -> None:
        self.game.apply_move((0, 0))
        self.assertEqual(self.game.board[0][0], PLAYER_X)
        self.assertEqual(self.game.move_count, 1)

    def test_turn_switches_after_move(self) -> None:
        self.assertEqual(self.game.current_player, PLAYER_X)
        self.game.apply_move((0, 0))
        self.assertEqual(self.game.current_player, PLAYER_O)

    def test_playing_on_occupied_cell_raises_error(self) -> None:
        self.game.apply_move((0, 0))
        with self.assertRaises(ValueError):
            self.game.apply_move((0, 0))

    def test_game_becomes_terminal_after_sixteen_moves(self) -> None:
        moves = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (3, 0),
        ]

        for move in moves:
            self.game.apply_move(move)

        self.assertTrue(self.game.is_terminal())
        self.assertEqual(self.game.move_count, 16)
        self.assertEqual(self.game.get_legal_moves(), [])

    def test_draw_winner_is_none(self) -> None:
        self.game.board = [
            ["X", "O", "X", "O", "X"],
            ["O", ".", ".", ".", "O"],
            ["X", ".", ".", ".", "X"],
            ["O", ".", ".", ".", "O"],
            ["X", "O", "X", "O", "X"],
        ]
        self.assertIsNone(self.game.get_winner())

    def test_captures_remove_stones_without_empty_neighbors(self) -> None:
        game = TerritoryCaptureGame(board_size=3, stones_per_player=5)
        setup_moves = [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        ]

        for move in setup_moves:
            game.apply_move(move)

        self.assertEqual(game.board[1][1], EMPTY)

        game.apply_move((1, 1))

        expected_board = [
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
        ]
        self.assertEqual(game.board, expected_board)
        self.assertEqual(len(game.last_captured_positions), 9)

    def test_territory_scoring_uses_board_after_captures(self) -> None:
        game = TerritoryCaptureGame(board_size=3, stones_per_player=5)
        setup_moves = [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
            (1, 1),
        ]

        for move in setup_moves:
            game.apply_move(move)

        result = game.evaluate_territory()

        self.assertEqual(result.scores[PLAYER_X], 0)
        self.assertEqual(result.scores[PLAYER_O], 0)


if __name__ == "__main__":
    unittest.main()
