"""Tests for the main Territory Capture game environment."""

from __future__ import annotations

import unittest

from src.game import TerritoryCaptureGame
from src.rules import EMPTY, PLAYER_O, PLAYER_X


class TestTerritoryCaptureGame(unittest.TestCase):
    """Validate move handling, captures, and final scoring."""

    def setUp(self) -> None:
        self.game = TerritoryCaptureGame()

    def test_initial_legal_moves_cover_entire_board(self) -> None:
        self.assertEqual(len(self.game.get_legal_moves()), 36)

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

    def test_game_becomes_terminal_after_twenty_moves(self) -> None:
        moves = [
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
            (0, 5), (1, 0), (1, 1), (1, 2), (1, 3),
            (1, 4), (1, 5), (2, 0), (2, 1), (2, 2),
            (2, 3), (2, 4), (2, 5), (3, 0), (3, 1),
        ]

        for move in moves:
            self.game.apply_move(move)

        self.assertTrue(self.game.is_terminal())
        self.assertEqual(self.game.move_count, 20)
        self.assertEqual(self.game.get_legal_moves(), [])

    def test_draw_winner_is_none_when_totals_and_tiebreak_match(self) -> None:
        self.game.board = [
            [PLAYER_X, PLAYER_O, PLAYER_X, PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_X, PLAYER_O, PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_X, PLAYER_O, PLAYER_X, PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_X, PLAYER_O, PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_X, PLAYER_O, PLAYER_X, PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_X, PLAYER_O, PLAYER_X, PLAYER_O, PLAYER_X],
        ]
        self.assertIsNone(self.game.get_winner())

    def test_captures_remove_stones_and_credit_bonus_to_opponent(self) -> None:
        game = TerritoryCaptureGame(board_size=3, stones_per_player=2)
        game.apply_move((1, 1))  # X
        game.apply_move((0, 1))  # O
        game.apply_move((2, 2))  # X
        game.apply_move((1, 0))  # O captures center X

        self.assertEqual(game.board[1][1], EMPTY)
        self.assertEqual(game.get_capture_scores()[PLAYER_O], 1)
        self.assertEqual(game.get_capture_scores()[PLAYER_X], 0)
        self.assertIn((1, 1), game.last_captured_positions)

    def test_get_result_combines_territory_and_capture_scores(self) -> None:
        self.game.board = [
            [EMPTY, PLAYER_X, PLAYER_X, PLAYER_X, PLAYER_X, PLAYER_X],
            [PLAYER_X, PLAYER_X, PLAYER_X, PLAYER_X, PLAYER_X, PLAYER_X],
            [PLAYER_X, PLAYER_X, PLAYER_X, PLAYER_X, PLAYER_X, PLAYER_X],
            [PLAYER_X, PLAYER_X, PLAYER_X, PLAYER_X, PLAYER_X, PLAYER_X],
            [PLAYER_X, PLAYER_X, PLAYER_X, PLAYER_X, PLAYER_X, PLAYER_X],
            [PLAYER_X, PLAYER_X, PLAYER_X, PLAYER_X, PLAYER_X, PLAYER_X],
        ]
        self.game.captured_by = {PLAYER_X: 0, PLAYER_O: 2}

        result = self.game.get_result()

        self.assertEqual(result.territory_scores[PLAYER_X], 1)
        self.assertEqual(result.capture_scores[PLAYER_O], 2)
        self.assertEqual(result.scores[PLAYER_X], 1)
        self.assertEqual(result.scores[PLAYER_O], 2)
        self.assertEqual(result.winner, PLAYER_O)


if __name__ == "__main__":
    unittest.main()
