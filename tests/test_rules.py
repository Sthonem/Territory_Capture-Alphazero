"""Tests for territory scoring rules."""

from __future__ import annotations

import unittest

from src.rules import EMPTY, PLAYER_O, PLAYER_X, evaluate_territory


class TestTerritoryRules(unittest.TestCase):
    """Check territory ownership and score totals."""

    def test_territory_evaluation_counts_neighbors(self) -> None:
        board = [
            [PLAYER_X, PLAYER_X, PLAYER_X, EMPTY, EMPTY],
            [PLAYER_X, EMPTY, PLAYER_O, EMPTY, EMPTY],
            [PLAYER_X, PLAYER_O, PLAYER_O, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
        ]

        result = evaluate_territory(board)

        self.assertEqual(result.territory_map[(1, 1)], PLAYER_X)
        self.assertEqual(result.scores[PLAYER_X], 1)

    def test_equal_neighbor_counts_create_neutral_cell(self) -> None:
        board = [
            [PLAYER_X, EMPTY, PLAYER_O, EMPTY, PLAYER_X],
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
            [PLAYER_O, EMPTY, PLAYER_X, EMPTY, PLAYER_O],
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
            [PLAYER_X, EMPTY, PLAYER_O, EMPTY, PLAYER_X],
        ]

        result = evaluate_territory(board)

        self.assertEqual(result.territory_map[(0, 1)], EMPTY)
        self.assertEqual(result.scores[PLAYER_X], 0)
        self.assertEqual(result.scores[PLAYER_O], 0)

    def test_draw_scores_are_detected(self) -> None:
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, EMPTY, EMPTY, EMPTY, PLAYER_O],
            [PLAYER_X, EMPTY, EMPTY, EMPTY, PLAYER_X],
            [PLAYER_O, EMPTY, EMPTY, EMPTY, PLAYER_O],
            [PLAYER_X, PLAYER_O, PLAYER_X, PLAYER_O, PLAYER_X],
        ]

        result = evaluate_territory(board)

        self.assertEqual(result.scores[PLAYER_X], result.scores[PLAYER_O])


if __name__ == "__main__":
    unittest.main()
