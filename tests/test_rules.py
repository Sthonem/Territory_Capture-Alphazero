"""Tests for Territory Capture scoring, capture, and tie-break rules."""

from __future__ import annotations

import unittest

from src.rules import (
    EMPTY,
    PLAYER_O,
    PLAYER_X,
    TerritoryResult,
    determine_winner,
    evaluate_territory,
    evaluate_tiebreaker,
    find_captured_stones,
)


class TestTerritoryRules(unittest.TestCase):
    """Check final scoring behavior for the 6x6 rule set."""

    def test_territory_evaluation_counts_neighbors(self) -> None:
        board = [
            [PLAYER_X, PLAYER_X, PLAYER_X, EMPTY, EMPTY, EMPTY],
            [PLAYER_X, EMPTY, PLAYER_O, EMPTY, EMPTY, EMPTY],
            [PLAYER_X, PLAYER_O, PLAYER_O, EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
        ]

        result = evaluate_territory(board)

        self.assertEqual(result.territory_map[(1, 1)], PLAYER_X)
        self.assertGreater(result.scores[PLAYER_X], 0)

    def test_equal_neighbor_counts_create_neutral_cell(self) -> None:
        board = [
            [PLAYER_X, EMPTY, PLAYER_O, EMPTY, PLAYER_X, EMPTY],
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
            [PLAYER_O, EMPTY, PLAYER_X, EMPTY, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
            [PLAYER_X, EMPTY, PLAYER_O, EMPTY, PLAYER_X, EMPTY],
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
        ]

        result = evaluate_territory(board)

        self.assertEqual(result.territory_map[(0, 1)], EMPTY)

    def test_capture_rule_can_remove_both_colors_simultaneously(self) -> None:
        board = [
            [PLAYER_X, PLAYER_O, EMPTY],
            [PLAYER_O, PLAYER_X, EMPTY],
            [EMPTY, EMPTY, PLAYER_O],
        ]

        result = find_captured_stones(board)

        self.assertCountEqual(
            result.positions,
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        )
        self.assertEqual(result.captured_counts[PLAYER_X], 2)
        self.assertEqual(result.captured_counts[PLAYER_O], 2)

    def test_tiebreaker_counts_only_claimed_empty_center_cells(self) -> None:
        territory_map = {
            (2, 2): PLAYER_X,
            (2, 3): EMPTY,
            (3, 2): PLAYER_O,
        }

        scores = evaluate_tiebreaker(territory_map, board_size=6)

        self.assertEqual(scores[PLAYER_X], 1)
        self.assertEqual(scores[PLAYER_O], 1)

    def test_determine_winner_uses_capture_bonus_before_tiebreak(self) -> None:
        territory_result = TerritoryResult(
            territory_map={},
            scores={PLAYER_X: 2, PLAYER_O: 2},
        )

        winner = determine_winner(
            territory_result=territory_result,
            captured_by={PLAYER_X: 1, PLAYER_O: 0},
            board_size=6,
        )

        self.assertEqual(winner, PLAYER_X)

    def test_determine_winner_uses_center_tiebreak_when_totals_match(self) -> None:
        territory_result = TerritoryResult(
            territory_map={
                (2, 2): PLAYER_X,
                (2, 3): PLAYER_X,
                (3, 2): PLAYER_O,
                (3, 3): EMPTY,
            },
            scores={PLAYER_X: 4, PLAYER_O: 4},
        )

        winner = determine_winner(
            territory_result=territory_result,
            captured_by={PLAYER_X: 0, PLAYER_O: 0},
            board_size=6,
        )

        self.assertEqual(winner, PLAYER_X)


if __name__ == "__main__":
    unittest.main()
