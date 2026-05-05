"""Rules helpers for Territory Capture scoring, captures, and tie-breaks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

Board = List[List[str]]
Position = Tuple[int, int]

PLAYER_X = "X"
PLAYER_O = "O"
EMPTY = "."

NEIGHBOR_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True)
class TerritoryResult:
    """Stores the final territory calculation for a finished game."""

    territory_map: Dict[Position, str]
    scores: Dict[str, int]


@dataclass(frozen=True)
class CaptureResult:
    """Stores simultaneous capture positions and captured-stone counts."""

    positions: List[Position]
    captured_counts: Dict[str, int]


def find_captured_stones(
    board: Sequence[Sequence[str]],
    empty_symbol: str = EMPTY,
) -> CaptureResult:
    """Return stones captured under the new local-majority pressure rule.

    A stone is captured when both conditions hold:
    1. it has at least two neighboring opponent stones
    2. opponent neighboring stones outnumber friendly neighboring stones

    Empty cells are ignored when counting neighbors. The result is based on the
    original board state so captures can be removed simultaneously.
    """

    board_size = len(board)
    captured_positions: List[Position] = []
    captured_counts = {PLAYER_X: 0, PLAYER_O: 0}

    for row in range(board_size):
        for col in range(board_size):
            owner = board[row][col]
            if owner == empty_symbol:
                continue
            opponent = PLAYER_O if owner == PLAYER_X else PLAYER_X

            friendly_neighbors = 0
            opponent_neighbors = 0
            for neighbor_row, neighbor_col in get_neighbors((row, col), board_size):
                value = board[neighbor_row][neighbor_col]
                if value == owner:
                    friendly_neighbors += 1
                elif value == opponent:
                    opponent_neighbors += 1

            if opponent_neighbors >= 2 and opponent_neighbors > friendly_neighbors:
                captured_positions.append((row, col))
                captured_counts[owner] += 1

    return CaptureResult(
        positions=captured_positions,
        captured_counts=captured_counts,
    )


def get_neighbors(position: Position, board_size: int) -> List[Position]:
    """Return all valid neighboring coordinates for a board position."""

    row, col = position
    neighbors: List[Position] = []

    for row_offset, col_offset in NEIGHBOR_OFFSETS:
        next_row = row + row_offset
        next_col = col + col_offset
        if 0 <= next_row < board_size and 0 <= next_col < board_size:
            neighbors.append((next_row, next_col))

    return neighbors


def evaluate_territory(
    board: Sequence[Sequence[str]],
    empty_symbol: str = EMPTY,
) -> TerritoryResult:
    """Evaluate all empty cells simultaneously based on adjacent stones only."""

    board_size = len(board)
    territory_map: Dict[Position, str] = {}
    scores = {PLAYER_X: 0, PLAYER_O: 0}

    for row in range(board_size):
        for col in range(board_size):
            if board[row][col] != empty_symbol:
                continue

            x_count = 0
            o_count = 0

            for neighbor_row, neighbor_col in get_neighbors((row, col), board_size):
                value = board[neighbor_row][neighbor_col]
                if value == PLAYER_X:
                    x_count += 1
                elif value == PLAYER_O:
                    o_count += 1

            if x_count > o_count:
                owner = PLAYER_X
                scores[PLAYER_X] += 1
            elif o_count > x_count:
                owner = PLAYER_O
                scores[PLAYER_O] += 1
            else:
                owner = empty_symbol

            territory_map[(row, col)] = owner

    return TerritoryResult(territory_map=territory_map, scores=scores)


def evaluate_tiebreaker(
    territory_map: Dict[Position, str],
    board_size: int,
    empty_symbol: str = EMPTY,
) -> Dict[str, int]:
    """Count claimed center cells for tie-breaking.

    Only center cells that remain empty and receive a territory owner count
    toward the tie-break. Occupied center cells and neutral cells do not count.
    """

    center_indices = (board_size // 2 - 1, board_size // 2)
    scores = {PLAYER_X: 0, PLAYER_O: 0}
    for row in center_indices:
        for col in center_indices:
            owner = territory_map.get((row, col), empty_symbol)
            if owner in scores:
                scores[owner] += 1
    return scores


def calculate_total_scores(
    territory_scores: Dict[str, int],
    captured_by: Dict[str, int],
) -> Dict[str, int]:
    """Combine territory points with capture bonuses."""

    return {
        PLAYER_X: territory_scores[PLAYER_X] + captured_by.get(PLAYER_X, 0),
        PLAYER_O: territory_scores[PLAYER_O] + captured_by.get(PLAYER_O, 0),
    }


def determine_winner(
    territory_result: TerritoryResult,
    captured_by: Dict[str, int],
    board_size: int,
) -> str | None:
    """Determine the winner from total score, then the center tie-break."""

    total_scores = calculate_total_scores(territory_result.scores, captured_by)
    if total_scores[PLAYER_X] > total_scores[PLAYER_O]:
        return PLAYER_X
    if total_scores[PLAYER_O] > total_scores[PLAYER_X]:
        return PLAYER_O

    tiebreak_scores = evaluate_tiebreaker(territory_result.territory_map, board_size)
    if tiebreak_scores[PLAYER_X] > tiebreak_scores[PLAYER_O]:
        return PLAYER_X
    if tiebreak_scores[PLAYER_O] > tiebreak_scores[PLAYER_X]:
        return PLAYER_O
    return None
