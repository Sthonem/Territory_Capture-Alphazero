"""Rules helpers for Territory Capture scoring and captures."""

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


def find_captured_stones(
    board: Sequence[Sequence[str]],
    empty_symbol: str = EMPTY,
) -> List[Position]:
    """Return stones that should be removed after a move.

    A stone is captured when none of its neighboring cells are empty.
    The result is computed from the original board state so captures
    can be removed simultaneously and deterministically.
    """

    board_size = len(board)
    captured_positions: List[Position] = []

    for row in range(board_size):
        for col in range(board_size):
            if board[row][col] == empty_symbol:
                continue

            neighbors = get_neighbors((row, col), board_size)
            has_empty_neighbor = any(
                board[neighbor_row][neighbor_col] == empty_symbol
                for neighbor_row, neighbor_col in neighbors
            )

            if not has_empty_neighbor:
                captured_positions.append((row, col))

    return captured_positions


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
