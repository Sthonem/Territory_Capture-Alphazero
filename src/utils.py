"""Utility helpers for board formatting and small display tasks."""

from __future__ import annotations

from typing import Iterable, Sequence


def format_board(board: Sequence[Sequence[str]]) -> str:
    """Return a readable text representation of the board."""

    lines = []
    for row in board:
        lines.append(" ".join(row))
    return "\n".join(lines)


def print_board(board: Sequence[Sequence[str]]) -> None:
    """Print the board using a compact text layout."""

    print(format_board(board))


def format_moves(moves: Iterable[tuple[int, int]]) -> str:
    """Format a move list for simple debugging output."""

    return ", ".join(f"({row}, {col})" for row, col in moves)
