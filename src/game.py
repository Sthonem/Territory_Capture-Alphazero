"""Core game environment for Territory Capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .rules import (
    EMPTY,
    PLAYER_O,
    PLAYER_X,
    TerritoryResult,
    evaluate_territory,
    find_captured_stones,
)

Position = Tuple[int, int]


@dataclass(frozen=True)
class GameResult:
    """Stores the final result of a completed game."""

    scores: Dict[str, int]
    winner: Optional[str]
    territory_map: Dict[Position, str]


class TerritoryCaptureGame:
    """A simple game environment that is easy to extend for future AI work."""

    def __init__(self, board_size: int = 5, stones_per_player: int = 8) -> None:
        self.board_size = board_size
        self.stones_per_player = stones_per_player
        self.players = (PLAYER_X, PLAYER_O)
        self.reset()

    def reset(self) -> None:
        """Reset the game to its initial empty state."""

        self.board: List[List[str]] = [
            [EMPTY for _ in range(self.board_size)] for _ in range(self.board_size)
        ]
        self.current_player = PLAYER_X
        self.move_count = 0
        self.stones_placed = {PLAYER_X: 0, PLAYER_O: 0}
        self.move_history: List[Tuple[str, Position]] = []
        self.last_captured_positions: List[Position] = []

    def copy_board(self) -> List[List[str]]:
        """Return a shallow copy of the board rows for safe external use."""

        return [row[:] for row in self.board]

    def clone(self) -> "TerritoryCaptureGame":
        """Return an independent copy of the current game state.

        This keeps the environment easy to use for future search-based agents
        such as MCTS, where many hypothetical game states must be explored
        without mutating the original game.
        """

        cloned_game = TerritoryCaptureGame(
            board_size=self.board_size,
            stones_per_player=self.stones_per_player,
        )
        cloned_game.board = self.copy_board()
        cloned_game.current_player = self.current_player
        cloned_game.move_count = self.move_count
        cloned_game.stones_placed = dict(self.stones_placed)
        cloned_game.move_history = list(self.move_history)
        cloned_game.last_captured_positions = list(self.last_captured_positions)
        return cloned_game

    def get_legal_moves(self) -> List[Position]:
        """Return all currently available empty cells."""

        if self.is_terminal():
            return []

        legal_moves: List[Position] = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == EMPTY:
                    legal_moves.append((row, col))
        return legal_moves

    def get_legal_actions(self) -> List[Position]:
        """Return legal actions using AI-friendly naming."""

        return self.get_legal_moves()

    def is_legal_move(self, move: Position) -> bool:
        """Check whether a move can be played."""

        row, col = move
        if self.is_terminal():
            return False
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False
        return self.board[row][col] == EMPTY

    def apply_move(self, move: Position) -> None:
        """Place a stone, resolve captures, and advance the turn."""

        if not self.is_legal_move(move):
            raise ValueError(f"Illegal move: {move}")

        player = self.current_player
        row, col = move
        self.board[row][col] = player
        self.stones_placed[player] += 1
        self.move_count += 1
        self.move_history.append((player, move))
        self.last_captured_positions = self._apply_captures()

        if not self.is_terminal():
            self.current_player = PLAYER_O if player == PLAYER_X else PLAYER_X

    def apply_action(self, action: Position) -> None:
        """Apply an action using AI-friendly naming."""

        self.apply_move(action)

    def _apply_captures(self) -> List[Position]:
        """Remove stones with no empty neighboring cells simultaneously."""

        captured_positions = find_captured_stones(self.board)
        for row, col in captured_positions:
            self.board[row][col] = EMPTY
        return captured_positions

    def is_terminal(self) -> bool:
        """Return True when both players have placed all their stones."""

        return self.move_count >= self.max_moves

    @property
    def max_moves(self) -> int:
        """Total number of moves in a complete game."""

        return self.stones_per_player * len(self.players)

    def evaluate_territory(self) -> TerritoryResult:
        """Delegate territory scoring to the rules module."""

        return evaluate_territory(self.board)

    def get_territory_scores(self) -> Dict[str, int]:
        """Return only the final territory totals for the current board."""

        return dict(self.evaluate_territory().scores)

    def get_winner(self) -> Optional[str]:
        """Return the winning player, or None for a draw."""

        result = self.evaluate_territory()
        x_score = result.scores[PLAYER_X]
        o_score = result.scores[PLAYER_O]

        if x_score > o_score:
            return PLAYER_X
        if o_score > x_score:
            return PLAYER_O
        return None

    def get_result(self) -> GameResult:
        """Return the final scoring summary for the current board."""

        territory_result = self.evaluate_territory()
        return GameResult(
            scores=territory_result.scores,
            winner=self.get_winner(),
            territory_map=territory_result.territory_map,
        )

    def get_state(self) -> Sequence[Sequence[str]]:
        """Expose the board state as a sequence for future encoders."""

        return self.board

    @property
    def board_state(self) -> List[List[str]]:
        """Expose a safe copy of the board for agent-side inspection."""

        return self.copy_board()
