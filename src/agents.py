"""Baseline agents for Territory Capture.

The goal of this module is not to build a strong agent yet. Instead, it
provides a few simple baselines that are easy to compare, explain, and
replace later when AlphaZero components are added.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

from .game import TerritoryCaptureGame
from .rules import EMPTY, PLAYER_O, PLAYER_X, get_neighbors

Position = Tuple[int, int]


class Agent(Protocol):
    """Simple protocol for agents that choose one legal move."""

    name: str

    def select_action(self, game: TerritoryCaptureGame) -> Position:
        """Choose one legal action for the current state."""


@dataclass(frozen=True)
class HeuristicBreakdown:
    """Readable breakdown of the heuristic features for one move."""

    center_preference: float
    empty_space: int
    friendly_support: int
    opponent_pressure: int

    @property
    def total_score(self) -> float:
        """Combine the feature values into one final heuristic score."""

        return (
            center_weight(self.center_preference)
            + empty_space_weight(self.empty_space)
            + support_weight(self.friendly_support)
            + pressure_weight(self.opponent_pressure)
        )


def center_weight(value: float) -> float:
    """Weight center preference more strongly than other signals."""

    return value * 3.0


def empty_space_weight(value: int) -> float:
    """Reward moves that keep local space open."""

    return value * 1.5


def support_weight(value: int) -> float:
    """Reward friendly local majority."""

    return value * 2.0


def pressure_weight(value: int) -> float:
    """Reward moves that challenge nearby opponent influence."""

    return float(value)


@dataclass
class RandomAgent:
    """Pick uniformly from all currently legal moves."""

    seed: Optional[int] = None
    name: str = "random"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def select_action(self, game: TerritoryCaptureGame) -> Position:
        """Return a random legal move."""

        legal_actions = game.get_legal_actions()
        if not legal_actions:
            raise ValueError("RandomAgent cannot act in a terminal state.")
        return self._rng.choice(legal_actions)


@dataclass
class HeuristicAgent:
    """Choose a move using a small set of easy-to-explain heuristics.

    The heuristic favors:
    - central positions
    - cells with more nearby empty space
    - moves supported by nearby friendly stones
    - moves that contest nearby opponent stones

    Ties are broken deterministically by board coordinate so repeated runs
    remain easy to reproduce and discuss.
    """

    name: str = "heuristic"

    def select_action(self, game: TerritoryCaptureGame) -> Position:
        """Return the best legal move according to a readable heuristic."""

        legal_actions = game.get_legal_actions()
        if not legal_actions:
            raise ValueError("HeuristicAgent cannot act in a terminal state.")

        scored_actions = [
            (self.explain_action(game, action).total_score, action)
            for action in legal_actions
        ]
        scored_actions.sort(key=lambda item: (-item[0], item[1]))
        return scored_actions[0][1]

    def explain_action(
        self,
        game: TerritoryCaptureGame,
        action: Position,
    ) -> HeuristicBreakdown:
        """Return the feature-level explanation for one candidate move."""

        return HeuristicBreakdown(
            center_preference=self._center_preference(game, action),
            empty_space=self._neighbor_empty_score(game, action),
            friendly_support=self._local_control_score(game, action),
            opponent_pressure=self._opponent_pressure_score(game, action),
        )

    def _center_preference(self, game: TerritoryCaptureGame, action: Position) -> float:
        """Prefer cells closer to the board center."""

        center = (game.board_size - 1) / 2.0
        row, col = action
        distance = abs(row - center) + abs(col - center)
        max_distance = center * 2 if center > 0 else 1.0
        return max_distance - distance

    def _neighbor_empty_score(self, game: TerritoryCaptureGame, action: Position) -> int:
        """Prefer actions surrounded by future expansion space."""

        empty_neighbors = 0
        for row, col in get_neighbors(action, game.board_size):
            if game.board[row][col] == EMPTY:
                empty_neighbors += 1
        return empty_neighbors

    def _local_control_score(self, game: TerritoryCaptureGame, action: Position) -> int:
        """Prefer moves that create a favorable local majority."""

        player = game.current_player
        opponent = PLAYER_O if player == PLAYER_X else PLAYER_X
        player_neighbors = 0
        opponent_neighbors = 0

        for row, col in get_neighbors(action, game.board_size):
            value = game.board[row][col]
            if value == player:
                player_neighbors += 1
            elif value == opponent:
                opponent_neighbors += 1

        return player_neighbors - opponent_neighbors

    def _opponent_pressure_score(self, game: TerritoryCaptureGame, action: Position) -> int:
        """Prefer moves near opponent stones that could weaken their local control."""

        player = game.current_player
        opponent = PLAYER_O if player == PLAYER_X else PLAYER_X
        pressure = 0

        for row, col in get_neighbors(action, game.board_size):
            if game.board[row][col] != opponent:
                continue

            empty_neighbors = 0
            for neighbor_row, neighbor_col in get_neighbors((row, col), game.board_size):
                if (neighbor_row, neighbor_col) == action:
                    continue
                if game.board[neighbor_row][neighbor_col] == EMPTY:
                    empty_neighbors += 1

            if empty_neighbors <= 2:
                pressure += 2
            else:
                pressure += 1

        return pressure


def create_agent(agent_name: str, seed: Optional[int] = None) -> Agent:
    """Create a baseline agent by name."""

    normalized_name = agent_name.strip().lower()
    if normalized_name == "random":
        return RandomAgent(seed=seed)
    if normalized_name == "heuristic":
        return HeuristicAgent()
    raise ValueError(f"Unknown agent type: {agent_name}")
