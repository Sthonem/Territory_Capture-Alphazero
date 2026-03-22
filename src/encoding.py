"""State and action representations for learning-based agents.

This module adds the fixed-size representation layer typically needed before
AlphaZero-style training. It does not implement any learning yet. Instead, it
converts game states and actions into stable numeric formats that can later be
consumed by neural networks, MCTS, and self-play pipelines.
"""

from __future__ import annotations

from typing import List, Tuple

from .game import TerritoryCaptureGame
from .rules import PLAYER_O, PLAYER_X

Position = Tuple[int, int]
EncodedState = List[List[List[int]]]
ActionMask = List[int]

BOARD_SIZE = 5
ACTION_SPACE_SIZE = BOARD_SIZE * BOARD_SIZE


def encode_state(game: TerritoryCaptureGame) -> EncodedState:
    """Encode the board from the current player's perspective.

    The output shape is always (2, 5, 5):
    - channel 0: stones belonging to the current player
    - channel 1: stones belonging to the opponent

    Empty cells are encoded as 0 in both channels.
    """

    _validate_board_size(game)

    current_player = game.current_player
    opponent = PLAYER_O if current_player == PLAYER_X else PLAYER_X

    current_player_channel = _encode_player_channel(game, current_player)
    opponent_channel = _encode_player_channel(game, opponent)
    return [current_player_channel, opponent_channel]


def action_to_index(action: Position) -> int:
    """Convert a board coordinate into a fixed action index."""

    row, col = action
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError(f"Action out of bounds: {action}")
    return row * BOARD_SIZE + col


def index_to_action(index: int) -> Position:
    """Convert a fixed action index back into a board coordinate."""

    if not (0 <= index < ACTION_SPACE_SIZE):
        raise ValueError(f"Action index out of bounds: {index}")
    return divmod(index, BOARD_SIZE)


def get_legal_action_mask(game: TerritoryCaptureGame) -> ActionMask:
    """Return a fixed-size mask over the 25 possible board actions."""

    _validate_board_size(game)

    mask = [0] * ACTION_SPACE_SIZE
    for action in game.get_legal_actions():
        mask[action_to_index(action)] = 1
    return mask


def _encode_player_channel(
    game: TerritoryCaptureGame,
    player: str,
) -> List[List[int]]:
    """Encode one player-specific binary board channel."""

    channel: List[List[int]] = []
    for row in game.board_state:
        channel.append([1 if cell == player else 0 for cell in row])
    return channel


def _validate_board_size(game: TerritoryCaptureGame) -> None:
    """Keep the representation layer explicit about its fixed input size."""

    if game.board_size != BOARD_SIZE:
        raise ValueError(
            f"Encoding expects a {BOARD_SIZE}x{BOARD_SIZE} board, "
            f"received {game.board_size}x{game.board_size}."
        )
