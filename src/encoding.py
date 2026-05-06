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

# Default board size — overridable per function call.
BOARD_SIZE = 6
ACTION_SPACE_SIZE = BOARD_SIZE * BOARD_SIZE

# Board-size configuration for multi-board support.
BOARD_CONFIGS = {
    5: {"stones_per_player": 8, "model_file": "model_5x5.pth"},
    6: {"stones_per_player": 10, "model_file": "model_6x6.pth"},
    7: {"stones_per_player": 12, "model_file": "model_7x7.pth"},
}


def action_space_for(board_size: int) -> int:
    """Return the action space size for a given board size."""
    return board_size * board_size


def encode_state(game: TerritoryCaptureGame) -> EncodedState:
    """Encode the board from the current player's perspective.

    The output shape is always (2, N, N) where N is the board size:
    - channel 0: stones belonging to the current player
    - channel 1: stones belonging to the opponent

    Empty cells are encoded as 0 in both channels.
    """

    current_player = game.current_player
    opponent = PLAYER_O if current_player == PLAYER_X else PLAYER_X

    current_player_channel = _encode_player_channel(game, current_player)
    opponent_channel = _encode_player_channel(game, opponent)
    return [current_player_channel, opponent_channel]


def encode_state_with_turn_plane(game: TerritoryCaptureGame) -> EncodedState:
    """Encode the board using fixed player channels plus a turn plane.

    The output shape is always (3, N, N):
    - channel 0: X stones
    - channel 1: O stones
    - channel 2: all ones if it is X's turn, else all zeros
    """

    bs = game.board_size
    x_channel = _encode_player_channel(game, PLAYER_X)
    o_channel = _encode_player_channel(game, PLAYER_O)
    turn_value = 1 if game.current_player == PLAYER_X else 0
    turn_channel = [[turn_value for _ in range(bs)] for _ in range(bs)]
    return [x_channel, o_channel, turn_channel]


def action_to_index(action: Position, board_size: int = BOARD_SIZE) -> int:
    """Convert a board coordinate into a fixed action index."""

    row, col = action
    if not (0 <= row < board_size and 0 <= col < board_size):
        raise ValueError(f"Action out of bounds: {action}")
    return row * board_size + col


def index_to_action(index: int, board_size: int = BOARD_SIZE) -> Position:
    """Convert a fixed action index back into a board coordinate."""

    action_space = board_size * board_size
    if not (0 <= index < action_space):
        raise ValueError(f"Action index out of bounds: {index}")
    return divmod(index, board_size)


def get_legal_action_mask(game: TerritoryCaptureGame) -> ActionMask:
    """Return a fixed-size mask over possible board actions."""

    bs = game.board_size
    action_space = bs * bs
    mask = [0] * action_space
    for action in game.get_legal_actions():
        mask[action_to_index(action, bs)] = 1
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
