"""Self-play dataset helpers for future AlphaZero-style training.

This module does not implement MCTS or a neural network yet. Instead, it
records training examples in the shape that later training code will need:

- encoded state
- policy target
- value target

For now, policy targets are simple one-hot vectors based on the action that
an existing baseline agent selected. Later, these one-hot targets can be
replaced by MCTS visit-count distributions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from .agents import Agent
from .encoding import (
    BOARD_SIZE,
    ActionMask,
    EncodedState,
    action_to_index,
    encode_state,
    get_legal_action_mask,
)
from .game import TerritoryCaptureGame


@dataclass(frozen=True)
class RecordedPosition:
    """One move-time snapshot collected during self-play."""

    encoded_state: EncodedState
    selected_action_index: int
    legal_action_mask: ActionMask
    player_to_move: str


@dataclass(frozen=True)
class TrainingSample:
    """A single training example for later policy/value learning."""

    encoded_state: EncodedState
    policy_target: List[int]
    value_target: int
    legal_action_mask: ActionMask
    selected_action_index: int
    player_to_move: str


@dataclass(frozen=True)
class SelfPlayEpisode:
    """Stores all training samples produced by one full self-play game."""

    samples: List[TrainingSample]
    winner: str | None
    final_scores: Dict[str, int]
    move_count: int


EncoderFn = Callable[[TerritoryCaptureGame], EncodedState]


def create_one_hot_policy_target(action_index: int, action_space: int = BOARD_SIZE * BOARD_SIZE) -> List[int]:
    """Convert one selected action into a fixed-size one-hot policy target."""

    if not (0 <= action_index < action_space):
        raise ValueError(f"Action index out of bounds: {action_index}")

    policy_target = [0] * action_space
    policy_target[action_index] = 1
    return policy_target


def play_self_play_episode(
    x_agent: Agent,
    o_agent: Agent,
    board_size: int = 6,
    stones_per_player: int = 10,
    encoder: EncoderFn = encode_state,
) -> SelfPlayEpisode:
    """Play one game and return the resulting training samples."""

    game = TerritoryCaptureGame(
        board_size=board_size,
        stones_per_player=stones_per_player,
    )
    recorded_positions: List[RecordedPosition] = []

    while not game.is_terminal():
        recorded_positions.append(record_position(game, encoder=encoder))
        agent = x_agent if game.current_player == "X" else o_agent
        action = agent.select_action(game.clone())
        game.apply_action(action)

        # Update the most recent record with the action that was actually played.
        last_record = recorded_positions[-1]
        recorded_positions[-1] = RecordedPosition(
            encoded_state=last_record.encoded_state,
            selected_action_index=action_to_index(action, game.board_size),
            legal_action_mask=last_record.legal_action_mask,
            player_to_move=last_record.player_to_move,
        )

    samples = build_training_samples(
        recorded_positions=recorded_positions,
        winner=game.get_winner(),
        board_size=board_size,
    )
    return SelfPlayEpisode(
        samples=samples,
        winner=game.get_winner(),
        final_scores=game.get_total_scores(),
        move_count=game.move_count,
    )


def record_position(
    game: TerritoryCaptureGame,
    encoder: EncoderFn = encode_state,
) -> RecordedPosition:
    """Record the representation of the current state before a move."""

    legal_action_mask = get_legal_action_mask(game)
    return RecordedPosition(
        encoded_state=encoder(game),
        selected_action_index=-1,
        legal_action_mask=legal_action_mask,
        player_to_move=game.current_player,
    )


def get_encoder_by_name(encoding_name: str) -> EncoderFn:
    """Return a named encoder used by dataset generation scripts."""

    normalized_name = encoding_name.strip().lower()
    if normalized_name in {"relative", "current-player", "2ch"}:
        return encode_state
    raise ValueError(f"Unknown encoding mode: {encoding_name}")


def has_expected_state_shape(
    encoded_state: EncodedState,
    channels: int = 2,
    board_size: int = 6,
) -> bool:
    """Return True when an encoded state matches the expected shape."""

    if len(encoded_state) != channels:
        return False
    return all(
        len(channel) == board_size
        and all(len(row) == board_size for row in channel)
        for channel in encoded_state
    )


def build_training_samples(
    recorded_positions: Sequence[RecordedPosition],
    winner: str | None,
    board_size: int = BOARD_SIZE,
) -> List[TrainingSample]:
    """Convert recorded move snapshots into full training samples."""

    action_space = board_size * board_size
    samples: List[TrainingSample] = []
    for position in recorded_positions:
        if position.selected_action_index < 0:
            raise ValueError("Recorded positions must include a selected action.")

        value_target = outcome_value_for_player(
            winner=winner,
            player=position.player_to_move,
        )
        policy_target = create_one_hot_policy_target(position.selected_action_index, action_space)
        samples.append(
            TrainingSample(
                encoded_state=position.encoded_state,
                policy_target=policy_target,
                value_target=value_target,
                legal_action_mask=position.legal_action_mask,
                selected_action_index=position.selected_action_index,
                player_to_move=position.player_to_move,
            )
        )
    return samples


def outcome_value_for_player(winner: str | None, player: str) -> int:
    """Return the final value target from one player's perspective."""

    if winner is None:
        return 0
    if winner == player:
        return 1
    return -1


def save_samples_to_json(samples: Sequence[TrainingSample], path: str | Path) -> None:
    """Save training samples in a simple JSON format."""

    destination = Path(path)
    payload = [asdict(sample) for sample in samples]
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_samples_from_json(path: str | Path) -> List[TrainingSample]:
    """Load training samples from the JSON export format."""

    source = Path(path)
    raw_payload: List[Dict[str, Any]] = json.loads(source.read_text(encoding="utf-8"))
    return [TrainingSample(**item) for item in raw_payload]
