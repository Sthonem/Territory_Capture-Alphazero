"""Simplified AlphaZero-style self-play data generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .ai_agent import AIAgent
from .dataset import outcome_value_for_player
from .encoding import encode_state
from .game import TerritoryCaptureGame


@dataclass(frozen=True)
class SelfPlaySummary:
    """Stores aggregate information for a self-play run."""

    total_games: int
    total_samples: int
    output_path: Path


@dataclass(frozen=True)
class SelfPlayConfig:
    """Keeps self-play generation settings together for cleaner experiments."""

    num_games: int = 20
    output_path: str | Path = "self_play_data.json"
    temperature_moves: int = 6
    opening_temperature: float = 1.0
    late_temperature: float = 0.0
    add_root_noise: bool = True


def generate_self_play_games(
    num_games: int = 20,
    output_path: str | Path = "self_play_data.json",
    agent: Optional[AIAgent] = None,
    temperature_moves: int = 6,
    opening_temperature: float = 1.0,
    late_temperature: float = 0.0,
    add_root_noise: bool = True,
    board_size: int = 6,
) -> SelfPlaySummary:
    """Generate self-play data using MCTS visit-count policies."""

    agent = agent or AIAgent(board_size=board_size)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    first_item = True

    with destination.open("w", encoding="utf-8") as output_file:
        output_file.write("[\n")

        for _ in range(num_games):
            agent.reset_search_tree()
            from .encoding import BOARD_CONFIGS
            config = BOARD_CONFIGS.get(board_size, {})
            stones = config.get("stones_per_player", 10)
            records = _play_self_play_game(
                agent,
                temperature_moves=temperature_moves,
                opening_temperature=opening_temperature,
                late_temperature=late_temperature,
                add_root_noise=add_root_noise,
                board_size=board_size,
                stones_per_player=stones,
            )
            total_samples += len(records)

            for record in records:
                if not first_item:
                    output_file.write(",\n")
                json.dump(record, output_file)
                first_item = False

        output_file.write("\n]\n")

    return SelfPlaySummary(
        total_games=num_games,
        total_samples=total_samples,
        output_path=destination,
    )


def generate_self_play_games_with_config(
    config: SelfPlayConfig,
    agent: Optional[AIAgent] = None,
) -> SelfPlaySummary:
    """Run self-play using a configuration object."""

    return generate_self_play_games(
        num_games=config.num_games,
        output_path=config.output_path,
        agent=agent,
        temperature_moves=config.temperature_moves,
        opening_temperature=config.opening_temperature,
        late_temperature=config.late_temperature,
        add_root_noise=config.add_root_noise,
    )


def _play_self_play_game(
    agent: AIAgent,
    temperature_moves: int,
    opening_temperature: float,
    late_temperature: float,
    add_root_noise: bool,
    board_size: int = 6,
    stones_per_player: int = 10,
) -> List[Dict]:
    """Play one self-play game and return JSON-ready move records."""

    game = TerritoryCaptureGame(board_size=board_size, stones_per_player=stones_per_player)
    move_records: List[Dict] = []

    while not game.is_terminal():
        temperature = (
            opening_temperature
            if game.move_count < temperature_moves
            else late_temperature
        )
        move_records.append(
            {
                "state": encode_state(game),
                "player_to_move": game.current_player,
            }
        )
        action, policy = agent.select_action_with_policy(
            game.clone(),
            temperature=temperature,
            add_exploration_noise=add_root_noise,
        )
        move_records[-1]["policy"] = policy
        game.apply_action(action)

    winner = game.get_winner()
    for record in move_records:
        record["value"] = outcome_value_for_player(winner, record["player_to_move"])
        del record["player_to_move"]

    return move_records
