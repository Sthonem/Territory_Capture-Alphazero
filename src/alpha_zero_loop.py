"""High-level self-play -> train -> arena loop."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .arena import ArenaResult, evaluate_model_against_best
from .replay_buffer import ReplayBuffer
from .self_play import SelfPlaySummary, generate_self_play_games
from .train import TrainingSummary, load_self_play_records, train_policy_value_model


@dataclass(frozen=True)
class AlphaZeroLoopSummary:
    """Stores the outputs of one full AlphaZero-style iteration."""

    self_play: SelfPlaySummary
    training: TrainingSummary
    arena: ArenaResult


def run_alpha_zero_iteration(
    self_play_games: int = 20,
    arena_games: int = 20,
    epochs: int = 5,
    self_play_path: str | Path = "self_play_data.json",
    replay_buffer_path: str | Path = "replay_buffer.json",
    candidate_model_path: str | Path = "src/latest_model.pth",
    incumbent_model_path: str | Path = "src/model.pth",
    acceptance_threshold: float = 0.55,
    replay_buffer_size: int = 10_000,
) -> AlphaZeroLoopSummary:
    """Run one full self-play, training, and arena evaluation cycle."""

    self_play_summary = generate_self_play_games(
        num_games=self_play_games,
        output_path=self_play_path,
    )
    replay_buffer = ReplayBuffer(max_samples=replay_buffer_size)
    replay_buffer.load_json(replay_buffer_path)
    replay_buffer.extend(load_self_play_records(self_play_summary.output_path))
    replay_buffer.save_json(replay_buffer_path)

    training_summary = train_policy_value_model(
        records=replay_buffer.records,
        output_path=candidate_model_path,
        epochs=epochs,
    )
    arena_result = evaluate_model_against_best(
        candidate_path=training_summary.output_path,
        incumbent_path=incumbent_model_path,
        num_games=arena_games,
        acceptance_threshold=acceptance_threshold,
        promote_on_win=True,
    )
    return AlphaZeroLoopSummary(
        self_play=self_play_summary,
        training=training_summary,
        arena=arena_result,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for one loop iteration."""

    parser = argparse.ArgumentParser(description="Run one AlphaZero-style training iteration.")
    parser.add_argument("--self-play-games", type=int, default=20)
    parser.add_argument("--arena-games", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--self-play-path", default="self_play_data.json")
    parser.add_argument("--replay-buffer-path", default="replay_buffer.json")
    parser.add_argument("--candidate-model", default="src/latest_model.pth")
    parser.add_argument("--incumbent-model", default="src/model.pth")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    """Run one AlphaZero-style iteration from the command line."""

    args = parse_args()
    summary = run_alpha_zero_iteration(
        self_play_games=args.self_play_games,
        arena_games=args.arena_games,
        epochs=args.epochs,
        self_play_path=args.self_play_path,
        replay_buffer_path=args.replay_buffer_path,
        candidate_model_path=args.candidate_model,
        incumbent_model_path=args.incumbent_model,
        acceptance_threshold=args.threshold,
        replay_buffer_size=args.buffer_size,
    )
    print("AlphaZero-style iteration complete")
    print(f"Self-play samples: {summary.self_play.total_samples}")
    print(f"Training output: {summary.training.output_path}")
    print(f"Arena accepted candidate: {summary.arena.accepted}")


if __name__ == "__main__":
    main()
