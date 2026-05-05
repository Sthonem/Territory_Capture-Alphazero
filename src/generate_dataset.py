"""Command-line dataset generation for large-scale self-play."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .agents import create_agent
from .dataset import get_encoder_by_name, has_expected_state_shape, play_self_play_episode

DEFAULT_PROGRESS_EVERY = 100
DEFAULT_GAMES_PER_TASK = 100


@dataclass(frozen=True)
class DatasetGenerationSummary:
    """Concise summary returned after dataset generation finishes."""

    matchup_code: str
    total_games: int
    total_samples: int
    skipped_samples: int
    output_path: Path

    @property
    def average_samples_per_game(self) -> float:
        """Return the average number of samples produced per game."""

        return self.total_samples / self.total_games


def parse_args() -> argparse.Namespace:
    """Parse command-line options for dataset generation."""

    parser = argparse.ArgumentParser(
        description="Generate Territory Capture self-play datasets."
    )
    parser.add_argument("--games", type=int, required=True, help="Number of games to run.")
    parser.add_argument(
        "--x-agent",
        default="random",
        choices=["random", "heuristic", "minimax"],
        help="Agent type for player X.",
    )
    parser.add_argument(
        "--o-agent",
        default="random",
        choices=["random", "heuristic", "minimax"],
        help="Agent type for player O.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Base output JSON filename. A timestamp will be appended automatically.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed for stochastic agents.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Print a progress update every N completed games.",
    )
    return parser.parse_args()


def generate_dataset(
    num_games: int,
    x_agent_name: str,
    o_agent_name: str,
    output_path: str | Path,
    seed: Optional[int] = None,
    workers: int = 1,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
) -> DatasetGenerationSummary:
    """Run many self-play games and stream the samples directly to JSON."""

    if num_games <= 0:
        raise ValueError("num_games must be greater than 0.")
    if workers <= 0:
        raise ValueError("workers must be greater than 0.")
    if progress_every < 0:
        raise ValueError("progress_every must be greater than or equal to 0.")

    matchup = get_matchup_code(x_agent_name, o_agent_name)
    resolved_output_path = build_timestamped_output_path(output_path)

    print(f"Matchup: {matchup} ({x_agent_name} vs {o_agent_name})")
    print(f"Games requested: {num_games}")
    print(f"Output file: {resolved_output_path}")

    tasks = build_generation_tasks(
        num_games=num_games,
        x_agent_name=x_agent_name,
        o_agent_name=o_agent_name,
        seed=seed,
    )

    total_samples, skipped_samples = stream_tasks_to_json(
        tasks=tasks,
        output_path=resolved_output_path,
        workers=workers,
        progress_every=progress_every,
    )

    return DatasetGenerationSummary(
        matchup_code=matchup,
        total_games=num_games,
        total_samples=total_samples,
        skipped_samples=skipped_samples,
        output_path=resolved_output_path,
    )


def build_timestamped_output_path(output_path: str | Path) -> Path:
    """Append a timestamp to the requested filename to avoid overwriting."""

    base_path = Path(output_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = base_path.stem
    suffix = base_path.suffix if base_path.suffix else ".json"
    filename = f"{stem}_{timestamp}{suffix}"
    return base_path.with_name(filename).resolve()


def get_matchup_code(x_agent_name: str, o_agent_name: str) -> str:
    """Return a short matchup code such as mh, hh, hr, or rr."""

    code_map = {
        "minimax": "m",
        "heuristic": "h",
        "random": "r",
    }
    return f"{code_map[x_agent_name]}{code_map[o_agent_name]}"


def build_generation_tasks(
    num_games: int,
    x_agent_name: str,
    o_agent_name: str,
    seed: Optional[int],
    games_per_task: int = DEFAULT_GAMES_PER_TASK,
) -> List[tuple[int, str, str, Optional[int]]]:
    """Split a dataset run into small stable tasks."""

    tasks: List[tuple[int, str, str, Optional[int]]] = []
    remaining_games = num_games
    task_index = 0

    while remaining_games > 0:
        task_games = min(games_per_task, remaining_games)
        task_seed = None if seed is None else seed + task_index * 1000
        tasks.append((task_games, x_agent_name, o_agent_name, task_seed))
        remaining_games -= task_games
        task_index += 1

    return tasks


def stream_tasks_to_json(
    tasks: Sequence[tuple[int, str, str, Optional[int]]],
    output_path: Path,
    workers: int,
    progress_every: int,
) -> tuple[int, int]:
    """Write task results incrementally to JSON to avoid large memory spikes."""

    completed_games = 0
    total_samples = 0
    skipped_samples = 0
    first_item = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write("[\n")

        if workers == 1:
            task_results: Iterable[tuple[int, int, list[dict]]] = map(
                _generate_task_samples,
                tasks,
            )
        else:
            with mp.Pool(processes=workers) as pool:
                task_results = pool.imap_unordered(_generate_task_samples, tasks)
                completed_games, total_samples, skipped_samples, first_item = _consume_task_results(
                    task_results=task_results,
                    output_file=output_file,
                    completed_games=completed_games,
                    total_samples=total_samples,
                    skipped_samples=skipped_samples,
                    progress_every=progress_every,
                    first_item=first_item,
                )
                output_file.write("\n]\n")
                return total_samples, skipped_samples

        completed_games, total_samples, skipped_samples, first_item = _consume_task_results(
            task_results=task_results,
            output_file=output_file,
            completed_games=completed_games,
            total_samples=total_samples,
            skipped_samples=skipped_samples,
            progress_every=progress_every,
            first_item=first_item,
        )
        output_file.write("\n]\n")

    return total_samples, skipped_samples


def _consume_task_results(
    task_results: Iterable[tuple[int, int, list[dict]]],
    output_file,
    completed_games: int,
    total_samples: int,
    skipped_samples: int,
    progress_every: int,
    first_item: bool,
) -> tuple[int, int, int, bool]:
    """Consume finished task chunks and stream them into one JSON array."""

    for task_games, chunk_skipped, chunk_payload in task_results:
        for sample in chunk_payload:
            if not first_item:
                output_file.write(",\n")
            json.dump(sample, output_file)
            first_item = False
        completed_games += task_games
        skipped_samples += chunk_skipped
        total_samples += len(chunk_payload)

        if progress_every and (completed_games % progress_every == 0):
            print(f"Completed {completed_games} games...")

    return completed_games, total_samples, skipped_samples, first_item


def _generate_task_samples(
    task: tuple[int, str, str, Optional[int]],
) -> tuple[int, int, list[dict]]:
    """Generate one task worth of samples and return JSON-ready dictionaries."""

    num_games, x_agent_name, o_agent_name, seed = task
    x_agent = create_agent(x_agent_name, seed=seed)
    o_seed: Optional[int] = None if seed is None else seed + 1
    o_agent = create_agent(o_agent_name, seed=o_seed)
    encoder = get_encoder_by_name("relative")

    payload: list[dict] = []
    skipped_samples = 0
    for _ in range(num_games):
        episode = play_self_play_episode(
            x_agent=x_agent,
            o_agent=o_agent,
            encoder=encoder,
        )
        for sample in episode.samples:
            try:
                assert has_expected_state_shape(sample.encoded_state, channels=2, board_size=6)
            except AssertionError:
                skipped_samples += 1
                print("Warning: skipped invalid sample with unexpected encoded_state shape.")
                continue
            payload.append(asdict(sample))

    return num_games, skipped_samples, payload


def main() -> None:
    """Generate a dataset and print a concise summary."""

    args = parse_args()
    summary = generate_dataset(
        num_games=args.games,
        x_agent_name=args.x_agent,
        o_agent_name=args.o_agent,
        output_path=args.output,
        seed=args.seed,
        workers=args.workers,
        progress_every=args.progress_every,
    )

    print("Territory Capture Dataset Generation")
    print(f"Matchup type: {summary.matchup_code}")
    print(f"Total games played: {summary.total_games}")
    print(f"Total samples collected: {summary.total_samples}")
    print(f"Skipped samples: {summary.skipped_samples}")
    print(f"Output file: {summary.output_path}")
    print(f"Average samples per game: {summary.average_samples_per_game:.2f}")


if __name__ == "__main__":
    main()
