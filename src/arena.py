"""Arena matches for comparing checkpoints."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from .ai_agent import AIAgent
from .ai_eval import EvaluationSummary, run_ai_vs_ai


@dataclass(frozen=True)
class ArenaResult:
    """Stores the outcome of a model-vs-model arena match."""

    summary: EvaluationSummary
    candidate_path: Path
    incumbent_path: Path
    accepted: bool


def evaluate_model_against_best(
    candidate_path: str | Path,
    incumbent_path: str | Path = "src/model.pth",
    num_games: int = 20,
    acceptance_threshold: float = 0.55,
    promote_on_win: bool = False,
) -> ArenaResult:
    """Compare a candidate model against the current incumbent."""

    candidate = Path(candidate_path).resolve()
    incumbent = Path(incumbent_path).resolve()

    summary = run_ai_vs_ai(
        num_games=num_games,
        ai1_factory=lambda: AIAgent(model_path=candidate),
        ai2_factory=lambda: AIAgent(model_path=incumbent),
        plot_path="results/arena_winrate.png",
    )
    accepted = summary.ai1_win_rate >= acceptance_threshold

    if accepted and promote_on_win:
        shutil.copyfile(candidate, incumbent)

    return ArenaResult(
        summary=summary,
        candidate_path=candidate,
        incumbent_path=incumbent,
        accepted=accepted,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for arena evaluation."""

    parser = argparse.ArgumentParser(description="Compare two Territory Capture checkpoints.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--incumbent", default="src/model.pth")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--promote", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the arena from the command line."""

    args = parse_args()
    result = evaluate_model_against_best(
        candidate_path=args.candidate,
        incumbent_path=args.incumbent,
        num_games=args.games,
        acceptance_threshold=args.threshold,
        promote_on_win=args.promote,
    )
    print("Arena result")
    print(f"Candidate: {result.candidate_path}")
    print(f"Incumbent: {result.incumbent_path}")
    print(f"Accepted: {result.accepted}")


if __name__ == "__main__":
    main()
