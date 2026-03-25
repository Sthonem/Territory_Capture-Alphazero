"""Arena matches for comparing checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
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
    metadata_path: Path


def evaluate_model_against_best(
    candidate_path: str | Path,
    incumbent_path: str | Path = "src/model.pth",
    num_games: int = 20,
    acceptance_threshold: float = 0.55,
    promote_on_win: bool = False,
    metadata_path: str | Path | None = None,
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
    resolved_metadata_path = (
        Path(metadata_path).resolve()
        if metadata_path is not None
        else candidate.with_suffix(".arena.json")
    )
    _save_arena_metadata(
        metadata_path=resolved_metadata_path,
        summary=summary,
        candidate=candidate,
        incumbent=incumbent,
        acceptance_threshold=acceptance_threshold,
        accepted=accepted,
    )

    if accepted and promote_on_win:
        shutil.copyfile(candidate, incumbent)
        candidate_metadata = candidate.with_suffix(".metadata.json")
        if candidate_metadata.exists():
            shutil.copyfile(candidate_metadata, incumbent.with_suffix(".metadata.json"))

    return ArenaResult(
        summary=summary,
        candidate_path=candidate,
        incumbent_path=incumbent,
        accepted=accepted,
        metadata_path=resolved_metadata_path,
    )


def _save_arena_metadata(
    metadata_path: Path,
    summary: EvaluationSummary,
    candidate: Path,
    incumbent: Path,
    acceptance_threshold: float,
    accepted: bool,
) -> None:
    """Persist one arena comparison result for checkpoint tracking."""

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_path": str(candidate),
        "incumbent_path": str(incumbent),
        "acceptance_threshold": acceptance_threshold,
        "accepted": accepted,
        "games": summary.total_games,
        "candidate_wins": summary.ai1_wins,
        "incumbent_wins": summary.ai2_wins,
        "draws": summary.draws,
        "candidate_win_rate": summary.ai1_win_rate,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for arena evaluation."""

    parser = argparse.ArgumentParser(description="Compare two Territory Capture checkpoints.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--incumbent", default="src/model.pth")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--metadata", default=None)
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
        metadata_path=args.metadata,
    )
    print("Arena result")
    print(f"Candidate: {result.candidate_path}")
    print(f"Incumbent: {result.incumbent_path}")
    print(f"Accepted: {result.accepted}")
    print(f"Saved metadata: {result.metadata_path}")


if __name__ == "__main__":
    main()
