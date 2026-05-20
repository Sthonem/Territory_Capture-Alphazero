"""Round-robin tournament for Territory Capture agents.

Two modes:
  - within-board:  Each board (5/6/7) runs its own agents vs each other
  - cross-board:   Hard model from one board plays on another board
                    (uses CrossBoardAgent adapter with state/policy resize)

Usage:
    python -m experiments.tournament --mode within --games 50
    python -m experiments.tournament --mode cross  --games 50
    python -m experiments.tournament --mode both   --games 50 --boards 5 6 7
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from src.agents import HeuristicAgent, MinimaxAgent, RandomAgent
from src.ai_agent import AIAgent
from src.encoding import BOARD_CONFIGS
from src.game import TerritoryCaptureGame


# ───────────────────────── data classes ─────────────────────────

@dataclass
class MatchResult:
    """Outcome of a multi-game match between two agents on a fixed board."""
    board_size: int
    x_agent: str
    o_agent: str
    games: int
    x_wins: int
    o_wins: int
    draws: int
    avg_x_score: float
    avg_o_score: float
    seconds: float

    @property
    def x_win_rate(self) -> float: return self.x_wins / max(1, self.games)
    @property
    def o_win_rate(self) -> float: return self.o_wins / max(1, self.games)
    @property
    def draw_rate(self) -> float: return self.draws / max(1, self.games)


# ───────────────────────── game runner ─────────────────────────

def play_match(
    board_size: int,
    x_factory: Callable,
    o_factory: Callable,
    x_label: str,
    o_label: str,
    games: int = 50,
    seed: int = 42,
    swap_colors: bool = True,
) -> MatchResult:
    """Play `games` games between two agents (split evenly with color swap).

    Agents are created via `x_factory` / `o_factory` to allow fresh MCTS trees
    per game (important for deterministic-ish play).
    """
    stones = BOARD_CONFIGS[board_size]["stones_per_player"]
    np.random.seed(seed)
    t0 = time.time()
    x_wins = o_wins = draws = 0
    x_scores: list[int] = []
    o_scores: list[int] = []

    if swap_colors:
        n_first_half = games // 2
        n_second_half = games - n_first_half
        sched = [(x_factory, o_factory, x_label, o_label)] * n_first_half + \
                [(o_factory, x_factory, o_label, x_label)] * n_second_half
    else:
        sched = [(x_factory, o_factory, x_label, o_label)] * games

    for i, (xf, of, xl, ol) in enumerate(sched):
        ax, ao = xf(), of()
        g = TerritoryCaptureGame(board_size=board_size, stones_per_player=stones)
        while not g.is_terminal():
            agent = ax if g.current_player == "X" else ao
            move = agent.select_action(g)
            g.apply_move(move)
        r = g.get_result()
        # Translate result back to the original (x_label, o_label) viewpoint
        if r.winner == "X":
            if xl == x_label:
                x_wins += 1
            else:
                o_wins += 1
        elif r.winner == "O":
            if ol == o_label:
                o_wins += 1
            else:
                x_wins += 1
        else:
            draws += 1
        if xl == x_label:
            x_scores.append(r.scores["X"]); o_scores.append(r.scores["O"])
        else:
            x_scores.append(r.scores["O"]); o_scores.append(r.scores["X"])

    return MatchResult(
        board_size=board_size,
        x_agent=x_label, o_agent=o_label,
        games=games, x_wins=x_wins, o_wins=o_wins, draws=draws,
        avg_x_score=float(np.mean(x_scores)) if x_scores else 0.0,
        avg_o_score=float(np.mean(o_scores)) if o_scores else 0.0,
        seconds=time.time() - t0,
    )


# ───────────────────────── agent factories ─────────────────────────

def factory_random():
    return lambda: RandomAgent()

def factory_heuristic():
    return lambda: HeuristicAgent()

def factory_minimax(depth: int = 3):
    return lambda: MinimaxAgent(depth=depth)

def factory_ai(board_size: int, model_path: str, num_simulations: int = 50):
    return lambda: AIAgent(
        board_size=board_size, model_path=model_path,
        num_simulations=num_simulations,
    )


def resolve_model_path(repo_root: Path, board_size: int, difficulty: str) -> Path | None:
    """Look up checkpoint path for (board_size, difficulty)."""
    prefix = "model_hard_" if difficulty == "hard" else "model_"
    candidate = repo_root / "src" / f"{prefix}{board_size}x{board_size}.pth"
    if candidate.exists():
        return candidate
    # fallback for 6x6 legacy names
    if board_size == 6:
        legacy = repo_root / "src" / ("model_hard.pth" if difficulty == "hard" else "model.pth")
        if legacy.exists():
            return legacy
    return None


# ───────────────────────── within-board tournament ─────────────────────────

def within_board_tournament(
    boards: list[int],
    games: int,
    sims_medium: int,
    sims_hard: int,
    include_baselines: bool,
    repo_root: Path,
) -> list[MatchResult]:
    results: list[MatchResult] = []
    for bs in boards:
        print(f"\n══════════ Within-board tournament: {bs}×{bs} ══════════", flush=True)
        med_path = resolve_model_path(repo_root, bs, "medium")
        hard_path = resolve_model_path(repo_root, bs, "hard")
        if med_path is None or hard_path is None:
            print(f"  ⚠ Missing models for {bs}×{bs} (med={med_path}, hard={hard_path}) — skip", flush=True)
            continue

        med = factory_ai(bs, str(med_path), sims_medium)
        hard = factory_ai(bs, str(hard_path), sims_hard)
        rnd = factory_random()
        heur = factory_heuristic()
        mm = factory_minimax(depth=3)

        # Core comparison: Medium vs Hard
        print(f"  Medium vs Hard ({games} games)...", flush=True)
        r = play_match(bs, hard, med, "Hard", "Medium", games=games)
        results.append(r)
        print(f"    Hard: {r.x_wins}/{games} ({r.x_win_rate*100:.0f}%), "
              f"Medium: {r.o_wins}/{games} ({r.o_win_rate*100:.0f}%), "
              f"Draws: {r.draws} | {r.seconds:.0f}s", flush=True)

        if include_baselines:
            for opp_label, opp_factory in [("Random", rnd), ("Heuristic", heur), ("Minimax", mm)]:
                print(f"  Hard vs {opp_label} ({games} games)...", flush=True)
                r = play_match(bs, hard, opp_factory, "Hard", opp_label, games=games)
                results.append(r)
                print(f"    Hard {r.x_wins}/{games} ({r.x_win_rate*100:.0f}%) | "
                      f"{opp_label} {r.o_wins}/{games} | D:{r.draws} | {r.seconds:.0f}s", flush=True)

                print(f"  Medium vs {opp_label} ({games} games)...", flush=True)
                r = play_match(bs, med, opp_factory, "Medium", opp_label, games=games)
                results.append(r)
                print(f"    Medium {r.x_wins}/{games} ({r.x_win_rate*100:.0f}%) | "
                      f"{opp_label} {r.o_wins}/{games} | D:{r.draws} | {r.seconds:.0f}s", flush=True)
    return results


# ───────────────────────── CLI ─────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["within", "cross", "both"], default="within")
    p.add_argument("--boards", type=int, nargs="+", default=[5, 6, 7])
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--sims-medium", type=int, default=50)
    p.add_argument("--sims-hard", type=int, default=50)
    p.add_argument("--include-baselines", action="store_true",
                   help="Add Random/Heuristic/Minimax baseline matches.")
    p.add_argument("--output", default="results/tournament.json")
    a = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results: list[MatchResult] = []

    if a.mode in ("within", "both"):
        results.extend(within_board_tournament(
            a.boards, a.games, a.sims_medium, a.sims_hard,
            a.include_baselines, repo_root,
        ))

    if a.mode in ("cross", "both"):
        # Imported lazily so this script works even without the adapter present.
        from experiments.cross_board import cross_board_tournament
        results.extend(cross_board_tournament(
            a.boards, a.games, a.sims_hard, repo_root,
        ))

    out_path = repo_root / a.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    print(f"\n\nWrote {len(results)} match results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
