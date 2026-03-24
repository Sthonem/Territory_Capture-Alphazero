"""Evaluation helpers for AI-vs-AI and AI-vs-Minimax benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Callable, List, Optional, Protocol, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "territory_capture_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .agents import MinimaxAgent
from .ai_agent import AIAgent
from .game import TerritoryCaptureGame
from .rules import PLAYER_O, PLAYER_X

Position = Tuple[int, int]


class MoveAgent(Protocol):
    """Minimal protocol for agents that can play the game loop."""

    def select_action(self, game: TerritoryCaptureGame) -> Position:
        """Return one legal move."""


@dataclass(frozen=True)
class EvaluationSummary:
    """Stores aggregate results for a benchmark run."""

    total_games: int
    ai1_wins: int
    ai2_wins: int
    draws: int
    running_winrate: List[float]

    @property
    def ai1_win_rate(self) -> float:
        return self.ai1_wins / self.total_games

    @property
    def ai2_win_rate(self) -> float:
        return self.ai2_wins / self.total_games

    @property
    def draw_rate(self) -> float:
        return self.draws / self.total_games


def run_ai_vs_ai(
    num_games: int = 50,
    alternate_starts: bool = True,
    plot_path: str | Path = "results/winrate.png",
    ai1_factory: Optional[Callable[[], AIAgent]] = None,
    ai2_factory: Optional[Callable[[], AIAgent]] = None,
) -> EvaluationSummary:
    """Run many games between two MCTS-based AI agents."""

    ai1_factory = ai1_factory or AIAgent
    ai2_factory = ai2_factory or AIAgent
    ai1 = ai1_factory()
    ai2 = ai2_factory()

    summary = _run_match_series(
        num_games=num_games,
        alternate_starts=alternate_starts,
        player_one_label="AI1",
        player_one_agent=ai1,
        player_two_label="AI2",
        player_two_agent=ai2,
    )
    _save_winrate_plot(summary.running_winrate, plot_path)
    _print_summary("AI vs AI", summary)
    return summary


def run_ai_vs_minimax(
    num_games: int = 30,
    alternate_starts: bool = True,
    plot_path: str | Path = "results/ai_vs_minimax_winrate.png",
    ai_factory: Optional[Callable[[], AIAgent]] = None,
    minimax_agent: Optional[MinimaxAgent] = None,
) -> EvaluationSummary:
    """Run many games between the neural-MCTS AI and the minimax baseline."""

    ai_factory = ai_factory or AIAgent
    ai_agent = ai_factory()
    minimax_agent = minimax_agent or MinimaxAgent(depth=2)

    summary = _run_match_series(
        num_games=num_games,
        alternate_starts=alternate_starts,
        player_one_label="AI",
        player_one_agent=ai_agent,
        player_two_label="Minimax",
        player_two_agent=minimax_agent,
    )
    _save_winrate_plot(summary.running_winrate, plot_path)
    _print_summary("AI vs Minimax", summary)
    return summary


def _run_match_series(
    num_games: int,
    alternate_starts: bool,
    player_one_label: str,
    player_one_agent: MoveAgent,
    player_two_label: str,
    player_two_agent: MoveAgent,
) -> EvaluationSummary:
    """Run a sequence of games and compute win-rate tracking."""

    ai1_wins = 0
    ai2_wins = 0
    draws = 0
    running_winrate: List[float] = []

    for game_index in range(num_games):
        if hasattr(player_one_agent, "reset_search_tree"):
            player_one_agent.reset_search_tree()
        if hasattr(player_two_agent, "reset_search_tree"):
            player_two_agent.reset_search_tree()

        if alternate_starts and game_index % 2 == 1:
            x_label, x_agent = player_two_label, player_two_agent
            o_label, o_agent = player_one_label, player_one_agent
        else:
            x_label, x_agent = player_one_label, player_one_agent
            o_label, o_agent = player_two_label, player_two_agent

        winner_label = _play_single_game(
            x_label=x_label,
            x_agent=x_agent,
            o_label=o_label,
            o_agent=o_agent,
        )

        if winner_label == player_one_label:
            ai1_wins += 1
            running_points = 1.0
        elif winner_label == player_two_label:
            ai2_wins += 1
            running_points = 0.0
        else:
            draws += 1
            running_points = 0.5

        if running_winrate:
            total_points = running_winrate[-1] * game_index + running_points
            running_winrate.append(total_points / (game_index + 1))
        else:
            running_winrate.append(running_points)

    return EvaluationSummary(
        total_games=num_games,
        ai1_wins=ai1_wins,
        ai2_wins=ai2_wins,
        draws=draws,
        running_winrate=running_winrate,
    )


def _play_single_game(
    x_label: str,
    x_agent: MoveAgent,
    o_label: str,
    o_agent: MoveAgent,
) -> Optional[str]:
    """Play one full game and return the winning side label."""

    game = TerritoryCaptureGame()
    while not game.is_terminal():
        if game.current_player == PLAYER_X:
            action = x_agent.select_action(game.clone())
        else:
            action = o_agent.select_action(game.clone())
        game.apply_action(action)

    winner = game.get_winner()
    if winner == PLAYER_X:
        return x_label
    if winner == PLAYER_O:
        return o_label
    return None


def _save_winrate_plot(running_winrate: List[float], path: str | Path) -> None:
    """Save a win-rate curve to the results folder."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(running_winrate) + 1), running_winrate, color="#2563eb")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Game index")
    plt.ylabel("Running winrate")
    plt.title("AI winrate over time")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(destination)
    plt.close()


def _print_summary(title: str, summary: EvaluationSummary) -> None:
    """Print a compact benchmark report."""

    print(title)
    print(f"Total games: {summary.total_games}")
    print(f"AI1 wins: {summary.ai1_wins} ({summary.ai1_win_rate:.1%})")
    print(f"AI2 wins: {summary.ai2_wins} ({summary.ai2_win_rate:.1%})")
    print(f"Draws: {summary.draws} ({summary.draw_rate:.1%})")
