"""Batch simulation helpers for baseline Territory Capture agents."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from .agents import Agent, create_agent
from .game import TerritoryCaptureGame
from .rules import PLAYER_O, PLAYER_X


@dataclass(frozen=True)
class SimulationSummary:
    """Aggregate statistics from a batch of games."""

    x_agent_name: str
    o_agent_name: str
    total_games: int
    x_wins: int
    o_wins: int
    draws: int
    average_x_score: float
    average_o_score: float
    average_score_margin: float

    @property
    def x_win_rate(self) -> float:
        """Return the fraction of games won by X."""

        return self.x_wins / self.total_games

    @property
    def o_win_rate(self) -> float:
        """Return the fraction of games won by O."""

        return self.o_wins / self.total_games

    @property
    def draw_rate(self) -> float:
        """Return the fraction of drawn games."""

        return self.draws / self.total_games


def play_agent_game(
    x_agent: Agent,
    o_agent: Agent,
    board_size: int = 6,
    stones_per_player: int = 10,
) -> TerritoryCaptureGame:
    """Play one full game between two agents."""

    game = TerritoryCaptureGame(
        board_size=board_size,
        stones_per_player=stones_per_player,
    )

    while not game.is_terminal():
        agent = x_agent if game.current_player == PLAYER_X else o_agent
        action = agent.select_action(game.clone())
        game.apply_action(action)

    return game


def run_simulation(
    x_agent: Agent,
    o_agent: Agent,
    num_games: int,
    board_size: int = 6,
    stones_per_player: int = 10,
) -> SimulationSummary:
    """Run a batch of games and return aggregate results."""

    if num_games <= 0:
        raise ValueError("num_games must be greater than 0.")

    x_wins = 0
    o_wins = 0
    draws = 0
    total_x_score = 0
    total_o_score = 0
    total_margin = 0

    for _ in range(num_games):
        game = play_agent_game(
            x_agent=x_agent,
            o_agent=o_agent,
            board_size=board_size,
            stones_per_player=stones_per_player,
        )
        scores = game.get_total_scores()
        winner = game.get_winner()

        total_x_score += scores[PLAYER_X]
        total_o_score += scores[PLAYER_O]
        total_margin += abs(scores[PLAYER_X] - scores[PLAYER_O])

        if winner == PLAYER_X:
            x_wins += 1
        elif winner == PLAYER_O:
            o_wins += 1
        else:
            draws += 1

    return SimulationSummary(
        x_agent_name=x_agent.name,
        o_agent_name=o_agent.name,
        total_games=num_games,
        x_wins=x_wins,
        o_wins=o_wins,
        draws=draws,
        average_x_score=total_x_score / num_games,
        average_o_score=total_o_score / num_games,
        average_score_margin=total_margin / num_games,
    )


def format_summary(summary: SimulationSummary) -> str:
    """Return a readable simulation report."""

    return "\n".join(
        [
            "Territory Capture Simulation",
            f"Matchup: X={summary.x_agent_name} vs O={summary.o_agent_name}",
            f"Total games: {summary.total_games}",
            f"X wins: {summary.x_wins} ({summary.x_win_rate:.1%})",
            f"O wins: {summary.o_wins} ({summary.o_win_rate:.1%})",
            f"Draws: {summary.draws} ({summary.draw_rate:.1%})",
            f"Average X total score: {summary.average_x_score:.2f}",
            f"Average O total score: {summary.average_o_score:.2f}",
            f"Average score margin: {summary.average_score_margin:.2f}",
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for batch simulation."""

    parser = argparse.ArgumentParser(description="Run Territory Capture simulations.")
    parser.add_argument(
        "--x-agent",
        default="random",
        choices=["random", "heuristic", "minimax"],
    )
    parser.add_argument(
        "--o-agent",
        default="random",
        choices=["random", "heuristic", "minimax"],
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run a simulation batch from the command line."""

    args = parse_args()
    x_agent = create_agent(args.x_agent, seed=args.seed)
    o_seed: Optional[int] = None if args.seed is None else args.seed + 1
    o_agent = create_agent(args.o_agent, seed=o_seed)
    summary = run_simulation(
        x_agent=x_agent,
        o_agent=o_agent,
        num_games=args.games,
    )
    print(format_summary(summary))


if __name__ == "__main__":
    main()
