"""Tests for baseline Territory Capture agents and simulation support."""

from __future__ import annotations

import unittest

from src.agents import HeuristicAgent, RandomAgent
from src.game import TerritoryCaptureGame
from src.simulate import format_summary, run_simulation


class TestAgentsAndSimulation(unittest.TestCase):
    """Validate baseline agents and batch simulation helpers."""

    def test_random_agent_returns_legal_move(self) -> None:
        game = TerritoryCaptureGame()
        agent = RandomAgent(seed=7)

        action = agent.select_action(game)

        self.assertIn(action, game.get_legal_actions())

    def test_heuristic_agent_returns_legal_move(self) -> None:
        game = TerritoryCaptureGame()
        agent = HeuristicAgent()

        action = agent.select_action(game)

        self.assertIn(action, game.get_legal_actions())

    def test_clone_does_not_mutate_original_game(self) -> None:
        game = TerritoryCaptureGame()
        cloned_game = game.clone()

        cloned_game.apply_action((2, 2))

        self.assertEqual(game.board[2][2], ".")
        self.assertEqual(cloned_game.board[2][2], "X")
        self.assertEqual(game.move_count, 0)
        self.assertEqual(cloned_game.move_count, 1)

    def test_simulation_batch_completes(self) -> None:
        summary = run_simulation(
            x_agent=HeuristicAgent(),
            o_agent=RandomAgent(seed=11),
            num_games=5,
        )

        self.assertEqual(summary.total_games, 5)
        self.assertEqual(summary.x_wins + summary.o_wins + summary.draws, 5)
        self.assertEqual(summary.x_agent_name, "heuristic")
        self.assertEqual(summary.o_agent_name, "random")

    def test_heuristic_explanation_returns_feature_breakdown(self) -> None:
        game = TerritoryCaptureGame()
        agent = HeuristicAgent()

        breakdown = agent.explain_action(game, (2, 2))

        self.assertGreater(breakdown.total_score, 0)
        self.assertEqual(breakdown.empty_space, 8)

    def test_simulation_summary_format_includes_rates(self) -> None:
        summary = run_simulation(
            x_agent=RandomAgent(seed=1),
            o_agent=RandomAgent(seed=2),
            num_games=3,
        )

        report = format_summary(summary)

        self.assertIn("Matchup:", report)
        self.assertIn("X wins:", report)
        self.assertIn("%", report)


if __name__ == "__main__":
    unittest.main()
