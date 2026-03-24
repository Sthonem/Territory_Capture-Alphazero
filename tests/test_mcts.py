"""Tests for the neural-network-guided MCTS helper."""

from __future__ import annotations

import unittest

import torch

from src.game import TerritoryCaptureGame
from src.mcts import MCTS


class DummyModel(torch.nn.Module):
    """Small deterministic model used for MCTS tests."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        logits = torch.zeros((batch_size, 25), dtype=torch.float32)
        value = torch.full((batch_size, 1), 0.25, dtype=torch.float32)
        return logits, value


class TestMCTS(unittest.TestCase):
    """Validate policy extraction and root reuse in MCTS."""

    def test_search_returns_legal_move_and_policy_distribution(self) -> None:
        mcts = MCTS(model=DummyModel(), num_simulations=5)
        game = TerritoryCaptureGame()

        action, policy = mcts.search(game, return_policy=True)

        self.assertIn(action, game.get_legal_actions())
        self.assertEqual(len(policy), 25)
        self.assertAlmostEqual(sum(policy), 1.0, places=5)

    def test_root_can_advance_to_selected_action(self) -> None:
        mcts = MCTS(model=DummyModel(), num_simulations=5)
        game = TerritoryCaptureGame()

        action = mcts.search(game)
        self.assertIsNotNone(mcts.root)

        mcts.advance_to_action(action)

        self.assertIsNotNone(mcts.root)
        self.assertEqual(mcts.root.game.move_count, 1)

    def test_dirichlet_noise_changes_root_priors(self) -> None:
        mcts = MCTS(model=DummyModel(), num_simulations=1, seed=7)
        game = TerritoryCaptureGame()

        mcts.search(game, add_exploration_noise=True)

        priors = [child.prior for child in mcts.root.children.values()]
        uniform_prior = 1.0 / len(game.get_legal_actions())
        self.assertTrue(any(abs(prior - uniform_prior) > 1e-6 for prior in priors))


if __name__ == "__main__":
    unittest.main()
