"""Wrapper for the trained neural-network + MCTS game agent."""

from __future__ import annotations

from pathlib import Path

import torch

from .game import TerritoryCaptureGame
from .mcts import MCTS
from .model import PolicyValueNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AIAgent:
    """Load the trained model and choose moves with MCTS."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        c_puct: float = 1.5,
        num_simulations: int = 50,
    ) -> None:
        resolved_model_path = (
            Path(model_path)
            if model_path is not None
            else Path(__file__).with_name("model.pth")
        )
        self.model = PolicyValueNet().to(device)
        state_dict = torch.load(resolved_model_path, map_location=device)
        # Training may have wrapped the model with torch.compile, which prefixes
        # parameter names with "_orig_mod.". Strip that prefix for inference.
        state_dict = {
            key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
        }
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.mcts = MCTS(
            model=self.model,
            c_puct=c_puct,
            num_simulations=num_simulations,
            device=device,
        )

    def select_action(self, game: TerritoryCaptureGame):
        """Return one legal move for the current state."""

        action = self.mcts.search(game.clone())
        self.mcts.advance_to_action(action)
        return action

    def select_action_with_policy(
        self,
        game: TerritoryCaptureGame,
        temperature: float = 0.0,
        add_exploration_noise: bool = False,
    ) -> tuple[tuple[int, int], list[float]]:
        """Return both the chosen action and the root visit-count policy."""

        action, policy = self.mcts.search(
            game.clone(),
            return_policy=True,
            temperature=temperature,
            add_exploration_noise=add_exploration_noise,
        )
        self.mcts.advance_to_action(action)
        return action, policy

    def reset_search_tree(self) -> None:
        """Clear any cached MCTS subtree between independent games."""

        self.mcts.reset()
