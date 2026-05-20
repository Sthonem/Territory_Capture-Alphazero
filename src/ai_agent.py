"""Wrapper for the trained neural-network + MCTS game agent."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .encoding import BOARD_CONFIGS
from .game import TerritoryCaptureGame
from .mcts import MCTS
from .model import PolicyValueNet


def _load_arch_from_metadata(model_path: Path) -> dict:
    """Read channels/num_blocks/dropout from sibling .metadata.json if present.

    Returns dict with keys: channels, num_blocks, dropout_p (defaults if missing).
    """
    meta_path = model_path.with_suffix(".metadata.json")
    arch = {"channels": 64, "num_blocks": 5, "dropout_p": 0.0}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for key in ("channels", "num_blocks", "dropout_p"):
                if key in meta:
                    arch[key] = meta[key]
        except (json.JSONDecodeError, OSError):
            pass
    return arch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class AIAgent:
    """Load the trained model and choose moves with MCTS."""

    def __init__(
        self,
        board_size: int = 6,
        model_path: str | Path | None = None,
        c_puct: float = 1.5,
        num_simulations: int = 50,
    ) -> None:
        self.board_size = board_size

        if model_path is not None:
            resolved_model_path = Path(model_path)
        else:
            config = BOARD_CONFIGS.get(board_size, {})
            model_file = config.get("model_file", f"model_{board_size}x{board_size}.pth")
            resolved_model_path = Path(__file__).parent / model_file

        # Auto-detect architecture (channels/blocks/dropout) from sibling metadata.
        arch = _load_arch_from_metadata(resolved_model_path)
        self.model = PolicyValueNet(
            board_size=board_size,
            channels=arch["channels"],
            num_blocks=arch["num_blocks"],
        ).to(device)
        self.model_arch = arch
        self.model_source = "random-init"

        if resolved_model_path.exists():
            state_dict = torch.load(resolved_model_path, map_location=device)
            state_dict = {
                key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
            }
            try:
                self.model.load_state_dict(state_dict, strict=False)
                # If dropout layers exist in checkpoint but missing in model, strict=False ignores
                self.model_source = str(resolved_model_path)
            except RuntimeError as error:
                if model_path is not None:
                    raise RuntimeError(
                        f"Checkpoint is incompatible with the {board_size}x{board_size} model. "
                        f"Architecture: {arch}. Re-train the network for this board size."
                    ) from error
        elif model_path is not None:
            raise FileNotFoundError(f"Checkpoint not found: {resolved_model_path}")
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
