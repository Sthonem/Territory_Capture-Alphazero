"""Cross-board generalization experiment.

Question: A policy-value network was trained on board size B_train.
What if we deploy it on a different board size B_play?

The Conv body of the ResNet is fully convolutional (3×3 conv with padding=1),
so it can technically accept any (2, B, B) input. The problem is the
policy/value heads end in Linear layers sized for B_train², not B_play².

This module provides ``CrossBoardAgent``: a wrapper that
1. Resizes the encoded state from (2, B_play, B_play) → (2, B_train, B_train)
   via nearest-neighbor interpolation (preserves stone positions in spatial
   approximation).
2. Runs the trained model.
3. Reshapes the policy logits from (B_train²,) → (B_train, B_train), then
   upsamples to (B_play, B_play) via bilinear interpolation.
4. Applies the legal-action mask of the real game and selects argmax.

This is an *approximate* transfer: it answers "does the learned representation
generalize across board sizes?" rather than "can this exact model play the
other game perfectly". The result is academically meaningful for the
multi-board scaling discussion required by the project.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from src.ai_agent import AIAgent, device, _load_arch_from_metadata
from src.encoding import BOARD_CONFIGS, encode_state
from src.game import TerritoryCaptureGame
from src.model import PolicyValueNet


class CrossBoardAgent:
    """A trained model from one board size playing on another."""

    def __init__(
        self,
        model_board_size: int,
        play_board_size: int,
        model_path: str | Path,
        num_simulations: int = 0,  # MCTS disabled by default — pure NN policy
        temperature: float = 0.0,
    ) -> None:
        self.model_board_size = model_board_size
        self.play_board_size = play_board_size
        self.temperature = temperature
        self.num_simulations = num_simulations

        model_path = Path(model_path)
        arch = _load_arch_from_metadata(model_path)
        self.model = PolicyValueNet(
            board_size=model_board_size,
            channels=arch["channels"],
            num_blocks=arch["num_blocks"],
        ).to(device)
        if model_path.exists():
            sd = torch.load(model_path, map_location=device)
            sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
            self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        self.model_source = str(model_path)
        self.model_arch = arch

    def _state_tensor(self, game: TerritoryCaptureGame) -> torch.Tensor:
        """Encode + resize the live state to the model's expected size."""
        enc = np.asarray(encode_state(game), dtype=np.float32)  # (2, B_play, B_play)
        t = torch.from_numpy(enc).unsqueeze(0).to(device)        # (1, 2, B_play, B_play)
        if self.play_board_size != self.model_board_size:
            t = F.interpolate(
                t,
                size=(self.model_board_size, self.model_board_size),
                mode="nearest",
            )
        return t

    def select_action(self, game: TerritoryCaptureGame) -> tuple[int, int]:
        """Pick a legal action using the cross-board policy logits."""

        with torch.no_grad():
            state = self._state_tensor(game)
            policy_logits, _ = self.model(state)
            B_train = self.model_board_size
            # (1, B_train²) → (1, 1, B_train, B_train)
            policy_2d = policy_logits.view(1, 1, B_train, B_train)
            if self.play_board_size != B_train:
                policy_2d = F.interpolate(
                    policy_2d,
                    size=(self.play_board_size, self.play_board_size),
                    mode="bilinear",
                    align_corners=False,
                )
            policy_2d = policy_2d.view(-1).cpu().numpy()
        # Mask illegal cells (occupied)
        mask = np.full(self.play_board_size * self.play_board_size, -1e9, dtype=np.float32)
        for (r, c) in game.get_legal_actions():
            mask[r * self.play_board_size + c] = 0.0
        scores = policy_2d + mask

        if self.temperature > 0:
            # softmax sampling
            probs = np.exp(scores - scores.max())
            probs = probs / probs.sum()
            idx = int(np.random.choice(len(probs), p=probs))
        else:
            idx = int(np.argmax(scores))
        return (idx // self.play_board_size, idx % self.play_board_size)


# ───────────────────────── tournament ─────────────────────────

def cross_board_tournament(
    boards: list[int],
    games: int,
    sims_hard: int,
    repo_root: Path,
) -> list:
    """Run Hard-model cross-board matches.

    For each pair (B_train, B_play) with B_train ≠ B_play in `boards`:
       Hard-trained-on-B_train (via CrossBoardAgent) vs Hard-native-B_play
       on a B_play board.
    """
    from experiments.tournament import play_match  # late import to avoid cycle

    results = []
    print(f"\n══════════ Cross-board tournament ══════════", flush=True)
    for b_play in boards:
        native_path = repo_root / "src" / f"model_hard_{b_play}x{b_play}.pth"
        if not native_path.exists() and b_play == 6:
            native_path = repo_root / "src" / "model_hard.pth"
        if not native_path.exists():
            print(f"  ⚠ no native hard model for {b_play}×{b_play} — skip", flush=True)
            continue

        # Native vs cross-board agents
        for b_train in boards:
            if b_train == b_play:
                continue
            train_path = repo_root / "src" / f"model_hard_{b_train}x{b_train}.pth"
            if not train_path.exists() and b_train == 6:
                train_path = repo_root / "src" / "model_hard.pth"
            if not train_path.exists():
                print(f"  ⚠ no hard model for {b_train}×{b_train} — skip cross", flush=True)
                continue

            # cross_agent = Hard trained on b_train, playing on b_play
            cross_factory = lambda b_train=b_train, b_play=b_play, p=str(train_path): \
                CrossBoardAgent(model_board_size=b_train, play_board_size=b_play, model_path=p)
            native_factory = lambda b_play=b_play, p=str(native_path), sims=sims_hard: \
                AIAgent(board_size=b_play, model_path=p, num_simulations=sims)

            label_cross = f"Hard-{b_train}→{b_play}"
            label_native = f"Hard-{b_play}(native)"
            print(f"  {label_cross} vs {label_native} ({games} games)...", flush=True)
            r = play_match(b_play, cross_factory, native_factory,
                           label_cross, label_native, games=games)
            results.append(r)
            print(f"    {label_cross}: {r.x_wins}/{games} ({r.x_win_rate*100:.0f}%) | "
                  f"{label_native}: {r.o_wins}/{games} ({r.o_win_rate*100:.0f}%) | "
                  f"D:{r.draws} | {r.seconds:.0f}s", flush=True)
    return results
