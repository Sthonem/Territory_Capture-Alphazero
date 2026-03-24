"""Monte Carlo Tree Search guided by the policy-value network."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .encoding import action_to_index, encode_state
from .game import TerritoryCaptureGame


@dataclass
class Node:
    """One MCTS node tied to a specific game state."""

    game: TerritoryCaptureGame
    parent: Optional["Node"] = None
    prior: float = 1.0
    base_prior: float = 1.0
    children: Dict[tuple[int, int], "Node"] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0

    def value(self) -> float:
        """Return the mean value from this node's perspective."""

        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        """Return True once legal children have been added."""

        return bool(self.children)


class MCTS:
    """A small PUCT-based search implementation for inference-time play."""

    def __init__(
        self,
        model: torch.nn.Module,
        c_puct: float = 1.5,
        num_simulations: int = 50,
        device: Optional[torch.device] = None,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = device or torch.device("cpu")
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.rng = random.Random(seed)
        self.root: Optional[Node] = None

    def search(
        self,
        game: TerritoryCaptureGame,
        return_policy: bool = False,
        temperature: float = 0.0,
        add_exploration_noise: bool = False,
    ) -> tuple[int, int] | tuple[tuple[int, int], list[float]]:
        """Run MCTS and return the most visited root action.

        When ``return_policy`` is True, also return the normalized visit-count
        distribution over the fixed 25-action space.
        """

        root = self._sync_root(game)
        self._expand(root)
        if add_exploration_noise:
            self._apply_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            while node.is_expanded() and not node.game.is_terminal():
                node = self._select_child(node)
                search_path.append(node)

            if node.game.is_terminal():
                leaf_value = self._terminal_value(node.game)
            else:
                leaf_value = self._expand(node)

            self._backpropagate(search_path, leaf_value)

        best_action = self._select_final_action(root, temperature=temperature)
        policy = self.get_visit_count_policy()
        if return_policy:
            return best_action, policy
        return best_action

    def advance_to_action(self, action: tuple[int, int]) -> None:
        """Reuse the subtree that matches a played action when available."""

        if self.root is None:
            return
        child = self.root.children.get(action)
        if child is None:
            self.root = None
            return
        child.parent = None
        self.root = child

    def reset(self) -> None:
        """Clear the cached search tree."""

        self.root = None

    def get_visit_count_policy(self) -> list[float]:
        """Return the root visit-count distribution over all 25 actions."""

        if self.root is None:
            return [0.0] * 25

        total_visits = sum(child.visit_count for child in self.root.children.values())
        policy = [0.0] * 25
        if total_visits == 0:
            return policy

        for action, child in self.root.children.items():
            policy[action_to_index(action)] = child.visit_count / total_visits
        return policy

    def _expand(self, node: Node) -> float:
        """Evaluate a state and add children using network priors."""

        legal_moves = node.game.get_legal_actions()
        if not legal_moves:
            return self._terminal_value(node.game)
        if node.is_expanded():
            _, value = self._evaluate(node.game)
            return value.item()

        policy_logits, value = self._evaluate(node.game)
        policy_probs = F.softmax(policy_logits, dim=0)

        move_priors = []
        for move in legal_moves:
            move_priors.append(policy_probs[action_to_index(move)].item())

        total_prior = sum(move_priors)
        if total_prior <= 0:
            move_priors = [1.0 / len(legal_moves)] * len(legal_moves)
        else:
            move_priors = [prior / total_prior for prior in move_priors]

        for move, prior in zip(legal_moves, move_priors):
            child_game = node.game.clone()
            child_game.apply_action(move)
            node.children[move] = Node(
                game=child_game,
                parent=node,
                prior=prior,
                base_prior=prior,
            )

        return value.item()

    def _sync_root(self, game: TerritoryCaptureGame) -> Node:
        """Reuse an existing subtree when it matches the current game state."""

        if self.root is None:
            self.root = Node(game=game.clone())
            return self.root

        if self._same_position(self.root.game, game):
            return self.root

        for child in self.root.children.values():
            if self._same_position(child.game, game):
                child.parent = None
                self.root = child
                return self.root

        self.root = Node(game=game.clone())
        return self.root

    def _select_child(self, node: Node) -> Node:
        """Select one child using the PUCT rule."""

        best_score = -math.inf
        best_child: Optional[Node] = None
        exploration_base = math.sqrt(node.visit_count + 1)

        for _, child in sorted(node.children.items()):
            prior_score = (
                self.c_puct * child.prior * exploration_base / (1 + child.visit_count)
            )
            # Child values are from the child's perspective, so flip the sign.
            score = -child.value() + prior_score
            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
            raise RuntimeError("MCTS selection failed to choose a child.")
        return best_child

    def _backpropagate(self, search_path: list[Node], value: float) -> None:
        """Propagate the evaluated leaf value back to the root."""

        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _apply_dirichlet_noise(self, node: Node) -> None:
        """Blend Dirichlet noise into root priors for self-play exploration."""

        if not node.children:
            return

        actions = sorted(node.children)
        noise = self._sample_dirichlet(len(actions))
        for action, noise_value in zip(actions, noise):
            child = node.children[action]
            child.prior = (
                (1.0 - self.dirichlet_epsilon) * child.base_prior
                + self.dirichlet_epsilon * noise_value
            )

    def _sample_dirichlet(self, size: int) -> list[float]:
        """Draw a small Dirichlet sample using the local RNG."""

        gamma_samples = [
            self.rng.gammavariate(self.dirichlet_alpha, 1.0) for _ in range(size)
        ]
        total = sum(gamma_samples)
        if total <= 0:
            return [1.0 / size] * size
        return [sample / total for sample in gamma_samples]

    def _select_final_action(
        self,
        node: Node,
        temperature: float,
    ) -> tuple[int, int]:
        """Choose the root action from visit counts with optional sampling."""

        if temperature <= 0:
            action, _ = max(
                sorted(node.children.items()),
                key=lambda item: item[1].visit_count,
            )
            return action

        actions = sorted(node.children)
        weights = []
        for action in actions:
            visits = node.children[action].visit_count
            # Keep zero-visit actions selectable only through their tiny floor.
            adjusted = max(float(visits), 1e-6) ** (1.0 / temperature)
            weights.append(adjusted)

        return self.rng.choices(actions, weights=weights, k=1)[0]

    def _evaluate(self, game: TerritoryCaptureGame) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the policy-value network on one encoded game state."""

        encoded_state = encode_state(game)
        state_tensor = torch.tensor(
            encoded_state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)

        return policy_logits[0], value[0]

    def _terminal_value(self, game: TerritoryCaptureGame) -> float:
        """Return the terminal outcome from the current state's perspective."""

        winner = game.get_winner()
        if winner is None:
            return 0.0
        if winner == game.current_player:
            return 1.0
        return -1.0

    def _same_position(
        self,
        first_game: TerritoryCaptureGame,
        second_game: TerritoryCaptureGame,
    ) -> bool:
        """Check whether two game objects describe the same position."""

        return (
            first_game.current_player == second_game.current_player
            and first_game.move_count == second_game.move_count
            and first_game.board == second_game.board
        )
