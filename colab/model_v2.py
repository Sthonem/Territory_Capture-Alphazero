"""Improved policy-value network (v2) with configurable depth/width + dropout.

Backward-compatible with src.model.PolicyValueNet weights when channels=64, num_blocks=5,
and dropout_p=0.0 (state_dict keys match).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.drop(x)
        x += residual
        return F.relu(x)


class PolicyValueNetV2(nn.Module):
    """Configurable AlphaZero-style policy-value network."""

    def __init__(
        self,
        board_size: int = 6,
        in_channels: int = 2,
        channels: int = 64,
        num_blocks: int = 5,
        dropout_p: float = 0.0,
        value_hidden: int = 32,  # paper uses 32 (was 64 in old code)
    ) -> None:
        super().__init__()
        self.board_size = board_size
        action_space = board_size * board_size

        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.res_blocks = nn.ModuleList(
            [ResBlock(channels, dropout_p=dropout_p) for _ in range(num_blocks)]
        )

        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_space)
        self.policy_drop = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)
        self.value_drop = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)

        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_drop(policy)
        policy = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = self.value_drop(value)
        value = torch.tanh(self.value_fc2(value))

        return policy, value
