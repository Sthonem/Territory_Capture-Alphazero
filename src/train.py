"""Training utilities for the Territory Capture policy-value network."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split

from .ai_agent import device
from .model import PolicyValueNet


class SelfPlayDataset(Dataset):
    """PyTorch dataset backed by exported self-play JSON records."""

    def __init__(self, records: Sequence[dict]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        state = torch.tensor(record["state"], dtype=torch.float32)
        policy = torch.tensor(record["policy"], dtype=torch.float32)
        value = torch.tensor([record["value"]], dtype=torch.float32)
        return state, policy, value


@dataclass(frozen=True)
class TrainingSummary:
    """Stores training metrics for one optimization run."""

    epochs: int
    train_loss: float
    validation_loss: float
    output_path: Path


def load_self_play_records(path: str | Path) -> List[dict]:
    """Load self-play records from JSON."""

    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8"))


def train_policy_value_model(
    data_path: str | Path | None = None,
    output_path: str | Path = "src/latest_model.pth",
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    validation_split: float = 0.1,
    model: PolicyValueNet | None = None,
    records: Sequence[dict] | None = None,
) -> TrainingSummary:
    """Train the policy-value network on self-play data."""

    if records is None:
        if data_path is None:
            raise ValueError("Provide either data_path or records for training.")
        records = load_self_play_records(data_path)

    dataset = SelfPlayDataset(records)
    model = (model or PolicyValueNet()).to(device)

    train_loader, validation_loader = _build_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        validation_split=validation_split,
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = 0.0
    validation_loss = 0.0

    for _ in range(epochs):
        train_loss = _run_epoch(model, train_loader, optimizer=optimizer, training=True)
        validation_loss = _run_epoch(
            model,
            validation_loader,
            optimizer=None,
            training=False,
        )

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)

    return TrainingSummary(
        epochs=epochs,
        train_loss=train_loss,
        validation_loss=validation_loss,
        output_path=destination,
    )


def _build_dataloaders(
    dataset: Dataset,
    batch_size: int,
    validation_split: float,
) -> tuple[DataLoader, DataLoader]:
    """Split the dataset into train/validation loaders."""

    validation_size = max(1, int(len(dataset) * validation_split))
    validation_size = min(validation_size, len(dataset) - 1)
    train_size = len(dataset) - validation_size

    if train_size <= 0:
        raise ValueError("Dataset is too small to create a train/validation split.")

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader


def _run_epoch(
    model: PolicyValueNet,
    dataloader: DataLoader,
    optimizer: optim.Optimizer | None,
    training: bool,
) -> float:
    """Run one training or validation epoch."""

    model.train(mode=training)
    total_loss = 0.0
    total_batches = 0

    for states, policy_targets, value_targets in dataloader:
        states = states.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)

        with torch.set_grad_enabled(training):
            policy_logits, predicted_values = model(states)
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(predicted_values, value_targets)
            loss = policy_loss + value_loss

            if training and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model training."""

    parser = argparse.ArgumentParser(description="Train the Territory Capture model.")
    parser.add_argument("--data", default="self_play_data.json")
    parser.add_argument("--output", default="src/latest_model.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    """Run model training from the command line."""

    args = parse_args()
    summary = train_policy_value_model(
        data_path=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    print("Training complete")
    print(f"Epochs: {summary.epochs}")
    print(f"Train loss: {summary.train_loss:.4f}")
    print(f"Validation loss: {summary.validation_loss:.4f}")
    print(f"Saved model: {summary.output_path}")


if __name__ == "__main__":
    main()
