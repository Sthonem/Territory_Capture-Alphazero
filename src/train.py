"""Training utilities for the Territory Capture policy-value network."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
        state = torch.tensor(record.get("encoded_state", record.get("state")), dtype=torch.float32)
        policy = torch.tensor(record.get("policy_target", record.get("policy")), dtype=torch.float32)
        value = torch.tensor([record.get("value_target", record.get("value"))], dtype=torch.float32)
        return state, policy, value


@dataclass(frozen=True)
class EpochMetrics:
    """Stores metrics for one training epoch."""

    epoch: int
    train_loss: float
    validation_loss: float
    learning_rate: float


@dataclass(frozen=True)
class TrainingSummary:
    """Stores training metrics and output artifacts for one optimization run."""

    epochs: int
    train_loss: float
    validation_loss: float
    history: List[EpochMetrics]
    records_used: int
    output_path: Path
    metadata_path: Path


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
    scheduler_step_size: int = 3,
    scheduler_gamma: float = 0.5,
    metadata_path: str | Path | None = None,
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
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma,
    )
    train_loss = 0.0
    validation_loss = 0.0
    history: List[EpochMetrics] = []

    for epoch_index in range(epochs):
        train_loss = _run_epoch(model, train_loader, optimizer=optimizer, training=True)
        validation_loss = _run_epoch(
            model,
            validation_loader,
            optimizer=None,
            training=False,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch_index + 1}/{epochs} — train_loss: {train_loss:.4f}, val_loss: {validation_loss:.4f}, lr: {current_lr:.6f}", flush=True)
        history.append(
            EpochMetrics(
                epoch=epoch_index + 1,
                train_loss=train_loss,
                validation_loss=validation_loss,
                learning_rate=current_lr,
            )
        )
        scheduler.step()

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)
    resolved_metadata_path = (
        Path(metadata_path).resolve()
        if metadata_path is not None
        else destination.with_suffix(".metadata.json")
    )
    _save_training_metadata(
        summary_path=resolved_metadata_path,
        output_path=destination,
        history=history,
        records_used=len(dataset),
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_split=validation_split,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
    )

    return TrainingSummary(
        epochs=epochs,
        train_loss=train_loss,
        validation_loss=validation_loss,
        history=history,
        records_used=len(dataset),
        output_path=destination,
        metadata_path=resolved_metadata_path,
    )


def _save_training_metadata(
    summary_path: Path,
    output_path: Path,
    history: Sequence[EpochMetrics],
    records_used: int,
    batch_size: int,
    learning_rate: float,
    validation_split: float,
    scheduler_step_size: int,
    scheduler_gamma: float,
) -> None:
    """Persist a JSON summary for experiment tracking and checkpoint review."""

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(output_path),
        "records_used": records_used,
        "epochs": len(history),
        "batch_size": batch_size,
        "initial_learning_rate": learning_rate,
        "validation_split": validation_split,
        "scheduler": {
            "name": "StepLR",
            "step_size": scheduler_step_size,
            "gamma": scheduler_gamma,
        },
        "history": [asdict(epoch_metrics) for epoch_metrics in history],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    parser.add_argument("--scheduler-step", type=int, default=3)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--metadata", default=None)
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
        scheduler_step_size=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        metadata_path=args.metadata,
    )
    print("Training complete")
    print(f"Epochs: {summary.epochs}")
    print(f"Train loss: {summary.train_loss:.4f}")
    print(f"Validation loss: {summary.validation_loss:.4f}")
    print(f"Records used: {summary.records_used}")
    print(f"Saved model: {summary.output_path}")
    print(f"Saved metadata: {summary.metadata_path}")


if __name__ == "__main__":
    main()
