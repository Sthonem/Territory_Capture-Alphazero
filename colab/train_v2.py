"""Improved training pipeline (v2) — paper-aligned hyperparameters.

Features:
  - NPZ data loader (fast, memory-mapped)
  - 8-fold dihedral symmetry augmentation
  - Early stopping on validation loss
  - Cosine annealing LR schedule (matches paper)
  - Composite loss: Lπ + λv·Lv − λe·H(π)   (paper: λv=0.1, λe=0.02)
  - Adam(lr=5e-4, wd=5e-4)                  (matches paper)
  - AMP + TF32 (cuda) for A100/T4 speedup    (matches paper)
  - Eval metrics: Top-1 / Top-3 policy accuracy + Value MAE (matches paper)
  - Configurable model architecture (channels, blocks, dropout, value_hidden)
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split

from model_v2 import PolicyValueNetV2


# ───────────────────────── data ─────────────────────────

class NpzSelfPlayDataset(Dataset):
    """Memory-efficient dataset reading from a .npz file produced by json_to_npz.py."""

    def __init__(self, npz_path: str | Path, augment: bool = False) -> None:
        data = np.load(npz_path)
        self.states = data["states"]      # (N, 2, B, B) float32
        self.policies = data["policies"]  # (N, B*B) float32
        self.values = data["values"]      # (N,) float32
        self.board_size = int(data["board_size"])
        self.augment = augment
        self.n = len(self.states)

    def __len__(self) -> int:
        return self.n

    def _apply_symmetry(self, state, policy, sym_id):
        B = self.board_size
        policy_2d = policy.reshape(B, B)
        if sym_id >= 4:
            state = state[:, :, ::-1]
            policy_2d = policy_2d[:, ::-1]
        k = sym_id % 4
        if k:
            state = np.rot90(state, k=k, axes=(1, 2))
            policy_2d = np.rot90(policy_2d, k=k)
        return np.ascontiguousarray(state), np.ascontiguousarray(policy_2d.reshape(-1))

    def __getitem__(self, index: int):
        state = self.states[index]
        policy = self.policies[index]
        value = self.values[index]
        if self.augment:
            sym_id = np.random.randint(0, 8)
            state, policy = self._apply_symmetry(state, policy, sym_id)
        return (
            torch.from_numpy(np.ascontiguousarray(state)),
            torch.from_numpy(np.ascontiguousarray(policy)),
            torch.tensor([value], dtype=torch.float32),
        )


# ───────────────────────── metrics ─────────────────────────

def _policy_topk_correct(logits: torch.Tensor, target_policy: torch.Tensor, k: int) -> int:
    """Count batch examples where argmax(target) is in topk(logits)."""
    target_idx = target_policy.argmax(dim=1)  # (B,)
    topk_idx = logits.topk(k, dim=1).indices   # (B, k)
    correct = (topk_idx == target_idx.unsqueeze(1)).any(dim=1)
    return int(correct.sum().item())


# ───────────────────────── training ─────────────────────────

@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_policy: float
    train_value: float
    train_entropy: float
    val_loss: float
    val_policy: float
    val_value: float
    val_top1: float
    val_top3: float
    val_value_mae: float
    lr: float


def _run_epoch(
    model: PolicyValueNetV2,
    loader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer | None,
    training: bool,
    value_weight: float,
    entropy_weight: float,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
) -> dict:
    model.train(mode=training)
    tot_loss = tot_pol = tot_val = tot_ent = 0.0
    n_batches = 0
    top1 = top3 = mae = total_n = 0

    for states, policies, values in loader:
        states = states.to(device, non_blocking=True)
        policies = policies.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)

        amp_ctx = torch.amp.autocast(device_type=device.type, enabled=use_amp)

        with torch.set_grad_enabled(training):
            with amp_ctx:
                policy_logits, predicted_values = model(states)
                log_probs = F.log_softmax(policy_logits, dim=1)
                probs = log_probs.exp()
                # Composite loss (paper Eq.4): Lπ + λv·Lv − λe·H(π)
                policy_loss = -(policies * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(predicted_values, values)
                entropy = -(probs * log_probs).sum(dim=1).mean()  # Shannon entropy
                loss = policy_loss + value_weight * value_loss - entropy_weight * entropy

            if training and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        # eval metrics (in fp32 — small overhead)
        with torch.no_grad():
            top1 += _policy_topk_correct(policy_logits.float(), policies, k=1)
            top3 += _policy_topk_correct(policy_logits.float(), policies, k=3)
            mae += float((predicted_values.float() - values).abs().sum().item())
            total_n += policies.size(0)

        tot_loss += loss.item()
        tot_pol += policy_loss.item()
        tot_val += value_loss.item()
        tot_ent += entropy.item()
        n_batches += 1

    n = max(1, n_batches)
    n_examples = max(1, total_n)
    return {
        "loss": tot_loss / n,
        "policy": tot_pol / n,
        "value": tot_val / n,
        "entropy": tot_ent / n,
        "top1": top1 / n_examples,
        "top3": top3 / n_examples,
        "value_mae": mae / n_examples,
    }


def train(
    npz_path: str,
    output_path: str,
    board_size: int,
    channels: int = 64,
    num_blocks: int = 5,
    dropout_p: float = 0.0,
    value_hidden: int = 32,           # paper: 32
    epochs: int = 25,                 # paper: 25
    batch_size: int = 1024,
    lr: float = 5e-4,                 # paper: 5e-4
    min_lr: float = 1e-6,
    weight_decay: float = 5e-4,       # paper: 5e-4
    value_weight: float = 0.1,        # paper: λv = 0.1
    entropy_weight: float = 0.02,     # paper: λe = 0.02
    val_split: float = 0.05,
    augment: bool = True,
    patience: int = 6,
    num_workers: int = 2,
    use_amp: bool = True,             # paper: AMP
    use_tf32: bool = True,            # paper: TF32
    device: str | None = None,
) -> None:
    """Train v2 model with paper-aligned hyperparameters."""

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {dev}", flush=True)

    if dev.type == "cuda" and use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("TF32 enabled, cudnn.benchmark=True", flush=True)
    can_amp = (dev.type == "cuda") and use_amp
    if can_amp:
        print("AMP (autocast + GradScaler) enabled", flush=True)

    full = NpzSelfPlayDataset(npz_path, augment=augment)
    assert full.board_size == board_size, (
        f"NPZ board_size={full.board_size} != requested {board_size}"
    )
    n_val = max(1, int(len(full) * val_split))
    n_train = len(full) - n_val
    train_set, val_set = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(dev.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(dev.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    model = PolicyValueNetV2(
        board_size=board_size,
        channels=channels,
        num_blocks=num_blocks,
        dropout_p=dropout_p,
        value_hidden=value_hidden,
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model: ch={channels}, blocks={num_blocks}, dp={dropout_p}, "
        f"value_hidden={value_hidden}, params={n_params:,}",
        flush=True,
    )

    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=min_lr
    )
    scaler = torch.amp.GradScaler() if can_amp else None

    history: List[EpochMetrics] = []
    best_val = float("inf")
    best_epoch = 0
    patience_left = patience
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        full.augment = augment
        tr = _run_epoch(
            model, train_loader, dev, optimizer, True,
            value_weight, entropy_weight, scaler, can_amp,
        )
        full.augment = False
        va = _run_epoch(
            model, val_loader, dev, None, False,
            value_weight, entropy_weight, None, can_amp,
        )
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history.append(EpochMetrics(
            epoch=epoch,
            train_loss=tr["loss"], train_policy=tr["policy"],
            train_value=tr["value"], train_entropy=tr["entropy"],
            val_loss=va["loss"], val_policy=va["policy"], val_value=va["value"],
            val_top1=va["top1"], val_top3=va["top3"],
            val_value_mae=va["value_mae"], lr=cur_lr,
        ))
        print(
            f"Ep {epoch}/{epochs} — "
            f"train_loss: {tr['loss']:.4f} (P:{tr['policy']:.4f} V:{tr['value']:.4f} H:{tr['entropy']:.3f}) "
            f"val_loss: {va['loss']:.4f} (P:{va['policy']:.4f} V:{va['value']:.4f}) "
            f"top1: {va['top1']*100:.1f}% top3: {va['top3']*100:.1f}% "
            f"MAE: {va['value_mae']:.3f} lr: {cur_lr:.6f}",
            flush=True,
        )

        if va["loss"] < best_val - 1e-4:
            best_val = va["loss"]
            best_epoch = epoch
            patience_left = patience
            torch.save(model.state_dict(), output)
            print(f"  ✓ best saved (val_loss={va['loss']:.4f})", flush=True)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(
                    f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)",
                    flush=True,
                )
                break

    meta_path = output.with_suffix(".metadata.json")
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(output),
        "board_size": board_size,
        "channels": channels,
        "num_blocks": num_blocks,
        "dropout_p": dropout_p,
        "value_hidden": value_hidden,
        "value_weight": value_weight,
        "entropy_weight": entropy_weight,
        "weight_decay": weight_decay,
        "use_amp": can_amp,
        "use_tf32": use_tf32 and dev.type == "cuda",
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "batch_size": batch_size,
        "initial_lr": lr,
        "min_lr": min_lr,
        "validation_split": val_split,
        "augment": augment,
        "patience": patience,
        "records_used": len(full),
        "paper_alignment": {
            "loss_form": "Lπ + λv·Lv − λe·H(π)",
            "lambda_v": value_weight,
            "lambda_e": entropy_weight,
            "optimizer": "Adam",
            "lr": lr,
            "weight_decay": weight_decay,
            "scheduler": "CosineAnnealingLR",
        },
        "history": [asdict(h) for h in history],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"\nTraining complete. Best epoch: {best_epoch} (val_loss={best_val:.4f})",
        flush=True,
    )
    print(f"Saved model: {output}", flush=True)
    print(f"Saved metadata: {meta_path}", flush=True)


# ───────────────────────── CLI ─────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--board-size", type=int, required=True, choices=[5, 6, 7])
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--value-hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--value-weight", type=float, default=0.1)
    p.add_argument("--entropy-weight", type=float, default=0.02)
    p.add_argument("--val-split", type=float, default=0.05)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-tf32", action="store_true")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    train(
        npz_path=a.data, output_path=a.output, board_size=a.board_size,
        channels=a.channels, num_blocks=a.blocks, dropout_p=a.dropout,
        value_hidden=a.value_hidden, epochs=a.epochs, batch_size=a.batch_size,
        lr=a.lr, min_lr=a.min_lr, weight_decay=a.weight_decay,
        value_weight=a.value_weight, entropy_weight=a.entropy_weight,
        val_split=a.val_split, augment=not a.no_augment, patience=a.patience,
        num_workers=a.workers, use_amp=not a.no_amp, use_tf32=not a.no_tf32,
        device=a.device,
    )


if __name__ == "__main__":
    main()
