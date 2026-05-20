"""Detailed training analysis — convergence curves, overfitting indicators,
paper baseline comparison. Generates PNG plots for the final report.

Usage:
    python -m experiments.analyze_training --output results/training_analysis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Paper baseline (from the report PDF, §VI-A)
PAPER_BASELINE = {"top1": 78.7, "top3": 83.6, "mae": 0.229}

# Match the GUI palette so figures feel native
COLORS = {
    5: "#FFD750",   # gold
    6: "#4CC9FF",   # cyan (X color)
    7: "#FF5FA2",   # pink (O color)
}
BG = "#05081A"
PANEL = "#0D1530"
GRID = "#21345D"
TEXT = "#ECF8FF"
MUTED = "#6482AA"


def load_history(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))["history"]


def style_axes(ax):
    """Apply dark-mode theme matching the GUI."""
    ax.set_facecolor(PANEL)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["left"].set_color(GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRID, alpha=0.3, linewidth=0.5)


def plot_loss_curves(histories: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.suptitle("Loss Convergence (lower = better)",
                 color=TEXT, fontsize=14, fontweight="bold")

    for col, (key, ylabel) in enumerate([("train_loss", "Train Loss"),
                                          ("val_loss", "Validation Loss")]):
        ax = axes[col]
        style_axes(ax)
        for bs, h in histories.items():
            ep = [e["epoch"] for e in h]
            vals = [e[key] for e in h]
            ax.plot(ep, vals, marker="o", markersize=4,
                    color=COLORS[bs], linewidth=2, label=f"{bs}×{bs}")
            best_idx = vals.index(min(vals))
            ax.scatter(ep[best_idx], vals[best_idx], s=80, color=COLORS[bs],
                       edgecolors=TEXT, linewidth=1.5, zorder=5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=10)

    plt.tight_layout()
    plt.savefig(out, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def plot_train_val_gap(histories: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
    style_axes(ax)
    ax.set_title("Train–Val Gap (negative = under-fit; positive = over-fit)",
                 fontsize=12, fontweight="bold")
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--", alpha=0.7)
    for bs, h in histories.items():
        ep = [e["epoch"] for e in h]
        gaps = [e["val_loss"] - e["train_loss"] for e in h]
        ax.plot(ep, gaps, marker="o", markersize=4,
                color=COLORS[bs], linewidth=2, label=f"{bs}×{bs}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss − Train Loss")
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
    plt.tight_layout()
    plt.savefig(out, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def plot_eval_metrics(histories: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG)
    fig.suptitle("Validation Metrics vs Paper Baseline (5×5)",
                 color=TEXT, fontsize=14, fontweight="bold")

    metrics = [
        ("val_top1", "Top-1 Policy Accuracy", PAPER_BASELINE["top1"] / 100, "higher = better"),
        ("val_top3", "Top-3 Policy Accuracy", PAPER_BASELINE["top3"] / 100, "higher = better"),
        ("val_value_mae", "Value MAE", PAPER_BASELINE["mae"], "lower = better"),
    ]

    for col, (key, title, baseline, direction) in enumerate(metrics):
        ax = axes[col]
        style_axes(ax)
        ax.set_title(f"{title}  ({direction})", fontsize=11)
        for bs, h in histories.items():
            ep = [e["epoch"] for e in h]
            vals = [e[key] for e in h]
            ax.plot(ep, vals, marker="o", markersize=4,
                    color=COLORS[bs], linewidth=2, label=f"{bs}×{bs}")
        ax.axhline(baseline, color="#FFD750", linewidth=1.5, linestyle="--",
                   alpha=0.7, label="Paper baseline")
        ax.set_xlabel("Epoch")
        if "Accuracy" in title:
            ax.set_ylabel("Accuracy")
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%")
            )
        else:
            ax.set_ylabel("MAE")
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    plt.tight_layout()
    plt.savefig(out, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def plot_policy_value_split(histories: dict, out: Path) -> None:
    """Decompose the loss into policy and value components per board."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG)
    fig.suptitle("Loss Decomposition  ·  Policy / Value / Entropy",
                 color=TEXT, fontsize=14, fontweight="bold")

    for col, bs in enumerate([5, 6, 7]):
        if bs not in histories:
            continue
        ax = axes[col]
        style_axes(ax)
        ax.set_title(f"{bs}×{bs}", fontsize=12, fontweight="bold",
                     color=COLORS[bs])
        h = histories[bs]
        ep = [e["epoch"] for e in h]
        ax.plot(ep, [e["train_policy"] for e in h],
                color="#4CC9FF", marker="o", markersize=3, label="train policy Lπ")
        ax.plot(ep, [e["val_policy"] for e in h],
                color="#4CC9FF", linestyle="--", linewidth=1.2, label="val policy Lπ")
        ax.plot(ep, [e["train_value"] for e in h],
                color="#FF5FA2", marker="o", markersize=3, label="train value Lv")
        ax.plot(ep, [e["val_value"] for e in h],
                color="#FF5FA2", linestyle="--", linewidth=1.2, label="val value Lv")
        ax.plot(ep, [e["train_entropy"] for e in h],
                color="#9B6DFF", marker="s", markersize=3, label="train entropy H(π)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss component")
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def plot_summary_bar(histories: dict, out: Path) -> None:
    """A single comparison chart against the paper baseline."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
    fig.suptitle("Peak Validation vs Paper Baseline",
                 color=TEXT, fontsize=14, fontweight="bold")

    metrics = [
        ("val_top1", "Top-1 Accuracy (%)", lambda v: v * 100, PAPER_BASELINE["top1"], "max"),
        ("val_top3", "Top-3 Accuracy (%)", lambda v: v * 100, PAPER_BASELINE["top3"], "max"),
        ("val_value_mae", "Value MAE", lambda v: v, PAPER_BASELINE["mae"], "min"),
    ]

    for col, (key, label, fmt, baseline, mode) in enumerate(metrics):
        ax = axes[col]
        style_axes(ax)
        ax.set_title(label, fontsize=11)
        boards = [f"{bs}×{bs}" for bs in histories.keys()]
        peak = []
        for bs, h in histories.items():
            vals = [e[key] for e in h]
            peak.append(fmt(min(vals) if mode == "min" else max(vals)))
        bars = ax.bar(boards, peak, color=[COLORS[bs] for bs in histories.keys()],
                      edgecolor=TEXT, linewidth=1)
        ax.axhline(baseline, color="#FFD750", linewidth=1.5,
                   linestyle="--", alpha=0.7, label=f"Paper {baseline}")
        for bar, val in zip(bars, peak):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(peak) * 0.02),
                    f"{val:.2f}" if mode == "min" else f"{val:.1f}",
                    ha="center", color=TEXT, fontsize=10, fontweight="bold")
        ax.set_ylabel(label)
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    plt.tight_layout()
    plt.savefig(out, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ───────────────────────── CLI ─────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata-dir", default="src")
    p.add_argument("--output-dir", default="results/training_analysis")
    a = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    meta_dir = repo_root / a.metadata_dir
    out_dir = repo_root / a.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    for bs in (5, 6, 7):
        path = meta_dir / f"model_hard_{bs}x{bs}.metadata.json"
        if path.exists():
            histories[bs] = load_history(path)
        else:
            print(f"  ⚠ {path} missing — skip")
    if not histories:
        print("No training metadata found.")
        return

    print(f"Loaded {len(histories)} board histories")
    plot_loss_curves(histories, out_dir / "01_loss_curves.png")
    plot_train_val_gap(histories, out_dir / "02_train_val_gap.png")
    plot_eval_metrics(histories, out_dir / "03_eval_metrics.png")
    plot_policy_value_split(histories, out_dir / "04_loss_decomposition.png")
    plot_summary_bar(histories, out_dir / "05_peak_vs_baseline.png")


if __name__ == "__main__":
    main()
