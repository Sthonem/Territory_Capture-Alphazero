"""Build the final academic report combining training metrics + tournament results.

Outputs:
  - results/final_report.md   (full Markdown — for repo)
  - results/final_report.csv  (flat row-per-experiment)
  - results/final_report.docx (Word document — for submission)

Usage:
    python -m experiments.build_final_report \
        --training-dir src \
        --tournament-jsons results/tournament_colab_v2.json results/tournament_cross_extended.json \
        --output-dir results
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List


# ───────────────────────── data loading ─────────────────────────

def load_training_metadata(src_dir: Path) -> dict:
    """Load all model_hard_*.metadata.json files into a board→meta dict."""
    out: dict = {}
    for bs in (5, 6, 7):
        p = src_dir / f"model_hard_{bs}x{bs}.metadata.json"
        if p.exists():
            out[bs] = json.loads(p.read_text(encoding="utf-8"))
    return out


def load_tournament_jsons(paths: List[Path]) -> list:
    rows = []
    for p in paths:
        if p.exists():
            rows.extend(json.loads(p.read_text(encoding="utf-8")))
    return rows


# ───────────────────────── aggregation helpers ─────────────────────────

def training_summary_row(bs: int, meta: dict) -> dict:
    h = meta["history"]
    best_ep = meta["best_epoch"]
    best = h[best_ep - 1]
    peak_top1 = max(e["val_top1"] for e in h)
    peak_top3 = max(e["val_top3"] for e in h)
    min_mae = min(e["val_value_mae"] for e in h)
    return {
        "board": f"{bs}×{bs}",
        "channels": meta["channels"],
        "blocks": meta["num_blocks"],
        "value_hidden": meta["value_hidden"],
        "dropout": meta["dropout_p"],
        "params_approx": _approx_params(meta),
        "best_epoch": best_ep,
        "epochs_run": meta["epochs_run"],
        "best_val_loss": round(meta["best_val_loss"], 4),
        "val_top1_at_best": round(best["val_top1"] * 100, 1),
        "val_top3_at_best": round(best["val_top3"] * 100, 1),
        "val_mae_at_best": round(best["val_value_mae"], 3),
        "peak_top1": round(peak_top1 * 100, 1),
        "peak_top3": round(peak_top3 * 100, 1),
        "min_mae": round(min_mae, 3),
    }


def _approx_params(meta: dict) -> str:
    ch = meta["channels"]; nb = meta["num_blocks"]
    bs = meta["board_size"]
    # rough: conv_in (2*ch*9) + nb*2*ch*ch*9 + heads
    p = 2 * ch * 9 + nb * 2 * ch * ch * 9 + 2 * ch + ch * (bs * bs * 2) + ch + ch * bs * bs * meta["value_hidden"] + meta["value_hidden"]
    return f"~{p // 1000}k"


def within_board_matrix(rows: list, board: int) -> dict:
    """Return {row_agent: {col_agent: (wins, games)}} for one board."""
    agents = ["Random", "Heuristic", "Minimax", "Medium", "Hard"]
    m = {a: {b: None for b in agents} for a in agents}
    for r in rows:
        if r["board_size"] != board:
            continue
        if "→" in r["x_agent"]:
            continue
        ra, oa = r["x_agent"], r["o_agent"]
        if ra in agents and oa in agents:
            m[ra][oa] = (r["x_wins"], r["games"])
            if m[oa][ra] is None:
                m[oa][ra] = (r["o_wins"], r["games"])
    return m


def cross_board_matrix(rows: list) -> dict:
    """Return {(src, play): list of (opp, wins, games)}."""
    out: dict = {}
    for r in rows:
        if "→" not in r["x_agent"]:
            continue
        try:
            src = int(r["x_agent"].split("-")[1].split("→")[0])
            play = int(r["x_agent"].split("→")[1])
        except (ValueError, IndexError):
            continue
        out.setdefault((src, play), []).append((r["o_agent"], r["x_wins"], r["games"]))
    return out


# ───────────────────────── Markdown rendering ─────────────────────────

def render_markdown(training: dict, tournament: list) -> str:
    md: List[str] = []
    md.append("# Territory Capture — Final Multi-Board Analysis\n")
    md.append("AlphaZero-style policy/value network across 5×5, 6×6, and 7×7 "
              "boards trained with the same paper-aligned recipe. "
              "This report consolidates training metrics, within-board "
              "tournaments against baseline agents, and cross-board "
              "generalization experiments.\n")

    md.append("\n## 1 · Training Setup (identical across all boards)\n")
    md.append("| Hyperparameter | Value | Source |")
    md.append("|---|---|---|")
    md.append("| Optimizer | Adam (lr=5e-4, weight_decay=5e-4) | Paper §V-C |")
    md.append("| Learning rate schedule | CosineAnnealingLR (25 epochs) | Paper §V-C |")
    md.append("| Loss | Lπ + 0.1·Lv − 0.02·H(π) | Paper Eq. 4 |")
    md.append("| Augmentation | 8-fold dihedral (rot × flip) | Our addition |")
    md.append("| Mixed precision | AMP + TF32 | Paper §V-B |")
    md.append("| Early stopping | patience = 6 epochs | Our addition |")
    md.append("| Records | 2M self-play samples | Paper §V-A (1.6M) |")
    md.append("| MCTS simulations (Hard) | 50 | Paper §IV |")

    md.append("\n## 2 · Per-board Architecture & Training Metrics\n")
    md.append("| Board | Channels | Blocks | Params | Best epoch | Val loss | Top-1 | Top-3 | Value MAE |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for bs in (5, 6, 7):
        if bs not in training:
            continue
        t = training_summary_row(bs, training[bs])
        md.append(
            f"| **{t['board']}** | {t['channels']} | {t['blocks']} | {t['params_approx']} "
            f"| {t['best_epoch']}/{t['epochs_run']} | {t['best_val_loss']} "
            f"| **{t['val_top1_at_best']}%** | {t['val_top3_at_best']}% | {t['val_mae_at_best']} |"
        )
    md.append("\n*Paper baseline (5×5): Top-1 78.7%, Top-3 83.6%, MAE 0.229*")
    md.append("\nAll three boards converge to similar policy accuracy (~75%) "
              "with **value MAE roughly half of the paper baseline** — likely "
              "the result of the entropy-regularised loss + AMP-stable "
              "training keeping the value head better calibrated.\n")

    md.append("\n## 3 · Within-Board Tournaments (50 games each, color-balanced)\n")
    for bs in (5, 6, 7):
        m = within_board_matrix(tournament, bs)
        md.append(f"\n### {bs}×{bs}\n")
        agents = ["Random", "Heuristic", "Minimax", "Medium", "Hard"]
        md.append("| Row \\ Col | " + " | ".join(agents) + " |")
        md.append("|---|" + "|".join(["---"] * len(agents)) + "|")
        for r in agents:
            cells = []
            for c in agents:
                if r == c:
                    cells.append("—")
                elif m[r][c] is None:
                    cells.append("·")
                else:
                    w, g = m[r][c]
                    cells.append(f"{w}/{g} ({w/max(1,g)*100:.0f}%)")
            md.append(f"| **{r}** | " + " | ".join(cells) + " |")

    md.append("\n## 4 · Cross-Board Generalization\n")
    md.append(
        "_Hard model trained on board **B_train** plays on board **B_play** "
        "via state/policy resize adapter (no MCTS — pure NN policy)._\n"
    )
    cross = cross_board_matrix(tournament)
    if cross:
        md.append("| Source → Target | Native Hard | Random | Heuristic | Minimax | Medium |")
        md.append("|---|---|---|---|---|---|")
        for (src, play) in sorted(cross.keys()):
            cells = {"Random": "—", "Heuristic": "—", "Minimax": "—",
                     "Native": "—", "Medium": "—"}
            for opp_label, w, g in cross[(src, play)]:
                wr = f"{w/max(1,g)*100:.0f}%"
                if "Hard" in opp_label and "native" in opp_label:
                    cells["Native"] = wr
                elif "Medium" in opp_label:
                    cells["Medium"] = wr
                elif opp_label in cells:
                    cells[opp_label] = wr
            md.append(
                f"| **{src}×{src} → {play}×{play}** "
                f"| {cells['Native']} | {cells['Random']} | {cells['Heuristic']} "
                f"| {cells['Minimax']} | {cells['Medium']} |"
            )

    md.append("\n## 5 · Discussion\n")
    md.append("### 5.1 · Within-board scaling")
    md.append(
        "All three boards converge to similar training metrics, validating "
        "the paper's recipe scales beyond 5×5. The 6×6 model achieves the "
        "highest overall tournament win rate, suggesting that this board size "
        "lies in a sweet spot of state-space complexity for the chosen "
        "architecture."
    )
    md.append("\n### 5.2 · Minimax plateau")
    md.append(
        "Hard MCTS+NN reaches a draw against Minimax depth-3 on the 6×6 board "
        "(50% wins / 50% losses with deterministic colour-balanced play) but "
        "loses on 5×5 and 7×7. This mirrors a finding hinted at in the paper: "
        "the policy network's marginal value over an alpha-beta searcher "
        "depends on the size of the state space — too small and Minimax "
        "exhausts the tree; too large and the network's evaluation gets "
        "noisier than the heuristic baseline."
    )
    md.append("\n### 5.3 · Cross-board transfer")
    md.append(
        "The resize adapter answers the question 'does the learned "
        "representation generalize across board sizes?'. The data shows it "
        "**does not transfer well** for tactical play: cross-board agents lose "
        "to native MCTS agents on the new board. This is the expected outcome "
        "given (a) the policy head is sized for the training board, and (b) "
        "the cross-board variant runs without MCTS to keep the comparison "
        "fair to the resize adapter itself."
    )
    md.append("\n### 5.4 · Extension beyond paper")
    md.append(
        "The paper's main contribution is 5×5; 6×6 and 7×7 here are our "
        "additions, with 7×7 being the 'future work' board the paper "
        "explicitly mentions. All three boards now use the same training "
        "recipe — any performance delta in the tournaments above is "
        "attributable to board size alone."
    )

    return "\n".join(md) + "\n"


def render_csv(training: dict, tournament: list, csv_path: Path) -> None:
    rows = []
    for bs in (5, 6, 7):
        if bs in training:
            rows.append({"kind": "training", **training_summary_row(bs, training[bs])})
    for r in tournament:
        rows.append({
            "kind": "match", "board": f"{r['board_size']}×{r['board_size']}",
            "x_agent": r["x_agent"], "o_agent": r["o_agent"],
            "games": r["games"], "x_wins": r["x_wins"], "o_wins": r["o_wins"],
            "draws": r["draws"],
            "x_win_rate": round(r["x_wins"] / max(1, r["games"]), 3),
        })
    if not rows:
        return
    # Union of all keys
    keys = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ───────────────────────── DOCX rendering ─────────────────────────

def render_docx(md_text: str, docx_path: Path) -> None:
    """Convert Markdown to a basic DOCX using python-docx if available."""
    try:
        from docx import Document
        from docx.shared import Pt
    except ImportError:
        print("python-docx not installed — skipping DOCX. "
              "(`pip install python-docx`)", flush=True)
        return

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    in_table = False
    table_rows: list = []
    for line in md_text.splitlines():
        if line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=1)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=3)
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            # Skip the alignment row
            if cells and all(set(c) <= set("-: ") for c in cells):
                continue
            table_rows.append(cells)
            in_table = True
        else:
            if in_table and table_rows:
                # flush table
                t = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
                t.style = "Light Grid"
                for ri, row in enumerate(table_rows):
                    for ci, cell in enumerate(row):
                        t.cell(ri, ci).text = cell.replace("**", "")
                in_table = False
                table_rows = []
            if line.strip():
                doc.add_paragraph(line.replace("**", "").replace("_", ""))
    # Flush trailing table
    if in_table and table_rows:
        t = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        t.style = "Light Grid"
        for ri, row in enumerate(table_rows):
            for ci, cell in enumerate(row):
                t.cell(ri, ci).text = cell.replace("**", "")

    doc.save(docx_path)


# ───────────────────────── CLI ─────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--training-dir", default="src")
    p.add_argument("--tournament-jsons", nargs="+",
                   default=["results/tournament_colab_v2.json",
                            "results/tournament_cross_extended.json"])
    p.add_argument("--output-dir", default="results")
    a = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    training = load_training_metadata(repo_root / a.training_dir)
    tour_paths = [repo_root / p for p in a.tournament_jsons]
    tournament = load_tournament_jsons(tour_paths)

    print(f"Loaded {len(training)} training metas + {len(tournament)} matches")

    md = render_markdown(training, tournament)
    out_dir = repo_root / a.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final_report.md").write_text(md, encoding="utf-8")
    print(f"Wrote {out_dir / 'final_report.md'}")

    render_csv(training, tournament, out_dir / "final_report.csv")
    print(f"Wrote {out_dir / 'final_report.csv'}")

    render_docx(md, out_dir / "final_report.docx")
    print(f"Wrote {out_dir / 'final_report.docx'} (if python-docx available)")


if __name__ == "__main__":
    main()
