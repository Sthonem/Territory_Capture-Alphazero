"""Build a human-readable tournament report (Markdown + CSV) from results JSON.

Usage:
    python -m experiments.build_report --input results/tournament.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/tournament.json")
    p.add_argument("--md-output", default="results/tournament_report.md")
    p.add_argument("--csv-output", default="results/tournament_results.csv")
    a = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data = json.loads((repo_root / a.input).read_text(encoding="utf-8"))

    # CSV
    csv_path = repo_root / a.csv_output
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if data:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            for row in data:
                w.writerow(row)

    # Markdown
    md_lines: list[str] = ["# Territory Capture — Tournament Report\n"]

    within = [r for r in data if not r["x_agent"].startswith("Hard-") or "→" not in r["x_agent"]]
    cross = [r for r in data if "→" in r["x_agent"]]

    if within:
        md_lines.append("## Within-Board Results\n")
        md_lines.append("Match-up | Board | X agent | O agent | Games | X wins | O wins | Draws | X win% | O win% | Avg X score | Avg O score | Time")
        md_lines.append("---|---|---|---|---|---|---|---|---|---|---|---|---")
        for r in within:
            md_lines.append(
                f"{r['x_agent']} vs {r['o_agent']} | {r['board_size']}×{r['board_size']} "
                f"| {r['x_agent']} | {r['o_agent']} | {r['games']} "
                f"| {r['x_wins']} | {r['o_wins']} | {r['draws']} "
                f"| {r['x_wins']/max(1,r['games'])*100:.0f}% | {r['o_wins']/max(1,r['games'])*100:.0f}% "
                f"| {r['avg_x_score']:.1f} | {r['avg_o_score']:.1f} | {r['seconds']:.0f}s"
            )

        # Win rate summary table (Medium vs Hard each board)
        md_lines.append("\n### Medium vs Hard — Within-Board Summary\n")
        md_lines.append("Board | Hard wins | Medium wins | Draws | Hard win%")
        md_lines.append("---|---|---|---|---")
        for r in within:
            if {r["x_agent"], r["o_agent"]} == {"Hard", "Medium"}:
                hard_wins = r["x_wins"] if r["x_agent"] == "Hard" else r["o_wins"]
                med_wins = r["o_wins"] if r["x_agent"] == "Hard" else r["x_wins"]
                hard_rate = hard_wins / max(1, r["games"]) * 100
                md_lines.append(
                    f"{r['board_size']}×{r['board_size']} | {hard_wins} | {med_wins} "
                    f"| {r['draws']} | {hard_rate:.0f}%"
                )

    if cross:
        md_lines.append("\n## Cross-Board Generalization\n")
        md_lines.append(
            "_How well does a Hard model trained on one board size play on a different board "
            "size (via state/policy resize adapter)? The cross-board model has **no MCTS** "
            "(pure NN), while the native opponent uses standard MCTS._\n"
        )
        md_lines.append("Cross agent | Play board | vs Native | Cross wins | Native wins | Draws | Cross win%")
        md_lines.append("---|---|---|---|---|---|---")
        for r in cross:
            cross_label = r["x_agent"]
            md_lines.append(
                f"{cross_label} | {r['board_size']}×{r['board_size']} | {r['o_agent']} "
                f"| {r['x_wins']} | {r['o_wins']} | {r['draws']} "
                f"| {r['x_wins']/max(1,r['games'])*100:.0f}%"
            )

    md_path = repo_root / a.md_output
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote Markdown report: {md_path}")
    print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
