"""Convert training JSON to compact NPZ for fast Colab loading.

Usage:
    python colab/json_to_npz.py --input training_2m_5x5.json --output colab/data_5x5.npz --board-size 5
    python colab/json_to_npz.py --input training_2m_7x7.json --output colab/data_7x7.npz --board-size 7
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def convert(input_path: Path, output_path: Path, board_size: int) -> None:
    print(f"Loading {input_path} ...", flush=True)
    with input_path.open() as f:
        records = json.load(f)
    n = len(records)
    print(f"Loaded {n:,} records.", flush=True)

    bs2 = board_size * board_size
    states = np.empty((n, 2, board_size, board_size), dtype=np.float32)
    policies = np.empty((n, bs2), dtype=np.float32)
    values = np.empty((n,), dtype=np.float32)
    masks = np.empty((n, bs2), dtype=np.float32)

    for i, r in enumerate(records):
        states[i] = np.asarray(r.get("encoded_state", r.get("state")), dtype=np.float32)
        policies[i] = np.asarray(r.get("policy_target", r.get("policy")), dtype=np.float32)
        values[i] = float(r.get("value_target", r.get("value", 0.0)))
        masks[i] = np.asarray(r["legal_action_mask"], dtype=np.float32)
        if (i + 1) % 200_000 == 0:
            print(f"  packed {i+1:,}/{n:,}", flush=True)

    print(f"Saving compressed NPZ to {output_path} ...", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        states=states,
        policies=policies,
        values=values,
        masks=masks,
        board_size=np.array(board_size, dtype=np.int32),
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Done. File size: {size_mb:.1f} MB ({size_mb/1024:.2f} GB)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--board-size", type=int, required=True)
    args = parser.parse_args()
    convert(Path(args.input), Path(args.output), args.board_size)


if __name__ == "__main__":
    main()
