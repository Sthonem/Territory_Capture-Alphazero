# Territory Capture with AlphaZero

This project implements the first phase of an AI course project: a small two-player board game environment that is designed to be extended later with an AlphaZero-style agent.

## Game Overview

Territory Capture is a turn-based strategy game played on a 5x5 board by two players, `X` and `O`.

- Players take turns placing stones on empty cells.
- Each player places exactly 8 stones.
- The game always lasts 16 total moves.
- After all stones are placed, the remaining 9 empty cells are scored as territory.

## Rules

1. The board size is 5x5.
2. Player `X` moves first.
3. Players alternate turns.
4. A move consists of placing one stone on an empty cell.
5. Each player may place at most 8 stones.
6. The game ends after both players have placed 8 stones.
7. Territory is evaluated only after the game ends.
8. For each empty cell, inspect all 8 neighboring positions around it, including diagonals.
9. Count only neighboring stones that were actually placed on the board.
10. If neighboring `X` stones are greater than neighboring `O` stones, that empty cell is territory for `X`.
11. If neighboring `O` stones are greater than neighboring `X` stones, that empty cell is territory for `O`.
12. If the counts are equal, the empty cell is neutral.
13. Territory evaluation is simultaneous: empty cells never influence the score of other empty cells.
14. Final score is based only on territory cells, not on stones placed.
15. The player with the higher territory score wins. Equal scores produce a draw.

## Project Structure

```text
territory-capture/
  README.md
  requirements.txt
  src/
    __init__.py
    game.py
    rules.py
    utils.py
    demo.py
    gui.py
    agents.py
    simulate.py
    encoding.py
    dataset.py
    generate_dataset.py
    ai_eval.py
    self_play.py
    train.py
    arena.py
    alpha_zero_loop.py
  tests/
    test_agents.py
    test_encoding.py
    test_dataset.py
    test_ai_eval.py
    test_self_play.py
    test_mcts.py
    test_train.py
    test_arena.py
    test_alpha_zero_loop.py
    test_game.py
    test_rules.py
```

## Running the Demo

From the project root:

```bash
python -m src.demo
```

This runs one or more random games, prints board states, and shows the final territory scores.

## Playing the Game Yourself

From the project root:

```bash
python -m src.gui
```

This opens a local desktop window where two human players can take turns by clicking on cells.

## Running the Tests

From the project root:

```bash
python -m unittest discover -s tests -v
```

## Running Agent Simulations

From the project root:

```bash
python -m src.simulate --x-agent random --o-agent random --games 100
python -m src.simulate --x-agent heuristic --o-agent random --games 100
python -m src.simulate --x-agent heuristic --o-agent heuristic --games 100
```

Available baseline agents:

- `random`
- `heuristic`
- `minimax`

Heuristic agent idea:

- prefers central cells
- prefers cells with more nearby empty space
- prefers moves near friendly stones
- prefers moves that contest nearby opponent stones

## Representation Layer

Before AlphaZero training, the project now includes a fixed representation layer:

- state encoding shape: `(2, 5, 5)`
- channel 1: current player stones
- channel 2: opponent stones
- fixed action space: `25` actions, one per board cell
- legal action mask: length `25`, with `1` for legal moves and `0` for illegal moves

This layer is intended for later use by:

- policy/value neural networks
- MCTS
- self-play data generation

## Self-Play Dataset Pipeline

The project also includes a simple self-play dataset layer for future training.

Each training sample currently contains:

- encoded state
- one-hot policy target for the selected move
- value target from the final game outcome
- legal action mask

For now, the policy target is based only on the action chosen by a baseline agent.
Later, this placeholder can be replaced by an MCTS visit-count distribution.

## Generating Dataset Files

To generate a larger dataset for later Colab training:

```bash
python -m src.generate_dataset --games 2000 --x-agent heuristic --o-agent random --output samples_2000.json
```

Recommended higher-quality examples:

```bash
python -m src.generate_dataset --games 2000 --x-agent heuristic --o-agent heuristic --output hh_samples_2000.json
python -m src.generate_dataset --games 2000 --x-agent minimax --o-agent heuristic --workers 4 --output mh_samples_2000.json
```

Precise dataset split examples:

```bash
python -m src.generate_dataset --games 40000 --x-agent minimax --o-agent heuristic --workers 4 --output mh_40k.json
python -m src.generate_dataset --games 30000 --x-agent heuristic --o-agent heuristic --workers 4 --output hh_30k.json
python -m src.generate_dataset --games 20000 --x-agent heuristic --o-agent random --workers 4 --output hr_20k.json
python -m src.generate_dataset --games 10000 --x-agent random --o-agent random --workers 4 --output rr_10k.json
```

You can change:

- `--games` to control dataset size
- `--x-agent` and `--o-agent` to choose `random`, `heuristic`, or `minimax`
- `--workers` to enable multiprocessing
- `--output` to choose the base JSON filename

Dataset generation now always uses the consistent `(2, 5, 5)` encoding.

The script automatically appends a timestamp to the output file, for example:

- `mh_40k_20260324_153000.json`
- `hh_30k_20260324_153500.json`

## Training Loop

The project now includes a simplified AlphaZero-style loop:

1. generate self-play data
2. train the policy-value network
3. evaluate the new checkpoint against the current best model
4. promote the new model if it wins often enough

Useful commands:

```bash
python -m src.train --data self_play_data.json --output src/latest_model.pth --epochs 5
python -m src.arena --candidate src/latest_model.pth --incumbent src/model.pth --games 20 --promote
python -m src.alpha_zero_loop --self-play-games 20 --arena-games 20 --epochs 5
```

The training pipeline now also includes:

- per-epoch loss history
- a learning-rate scheduler
- JSON metadata next to checkpoints and arena runs
- replay-buffer-backed iterative training
- configurable self-play temperature and root-noise settings

Examples:

```bash
python -m src.train --data self_play_data.json --output src/latest_model.pth --epochs 5 --scheduler-step 3 --scheduler-gamma 0.5
python -m src.arena --candidate src/latest_model.pth --incumbent src/model.pth --games 20 --promote --metadata results/arena_run.json
python -m src.alpha_zero_loop --self-play-games 20 --arena-games 20 --epochs 5 --buffer-size 10000 --temperature-moves 6 --opening-temperature 1.0 --late-temperature 0.0
```

## Extension Plan

This version includes only the game environment and tests. A later phase can build AlphaZero components on top of this codebase, such as:

- state encoding for neural network input
- action masking for legal moves
- self-play data generation
- Monte Carlo Tree Search (MCTS)
- a policy/value neural network
