# Territory Capture with AlphaZero

Territory Capture is a custom 6x6 strategy board game built for an AI course project.  
The project now includes:

- a complete game engine
- a desktop GUI
- baseline agents (`random`, `heuristic`, `minimax`)
- a policy-value neural network
- MCTS-based AI play
- self-play data generation
- replay-buffer-based training
- arena evaluation and iterative model promotion

The current state of the project is best described as a **simplified AlphaZero-style pipeline for an original strategy game**.

## Game Overview

Territory Capture is played on a `6x6` board by two players: `X` and `O`.

- Players alternate turns.
- A move consists of placing one stone on an empty cell.
- Each player places exactly `10` stones.
- The game lasts `20` total moves.
- `16` cells remain empty at the end.
- Final scoring is based on **territory plus capture bonus**.

## Rules

1. The board size is `6x6`.
2. Player `X` moves first.
3. Players alternate turns.
4. A move places one stone on an empty cell.
5. Each player may place at most `10` stones.
6. The game ends after both players have placed all `10` stones.
7. Territory is evaluated only after the game ends.
8. For each empty cell, check all `8` neighbors, including diagonals.
9. Only placed stones count during territory evaluation.
10. If neighboring `X` stones are greater than neighboring `O` stones, the cell belongs to `X`.
11. If neighboring `O` stones are greater than neighboring `X` stones, the cell belongs to `O`.
12. If the counts are equal, the cell is neutral.
13. Territory evaluation is simultaneous: empty cells do not affect one another.
14. Each captured opponent stone is worth `+1` bonus point.
15. Final score is `territory + capture bonus`.
16. If total scores tie, only the four empty center cells that receive territory assignments are used as a tie-break.
17. If the center tie-break also ties, the result is a true draw.

### Capture Rule

The current version uses a local-pressure capture rule:

- After each move, all stones on the board are checked.
- A stone is removed only when **both** conditions hold:
  - it has at least `2` neighboring opponent stones
  - opponent neighboring stones outnumber friendly neighboring stones
- Empty neighbors do not count in this comparison.
- Removals happen **simultaneously**, not sequentially.

## Project Structure

```text
territory-capture/
  README.md
  requirements.txt
  reports/
    build_reports.py
    report_1_game_mechanics.docx
    report_2_implementation_and_ai_foundation.docx
    report_3_final_ai_system.docx
  results/
    winrate.png
    arena_winrate.png
    alpha_zero_iteration.json
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
    model.py
    mcts.py
    ai_agent.py
    ai_eval.py
    self_play.py
    replay_buffer.py
    train.py
    arena.py
    alpha_zero_loop.py
    model.pth
  tests/
    test_agents.py
    test_ai_eval.py
    test_alpha_zero_loop.py
    test_arena.py
    test_dataset.py
    test_encoding.py
    test_game.py
    test_generate_dataset.py
    test_mcts.py
    test_replay_buffer.py
    test_rules.py
    test_self_play.py
    test_train.py
```

## Running the GUI

From the project root:

```bash
python -m src.gui
```

The GUI supports:

- `Human vs Human`
- `Human vs AI`
- `AI vs AI`

### AI Difficulty Levels

The GUI includes three difficulty presets:

- `Easy`: random play
- `Medium`: minimax search (depth 2)
- `Hard`: MCTS + trained policy-value network (100k training samples)

In `AI vs AI` mode, `X` and `O` can use different difficulty levels.

## Running the Demo

```bash
python -m src.demo
```

This runs random sample games in the terminal and prints final boards and scores.

## Running the Tests

```bash
python -m unittest discover -s tests -v
```

## Baseline Agent Simulations

```bash
python -m src.simulate --x-agent random --o-agent random --games 100
python -m src.simulate --x-agent heuristic --o-agent random --games 100
python -m src.simulate --x-agent heuristic --o-agent heuristic --games 100
python -m src.simulate --x-agent minimax --o-agent heuristic --games 100
```

Available baseline agents:

- `random`
- `heuristic`
- `minimax`

## Representation Layer

The neural-network representation layer is fixed and consistent:

- state encoding shape: `(2, 6, 6)`
- channel `0`: current player stones
- channel `1`: opponent stones
- fixed action space: `36`
- legal action mask: length `36`

This representation is used by:

- policy/value neural networks
- MCTS
- self-play
- dataset generation

## Dataset Pipeline

The project includes a dataset/export pipeline for large-scale game generation.

Each recorded training sample can include:

- encoded state
- policy target
- value target
- legal action mask

### Generate Datasets

```bash
python -m src.generate_dataset --games 2000 --x-agent heuristic --o-agent random --output samples_2000.json
python -m src.generate_dataset --games 2000 --x-agent heuristic --o-agent heuristic --output hh_samples_2000.json
python -m src.generate_dataset --games 2000 --x-agent minimax --o-agent heuristic --workers 4 --output mh_samples_2000.json
```

### Exact Distribution Examples

```bash
python -m src.generate_dataset --games 40000 --x-agent minimax --o-agent heuristic --workers 4 --output mh_40k.json
python -m src.generate_dataset --games 30000 --x-agent heuristic --o-agent heuristic --workers 4 --output hh_30k.json
python -m src.generate_dataset --games 20000 --x-agent heuristic --o-agent random --workers 4 --output hr_20k.json
python -m src.generate_dataset --games 10000 --x-agent random --o-agent random --workers 4 --output rr_10k.json
```

Notes:

- dataset generation always uses the consistent `(2, 6, 6)` encoding
- output filenames automatically receive timestamps
- multiprocessing is supported through `--workers`

## Neural Network and MCTS

The project now includes:

- a residual policy-value network in [model.py](/Users/erdem/Downloads/Ai%20Project/territory-capture/src/model.py)
- a PUCT-style MCTS implementation in [mcts.py](/Users/erdem/Downloads/Ai%20Project/territory-capture/src/mcts.py)
- root reuse
- visit-count policy extraction
- Dirichlet root noise for self-play exploration
- temperature-based self-play action selection

## Self-Play and Training

The project includes a simplified AlphaZero-style training loop:

1. generate self-play games
2. store records in a replay buffer
3. train the policy-value network
4. compare the candidate model against the current best checkpoint
5. promote the candidate if it performs well enough

### Self-Play

```bash
python - <<'PY'
from src.self_play import generate_self_play_games
generate_self_play_games(num_games=20)
PY
```

### Train a Model

```bash
python -m src.train --data self_play_data.json --output src/latest_model.pth --epochs 5
```

### Arena Evaluation

```bash
python -m src.arena --candidate src/latest_model.pth --incumbent src/model.pth --games 20 --promote
```

Note:

- old `5x5` checkpoints are intentionally incompatible with the current `6x6 / 36-action` game
- the current model was trained on 100k samples (40% minimax-heuristic, 30% heuristic-heuristic, 20% heuristic-random, 10% random-random)

### Full AlphaZero-Style Iteration

```bash
python -m src.alpha_zero_loop --self-play-games 20 --arena-games 20 --epochs 5
```

## Training Improvements Already Included

The current training pipeline includes:

- per-epoch loss history
- replay-buffer-backed training
- configurable self-play temperature
- configurable root-noise exploration
- learning-rate scheduling
- JSON metadata files next to checkpoints and arena runs
- iterative model evaluation and promotion

Example commands:

```bash
python -m src.train --data self_play_data.json --output src/latest_model.pth --epochs 5 --scheduler-step 3 --scheduler-gamma 0.5
python -m src.arena --candidate src/latest_model.pth --incumbent src/model.pth --games 20 --promote --metadata results/arena_run.json
python -m src.alpha_zero_loop --self-play-games 20 --arena-games 20 --epochs 5 --buffer-size 10000 --temperature-moves 6 --opening-temperature 1.0 --late-temperature 0.0
```

## Evaluation Tools

The project includes evaluation scripts for:

- `AI vs AI`
- `AI vs Minimax`
- running win-rate tracking
- plot export to `results/`

Example:

```bash
python - <<'PY'
from src.ai_eval import run_ai_vs_ai
run_ai_vs_ai(num_games=50)
PY
```

## Reports

The repository also contains course-report files in Word format:

- `report_1_game_mechanics.docx`
- `report_2_implementation_and_ai_foundation.docx`
- `report_3_final_ai_system.docx`

These are stored in the `reports/` directory.

## Colab / Notebook Training

A companion notebook version of the training workflow is also available in the workspace:

- [/Users/erdem/Downloads/Ai Project/Territory_Capture-Alphazero/aiModel.ipynb](/Users/erdem/Downloads/Ai%20Project/Territory_Capture-Alphazero/aiModel.ipynb)

This notebook can be used for:

- larger-scale dataset handling
- Colab or Drive-based training
- exporting trained checkpoints such as `model.pth`

## Project Status

This repository is no longer only the “first phase” of the project.  
It now includes a working **simplified AlphaZero-style pipeline** for a custom strategy game:

- original game environment
- encoded state/action representation
- policy-value network
- MCTS
- self-play
- replay buffer
- training
- arena evaluation
- model promotion
- GUI integration

## Academic Positioning

The most accurate way to describe the project is:

> A simplified AlphaZero-style implementation for an original 6x6 strategy game.

This is stronger and more accurate than calling it only a basic game environment, while still being academically honest about not being a full paper-level reproduction of AlphaZero or AlphaStar.
