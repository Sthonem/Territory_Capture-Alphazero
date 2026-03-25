🧠 Territory Capture with AlphaZero

Territory Capture is a custom 5×5 strategy board game developed as part of an AI course project.
The project has evolved into a complete AlphaZero-style learning system built around an original game environment.

It includes:
	•	🎮 fully implemented game engine
	•	🖥️ desktop GUI
	•	🤖 baseline agents (random, heuristic, minimax)
	•	🧠 policy-value neural network (ResNet-based)
	•	🌳 MCTS (PUCT-based) search
	•	🔁 self-play data generation
	•	🗂️ replay buffer training
	•	⚔️ arena evaluation & model promotion

🔥 The project represents a simplified AlphaZero-style pipeline applied to a custom-designed strategy game.

⸻

🎯 Game Overview

Territory Capture is played on a 5×5 grid between two players: X and O.
	•	Players alternate turns.
	•	Each move places one stone.
	•	Each player places exactly 8 stones.
	•	The game lasts 16 total moves.
	•	Remaining 9 cells are empty.
	•	The winner is determined by territory control, not stone count.

⸻

📜 Rules
	1.	Board size is 5×5
	2.	Player X starts
	3.	Players alternate turns
	4.	One stone per move
	5.	Max 8 stones per player
	6.	Game ends after 16 moves
	7.	Territory evaluated only at the end
	8.	Each empty cell checks 8 neighbors (including diagonals)
	9.	Only stones count in evaluation
	10.	More neighboring X → belongs to X
	11.	More neighboring O → belongs to O
	12.	Equal → neutral
	13.	Evaluation is simultaneous
	14.	Final score = number of controlled cells
	15.	Higher score wins
	16.	Equal → draw

⸻

⚡ Capture Rule

After each move:
	•	Stones with no empty neighbors are removed
	•	Removals happen simultaneously

⸻

🗂️ Project Structure

territory-capture/
  README.md
  requirements.txt
  reports/
  results/
  src/
    game.py
    gui.py
    agents.py
    model.py
    mcts.py
    ai_agent.py
    ai_eval.py
    self_play.py
    train.py
    arena.py
    alpha_zero_loop.py
    model.pth
  tests/


⸻

▶️ Running the GUI

python -m src.gui

Modes
	•	Human vs Human
	•	Human vs AI
	•	AI vs AI

Difficulty Levels
	•	🟢 Easy → Random
	•	🟡 Medium → NN + shallow MCTS
	•	🔴 Hard → NN + deeper MCTS

⸻

🎮 Running Demo

python -m src.demo


⸻

🧪 Running Tests

python -m unittest discover -s tests -v


⸻

🤖 Baseline Agent Simulation

python -m src.simulate --x-agent random --o-agent random --games 100
python -m src.simulate --x-agent heuristic --o-agent random --games 100
python -m src.simulate --x-agent minimax --o-agent heuristic --games 100


⸻

🧠 Representation
	•	State shape: (2, 5, 5)
	•	Channel 0 → current player
	•	Channel 1 → opponent
	•	Action space → 25
	•	Legal move mask → length 25

Used across:
	•	Neural network
	•	MCTS
	•	Self-play
	•	Dataset pipeline

⸻

📊 Dataset Pipeline

Each sample includes:
	•	state
	•	policy target
	•	value target
	•	legal mask

Generate Dataset

python -m src.generate_dataset --games 2000 --x-agent heuristic --o-agent random


⸻

🧠 Neural Network + MCTS

Includes:
	•	ResNet policy-value network
	•	PUCT-based MCTS
	•	Root reuse
	•	Visit-count policy extraction
	•	Dirichlet noise (exploration)
	•	Temperature-based action selection

⸻

🔁 Self-Play & Training

Pipeline:
	1.	Self-play games
	2.	Store in replay buffer
	3.	Train network
	4.	Arena evaluation
	5.	Promote better model

⸻

▶️ Self-Play

python - <<'PY'
from src.self_play import generate_self_play_games
generate_self_play_games(num_games=20)
PY


⸻

🏋️ Training

python -m src.train --data self_play_data.json --output src/latest_model.pth --epochs 5


⸻

⚔️ Arena Evaluation

python -m src.arena --candidate src/latest_model.pth --incumbent src/model.pth --games 20 --promote


⸻

🔥 Full AlphaZero Loop

python -m src.alpha_zero_loop --self-play-games 20 --arena-games 20 --epochs 5


⸻

📈 Training Features
	•	Replay buffer
	•	Loss tracking
	•	Learning rate scheduling
	•	Self-play temperature control
	•	Metadata logging
	•	Model promotion system

⸻

📊 Evaluation Tools
	•	AI vs AI
	•	AI vs Minimax
	•	Win-rate tracking
	•	Plot export

python - <<'PY'
from src.ai_eval import run_ai_vs_ai
run_ai_vs_ai(num_games=50)
PY


⸻

📄 Reports

Located in /reports:
	•	Game mechanics
	•	Implementation details
	•	Final AI system

⸻

☁️ Colab Training Setup & Dataset Scale


Model training was performed in a dedicated Google Colab environment using GPU acceleration (NVIDIA A100).

Dataset Composition

The training dataset was generated using multiple agent configurations to ensure both strategic depth and robustness:
	•	Minimax vs Heuristic: 40,000 games
	•	Heuristic vs Heuristic: 30,000 games
	•	Heuristic vs Random: 20,000 games
	•	Random vs Random: 10,000 games

Total games: 100,000

From these games, a total of:
	•	1,600,000 training samples were generated

Each sample contains:
	•	state: (2, 5, 5) tensor
	•	policy target: (25)
	•	value target: (1)
	•	legal action mask: (25)

Training Configuration
	•	GPU: NVIDIA A100 (Colab)
	•	Batch size: 1024
	•	Epochs: 25
	•	Optimizer: Adam
	•	Learning rate: 5e-4
	•	Weight decay: 5e-4
	•	Scheduler: Cosine Annealing
	•	Mixed precision: enabled (torch.amp)

Loss Function

The model is trained using a combined objective:
	•	Policy loss: Cross-entropy
	•	Value loss: Mean squared error
	•	Entropy regularization: to encourage exploration

Final loss:

Loss = Policy + 0.1 × Value − 0.02 × Entropy

Training Performance

Training was completed in approximately:
	•	~450 seconds total on A100 GPU

Final validation metrics:
	•	Top-1 Accuracy: ~78.7%
	•	Top-3 Accuracy: ~83.6%
	•	MAE (Value Head): ~0.23
	•	Validation Loss: ~0.669

Evaluation

The trained model was evaluated using MCTS-based gameplay:
	•	AI vs Random:
	•	Win Rate: 100% (30/30 games)

Note:
The random agent is a weak baseline. Future evaluation includes stronger opponents such as minimax and self-play agents.

⸻

Why This Matters
	•	Large-scale dataset (1.6M samples) ensures stable learning
	•	Mixed data sources (minimax + random) improve generalization
	•	GPU-optimized training enables fast experimentation
	•	Results demonstrate strong policy learning and consistent gameplay performance

⸻

🚀 Project Status

This project has evolved into:

💥 A working simplified AlphaZero-style system for a custom board game

Includes:
	•	game environment
	•	neural network
	•	MCTS
	•	self-play
	•	training loop
	•	evaluation
	•	GUI integration

⸻

🎓 Academic Positioning

A simplified AlphaZero-style implementation for an original 5×5 strategy game.

✔ Not a full reproduction of AlphaZero
✔ But a complete and functional learning pipeline

⸻

🔥 Final Note

This project demonstrates:
	•	reinforcement learning concepts
	•	search + neural network integration
	•	self-improving AI systems
	•	end-to-end pipeline design

⸻

⭐ If you like this project, consider giving it a star!
