Territory Capture with AlphaZero

Territory Capture is a custom-designed 5×5 strategy board game developed as part of an artificial intelligence project.
The system evolves beyond a simple game implementation into a complete learning framework inspired by the core principles of AlphaZero.

It integrates neural networks, search algorithms, and self-play to produce a fully autonomous decision-making agent in a novel game environment.

Overview

The project combines:
	•	a fully defined game environment
	•	multiple baseline agents (random, heuristic, minimax)
	•	a policy–value neural network
	•	a PUCT-based Monte Carlo Tree Search (MCTS)
	•	self-play data generation
	•	replay buffer training
	•	iterative model evaluation and promotion

The result is a unified pipeline where the agent improves through experience rather than hard-coded strategies.

Game Description

Territory Capture is played on a 5×5 board by two players (X and O).
	•	Players alternate turns
	•	Each move places a stone on an empty cell
	•	Each player places exactly 8 stones (16 total moves)
	•	The remaining cells form neutral territory candidates
	•	The winner is determined by territorial control, not piece count

Key Mechanics
	•	Territory is evaluated only at the end of the game
	•	Each empty cell considers its 8 neighboring cells
	•	Majority ownership determines territory assignment
	•	Equal influence results in neutral cells
	•	A capture rule removes stones with no adjacent empty spaces

This design creates a compact but strategically rich environment suitable for learning-based approaches.

System Architecture

The project follows a structured AlphaZero-style pipeline:
	1.	State Representation
	•	Shape: (2, 5, 5)
	•	Channel 0: current player
	•	Channel 1: opponent
	•	Action space: 25
	2.	Neural Network
	•	Residual convolutional architecture
	•	Dual-head output:
	•	policy (action probabilities)
	•	value (position evaluation)
	3.	Monte Carlo Tree Search (MCTS)
	•	PUCT selection strategy
	•	neural prior integration
	•	value-based backpropagation
	•	root reuse between moves
	•	visit-count based policy extraction
	4.	Self-Play Learning
	•	games generated using current policy
	•	exploration via temperature and Dirichlet noise
	•	training targets:
	•	policy (visit distribution)
	•	value (game outcome)
	5.	Training Loop
	•	replay buffer sampling
	•	supervised updates from self-play data
	•	periodic evaluation against previous model
	•	model promotion based on win-rate

Running the Project

GUI

python -m src.gui

Modes available:
	•	Human vs Human
	•	Human vs AI
	•	AI vs AI

Demo

python -m src.demo

Tests

python -m unittest discover -s tests -v

Baseline Agents

The project includes multiple baseline strategies:
	•	random
	•	heuristic
	•	minimax

Example:

python -m src.simulate --x-agent minimax --o-agent heuristic --games 100

These baselines serve both as training data sources and evaluation benchmarks.

Dataset Pipeline

Training samples consist of:
	•	encoded state
	•	policy target
	•	value target
	•	legal action mask

Large-scale datasets can be generated using parallel simulations.

Training Workflow

Self-Play

python - <<'PY'
from src.self_play import generate_self_play_games
generate_self_play_games(num_games=20)
PY

Model Training

python -m src.train --data self_play_data.json --output src/latest_model.pth --epochs 5

Arena Evaluation

python -m src.arena --candidate src/latest_model.pth --incumbent src/model.pth --games 20 --promote

Full Iterative Loop

python -m src.alpha_zero_loop --self-play-games 20 --arena-games 20 --epochs 5

Evaluation

The system supports:
	•	AI vs AI benchmarking
	•	AI vs minimax comparison
	•	win-rate tracking
	•	automatic result plotting

These tools allow continuous monitoring of model improvement across training iterations.

Project Status

This project represents a complete and functional learning system that demonstrates:
	•	integration of neural networks with search algorithms
	•	policy and value learning from self-play
	•	iterative improvement without human-labeled data
	•	application of AlphaZero principles to a custom game

Academic Context

The project is best described as:

A simplified AlphaZero-style implementation applied to an original 5×5 strategy game.

While not a full-scale reproduction of AlphaZero, it captures the essential ideas:
	•	search-guided policy improvement
	•	value-based evaluation
	•	self-play driven learning
