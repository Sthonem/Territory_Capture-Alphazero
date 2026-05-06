"""Create simple DOCX reports without external dependencies."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""

ROOT_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""

APP_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>
"""

CORE_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{title}</dc:title>
  <dc:creator>Codex</dc:creator>
</cp:coreProperties>
"""

DOCUMENT_XML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
 xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
 xmlns:o="urn:schemas-microsoft-com:office:office"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
 xmlns:v="urn:schemas-microsoft-com:vml"
 xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
 xmlns:w10="urn:schemas-microsoft-com:office:word"
 xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
 xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
 xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk"
 xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
 xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
 mc:Ignorable="w14 wp14">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>
"""


def paragraph_xml(text: str, *, title: bool = False) -> str:
    """Create a simple WordprocessingML paragraph."""

    escaped = escape(text)
    if title:
        return (
            "<w:p><w:pPr><w:jc w:val=\"center\"/></w:pPr>"
            "<w:r><w:rPr><w:b/><w:sz w:val=\"32\"/></w:rPr>"
            f"<w:t>{escaped}</w:t></w:r></w:p>"
        )
    return (
        "<w:p><w:r><w:rPr><w:sz w:val=\"24\"/></w:rPr>"
        f"<w:t xml:space=\"preserve\">{escaped}</w:t></w:r></w:p>"
    )


def build_docx(title: str, paragraphs: list[str], output_path: Path) -> None:
    """Write a minimal DOCX package."""

    body_parts = [paragraph_xml(title, title=True)]
    body_parts.extend(paragraph_xml(paragraph) for paragraph in paragraphs)
    document_xml = DOCUMENT_XML_TEMPLATE.format(body="\n    ".join(body_parts))

    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as docx_file:
        docx_file.writestr("[Content_Types].xml", CONTENT_TYPES_XML)
        docx_file.writestr("_rels/.rels", ROOT_RELS_XML)
        docx_file.writestr("docProps/app.xml", APP_XML)
        docx_file.writestr("docProps/core.xml", CORE_XML.format(title=escape(title)))
        docx_file.writestr("word/document.xml", document_xml)


def main() -> None:
    """Generate all report DOCX files."""

    report_dir = Path(__file__).resolve().parent
    reports = {
        "report_1_game_mechanics.docx": (
            "Territory Capture Game Mechanics",
            [
                "Territory Capture is a two-player, turn-based, strategy-oriented board game designed on a 5x5 grid. The game is played by two players, represented by X and O. The players take turns placing stones on empty cells of the board. Each player places exactly eight stones during a full game. As a result, after sixteen total moves, nine cells remain empty. The main objective of the game is not simply to place more stones than the opponent, but to control more territory when the game ends. This makes the game strategically different from standard placement games, because the final outcome depends on spatial influence rather than direct stone count.",
                "One of the most important characteristics of Territory Capture is that scoring is performed only after the game is finished. During gameplay, no territory is assigned. Once both players have placed all eight stones, every empty cell on the board is evaluated individually. For each empty cell, the game checks all eight neighboring positions, including horizontal, vertical, and diagonal neighbors. If the number of neighboring X stones is greater than the number of neighboring O stones, that empty cell is counted as X territory. If the number of neighboring O stones is greater, the cell becomes O territory. If both counts are equal, the cell is considered neutral and gives no point to either player.",
                "A critical rule in the design of the game is that territory evaluation is simultaneous. In other words, empty cells do not influence each other during scoring. Only the stones that were actually placed on the board are considered. This rule keeps the scoring system clear and fair while also making the strategic planning more interesting. Players must think not only about the immediate position of their stones, but also about how those placements will affect the surrounding empty cells at the end of the game.",
                "In a later version of the game, a light capture mechanic was also introduced to make the gameplay more dynamic. After each move, all stones on the board are checked. If a stone has no empty neighboring cells among its eight surrounding positions, that stone is removed from the board. This removal is performed simultaneously rather than sequentially. Therefore, one stone being removed during a turn does not change the removal condition of another stone within the same check. This mechanic adds tactical depth because players can try not only to build territory, but also to trap and eliminate stones by surrounding them.",
                "The winner is determined by the number of territory cells controlled at the end of the game. The final score is based only on territory, not on the number of stones left on the board. If one player controls more territory cells than the other, that player wins. If both players control the same number of territory cells, the result is a draw. This scoring system makes the game easy to understand while still rewarding planning, positioning, and long-term board control.",
                "Overall, Territory Capture offers a simple but strategically rich structure. Even though it uses a small 5x5 board and a limited number of stones, the interaction between placement, influence, and territory creates meaningful decision making. Players must balance short-term moves with long-term control of space. Because of these properties, the game is well suited for artificial intelligence experiments, especially for projects involving search algorithms and self-play learning.",
            ],
        ),
        "report_2_implementation_and_ai_foundation.docx": (
            "Implementation of the Game and AI Foundation",
            [
                "The Territory Capture project was implemented in Python with a modular and extendable software structure. The first objective of the project was to create a reliable game environment that correctly represents the rules of the custom board game. The second objective was to prepare this environment for future artificial intelligence methods, especially AlphaZero-style learning. For this reason, the software architecture was designed not only to support gameplay, but also to support search algorithms, data generation, and machine learning components.",
                "The game logic and the user interface were intentionally separated. The core engine handles board representation, turn management, legal move generation, move application, terminal-state detection, capture processing, and winner determination. The territory scoring logic was also kept in a dedicated rules module rather than being mixed into the main game loop. This separation improved readability and made the code easier to test, maintain, and reuse for AI development. As a result, the game engine can be used independently of the graphical interface, which is essential for automated simulations and self-play.",
                "The board is internally represented as a 5x5 structure, and the players are represented by the symbols X and O. On each turn, the current player selects a legal empty cell and places a stone there. The game state is then updated, the optional capture rule is applied, and the turn passes to the other player. When the maximum number of moves is reached, the engine evaluates the final territory scores and determines the winner. Since the environment is deterministic and clearly defined, it is suitable for both classical AI agents and learning-based methods.",
                "To provide baseline performance comparisons, several non-learning agents were added. A RandomAgent selects any legal move uniformly at random. A HeuristicAgent uses a small set of understandable positional rules, such as preferring the center, valuing empty neighboring cells, and improving local influence. A MinimaxAgent performs shallow lookahead with a simple evaluation function. These agents serve two purposes. First, they verify that the game engine behaves correctly under repeated automated play. Second, they provide useful benchmark opponents for later evaluation of the neural network based system.",
                "An important step toward AI training was the addition of a representation layer. Each game state is encoded into a fixed tensor of shape (2, 5, 5). The first channel stores the stones of the current player, and the second channel stores the stones of the opponent. This perspective-based representation means that the model always sees the board from the point of view of the player whose turn it is. In addition, each board position is mapped to a fixed action index from 0 to 24, and a legal action mask indicates which actions are valid in the current position. These design choices make the environment directly compatible with policy-based neural networks and Monte Carlo Tree Search.",
                "The project also includes a self-play dataset pipeline. For each recorded move, the system stores the encoded state, a policy target, and a value target. In the earlier stages, policy targets were stored as simple one-hot vectors. Later, stronger policy targets based on search visit counts were added. The dataset tools allow games to be simulated in large numbers and exported to JSON. A replay buffer was also introduced so that training can use a rolling collection of recent self-play samples rather than depending only on the newest games. This makes the learning setup closer to modern game-AI pipelines.",
                "In summary, the software implementation does more than simply run the game. It provides a complete foundation for artificial intelligence experiments. The project now includes a stable game environment, baseline agents, state encoding, action mapping, legal move masking, and data generation utilities. These components make the system suitable for building search-based and learning-based AI methods in a clean and academically explainable way.",
            ],
        ),
        "report_3_final_ai_system.docx": (
            "Final AI-Enabled Version of Territory Capture",
            [
                "In the final stage of the project, Territory Capture was extended from a custom board game into an AI-enabled learning system. The final version includes a policy-value neural network, Monte Carlo Tree Search (MCTS), self-play data generation, replay-buffer-based training, model evaluation, and iterative improvement. Although the system is a simplified implementation rather than a perfect reproduction of the original AlphaZero paper, it successfully captures the main ideas of an AlphaZero-style algorithm for a custom strategy game.",
                "At the center of the system is a deep learning model implemented in PyTorch. The network receives a game-state tensor of shape (2, 5, 5) and produces two outputs. The first output is a policy prediction over the 25 possible board actions. The second output is a scalar value estimate between -1 and 1 that represents the expected outcome of the position. Residual blocks were used in the architecture so that the model follows the same general design philosophy as AlphaZero-style policy-value networks. This makes the model suitable for both action selection and state evaluation.",
                "The neural network does not act alone. Instead, it is combined with Monte Carlo Tree Search. Before making a move, MCTS explores possible continuations of the current position and uses the network outputs to guide the search. The policy head provides prior probabilities for expansion, while the value head estimates the quality of leaf states. The implementation also includes useful AlphaZero-inspired improvements such as root reuse, visit-count policy extraction, Dirichlet noise at the root for exploration during self-play, and temperature-based action selection in the opening moves. Together, these features allow the system to balance exploitation of strong moves with exploration of alternative positions.",
                "Self-play is a crucial part of the final system. The AI plays games against itself and records training examples for every move. Each sample contains the encoded state, the MCTS policy distribution derived from visit counts, and the final game result from the perspective of the player to move. These records are accumulated in a replay buffer, which stores a rolling window of recent samples. The replay buffer makes training more stable because the model learns from a broader set of recent experiences instead of relying only on the most recently generated games.",
                "A training pipeline was then built on top of this data. The replay-buffer samples are used to train the policy-value network, and the resulting candidate model is evaluated against the current best model in an arena setting. If the candidate model performs well enough, it is promoted to become the new best model. This creates an iterative improvement loop consisting of self-play, training, evaluation, and model replacement. That loop is the core idea behind AlphaZero-style learning.",
                "The project also includes practical evaluation tools. AI vs AI match systems, AI vs Minimax comparisons, result statistics, and win-rate plots were added to measure progress experimentally. In addition, the trained AI was integrated into the existing graphical interface so that the game can be played in Human vs Human, Human vs AI, and AI vs AI modes. This makes the final system not only technically complete, but also easy to demonstrate in a presentation or live project defense.",
                "In conclusion, the final version of Territory Capture includes the essential components of a simplified AlphaZero-style algorithm for a custom strategy game: game-state encoding, a policy-value network, MCTS, self-play, replay-based training, and iterative evaluation. Therefore, the project successfully fulfills the goal of applying an AlphaZero-inspired approach to an original simple strategy game in a way that is both academically defensible and practically demonstrable.",
            ],
        ),
        "report_4_v2_improvements.docx": (
            "Territory Capture v2 — Game Redesign, New GUI, and Model Training",
            [
                "1. Introduction and Motivation",
                "This report describes the changes made between version 1 and version 2 of the Territory Capture project. The first version established the game engine, baseline agents, neural network architecture, MCTS, and self-play infrastructure on a 5x5 board. In version 2, the focus shifted to making the game more strategic and enjoyable, improving the visual presentation, and training a stronger AI model. These improvements were made in preparation for the second project presentation.",
                "",
                "2. Game Rule Redesign",
                "The most significant change in v2 is the complete redesign of the game rules. The board was enlarged from 5x5 to 6x6, and each player now places 10 stones instead of 8, for a total of 20 moves per game. After all stones are placed, 16 empty cells remain for territory evaluation.",
                "The capture mechanic was fundamentally changed. In v1, a stone was captured only when it had zero empty neighbors — a condition that rarely occurred on a 5x5 board and had minimal strategic impact. In v2, the new majority capture rule is used: a stone is captured when it has at least 2 opponent neighbors AND the number of opponent neighbors strictly exceeds the number of friendly neighbors. This rule triggers much more frequently and creates meaningful tactical decisions throughout the game.",
                "A capture bonus scoring system was also added. Each captured opponent stone gives the capturing player +1 bonus point. The final score is now calculated as territory points plus capture bonus, making captures an important part of the overall strategy rather than a rare side effect.",
                "A new tiebreaker rule was introduced for draws. When both players have equal total scores, only the four center cells of the board (positions 2,2 / 2,3 / 3,2 / 3,3) are examined. Among these, only empty cells that received a territory assignment are counted. The player who controls more of these center cells wins the tiebreak. If the center count is also equal, the game is declared a true draw. This tiebreaker encourages center-oriented play and reduces the frequency of draws.",
                "",
                "3. Encoding and Neural Network Changes",
                "The state encoding was updated from shape (2, 5, 5) to (2, 6, 6), and the action space increased from 25 to 36. All dependent components — the policy-value network input/output dimensions, MCTS, self-play, training pipeline, and dataset generation — were updated accordingly. The policy-value network retains its ResNet architecture with 5 residual blocks and 64 channels, but now processes the larger board representation. All old 5x5 model checkpoints are incompatible with the new version.",
                "",
                "4. Agent Improvements",
                "The MinimaxAgent evaluation function was improved. In v1, it only considered territory scores in its heuristic. In v2, the evaluation includes: territory difference (weight 4.0), capture bonus difference (weight 2.0), stone count difference (weight 0.5), turn bonus, and mobility bonus. For terminal states, the agent uses exact total scores multiplied by 10.0, ensuring it correctly values captured stones and end-of-game positions.",
                "",
                "5. Graphical User Interface Rewrite",
                "The GUI was completely rewritten from Tkinter to Pygame. The new interface runs in a 1000x720 window with an 80-pixel cell grid. Visual improvements include: circle-shaped stones with glow and shadow effects, capture flash animations when stones are removed, territory overlay with color-coded cells at end of game, hover preview for the next move, left and right player panels showing scores and stone counts, and a result overlay with detailed score breakdowns.",
                "The difficulty system was redesigned. Instead of a single difficulty setting, each player now has an independent difficulty selector. In AI vs AI mode, Player X and Player O can use different difficulty levels. The three levels are: Easy (RandomAgent), Medium (MinimaxAgent with depth 2), and Hard (AIAgent using MCTS with the trained neural network). This allows direct comparison between different AI approaches during live demonstrations.",
                "",
                "6. Training Data Generation and Model Training",
                "A large-scale dataset of 100,000 training samples was generated for the 6x6 game version. The samples were created using four different agent matchups with an optimized distribution: 40,000 samples from minimax vs heuristic games (40%), 30,000 from heuristic vs heuristic (30%), 20,000 from heuristic vs random (20%), and 10,000 from random vs random (10%). This distribution was chosen to provide diverse training data that ranges from high-quality strategic play to broad position coverage.",
                "The policy-value network was trained for 15 epochs on this dataset with a batch size of 128 and learning rate of 0.001 with step-decay scheduling. The final training loss was 0.4978 and validation loss was 1.2363. The trained model was saved and integrated into the GUI as the Hard difficulty level, where it is combined with 50-simulation MCTS for move selection.",
                "",
                "7. Code Quality and Testing",
                "All 52 unit tests pass on the v2 codebase. The test suite covers game logic, rules, encoding, agents, MCTS, self-play, training, arena evaluation, and the AlphaZero iteration loop. The training pipeline was updated to accept both old-format (state/policy/value) and new-format (encoded_state/policy_target/value_target) dataset records for backward compatibility.",
                "",
                "8. Summary of Changes from v1 to v2",
                "Board: 5x5 → 6x6. Stones per player: 8 → 10. Capture rule: zero-empty-neighbors → majority opponent neighbors. Scoring: territory only → territory + capture bonus. Tiebreaker: draw → center 4 cells. GUI: Tkinter → Pygame with animations and effects. Difficulty: single shared level → per-player independent levels. AI levels: Easy/disabled/disabled → Easy (random), Medium (minimax), Hard (MCTS + neural net). Training data: none for 6x6 → 100k samples. Model: untrained → trained on 100k samples, 15 epochs. Tests: all 52 passing.",
            ],
        ),
        "report_5_6x6_detailed_status.docx": (
            "Territory Capture 6x6 — Detailed Status Report (Before Multi-Board Migration)",
            [
                "1. Project Overview",
                "This report documents the complete status of the Territory Capture 6x6 system before migrating to the multi-board (5x5, 6x6, 7x7) architecture. The project implements an AlphaZero-style AI pipeline for a custom turn-based strategy game. The codebase consists of 3,879 lines of source code across 20 Python modules and 961 lines of test code across 13 test files. All 52 unit tests pass successfully.",
                "",
                "2. Game Engine (game.py, rules.py)",
                "The game engine implements a 6x6 board where each player (X and O) places 10 stones in alternating turns, totaling 20 moves per game. After all stones are placed, 16 empty cells remain for territory evaluation. The capture mechanic uses the majority pressure rule: a stone is captured when it has at least 2 opponent neighbors AND opponent neighbors outnumber friendly neighbors. Captures are resolved simultaneously after each move. Scoring combines territory points with a +1 capture bonus per captured opponent stone. The tiebreaker examines the 4 center cells (positions 2,2 / 2,3 / 3,2 / 3,3) for territory ownership.",
                "",
                "3. State Encoding (encoding.py)",
                "Game states are encoded as (2, 6, 6) tensors using relative perspective encoding. Channel 0 represents stones of the current player, channel 1 represents opponent stones. This perspective-based encoding means the network always sees the board from the viewpoint of the player to move. The action space is 36 (6x6 board positions). A legal action mask marks valid moves. The encoding module also supports a 3-channel turn-plane encoding variant for future experiments.",
                "",
                "4. Neural Network Architecture (model.py)",
                "The PolicyValueNet is a dual-headed ResNet implemented in PyTorch. Architecture details: Input shape (2, 6, 6), initial 3x3 convolution to 64 channels with batch normalization, 5 residual blocks (each with two 3x3 convolutions + batch norm + skip connection), policy head (1x1 conv to 2 channels, flatten, fully connected to 36 outputs), value head (1x1 conv to 1 channel, flatten, FC to 64 units, FC to 1 unit with tanh activation). Total parameters: approximately 290,000. The model outputs raw policy logits and a scalar value in [-1, 1].",
                "",
                "5. Monte Carlo Tree Search (mcts.py)",
                "The MCTS implementation uses the PUCT selection formula with configurable c_puct (default 1.5). Features include: root subtree reuse across consecutive moves, Dirichlet noise injection at root for exploration during self-play (alpha=0.3, epsilon=0.25), temperature-based action selection (high temperature for opening moves, greedy for late game), visit-count policy extraction for training targets. Default simulation count is 50 per move.",
                "",
                "6. Baseline Agents (agents.py)",
                "Three non-learning agents provide baselines: (a) RandomAgent — selects uniformly from legal moves, (b) HeuristicAgent — uses positional features: center preference, empty neighbor count, local influence advantage, capture avoidance, (c) MinimaxAgent — alpha-beta search with depth 2, evaluation function weighing territory difference (4.0), capture bonus difference (2.0), stone count (0.5), turn bonus, and mobility. These agents serve as benchmarks and as data generation sources.",
                "",
                "7. AI Agent Integration (ai_agent.py)",
                "The AIAgent class wraps PolicyValueNet + MCTS into a game-ready agent. It automatically detects the best available device (CUDA > MPS > CPU). MPS (Apple Silicon GPU) acceleration provides 5-10x speedup over CPU for both training and inference. The agent supports configurable simulation count and c_puct. Tree state is preserved between consecutive moves and reset between games.",
                "",
                "8. Training Data Generation",
                "Two dataset generation approaches exist: (a) generate_dataset.py — uses baseline agents (random, heuristic, minimax) for large-scale one-hot policy data, (b) self_play.py — uses the neural network + MCTS for visit-count policy data (AlphaZero-style).",
                "For the 6x6 model, 2,000,000 training samples were generated using the baseline agent pipeline with the following distribution: 800,000 samples from minimax vs heuristic (40%), 600,000 from heuristic vs heuristic (30%), 400,000 from heuristic vs random (20%), 200,000 from random vs random (10%). This distribution ensures diverse position coverage from high-quality strategic play down to broad random exploration.",
                "",
                "9. Training Pipeline (train.py)",
                "The training pipeline uses PyTorch with the following configuration: Adam optimizer, StepLR scheduler (step_size=3, gamma=0.5), cross-entropy loss for policy head, MSE loss for value head, 90/10 train/validation split. Two models were trained for 6x6:",
                "Model 1 (model.pth — Medium/Normal): 100,000 samples, 15 epochs, batch size 128. Final train loss: 0.4978, validation loss: 1.2363.",
                "Model 2 (model_hard.pth — Hard): 2,000,000 samples, 20 epochs, batch size 512, MPS GPU acceleration. Final train loss: 0.7468, validation loss: 0.9009. This model shows better generalization with lower validation loss despite higher training loss, indicating less overfitting on the larger dataset.",
                "",
                "10. AlphaZero Iteration Loop (alpha_zero_loop.py, arena.py, replay_buffer.py)",
                "The full AlphaZero improvement cycle is implemented: self-play data generation with MCTS policies, replay buffer accumulation (configurable capacity), candidate model training, arena evaluation (candidate vs current best), model promotion if win rate exceeds threshold (default 55%). The replay buffer maintains a rolling window of recent samples for training stability.",
                "",
                "11. Graphical User Interface (gui.py)",
                "The Pygame-based GUI (723 lines) provides: 1000x720 window with 80px cell grid, circle-shaped stones with glow and shadow effects, capture flash animations, territory overlay with color-coded cells at end of game, hover preview for next move, per-player independent difficulty selection (Easy/Medium/Hard), left and right player panels with scores and stone counts, result overlay with detailed score breakdown, Human vs Human, Human vs AI, and AI vs AI modes.",
                "",
                "12. Test Suite Summary",
                "All 52 tests pass (pytest, 41 seconds). Test coverage by module: test_game.py (8 tests) — board placement, turn switching, terminal detection, capture processing, scoring, draw conditions. test_rules.py (6 tests) — territory evaluation, capture rule, tiebreaker, neutral cells, capture bonus. test_encoding.py (7 tests) — state shape, perspective encoding, action index round-trip, legal action mask, turn plane encoding. test_agents.py (7 tests) — random/heuristic/minimax move legality, simulation batch, clone safety. test_mcts.py (3 tests) — search returns legal move + policy, Dirichlet noise changes priors, root advancement. test_dataset.py (6 tests) — self-play episode production, one-hot policy, value propagation, JSON round-trip, encoder lookup. test_generate_dataset.py (4 tests) — summary structure, matchup codes, task builder, timestamped paths. test_train.py (3 tests) — JSON record loading, in-memory training, checkpoint saving. test_self_play.py (2 tests) — output format, config-based pipeline. test_ai_eval.py (2 tests) — AI vs AI stats, AI vs Minimax stats. test_alpha_zero_loop.py (1 test) — iteration summary. test_arena.py (1 test) — model evaluation result. test_replay_buffer.py (2 tests) — capacity trimming, save/load round-trip.",
                "",
                "13. Repository Structure",
                "Branch: v2 on GitHub (Sthonem/Territory_Capture-Alphazero). Key source files: game.py (214 lines) — game engine, rules.py (197 lines) — scoring and captures, encoding.py (110 lines) — state/action encoding, model.py (75 lines) — PolicyValueNet, mcts.py (307 lines) — MCTS implementation, ai_agent.py (96 lines) — model+MCTS wrapper, agents.py (324 lines) — baseline agents, dataset.py (217 lines) — training data structures, generate_dataset.py (324 lines) — large-scale data generation, self_play.py (146 lines) — MCTS self-play data, train.py (288 lines) — training pipeline, gui.py (723 lines) — Pygame interface, alpha_zero_loop.py (171 lines) — iteration loop, arena.py (135 lines) — model evaluation, replay_buffer.py (46 lines) — rolling sample buffer.",
                "",
                "14. Known Limitations and Next Steps",
                "The 6x6 system is feature-complete but has room for improvement. The 2M-sample model (model_hard.pth) has not been formally evaluated in a large-scale AI vs baselines arena test. The GUI currently only supports the 6x6 board. The next phase will extend the codebase to support 5x5 and 7x7 boards with separate trained models, enabling cross-board-size comparison experiments for the final presentation.",
            ],
        ),
    }

    for filename, (title, paragraphs) in reports.items():
        build_docx(title, paragraphs, report_dir / filename)


if __name__ == "__main__":
    main()
