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
    }

    for filename, (title, paragraphs) in reports.items():
        build_docx(title, paragraphs, report_dir / filename)


if __name__ == "__main__":
    main()
