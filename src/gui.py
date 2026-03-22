"""Tkinter user interface for playing Territory Capture locally."""

from __future__ import annotations

import tkinter as tk
from typing import Dict, Optional, Tuple

from .game import GameResult, TerritoryCaptureGame
from .rules import EMPTY, PLAYER_O, PLAYER_X

Position = Tuple[int, int]

WINDOW_BG = "#050816"
WINDOW_BG_ALT = "#0b1228"
PANEL_BG = "#0d1630"
PANEL_GLOW = "#1b2d57"
TEXT_PRIMARY = "#ecf8ff"
TEXT_MUTED = "#8ea6c9"
TEXT_ACCENT = "#9ecbff"
NEON_BLUE = "#4cc9ff"
NEON_BLUE_SOFT = "#143a5c"
NEON_PINK = "#ff5fa2"
NEON_PINK_SOFT = "#4c1630"
NEON_VIOLET = "#9b6dff"
CELL_EMPTY = "#111a33"
CELL_EMPTY_HOVER = "#16264a"
CELL_BORDER = "#21345d"
CELL_BORDER_HOVER = "#4cc9ff"
CELL_X = "#081a2b"
CELL_O = "#2a0c19"
TERRITORY_X = "#122b46"
TERRITORY_O = "#3a1424"
TERRITORY_NEUTRAL = "#1b2743"
TERRITORY_X_WIN = "#183962"
TERRITORY_O_WIN = "#4b1b31"
CAPTURE_FLASH = "#6b4f0b"
CAPTURE_FLASH_TEXT = "#ffd76a"


class TerritoryCaptureGUI:
    """Polished desktop interface for a local two-player game."""

    def __init__(self) -> None:
        self.game = TerritoryCaptureGame()
        self.root = tk.Tk()
        self.root.title("Territory Capture")
        self.root.configure(bg=WINDOW_BG)
        self.root.resizable(False, False)

        self.buttons: Dict[Position, tk.Button] = {}
        self.last_move: Optional[Position] = None
        self.current_result: Optional[GameResult] = None
        self.rules_window: Optional[tk.Toplevel] = None
        self.recently_captured_positions: set[Position] = set()

        self.status_var = tk.StringVar()
        self.score_var = tk.StringVar()
        self.info_var = tk.StringVar(
            value="Place 8 stones each. Stones with no empty neighbors are removed."
        )
        self.legend_var = tk.StringVar(
            value="Stone colors: blue = X, pink = O. Territory appears after the game."
        )

        self.main_frame = tk.Frame(self.root, bg=WINDOW_BG, padx=22, pady=18)
        self.main_frame.pack()

        self.top_glow = tk.Frame(self.root, bg=WINDOW_BG_ALT, height=6)
        self.top_glow.pack(fill="x", side="top")

        self.header_frame = tk.Frame(self.main_frame, bg=WINDOW_BG)
        self.header_frame.pack(fill="x")

        self.board_shell = tk.Frame(
            self.main_frame,
            bg=PANEL_BG,
            highlightthickness=2,
            highlightbackground=PANEL_GLOW,
            padx=16,
            pady=16,
        )
        self.board_shell.pack(pady=(14, 18))

        self.board_frame = tk.Frame(self.board_shell, bg=PANEL_BG)
        self.board_frame.pack()

        self.footer_frame = tk.Frame(self.main_frame, bg=WINDOW_BG)
        self.footer_frame.pack(fill="x")

        self._build_header()
        self._build_board()
        self._build_footer()
        self._refresh_view()

    def _build_header(self) -> None:
        """Create the title, status, and top-level actions."""

        title = tk.Label(
            self.header_frame,
            text="Territory Capture",
            font=("Helvetica", 28, "bold"),
            fg=TEXT_PRIMARY,
            bg=WINDOW_BG,
        )
        title.pack()

        title_glow = tk.Label(
            self.header_frame,
            text="Territory Capture",
            font=("Helvetica", 12, "bold"),
            fg=NEON_BLUE,
            bg=WINDOW_BG,
            pady=2,
        )
        title_glow.pack()

        subtitle = tk.Label(
            self.header_frame,
            text="A compact strategy game with endgame territory scoring",
            font=("Helvetica", 12),
            fg=TEXT_MUTED,
            bg=WINDOW_BG,
            pady=4,
        )
        subtitle.pack()

        action_row = tk.Frame(self.header_frame, bg=WINDOW_BG, pady=10)
        action_row.pack()

        how_to_play_button = tk.Button(
            action_row,
            text="How to Play",
            command=self._show_rules_modal,
            font=("Helvetica", 12, "bold"),
            bg=NEON_VIOLET,
            fg="#f8fafc",
            activebackground="#8353ff",
            activeforeground="#f8fafc",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground="#c4b5fd",
            padx=18,
            pady=8,
            cursor="hand2",
        )
        how_to_play_button.pack(side="left", padx=6)

        new_game_button = tk.Button(
            action_row,
            text="New Game",
            command=self._reset_game,
            font=("Helvetica", 12, "bold"),
            bg="#0f2742",
            fg="#d8f3ff",
            activebackground="#14355c",
            activeforeground="#f8fdff",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=NEON_BLUE,
            padx=18,
            pady=8,
            cursor="hand2",
        )
        new_game_button.pack(side="left", padx=6)

        status = tk.Label(
            self.header_frame,
            textvariable=self.status_var,
            font=("Helvetica", 14, "bold"),
            fg="#f5fbff",
            bg=WINDOW_BG,
            pady=8,
        )
        status.pack()

        score = tk.Label(
            self.header_frame,
            textvariable=self.score_var,
            font=("Helvetica", 12),
            fg="#d6e6ff",
            bg=WINDOW_BG,
        )
        score.pack()

        info = tk.Label(
            self.header_frame,
            textvariable=self.info_var,
            font=("Helvetica", 11),
            fg=TEXT_ACCENT,
            bg=WINDOW_BG,
            pady=6,
        )
        info.pack()

        legend = tk.Label(
            self.header_frame,
            textvariable=self.legend_var,
            font=("Helvetica", 10),
            fg=TEXT_MUTED,
            bg=WINDOW_BG,
            pady=4,
        )
        legend.pack()

    def _build_board(self) -> None:
        """Create clickable board cells."""

        for row in range(self.game.board_size):
            for col in range(self.game.board_size):
                button = tk.Button(
                    self.board_frame,
                    text="",
                    width=6,
                    height=3,
                    font=("Helvetica", 20, "bold"),
                    relief=tk.FLAT,
                    bd=0,
                    bg=CELL_EMPTY,
                    fg=TEXT_PRIMARY,
                    activebackground=CELL_EMPTY_HOVER,
                    activeforeground=TEXT_PRIMARY,
                    highlightthickness=1,
                    highlightbackground=CELL_BORDER,
                    highlightcolor=CELL_BORDER_HOVER,
                    cursor="hand2",
                    command=lambda position=(row, col): self._handle_move(position),
                )
                button.grid(row=row, column=col, padx=5, pady=5)
                button.bind(
                    "<Enter>",
                    lambda _event, pos=(row, col): self._handle_hover(pos, True),
                )
                button.bind(
                    "<Leave>",
                    lambda _event, pos=(row, col): self._handle_hover(pos, False),
                )
                self.buttons[(row, col)] = button

    def _build_footer(self) -> None:
        """Create a small presentation-friendly footer."""

        footer_note = tk.Label(
            self.footer_frame,
            text="Tip: open 'How to Play' for a quick visual explanation before presenting.",
            font=("Helvetica", 10),
            fg=TEXT_MUTED,
            bg=WINDOW_BG,
        )
        footer_note.pack()

    def _handle_hover(self, position: Position, is_hovered: bool) -> None:
        """Apply a soft glow to hoverable empty cells."""

        if self.game.is_terminal() or not self.game.is_legal_move(position):
            return

        button = self.buttons[position]
        if is_hovered:
            button.configure(
                bg=CELL_EMPTY_HOVER,
                activebackground=CELL_EMPTY_HOVER,
                highlightbackground=CELL_BORDER_HOVER,
            )
        else:
            button.configure(
                bg=CELL_EMPTY,
                activebackground=CELL_EMPTY_HOVER,
                highlightbackground=CELL_BORDER,
            )

    def _handle_move(self, position: Position) -> None:
        """Apply a human move and update the interface."""

        if not self.game.is_legal_move(position):
            return

        self.game.apply_move(position)
        self.last_move = position
        self.current_result = None
        self.recently_captured_positions = set(self.game.last_captured_positions)
        if self.recently_captured_positions:
            removed_count = len(self.recently_captured_positions)
            noun = "stone" if removed_count == 1 else "stones"
            self.info_var.set(f"Capture resolved: {removed_count} {noun} removed.")
        else:
            self.info_var.set(
                "No capture this turn. Stones with no empty neighbors are removed."
            )
        self._refresh_view()

        if self.game.is_terminal():
            self._show_final_result()

    def _reset_game(self) -> None:
        """Start a fresh match."""

        self.game.reset()
        self.last_move = None
        self.current_result = None
        self.recently_captured_positions = set()
        self.info_var.set(
            "Place 8 stones each. Stones with no empty neighbors are removed."
        )
        self.legend_var.set(
            "Stone colors: blue = X, pink = O. Territory appears after the game."
        )
        self._refresh_view()

    def _refresh_view(self) -> None:
        """Redraw board cells and status text."""

        territory_map = (
            self.current_result.territory_map if self.current_result is not None else {}
        )

        for row in range(self.game.board_size):
            for col in range(self.game.board_size):
                button = self.buttons[(row, col)]
                value = self.game.board[row][col]
                territory_owner = territory_map.get((row, col))
                self._style_cell(button, value, territory_owner, (row, col))

        if self.game.is_terminal():
            self.status_var.set("Game finished")
        else:
            self.status_var.set(f"Current turn: Player {self.game.current_player}")

        self.score_var.set(
            f"Moves: {self.game.move_count}/{self.game.max_moves}    "
            f"X stones: {self.game.stones_placed[PLAYER_X]}/8    "
            f"O stones: {self.game.stones_placed[PLAYER_O]}/8"
        )

    def _style_cell(
        self,
        button: tk.Button,
        value: str,
        territory_owner: Optional[str],
        position: Position,
    ) -> None:
        """Update colors and labels for one board cell."""

        if value == PLAYER_X:
            highlight = "#143354" if position == self.last_move else CELL_X
            button.configure(
                text="X",
                bg=highlight,
                fg=NEON_BLUE,
                activebackground=highlight,
                activeforeground=NEON_BLUE,
                state=tk.DISABLED,
                disabledforeground=NEON_BLUE,
                highlightbackground="#4fbfff",
            )
            return

        if value == PLAYER_O:
            highlight = "#441427" if position == self.last_move else CELL_O
            button.configure(
                text="O",
                bg=highlight,
                fg=NEON_PINK,
                activebackground=highlight,
                activeforeground=NEON_PINK,
                state=tk.DISABLED,
                disabledforeground=NEON_PINK,
                highlightbackground="#ff7caf",
            )
            return

        if territory_owner == PLAYER_X:
            winner_boost = self.current_result is not None and self.current_result.winner == PLAYER_X
            button.configure(
                text="x",
                bg=TERRITORY_X_WIN if winner_boost else TERRITORY_X,
                fg="#8fdfff",
                activebackground=TERRITORY_X_WIN if winner_boost else TERRITORY_X,
                activeforeground="#8fdfff",
                state=tk.DISABLED,
                disabledforeground="#8fdfff",
                highlightbackground="#4076b1" if winner_boost else "#2d5c87",
            )
        elif territory_owner == PLAYER_O:
            winner_boost = self.current_result is not None and self.current_result.winner == PLAYER_O
            button.configure(
                text="o",
                bg=TERRITORY_O_WIN if winner_boost else TERRITORY_O,
                fg="#ff9fc1",
                activebackground=TERRITORY_O_WIN if winner_boost else TERRITORY_O,
                activeforeground="#ff9fc1",
                state=tk.DISABLED,
                disabledforeground="#ff9fc1",
                highlightbackground="#944566" if winner_boost else "#6f314a",
            )
        elif territory_owner == EMPTY:
            button.configure(
                text="·",
                bg=TERRITORY_NEUTRAL,
                fg="#b7c9e8",
                activebackground=TERRITORY_NEUTRAL,
                activeforeground="#b7c9e8",
                state=tk.DISABLED,
                disabledforeground="#b7c9e8",
                highlightbackground="#32476f",
            )
        elif position in self.recently_captured_positions:
            button.configure(
                text="*",
                bg=CAPTURE_FLASH,
                fg=CAPTURE_FLASH_TEXT,
                activebackground=CAPTURE_FLASH,
                activeforeground=CAPTURE_FLASH_TEXT,
                state=tk.DISABLED if self.game.is_terminal() else tk.NORMAL,
                disabledforeground=CAPTURE_FLASH_TEXT,
                highlightbackground="#c7951d",
            )
        else:
            button.configure(
                text="",
                bg=CELL_EMPTY,
                fg=TEXT_PRIMARY,
                activebackground=CELL_EMPTY_HOVER,
                activeforeground=TEXT_PRIMARY,
                state=tk.NORMAL,
                highlightbackground=CELL_BORDER,
            )

    def _show_rules_modal(self) -> None:
        """Open a compact modal that explains the game rules."""

        if self.rules_window is not None and self.rules_window.winfo_exists():
            self.rules_window.focus_set()
            return

        modal = tk.Toplevel(self.root)
        modal.title("How to Play")
        modal.configure(bg=WINDOW_BG_ALT)
        modal.resizable(False, False)
        modal.transient(self.root)
        modal.grab_set()
        self.rules_window = modal

        shell = tk.Frame(
            modal,
            bg=PANEL_BG,
            highlightthickness=2,
            highlightbackground=NEON_VIOLET,
            padx=18,
            pady=18,
        )
        shell.pack(padx=18, pady=18)

        title = tk.Label(
            shell,
            text="How to Play",
            font=("Helvetica", 20, "bold"),
            fg=TEXT_PRIMARY,
            bg=PANEL_BG,
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            shell,
            text="Fast explanation for first-time players",
            font=("Helvetica", 11),
            fg="#c5b8ff",
            bg=PANEL_BG,
            pady=2,
        )
        subtitle.pack(anchor="w")

        summary_text = (
            "1. Play on a 5x5 board.\n"
            "2. X and O alternate turns placing one stone.\n"
            "3. After each move, any stone with no empty neighbors is removed.\n"
            "4. Each player places exactly 8 stones.\n"
            "5. The game ends after 16 total moves.\n"
            "6. The remaining empty cells are then scored as territory."
        )
        summary = tk.Label(
            shell,
            text=summary_text,
            justify="left",
            font=("Helvetica", 11),
            fg=TEXT_PRIMARY,
            bg=PANEL_BG,
            pady=10,
        )
        summary.pack(anchor="w")

        example_title = tk.Label(
            shell,
            text="How territory is decided",
            font=("Helvetica", 13, "bold"),
            fg="#f8fafc",
            bg=PANEL_BG,
            pady=6,
        )
        example_title.pack(anchor="w")

        example_grid = tk.Frame(shell, bg=PANEL_BG)
        example_grid.pack(anchor="w", pady=(0, 10))

        self._build_example_grid(
            example_grid,
            cells=[
                ["X", "X", "."],
                ["O", "?", "O"],
                [".", "X", "."],
            ],
        )

        explanation = tk.Label(
            shell,
            text=(
                "Look at the center '?'. Count all 8 neighbors, including diagonals.\n"
                "Here X has 3 neighbors and O has 2, so the empty cell becomes X territory."
            ),
            justify="left",
            font=("Helvetica", 11),
            fg=TEXT_PRIMARY,
            bg=PANEL_BG,
            wraplength=430,
        )
        explanation.pack(anchor="w")

        legend = tk.Label(
            shell,
            text=(
                "Capture rule: after every move, stones with zero empty neighbors are removed.\n"
                "If O had more neighbors, it would become O territory.\n"
                "If the counts were equal, the cell would stay neutral."
            ),
            justify="left",
            font=("Helvetica", 10),
            fg=TEXT_ACCENT,
            bg=PANEL_BG,
            pady=10,
        )
        legend.pack(anchor="w")

        result_note = tk.Label(
            shell,
            text="Winner = player with more territory cells after scoring.",
            font=("Helvetica", 12, "bold"),
            fg="#fde68a",
            bg=PANEL_BG,
            pady=4,
        )
        result_note.pack(anchor="w")

        close_button = tk.Button(
            shell,
            text="Close",
            command=modal.destroy,
            font=("Helvetica", 11, "bold"),
            bg="#8b5cf6",
            fg="#f8fafc",
            activebackground="#7c3aed",
            activeforeground="#f8fafc",
            relief=tk.FLAT,
            padx=16,
            pady=7,
            cursor="hand2",
        )
        close_button.pack(anchor="e", pady=(14, 0))

        modal.protocol("WM_DELETE_WINDOW", modal.destroy)

    def _build_example_grid(self, parent: tk.Frame, cells: list[list[str]]) -> None:
        """Render a tiny visual example inside the rules modal."""

        colors = {
            "X": CELL_X,
            "O": CELL_O,
            ".": CELL_EMPTY,
            "?": "#fde68a",
        }
        text_colors = {
            "X": "#082f49",
            "O": "#4c0519",
            ".": "#cbd5e1",
            "?": "#78350f",
        }

        for row_index, row in enumerate(cells):
            for col_index, value in enumerate(row):
                label = tk.Label(
                    parent,
                    text=value,
                    width=4,
                    height=2,
                    font=("Helvetica", 15, "bold"),
                    bg=colors[value],
                    fg=text_colors[value],
                    relief=tk.FLAT,
                    highlightthickness=1,
                    highlightbackground=CELL_BORDER,
                )
                label.grid(row=row_index, column=col_index, padx=4, pady=4)

    def _show_final_result(self) -> None:
        """Display the winner and highlight territory cells."""

        self.current_result = self.game.get_result()
        result = self.current_result
        self._refresh_view()

        self.legend_var.set(
            "Board legend: bold X/O = stones, soft x/o = territory, dot = neutral."
        )

        if result.winner is None:
            self.info_var.set(
                f"Draw. Territory score X:{result.scores[PLAYER_X]} O:{result.scores[PLAYER_O]}"
            )
            title = "Draw game"
        else:
            self.info_var.set(
                f"Player {result.winner} wins with territory "
                f"{result.scores[PLAYER_X]} - {result.scores[PLAYER_O]}"
            )
            title = f"Player {result.winner} wins"

        self._show_result_modal(result, title)

    def _show_result_modal(self, result: GameResult, title_text: str) -> None:
        """Open a compact result popup after the game ends."""

        modal = tk.Toplevel(self.root)
        modal.title("Game Over")
        modal.configure(bg=WINDOW_BG_ALT)
        modal.resizable(False, False)
        modal.transient(self.root)
        modal.grab_set()

        shell = tk.Frame(
            modal,
            bg=PANEL_BG,
            highlightthickness=2,
            highlightbackground=NEON_BLUE if result.winner != PLAYER_O else NEON_PINK,
            padx=18,
            pady=18,
        )
        shell.pack(padx=18, pady=18)

        title = tk.Label(
            shell,
            text=title_text,
            font=("Helvetica", 18, "bold"),
            fg=NEON_BLUE if result.winner != PLAYER_O else NEON_PINK,
            bg=PANEL_BG,
        )
        title.pack(anchor="w")

        score = tk.Label(
            shell,
            text=(
                f"Territory score\n"
                f"X: {result.scores[PLAYER_X]}\n"
                f"O: {result.scores[PLAYER_O]}"
            ),
            justify="left",
            font=("Helvetica", 12),
            fg=TEXT_PRIMARY,
            bg=PANEL_BG,
            pady=12,
        )
        score.pack(anchor="w")

        explanation = tk.Label(
            shell,
            text="Soft board colors now show territory ownership directly on the grid.",
            font=("Helvetica", 10),
            fg=TEXT_MUTED,
            bg=PANEL_BG,
            wraplength=300,
            justify="left",
        )
        explanation.pack(anchor="w")

        close_button = tk.Button(
            shell,
            text="OK",
            command=modal.destroy,
            font=("Helvetica", 11, "bold"),
            bg="#2563eb",
            fg="#f8fafc",
            activebackground="#1d4ed8",
            activeforeground="#f8fafc",
            relief=tk.FLAT,
            padx=16,
            pady=7,
            cursor="hand2",
        )
        close_button.pack(anchor="e", pady=(14, 0))

    def run(self) -> None:
        """Start the Tkinter event loop."""

        self.root.mainloop()


def main() -> None:
    """Launch the game window."""

    app = TerritoryCaptureGUI()
    app.run()


if __name__ == "__main__":
    main()
