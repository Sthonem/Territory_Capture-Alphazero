"""Pygame desktop interface for Territory Capture."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pygame

from .agents import HeuristicAgent, MinimaxAgent, RandomAgent
from .ai_agent import AIAgent
from .encoding import BOARD_CONFIGS
from .game import GameResult, TerritoryCaptureGame
from .rules import EMPTY, PLAYER_O, PLAYER_X

# ── Geometry ──────────────────────────────────────────────────────────────────
WIN_W, WIN_H = 1000, 720
BOARD_Y      = 100

BOARD_SIZES  = [5, 6, 7]
DEFAULT_BS   = 6


FOOTER_H = 124  # fixed footer height per design system handoff


def _calc_geometry(bs: int) -> dict:
    """Compute dynamic layout values for a given board size.

    Footer is fixed at the bottom (FOOTER_H = 124px) regardless of board
    size — matches the design handoff (ui_kits/desktop/Footer.jsx).
    Panels span vertically from BOARD_Y down to the top of the footer.
    """
    cell = max(60, min(80, 480 // bs))
    board_px = cell * bs
    board_x = (WIN_W - board_px) // 2
    panel_w = board_x - 20
    footer_y = WIN_H - FOOTER_H
    return {
        "cell": cell,
        "bs": bs,
        "board_px": board_px,
        "board_x": board_x,
        "panel_w": panel_w,
        "l_panel_x": 10,
        "r_panel_x": board_x + board_px + 10,
        "panel_y": BOARD_Y,
        "panel_h": footer_y - BOARD_Y - 12,  # leave a 12px gap above footer
        "footer_y": footer_y,
        "footer_h": FOOTER_H,
    }

# ── Palette ───────────────────────────────────────────────────────────────────
# Source of truth: colors_and_type.css from the design system handoff.
BG         = (5,   8,  22)
PANEL_BG   = (13,  22,  48)
BOARD_BG   = (11,  17,  40)
GRID_C     = (33,  52,  93)
LINE_SOFT  = (100, 130, 170)  # blit via alpha surface @ α = 46 for soft organisers
FOOTER_BG  = (7,   12,  28)   # tc-footer-bg, slightly darker than tc-bg
COL_X      = (76, 201, 255)
COL_O      = (255,  95, 162)
GOLD       = (255, 215,  80)
TER_X      = (76, 201, 255)
TER_O      = (255,  95, 162)
TER_N      = (158, 180, 215)
CAP_COL    = (255, 200,  60)
TEXT_PRI   = (236, 248, 255)
TEXT_MUT   = (100, 130, 170)
TEXT_ACC   = (158, 203, 255)
VIOLET     = (155, 109, 255)
LAST_C     = (40,  70, 120)
HOVER_BG   = (25,  40,  75)   # darker hover state for buttons

MODE_HVH   = "Human vs Human"
MODE_HVAI  = "Human vs AI"
MODE_AIVAI = "AI vs AI"
MODES      = [MODE_HVH, MODE_HVAI, MODE_AIVAI]

DIFF_EASY  = "Easy"
DIFF_MED   = "Medium"
DIFF_HARD  = "Hard"
DIFFS      = [DIFF_EASY, DIFF_MED, DIFF_HARD]

AI_DELAY_MS = 600
FLASH_MS    = 800
TER_SPEED   = 5

Position = Tuple[int, int]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rrect(
    surf: pygame.Surface,
    color: tuple,
    rect: pygame.Rect,
    radius: int = 10,
    border: int = 0,
    border_color: tuple | None = None,
) -> None:
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border and border_color:
        pygame.draw.rect(surf, border_color, rect, border, border_radius=radius)


def coord_label(row: int, col: int) -> str:
    """Convert (row, col) to chess-like coordinate (a1, b2, ...)."""
    return f"{chr(ord('a') + col)}{row + 1}"


def _soft_vline(surf: pygame.Surface, x: int, y: int, h: int) -> None:
    """Draw a soft 1×h vertical line at α=46 (tc-line-soft token)."""
    line = pygame.Surface((1, h), pygame.SRCALPHA)
    line.fill((*LINE_SOFT, 46))
    surf.blit(line, (x, y))


def _soft_hline(surf: pygame.Surface, x: int, y: int, w: int) -> None:
    """Draw a soft w×1 horizontal line at α=46 (tc-line-soft token)."""
    line = pygame.Surface((w, 1), pygame.SRCALPHA)
    line.fill((*LINE_SOFT, 46))
    surf.blit(line, (x, y))


def _draw_group_label(
    surf: pygame.Surface,
    text: str,
    x: int, y: int, w: int,
    font: pygame.font.Font,
) -> None:
    """Render a tiny CAPS group label centered in (x, y, w).

    Specs: 9px bold, letter-spacing 0.12em, uppercase, muted color.
    Pygame approximates letter-spacing by widening the rendered text.
    """
    label = text.upper()
    # Render letter-spaced by drawing letters with extra pixel gaps.
    rendered_chars = [font.render(ch, True, TEXT_MUT) for ch in label]
    widths = [r.get_width() for r in rendered_chars]
    gap = 1  # approximates 0.12em at 9px
    total = sum(widths) + gap * max(0, len(label) - 1)
    cx = x + (w - total) // 2
    for i, rc in enumerate(rendered_chars):
        surf.blit(rc, (cx, y))
        cx += rc.get_width() + gap


def _draw_chip(
    surf: pygame.Surface,
    rect: pygame.Rect,
    label: str,
    color: tuple,
    font: pygame.font.Font,
    *,
    active: bool = False,
    dim: bool = False,
    extra_glyph: pygame.Surface | None = None,
    extra_gap: int = 6,
) -> None:
    """Generic chip primitive used by AI Setup, model-source, and inspector.

    - Inactive: panel bg + 2px border in `color`, label in `color`
    - Active: filled with `color`, label in TEXT_PRI
    - Dim: 70% opacity (for missing checkpoint chips)
    """
    bg = color if active else PANEL_BG
    fg = TEXT_PRI if active else color
    if dim:
        # Pre-blit to alpha surface so border + fill dim together.
        tmp = pygame.Surface(rect.size, pygame.SRCALPHA)
        local = pygame.Rect(0, 0, rect.w, rect.h)
        _rrect(tmp, bg, local, 8, 2, color)
        text = font.render(label, True, fg)
        extra_w = extra_glyph.get_width() + extra_gap if extra_glyph is not None else 0
        total = text.get_width() + extra_w
        tx = (rect.w - total) // 2
        tmp.blit(text, (tx, (rect.h - text.get_height()) // 2))
        if extra_glyph is not None:
            tmp.blit(extra_glyph, (tx + text.get_width() + extra_gap,
                                   (rect.h - extra_glyph.get_height()) // 2))
        tmp.set_alpha(int(255 * 0.7))
        surf.blit(tmp, rect.topleft)
        return

    _rrect(surf, bg, rect, 8, 2, color)
    text = font.render(label, True, fg)
    extra_w = extra_glyph.get_width() + extra_gap if extra_glyph is not None else 0
    total = text.get_width() + extra_w
    tx = rect.x + (rect.w - total) // 2
    surf.blit(text, (tx, rect.y + (rect.h - text.get_height()) // 2))
    if extra_glyph is not None:
        surf.blit(extra_glyph, (tx + text.get_width() + extra_gap,
                                rect.y + (rect.h - extra_glyph.get_height()) // 2))


def _draw_availability_dot(
    surf: pygame.Surface,
    cx: int, cy: int,
    *,
    exists: bool,
    is_loaded: bool = False,
    ticks: int | None = None,
) -> None:
    """Filled (exists) / hollow (missing) / pulse ring (loaded).

    Pulse formula from design handoff:
       alpha = 0.35 + 0.65 * (sin(ticks * 0.0045) + 1) / 2
       scale = 1.0 + 0.04 * sin(...)
    """
    import math
    r = 4
    if exists:
        pygame.draw.circle(surf, GOLD, (cx, cy), r)
    else:
        pygame.draw.circle(surf, BG, (cx, cy), r)
        pygame.draw.circle(surf, TEXT_MUT, (cx, cy), r, 1)
    if is_loaded:
        t = ticks if ticks is not None else pygame.time.get_ticks()
        s = math.sin(t * 0.0045)
        alpha = int(255 * (0.35 + 0.65 * (s + 1) / 2))
        scale = 1.0 + 0.04 * s
        rr = int((r + 4) * scale)
        ring = pygame.Surface((rr * 2 + 4, rr * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(ring, (*COL_X, alpha), (rr + 2, rr + 2), rr, 2)
        surf.blit(ring, (cx - rr - 2, cy - rr - 2))


def _draw_pill(
    surf: pygame.Surface,
    rect: pygame.Rect,
    segments: list[tuple[str, tuple]],
    font: pygame.font.Font,
    border_color: tuple,
    bg_color: tuple | None = None,
) -> None:
    """Pill-shaped multi-segment chip (rounded fully).

    `segments` is a list of (text, color) — segments rendered sequentially
    with a single space between, centered in the pill.
    """
    radius = rect.h // 2
    if bg_color is None:
        bg_color = PANEL_BG
    _rrect(surf, bg_color, rect, radius, 1, border_color)
    parts = [font.render(t, True, c) for t, c in segments]
    gap = 4
    total = sum(p.get_width() for p in parts) + gap * (len(parts) - 1)
    tx = rect.x + (rect.w - total) // 2
    for p in parts:
        surf.blit(p, (tx, rect.y + (rect.h - p.get_height()) // 2))
        tx += p.get_width() + gap


class _Flash:
    def __init__(self, pos: Position) -> None:
        self.pos = pos
        self._t  = pygame.time.get_ticks()

    def alpha(self) -> int:
        elapsed = pygame.time.get_ticks() - self._t
        return max(0, int(230 * (1 - elapsed / FLASH_MS)))

    def done(self) -> bool:
        return pygame.time.get_ticks() - self._t >= FLASH_MS


class _Btn:
    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        border: tuple = VIOLET,
        fg: tuple = TEXT_PRI,
        font: pygame.font.Font | None = None,
    ) -> None:
        self.rect   = rect
        self.text   = text
        self.border = border
        self.fg     = fg
        self.font   = font

    def draw(self, surf: pygame.Surface) -> None:
        hov = self.rect.collidepoint(pygame.mouse.get_pos())
        bg  = HOVER_BG if hov else PANEL_BG
        _rrect(surf, bg, self.rect, 8, 1, self.border)
        if self.font:
            lbl = self.font.render(self.text, True, self.fg)
            surf.blit(lbl, lbl.get_rect(center=self.rect.center))

    def hit(self, pos: tuple) -> bool:
        return self.rect.collidepoint(pos)


# ── Main GUI ──────────────────────────────────────────────────────────────────

class PygameGUI:
    """Pygame interface for Territory Capture."""

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Territory Capture")
        self.clock  = pygame.time.Clock()

        self.bs_index = BOARD_SIZES.index(DEFAULT_BS)
        self.bs       = DEFAULT_BS
        self.geo      = _calc_geometry(self.bs)

        config = BOARD_CONFIGS.get(self.bs, {})
        stones = config.get("stones_per_player", 10)
        self.game      = TerritoryCaptureGame(board_size=self.bs, stones_per_player=stones)
        self.result:   Optional[GameResult] = None
        self.last_move: Optional[Position]  = None
        self.flashes:  List[_Flash]         = []
        self.ter_alpha = 0
        self.status    = "Player X — place your first stone"

        self.mode_i  = 1   # default: Human vs AI
        self.diff_x  = 0   # X difficulty index (AI vs AI)
        self.diff_o  = 2   # O difficulty index (H vs AI / AI vs AI)
        self.show_rules  = False
        self.show_result = False
        self.show_advanced = False
        self.show_tournament = False
        self.review_mode = False
        self.history_scroll = 0   # scroll offset for move history strip
        self.ai_waiting  = False
        self.ai_at       = 0
        # Pending AI move + inspection data (computed before AI_DELAY_MS visual pause)
        self.ai_pending_action: Optional[Position] = None
        self.ai_inspection: Optional[dict] = None
        self.ai_inspection_player: Optional[str] = None

        # Cross-board model overrides per player (None = native to current board).
        # When set to 5/6/7, the AI for that player uses a model trained on a
        # different board size, wrapped in CrossBoardAgent.
        self.x_src_bs: Optional[int] = None
        self.o_src_bs: Optional[int] = None
        self._adv_buttons: List["_Btn"] = []  # populated when overlay opens

        self._init_fonts()
        self._init_agents()
        self._init_buttons()

    # ── Init ─────────────────────────────────────────────────────────────────

    def _init_fonts(self) -> None:
        def f(size: int, bold: bool = False) -> pygame.font.Font:
            return pygame.font.SysFont("helvetica", size, bold=bold)

        self.F_TITLE = f(34, True)
        self.F_H1    = f(18, True)
        self.F_H2    = f(14, True)
        self.F_BODY  = f(14)
        self.F_SMALL = f(12)
        self.F_SCORE = f(28, True)
        self.F_GROUP = f(9, True)   # CAPS group labels: MATCH / CONFIG / ADVANCED
        self.F_CHIP  = f(13, True)  # 36-tall chip labels (AI Setup, model chip)
        self.F_MICRO = f(10, True)  # 10px CAPS micro-text (status, "MODEL n×n →")
        self.F_MONO  = f(11, False) # mono-ish 11px for AI Inspector readouts

    def _init_agents(self) -> None:
        """Build the native agent set for the current board size.

        Cross-board agents (when source≠current board) are constructed lazily
        in ``_do_ai_move`` so the per-player overrides can change at any time
        without re-initializing the native agent cache.
        """
        hard_path = f"src/model_hard_{self.bs}x{self.bs}.pth"
        # 6x6 has a legacy unpostfixed name
        if self.bs == 6 and not Path(hard_path).exists() and Path("src/model_hard.pth").exists():
            hard_path = "src/model_hard.pth"
        self.agents = {
            DIFF_EASY: RandomAgent(),
            DIFF_MED:  AIAgent(board_size=self.bs),
            DIFF_HARD: AIAgent(board_size=self.bs, model_path=hard_path, num_simulations=100)
                       if Path(hard_path).exists()
                       else AIAgent(board_size=self.bs, num_simulations=100),
        }

    def _build_cross_agent(self, src_bs: int, difficulty_idx: int):
        """Return a CrossBoardAgent for the given source board + difficulty."""
        from experiments.cross_board import CrossBoardAgent
        if DIFFS[difficulty_idx] == DIFF_HARD:
            model_path = f"src/model_hard_{src_bs}x{src_bs}.pth"
            if src_bs == 6 and not Path(model_path).exists() and Path("src/model_hard.pth").exists():
                model_path = "src/model_hard.pth"
        else:
            model_path = f"src/model_{src_bs}x{src_bs}.pth"
            if src_bs == 6 and not Path(model_path).exists() and Path("src/model.pth").exists():
                model_path = "src/model.pth"
        return CrossBoardAgent(
            model_board_size=src_bs,
            play_board_size=self.bs,
            model_path=model_path,
        )

    def _init_buttons(self) -> None:
        """Footer B layout — three labelled clusters + an anchor.

        Pixel positions mirror ui_kits/desktop/Footer.jsx:
          - New Game anchor at x=12
          - Sep @ x=116
          - MATCH       (x=124, w=244): Mode (132) + Board (92), centered
          - Sep @ x=372
          - CONFIG      (x=380, w=204): Diff X (96) + Diff O (96), centered
          - Sep @ x=588
          - ADVANCED    (x=596, w=392): AI Setup (110) + Tournament (120) + Rules (92)
          - Info readout: bottom-right, second line
        Buttons are 42px tall starting at footer_y + 24.
        """
        g = self.geo
        bh = 42
        by = g["footer_y"] + 24  # 24px from footer top, leaves 8px for group label

        # Anchor
        self.btn_new = _Btn(pygame.Rect(12, by, 92, bh), "New Game",
                            COL_X, COL_X, self.F_H2)

        # MATCH cluster — group at x=124, w=244, content centred (Mode 132 + 6 gap + Board 92 = 230, pad 7 each side)
        self.btn_mode  = _Btn(pygame.Rect(131, by, 132, bh), self._mode_lbl(),
                              VIOLET, TEXT_PRI, self.F_H2)
        self.btn_board = _Btn(pygame.Rect(269, by,  92, bh), self._board_lbl(),
                              GOLD, GOLD, self.F_H2)

        # CONFIGURATION cluster — group at x=380, w=204 (Diff X 96 + 6 + Diff O 96 = 198, pad 3)
        self.btn_diff_x = _Btn(pygame.Rect(383, by, 96, bh), self._diff_x_lbl(),
                               COL_X, COL_X, self.F_H2)
        self.btn_diff_o = _Btn(pygame.Rect(485, by, 96, bh), self._diff_o_lbl(),
                               COL_O, COL_O, self.F_H2)

        # ADVANCED cluster — group at x=596, w=392 (AI Setup 110 + 6 + Tournament 120 + 6 + Rules 92 = 334, pad 29)
        self.btn_adv   = _Btn(pygame.Rect(625, by, 110, bh), "AI Setup",
                              GOLD, GOLD, self.F_H2)
        self.btn_tour  = _Btn(pygame.Rect(741, by, 120, bh), "Tournament",
                              GOLD, GOLD, self.F_H2)
        self.btn_rules = _Btn(pygame.Rect(867, by,  92, bh), "Rules",
                              VIOLET, TEXT_PRI, self.F_H2)

        self._btns = [self.btn_new, self.btn_mode, self.btn_board,
                      self.btn_diff_x, self.btn_diff_o,
                      self.btn_adv, self.btn_tour, self.btn_rules]

    def _mode_lbl(self) -> str:
        short = {MODE_HVH: "H vs H", MODE_HVAI: "H vs AI", MODE_AIVAI: "AI vs AI"}
        return f"Mode: {short[MODES[self.mode_i]]}"

    def _board_lbl(self) -> str:
        return f"Board: {self.bs}x{self.bs}"

    def _diff_x_lbl(self) -> str:
        # Cross-board override is now shown as a pill under the player name
        # in the side panel (see _draw_panel). Difficulty button stays clean.
        return f"X · {DIFFS[self.diff_x]}"

    def _diff_o_lbl(self) -> str:
        return f"O · {DIFFS[self.diff_o]}"

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        while True:
            now = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._click(event.pos)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    if (self.show_rules or self.show_result or
                            self.show_advanced or self.show_tournament):
                        self.show_rules = False
                        self.show_result = False
                        self.show_advanced = False
                        self.show_tournament = False

            if self.ai_waiting and now >= self.ai_at and not self.game.is_terminal():
                self.ai_waiting = False
                self._do_ai_move()

            if self.result and self.ter_alpha < 200:
                self.ter_alpha = min(200, self.ter_alpha + TER_SPEED)

            self.flashes = [fl for fl in self.flashes if not fl.done()]

            self._draw()
            pygame.display.flip()
            self.clock.tick(60)

    # ── Input ─────────────────────────────────────────────────────────────────

    def _click(self, pos: tuple) -> None:
        if self.show_tournament:
            self._click_tournament(pos)
            return
        if self.show_advanced:
            self._click_advanced(pos)
            return
        if self.show_rules:
            self.show_rules = False
            return
        if self.show_result:
            self._click_result(pos)
            return
        # In review mode, the footer is replaced by the history strip — route
        # footer-area clicks to the history handler.
        if self.review_mode and pos[1] >= self.geo["footer_y"]:
            self._click_history(pos)
            return

        if self.btn_new.hit(pos):
            self._reset()
            return
        if self.btn_mode.hit(pos):
            self.mode_i = (self.mode_i + 1) % len(MODES)
            self.btn_mode.text = self._mode_lbl()
            self._reset()
            return
        if self.btn_board.hit(pos):
            self.bs_index = (self.bs_index + 1) % len(BOARD_SIZES)
            self.bs = BOARD_SIZES[self.bs_index]
            self.geo = _calc_geometry(self.bs)
            self._init_agents()
            self._init_buttons()
            self._reset()
            return
        if self.btn_diff_x.hit(pos):
            self.diff_x = (self.diff_x + 1) % len(DIFFS)
            self.btn_diff_x.text = self._diff_x_lbl()
            self._reset()
            return
        if self.btn_diff_o.hit(pos):
            self.diff_o = (self.diff_o + 1) % len(DIFFS)
            self.btn_diff_o.text = self._diff_o_lbl()
            self._reset()
            return
        if self.btn_rules.hit(pos):
            self.show_rules = True
            return
        if self.btn_adv.hit(pos):
            self.show_advanced = True
            self._build_advanced_buttons()
            return
        if self.btn_tour.hit(pos):
            self.show_tournament = True
            self._load_tournament_data()
            return

        if not self.game.is_terminal() and not self._is_ai_turn():
            cell = self._px_to_cell(pos)
            if cell and self.game.is_legal_move(cell):
                self._notify_agents_of_move(cell)
                self._apply_move(cell)

    def _px_to_cell(self, pos: tuple) -> Optional[Position]:
        g = self.geo
        x, y = pos
        bx, bpx, cell, bs = g["board_x"], g["board_px"], g["cell"], g["bs"]
        if bx <= x < bx + bpx and BOARD_Y <= y < BOARD_Y + bpx:
            col = (x - bx) // cell
            row = (y - BOARD_Y) // cell
            if 0 <= row < bs and 0 <= col < bs:
                return (row, col)
        return None

    def _apply_move(self, pos: Position) -> None:
        self.game.apply_move(pos)
        self.last_move = pos

        caps = self.game.last_captured_positions
        for c in caps:
            self.flashes.append(_Flash(c))

        if self.game.is_terminal():
            self.result    = self.game.get_result()
            self.ter_alpha = 0
            w = self.result.winner
            self.status = f"Player {w} wins!" if w else "Draw!"
            self.show_result = True
        else:
            noun = "stone" if len(caps) == 1 else "stones"
            self.status = (
                f"{len(caps)} {noun} captured!"
                if caps
                else f"Player {self.game.current_player}'s turn"
            )
            if self._is_ai_turn():
                self.status    = f"AI thinking...  (Player {self.game.current_player})"
                self.ai_waiting = True
                self.ai_at     = pygame.time.get_ticks() + AI_DELAY_MS
                # Pre-compute the AI's move and inspection data NOW so the
                # Inspector tile can show what the AI is "thinking" during
                # the visual AI_DELAY_MS pause. The move itself isn't applied
                # until the timer fires.
                self._start_ai_turn()

    def _notify_agents_of_move(self, action: Position) -> None:
        for agent in self.agents.values():
            if hasattr(agent, "mcts"):
                agent.mcts.advance_to_action(action)

    def _do_ai_move(self) -> None:
        """Apply the AI move pre-computed at the start of the AI turn.

        MCTS is now run eagerly when the AI turn starts (see _start_ai_turn);
        this method just applies the cached move after the visual AI_DELAY_MS.
        """
        if self.ai_pending_action is None:
            self._start_ai_turn()  # fallback if for any reason we missed it
        action = self.ai_pending_action
        self.ai_pending_action = None
        if action is not None:
            self._apply_move(action)

    def _start_ai_turn(self) -> None:
        """Run MCTS / agent selection eagerly so the AI Inspector can display
        the model's reasoning during the AI_DELAY_MS visual pause."""
        player = self.game.current_player
        diff_idx = self.diff_x if player == PLAYER_X else self.diff_o
        src_bs   = self.x_src_bs if player == PLAYER_X else self.o_src_bs
        diff = DIFFS[diff_idx]

        if src_bs is not None and src_bs != self.bs and diff != DIFF_EASY:
            agent = self._build_cross_agent(src_bs, diff_idx)
        else:
            agent = self.agents[diff]

        action = agent.select_action(self.game.clone())
        self.ai_pending_action = action
        self.ai_inspection_player = player
        self.ai_inspection = self._extract_inspection(agent, diff, src_bs)

    def _extract_inspection(self, agent, diff: str, src_bs):
        """Pull MCTS root introspection (value, candidates, visit heatmap).

        Returns None for Random/Easy or non-MCTS agents (e.g. CrossBoardAgent
        which is pure NN).
        """
        if not hasattr(agent, "mcts") or agent.mcts.root is None:
            return None
        root = agent.mcts.root
        total_visits = sum(c.visit_count for c in root.children.values())
        if total_visits == 0:
            return None

        # Root value estimate (from current player's perspective)
        root_value = root.value_sum / max(1, root.visit_count)

        # Top-3 children by visit count
        items = sorted(
            root.children.items(),
            key=lambda kv: -kv[1].visit_count,
        )[:3]
        candidates = []
        for (r, c), node in items:
            n = node.visit_count
            p = node.base_prior
            q = node.value_sum / max(1, node.visit_count)
            candidates.append({
                "rc": (r, c),
                "move": coord_label(r, c),
                "n": n, "p": p, "q": q,
            })

        # Visit-count heatmap over the full board
        heat = {}
        max_n = max((c.visit_count for c in root.children.values()), default=1)
        for (r, c), node in root.children.items():
            heat[(r, c)] = node.visit_count / max_n if max_n > 0 else 0

        sims = total_visits
        return {
            "value": root_value,
            "candidates": candidates,
            "heat": heat,
            "sims": sims,
            "label": f"{diff}" + (f" · src {src_bs}×{src_bs}" if src_bs else ""),
        }

    def _reset(self) -> None:
        config = BOARD_CONFIGS.get(self.bs, {})
        stones = config.get("stones_per_player", 10)
        self.game = TerritoryCaptureGame(board_size=self.bs, stones_per_player=stones)
        self.result      = None
        self.last_move   = None
        self.flashes.clear()
        self.ter_alpha   = 0
        self.show_result = False
        self.ai_waiting  = False
        self.status      = "Player X — place your first stone"
        for agent in self.agents.values():
            if hasattr(agent, "reset_search_tree"):
                agent.reset_search_tree()

        if self._is_ai_turn():
            self.status    = "AI thinking...  (Player X)"
            self.ai_waiting = True
            self.ai_at     = pygame.time.get_ticks() + AI_DELAY_MS

    def _is_ai_turn(self) -> bool:
        mode = MODES[self.mode_i]
        if mode == MODE_AIVAI:
            return True
        if mode == MODE_HVAI:
            return self.game.current_player == PLAYER_O
        return False

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        g = self.geo
        self.screen.fill(BG)
        self._draw_header()
        self._draw_panel(PLAYER_X, g["l_panel_x"])
        self._draw_board()
        self._draw_panel(PLAYER_O, g["r_panel_x"])
        if self.review_mode and self.result:
            self._draw_history_strip()
        else:
            self._draw_footer()
        # In-game AI Inspector tile — shown only while AI is thinking
        # and no modal is open.
        if (self.ai_waiting and self.ai_inspection
                and not (self.show_tournament or self.show_advanced
                         or self.show_rules or self.show_result)):
            self._draw_inspector_tile()

        if self.show_tournament:
            self._draw_tournament_overlay()
        elif self.show_advanced:
            self._draw_advanced_overlay()
        elif self.show_rules:
            self._draw_rules_overlay()
        elif self.show_result and self.result:
            self._draw_result_overlay()

    # ── Header ────────────────────────────────────────────────────────────────

    def _draw_header(self) -> None:
        title = self.F_TITLE.render("Territory Capture", True, TEXT_PRI)
        self.screen.blit(title, title.get_rect(centerx=WIN_W // 2, y=14))

        status = self.F_BODY.render(self.status, True, TEXT_ACC)
        self.screen.blit(status, status.get_rect(centerx=WIN_W // 2, y=58))

        pygame.draw.line(self.screen, GRID_C, (20, 86), (WIN_W - 20, 86), 1)

    # ── Player panel ─────────────────────────────────────────────────────────

    def _draw_panel(self, player: str, px: int) -> None:
        g = self.geo
        pw, ph, py_top = g["panel_w"], g["panel_h"], g["panel_y"]
        color  = COL_X if player == PLAYER_X else COL_O
        rect   = pygame.Rect(px, py_top, pw, ph)
        border = tuple(c // 2 for c in color)
        _rrect(self.screen, PANEL_BG, rect, 12, 1, border)

        # Active glow
        is_current = (not self.game.is_terminal()
                      and self.game.current_player == player)
        if is_current:
            gsurf = pygame.Surface((pw, ph), pygame.SRCALPHA)
            pygame.draw.rect(gsurf, (*color, 14), (0, 0, pw, ph),
                             border_radius=12)
            self.screen.blit(gsurf, rect.topleft)

        cx = px + pw // 2
        y  = py_top + 22

        # Player name
        lbl = self.F_H1.render(f"Player  {player}", True, color)
        self.screen.blit(lbl, lbl.get_rect(centerx=cx, y=y))
        y += 28

        # Model-source chip — shown only when an AI override is active and
        # the source board ≠ the live play board (cross-board transfer).
        # Spec: PlayerPanel.jsx — pill at top:50, "MODEL n×n → m×m"
        src_bs = self.x_src_bs if player == PLAYER_X else self.o_src_bs
        if src_bs is not None and src_bs != self.bs:
            chip_h = 22
            chip_w = 150
            chip_rect = pygame.Rect(cx - chip_w // 2, y, chip_w, chip_h)
            _draw_pill(
                self.screen,
                chip_rect,
                segments=[
                    ("MODEL", TEXT_MUT),
                    (f"{src_bs}×{src_bs}", TEXT_PRI),
                    ("→", TEXT_MUT),
                    (f"{self.bs}×{self.bs}", GOLD),
                ],
                font=self.F_MICRO,
                border_color=color,
                bg_color=tuple(int(c * 0.18 + b * 0.82)
                               for c, b in zip(color, PANEL_BG)),
            )
            y += chip_h + 6
        else:
            y += 2  # small gap when no chip

        # Turn / status indicator
        mode = MODES[self.mode_i]
        ai_this = (mode == MODE_AIVAI) or (mode == MODE_HVAI and player == PLAYER_O)
        if self.result and self.result.winner == player:
            ind_txt, ind_col = "★  WINNER", GOLD
        elif self.result:
            ind_txt, ind_col = "─  FINISHED", TEXT_MUT
        elif is_current:
            ind_txt, ind_col = ("▶  THINKING" if ai_this else "▶  YOUR TURN"), color
        else:
            ind_txt, ind_col = "○  WAITING", TEXT_MUT

        ind = self.F_SMALL.render(ind_txt, True, ind_col)
        self.screen.blit(ind, ind.get_rect(centerx=cx, y=y))
        y += 28

        def divider() -> None:
            pygame.draw.line(
                self.screen,
                tuple(c // 4 for c in color),
                (px + 18, y),
                (px + self.geo["panel_w"] - 18, y),
                1,
            )

        divider()
        y += 18

        # ── Stones ────────────────────────────────────────────────────────────
        self.screen.blit(
            self.F_H2.render("Stones", True, TEXT_MUT),
            self.F_H2.render("Stones", True, TEXT_MUT).get_rect(centerx=cx, y=y),
        )
        y += 20
        stones = self.game.stones_placed[player]
        max_stones = self.game.stones_per_player
        sv = self.F_SCORE.render(f"{stones} / {max_stones}", True, color)
        self.screen.blit(sv, sv.get_rect(centerx=cx, y=y))
        y += 36

        # Progress bar
        bw = pw - 30
        bar = pygame.Rect(px + 15, y, bw, 7)
        pygame.draw.rect(self.screen, GRID_C, bar, border_radius=3)
        fw = int(bw * stones / max_stones) if max_stones > 0 else 0
        if fw > 0:
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(px + 15, y, fw, 7), border_radius=3)
        y += 22
        divider()
        y += 18

        # ── Captures ──────────────────────────────────────────────────────────
        self.screen.blit(
            self.F_H2.render("Captures", True, TEXT_MUT),
            self.F_H2.render("Captures", True, TEXT_MUT).get_rect(centerx=cx, y=y),
        )
        y += 20
        caps = self.game.captured_by[player]
        cv = self.F_SCORE.render(f"+{caps}", True, GOLD)
        self.screen.blit(cv, cv.get_rect(centerx=cx, y=y))
        y += 36
        divider()
        y += 18

        # ── Territory ─────────────────────────────────────────────────────────
        self.screen.blit(
            self.F_H2.render("Territory", True, TEXT_MUT),
            self.F_H2.render("Territory", True, TEXT_MUT).get_rect(centerx=cx, y=y),
        )
        y += 20

        if self.result:
            ter = self.result.territory_scores[player]
            tv  = self.F_SCORE.render(str(ter), True, color)
        else:
            tv  = self.F_SCORE.render("?", True, TEXT_MUT)
        self.screen.blit(tv, tv.get_rect(centerx=cx, y=y))
        y += 36

        if self.result:
            divider()
            y += 18
            self.screen.blit(
                self.F_H2.render("Total Score", True, TEXT_MUT),
                self.F_H2.render("Total Score", True, TEXT_MUT).get_rect(centerx=cx, y=y),
            )
            y += 20
            total   = self.result.scores[player]
            is_win  = self.result.winner == player
            t_color = GOLD if is_win else TEXT_PRI
            tot_v   = self.F_SCORE.render(str(total), True, t_color)
            self.screen.blit(tot_v, tot_v.get_rect(centerx=cx, y=y))

    # ── Board ─────────────────────────────────────────────────────────────────

    def _draw_board(self) -> None:
        g = self.geo
        bx, bpx, cell, bs = g["board_x"], g["board_px"], g["cell"], g["bs"]
        board_rect = pygame.Rect(
            bx - 14, BOARD_Y - 14,
            bpx + 28, bpx + 28,
        )
        _rrect(self.screen, PANEL_BG, board_rect, 14, 1, GRID_C)

        mouse = pygame.mouse.get_pos()

        for row in range(bs):
            for col in range(bs):
                cx = bx + col * cell + cell // 2
                cy = BOARD_Y + row * cell + cell // 2
                crect = pygame.Rect(bx + col * cell, BOARD_Y + row * cell, cell, cell)
                val   = self.game.board[row][col]
                is_last = self.last_move == (row, col)

                # Cell background
                bg = LAST_C if is_last else BOARD_BG
                pygame.draw.rect(self.screen, bg, crect)
                pygame.draw.rect(self.screen, GRID_C, crect, 1)

                # Territory overlay (fade-in after game ends)
                if self.result and val == EMPTY and self.ter_alpha > 0:
                    t_owner = self.result.territory_map.get((row, col))
                    tcol = (
                        TER_X if t_owner == PLAYER_X else
                        TER_O if t_owner == PLAYER_O else
                        TER_N  if t_owner == EMPTY   else None
                    )
                    if tcol:
                        ov = pygame.Surface((cell, cell), pygame.SRCALPHA)
                        ov.fill((*tcol, self.ter_alpha // 3))
                        self.screen.blit(ov, crect.topleft)

                # Capture flash
                for fl in self.flashes:
                    if fl.pos == (row, col):
                        a = fl.alpha()
                        fs = pygame.Surface((cell, cell), pygame.SRCALPHA)
                        fs.fill((*CAP_COL, a))
                        self.screen.blit(fs, crect.topleft)

                # Stone
                if val == PLAYER_X:
                    self._draw_stone(cx, cy, COL_X, is_last)
                elif val == PLAYER_O:
                    self._draw_stone(cx, cy, COL_O, is_last)
                elif self.result and val == EMPTY:
                    # Territory label
                    t_owner = self.result.territory_map.get((row, col))
                    if t_owner == PLAYER_X:
                        sym = self.F_H2.render("x", True, COL_X)
                    elif t_owner == PLAYER_O:
                        sym = self.F_H2.render("o", True, COL_O)
                    else:
                        sym = self.F_SMALL.render("·", True, TEXT_MUT)
                    self.screen.blit(sym, sym.get_rect(center=(cx, cy)))
                else:
                    # Hover preview for legal empty cells
                    if (crect.collidepoint(mouse)
                            and not self.game.is_terminal()
                            and not self._is_ai_turn()
                            and self.game.is_legal_move((row, col))):
                        hov = pygame.Surface((cell, cell), pygame.SRCALPHA)
                        hov.fill((255, 255, 255, 16))
                        self.screen.blit(hov, crect.topleft)
                        hint_col = COL_X if self.game.current_player == PLAYER_X else COL_O
                        pygame.draw.circle(self.screen, (*hint_col, 70), (cx, cy), 10)

    def _draw_stone(self, cx: int, cy: int, color: tuple, highlighted: bool) -> None:
        cell = self.geo["cell"]
        radius = cell // 2 - 8

        # Glow layers
        glow_a = 95 if highlighted else 55
        for i in range(4, 0, -1):
            r = radius + i * 4
            a = glow_a // i
            gsurf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(gsurf, (*color, a), (r, r), r)
            self.screen.blit(gsurf, (cx - r, cy - r))

        # Drop shadow
        pygame.draw.circle(self.screen, (0, 0, 0), (cx + 2, cy + 3), radius)

        # Dark border ring
        dark = tuple(max(0, c - 55) for c in color)
        pygame.draw.circle(self.screen, dark, (cx, cy), radius)

        # Main stone face
        pygame.draw.circle(self.screen, color, (cx, cy), radius - 2)

        # Specular highlight
        hl = tuple(min(255, c + 95) for c in color)
        pygame.draw.circle(self.screen, hl,
                           (cx - radius // 3, cy - radius // 3), radius // 4)

    # ── Footer ────────────────────────────────────────────────────────────────

    def _draw_footer(self) -> None:
        """Footer B — three labelled clusters with soft dividers.

        Layout from ui_kits/desktop/Footer.jsx:
          - FOOTER_BG fill + 1px GRID top border
          - Group labels (9px CAPS) at top:8 above each cluster
          - Soft separators at x=116 / 372 / 588, span y:14→70
          - Buttons at top:24, h=42
          - Info readout at bottom-right, second line (tabular nums)
        """
        g = self.geo
        fy, fh = g["footer_y"], g["footer_h"]
        pygame.draw.rect(self.screen, FOOTER_BG,
                         pygame.Rect(0, fy, WIN_W, fh))
        pygame.draw.line(self.screen, GRID_C, (0, fy), (WIN_W, fy), 1)

        # Group labels (top:8 relative to footer)
        label_y = fy + 8
        _draw_group_label(self.screen, "MATCH",          124, label_y, 244, self.F_GROUP)
        _draw_group_label(self.screen, "CONFIGURATION",  380, label_y, 204, self.F_GROUP)
        _draw_group_label(self.screen, "ADVANCED",       596, label_y, 392, self.F_GROUP)

        # Soft separators between clusters
        _soft_vline(self.screen, 116, fy + 14, 56)
        _soft_vline(self.screen, 372, fy + 14, 56)
        _soft_vline(self.screen, 588, fy + 14, 56)

        # Buttons
        for btn in self._btns:
            btn.draw(self.screen)

        # Info readout — second line, bottom-right, tabular nums
        max_moves = self.game.stones_per_player * 2
        info = (
            f"{self.bs}x{self.bs}  ·  Move {self.game.move_count}/{max_moves}  ·  "
            f"Capture bonus  X +{self.game.captured_by[PLAYER_X]}  ·  "
            f"O +{self.game.captured_by[PLAYER_O]}"
        )
        inf_surf = self.F_SMALL.render(info, True, TEXT_MUT)
        self.screen.blit(
            inf_surf,
            inf_surf.get_rect(right=WIN_W - 20, bottom=fy + fh - 14),
        )

    # ── Move history strip (review mode) ────────────────────────────────────

    def _draw_history_strip(self) -> None:
        """Replace the footer with a move history strip in review mode.

        Spec: ui_kits/desktop/MoveHistoryStrip.jsx
          - Anchored to footer area (replaces it)
          - Header micro: "MOVE HISTORY"  + counter on right
          - Horizontal chip row with scroll affordance (chevrons + edge fades)
          - Each chip: index + player dot + cell coord (+ capture badge if any)
          - Last move chip highlighted with last-cell bg + player-colored border
          - "Done" pill on the far right exits review mode
        """
        g = self.geo
        fy, fh = g["footer_y"], g["footer_h"]
        rect = pygame.Rect(0, fy, WIN_W, fh)
        pygame.draw.rect(self.screen, PANEL_BG, rect)
        pygame.draw.line(self.screen, GRID_C, (0, fy), (WIN_W, fy), 1)

        # Header row
        title = self.F_MICRO.render("MOVE HISTORY", True, TEXT_MUT)
        self.screen.blit(title, (20, fy + 14))
        moves = self.game.move_history
        total = self.game.stones_per_player * 2
        counter = self.F_SMALL.render(
            f"REVIEW MODE  ·  {len(moves)} / {total} MOVES", True, TEXT_MUT,
        )
        self.screen.blit(counter,
                         counter.get_rect(right=WIN_W - 130, y=fy + 14))

        # "Done" pill on the far right
        self._hist_btn_done = pygame.Rect(WIN_W - 110, fy + 10, 90, 30)
        _rrect(self.screen, PANEL_BG, self._hist_btn_done, 8, 1, GOLD)
        done_lbl = self.F_H2.render("Done", True, GOLD)
        self.screen.blit(done_lbl, done_lbl.get_rect(center=self._hist_btn_done.center))

        # Chips area
        chip_area_y = fy + 44
        chip_area_h = 56
        chip_area_left = 36
        chip_area_right = WIN_W - 36

        # Chevrons (no scrolling yet; place but inactive for now)
        self._hist_btn_left = pygame.Rect(8, chip_area_y, 28, chip_area_h)
        self._hist_btn_right = pygame.Rect(WIN_W - 36, chip_area_y, 28, chip_area_h)

        # Chip layout
        chip_h = 36
        chip_y = chip_area_y + (chip_area_h - chip_h) // 2
        gap = 8
        sep_w = 1
        # Pre-render chips and their widths
        chip_specs: list[tuple[str, str, int, bool]] = []  # (player, cell, idx, is_last)
        for i, (p, mv) in enumerate(moves):
            cell = coord_label(*mv)
            chip_specs.append((p, cell, i + 1, i == len(moves) - 1))

        # Width estimation: index(20) + dot(18) + cell(28) + padding(18) ≈ 84
        chip_w = 88

        # Sliding window of chips that fit
        available = chip_area_right - chip_area_left
        max_visible = max(1, (available + gap) // (chip_w + gap + sep_w + gap))
        scroll = min(self.history_scroll, max(0, len(chip_specs) - max_visible))
        visible = chip_specs[scroll:scroll + max_visible]

        # Chevron activity
        left_active = scroll > 0
        right_active = scroll + max_visible < len(chip_specs)

        # Left chevron + soft right border
        _rrect(self.screen, PANEL_BG, self._hist_btn_left, 6,
               border=1, border_color=LINE_SOFT if left_active else GRID_C)
        ch_l = self.F_H2.render("<", True, TEXT_PRI if left_active else TEXT_MUT)
        if not left_active:
            ch_l.set_alpha(int(255 * 0.4))
        self.screen.blit(ch_l, ch_l.get_rect(center=self._hist_btn_left.center))

        # Right chevron
        _rrect(self.screen, PANEL_BG, self._hist_btn_right, 6,
               border=1, border_color=LINE_SOFT if right_active else GRID_C)
        ch_r = self.F_H2.render(">", True, TEXT_PRI if right_active else TEXT_MUT)
        if not right_active:
            ch_r.set_alpha(int(255 * 0.4))
        self.screen.blit(ch_r, ch_r.get_rect(center=self._hist_btn_right.center))

        # Draw chips
        cx = chip_area_left + 8
        for (p, cell, idx, is_last) in visible:
            chip_color = COL_X if p == PLAYER_X else COL_O
            border_color = chip_color if is_last else GRID_C
            bg_color = LAST_C if is_last else BOARD_BG
            chip_rect = pygame.Rect(cx, chip_y, chip_w, chip_h)
            _rrect(self.screen, bg_color, chip_rect, 8, 1, border_color)
            # Index (e.g. "01")
            idx_lbl = self.F_MICRO.render(f"{idx:02d}", True, TEXT_MUT)
            self.screen.blit(idx_lbl,
                             (chip_rect.x + 8,
                              chip_rect.y + (chip_h - idx_lbl.get_height()) // 2))
            # Player dot
            dot_cx = chip_rect.x + 30
            dot_cy = chip_rect.y + chip_h // 2
            pygame.draw.circle(self.screen, chip_color, (dot_cx, dot_cy), 7)
            # darker outline
            dark = (chip_color[0] // 2, chip_color[1] // 2, chip_color[2] // 2)
            pygame.draw.circle(self.screen, dark, (dot_cx, dot_cy), 7, 1)
            # Cell label
            cell_lbl = self.F_CHIP.render(cell, True, TEXT_PRI)
            self.screen.blit(cell_lbl,
                             (chip_rect.x + 42,
                              chip_rect.y + (chip_h - cell_lbl.get_height()) // 2))
            cx += chip_w + gap
            # Soft separator between chips
            if (p, cell, idx, is_last) != visible[-1]:
                _soft_vline(self.screen, cx, chip_y + 8, chip_h - 16)
                cx += sep_w + gap

    def _click_history(self, pos: tuple) -> None:
        """Handle clicks on the move history strip (review mode)."""
        if hasattr(self, "_hist_btn_done") and self._hist_btn_done.collidepoint(pos):
            self.review_mode = False
            self._reset()
            return
        if hasattr(self, "_hist_btn_left") and self._hist_btn_left.collidepoint(pos):
            if self.history_scroll > 0:
                self.history_scroll -= 1
            return
        if hasattr(self, "_hist_btn_right") and self._hist_btn_right.collidepoint(pos):
            self.history_scroll += 1  # _draw_history_strip clamps
            return

    # ── Rules overlay ─────────────────────────────────────────────────────────

    # ── Advanced (AI Setup) overlay ──────────────────────────────────────────

    # ── AI Setup: Hybrid (pipeline lane + availability dots) ─────────────────

    def _hard_model_exists(self, src_bs: int) -> bool:
        """Check if the Hard checkpoint for a given source board is on disk."""
        if src_bs == 6:
            if Path("src/model_hard_6x6.pth").exists():
                return True
            if Path("src/model_hard.pth").exists():
                return True
            return False
        return Path(f"src/model_hard_{src_bs}x{src_bs}.pth").exists()

    def _build_advanced_buttons(self) -> None:
        """Lay out the AI Setup hybrid modal — two pipeline lanes per player.

        Spec: ui_kits/desktop/AISetupModal.jsx
          - Modal 720×460, centered, gold border
          - Lane per player with: Native/5×5/6×6/7×7 chips on left,
            "→" arrow, gold "Playing on N×N" chip on right
          - Availability dot on each non-native chip (filled/hollow/pulse)
          - Missing-checkpoint hint when selected chip has no file
          - Reset (left) and Apply (right) at bottom; Apply disabled when missing
        """
        pw, ph = 720, 460
        px = (WIN_W - pw) // 2
        py = (WIN_H - ph) // 2
        self._adv_overlay_rect = pygame.Rect(px, py, pw, ph)

        self._adv_chips: list = []
        labels = [("Native", None), ("5×5", 5), ("6×6", 6), ("7×7", 7)]

        # Two lanes — X at y_x, O at y_o. Within each lane:
        #   left:   "Trained on" label + 4 chips
        #   middle: "→" arrow
        #   right:  gold "Playing on" chip
        chip_h = 36
        chip_gap = 6
        chip_w = 78  # tighter so all four fit comfortably

        for lane_idx, player in enumerate([PLAYER_X, PLAYER_O]):
            color = COL_X if player == PLAYER_X else COL_O
            lane_top = py + 110 + lane_idx * 130

            # Chips area starts at px+170 (gives room for player label + native pill)
            chips_x = px + 170
            chips_y = lane_top + 30
            for i, (lbl, bs_val) in enumerate(labels):
                rect = pygame.Rect(chips_x + i * (chip_w + chip_gap),
                                   chips_y, chip_w, chip_h)
                self._adv_chips.append({
                    "rect": rect, "label": lbl, "bs_val": bs_val,
                    "player": player, "color": color, "native": bs_val is None,
                })

        # Reset / Apply buttons
        btn_h = 38
        btn_w = 132
        btn_y = py + ph - 18 - btn_h
        self._adv_btn_reset = pygame.Rect(px + 24, btn_y, btn_w, btn_h)
        self._adv_btn_apply = pygame.Rect(px + pw - 24 - btn_w, btn_y, btn_w, btn_h)

    def _adv_apply_disabled(self) -> bool:
        """Apply is disabled when any selection points to a missing checkpoint."""
        for src_bs in (self.x_src_bs, self.o_src_bs):
            if src_bs is not None and src_bs != self.bs:
                if not self._hard_model_exists(src_bs):
                    return True
        return False

    def _click_advanced(self, pos: tuple) -> None:
        """Handle clicks while the AI Setup overlay is visible."""
        if not self._adv_overlay_rect.collidepoint(pos):
            self.show_advanced = False
            return

        for chip in self._adv_chips:
            if chip["rect"].collidepoint(pos):
                if chip["player"] == PLAYER_X:
                    self.x_src_bs = chip["bs_val"]
                else:
                    self.o_src_bs = chip["bs_val"]
                return

        if self._adv_btn_reset.collidepoint(pos):
            self.x_src_bs = None
            self.o_src_bs = None
            return

        if self._adv_btn_apply.collidepoint(pos):
            if self._adv_apply_disabled():
                return  # silent reject; the disabled state is visible
            self.show_advanced = False
            self._reset()
            return

    def _draw_advanced_overlay(self) -> None:
        """AI Setup Hybrid — pipeline lane + availability dots per player."""
        # Dim scrim
        dim = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 170))
        self.screen.blit(dim, (0, 0))

        rect = self._adv_overlay_rect
        _rrect(self.screen, PANEL_BG, rect, 16, 2, GOLD)
        cx = WIN_W // 2

        # Title
        title = self.F_H1.render("AI Setup — Transfer Pipeline", True, TEXT_PRI)
        self.screen.blit(title, title.get_rect(centerx=cx, y=rect.y + 22))

        # Legend line (dot semantics)
        legend_parts = [
            ("Dot per chip:", TEXT_MUT),
            ("filled = available", GOLD),
            ("·", TEXT_MUT),
            ("hollow = missing", TEXT_MUT),
            ("·", TEXT_MUT),
            ("ring = currently loaded", COL_X),
        ]
        # Render legend manually with gaps
        ly = rect.y + 56
        rendered = [(self.F_SMALL.render(t, True, c), c) for t, c in legend_parts]
        total_w = sum(s.get_width() for s, _ in rendered) + 8 * (len(rendered) - 1)
        lx = cx - total_w // 2
        for surf, _ in rendered:
            self.screen.blit(surf, (lx, ly))
            lx += surf.get_width() + 8

        ticks = pygame.time.get_ticks()

        # Two lanes
        for lane_idx, player in enumerate([PLAYER_X, PLAYER_O]):
            color = COL_X if player == PLAYER_X else COL_O
            color_rgb = color  # tuple alias
            src_bs = self.x_src_bs if player == PLAYER_X else self.o_src_bs
            loaded = src_bs  # for now, loaded == currently selected
            is_cross = src_bs is not None and src_bs != self.bs

            lane_top = rect.y + 110 + lane_idx * 130

            # Divider between lanes
            if lane_idx == 1:
                _soft_hline(self.screen, rect.x + 24, lane_top - 14,
                            rect.w - 48)

            # Player label (left side)
            p_lbl = self.F_H2.render(f"Player {player}", True, color)
            self.screen.blit(p_lbl, (rect.x + 24, lane_top + 4))

            # Mode pill: "Native · MCTS" or "Transfer · no MCTS"
            mode_text = "TRANSFER · NO MCTS" if is_cross else "NATIVE · MCTS"
            mode_color = color if is_cross else TEXT_MUT
            mode_bg = (
                tuple(int(c * 0.15 + b * 0.85) for c, b in zip(color, PANEL_BG))
                if is_cross else PANEL_BG
            )
            mode_lbl_surf = self.F_MICRO.render(mode_text, True, mode_color)
            mode_pill_w = mode_lbl_surf.get_width() + 14
            mode_pill_rect = pygame.Rect(rect.x + 24, lane_top + 24,
                                         mode_pill_w, 18)
            _rrect(self.screen, mode_bg, mode_pill_rect, 4, 1, mode_color)
            self.screen.blit(mode_lbl_surf,
                             mode_lbl_surf.get_rect(center=mode_pill_rect.center))

            # "TRAINED ON" micro label above chips
            train_lbl = self.F_MICRO.render("TRAINED ON", True, TEXT_MUT)
            self.screen.blit(train_lbl, (rect.x + 170, lane_top + 14))

            # Chips for this player
            for chip in self._adv_chips:
                if chip["player"] != player:
                    continue
                active = src_bs == chip["bs_val"]
                exists = chip["native"] or self._hard_model_exists(chip["bs_val"])
                # Build availability dot if not native
                dot_surface = None
                if not chip["native"]:
                    is_loaded = active and not chip["native"]
                    dot_surface = pygame.Surface((20, 20), pygame.SRCALPHA)
                    _draw_availability_dot(dot_surface, 10, 10,
                                           exists=exists, is_loaded=is_loaded,
                                           ticks=ticks)
                _draw_chip(
                    self.screen, chip["rect"], chip["label"], color,
                    self.F_CHIP, active=active,
                    dim=not chip["native"] and not exists,
                    extra_glyph=dot_surface, extra_gap=6,
                )

            # → arrow
            arrow_x = rect.x + 170 + 4 * (78 + 6) - 2
            arrow_lbl = self.F_H1.render("→", True,
                                          color if is_cross else TEXT_MUT)
            self.screen.blit(arrow_lbl, (arrow_x, lane_top + 36))

            # "PLAYING ON" micro label above target chip
            target_lbl = self.F_MICRO.render("PLAYING ON", True, TEXT_MUT)
            self.screen.blit(target_lbl, (arrow_x + 40, lane_top + 14))

            # Gold target chip
            target_rect = pygame.Rect(arrow_x + 40, lane_top + 30, 64, 36)
            _rrect(self.screen,
                   tuple(int(c * 0.10 + b * 0.90) for c, b in zip(GOLD, PANEL_BG)),
                   target_rect, 8, 2, GOLD)
            t_surf = self.F_CHIP.render(f"{self.bs}×{self.bs}", True, GOLD)
            self.screen.blit(t_surf, t_surf.get_rect(center=target_rect.center))

            # Missing-checkpoint hint
            if is_cross and not self._hard_model_exists(src_bs):
                hint_text = f"checkpoint missing:  src/model_hard_{src_bs}x{src_bs}.pth"
                hint_surf = self.F_SMALL.render(hint_text, True, COL_O)
                hint_pad = 8
                hint_rect = pygame.Rect(rect.x + 24, lane_top + 76,
                                         hint_surf.get_width() + 2 * hint_pad, 22)
                _rrect(self.screen,
                       tuple(int(c * 0.10 + b * 0.90) for c, b in zip(COL_O, PANEL_BG)),
                       hint_rect, 6, 1, COL_O)
                self.screen.blit(hint_surf,
                                 (hint_rect.x + hint_pad,
                                  hint_rect.y + (hint_rect.h - hint_surf.get_height()) // 2))

        # Action buttons (Reset left, Apply right)
        apply_disabled = self._adv_apply_disabled()

        # Reset — outlined
        _rrect(self.screen, PANEL_BG, self._adv_btn_reset, 8, 1, TEXT_MUT)
        r_lbl = self.F_H2.render("Reset", True, TEXT_PRI)
        self.screen.blit(r_lbl, r_lbl.get_rect(center=self._adv_btn_reset.center))

        # Apply — gold outline when enabled, muted when disabled
        apply_border = TEXT_MUT if apply_disabled else GOLD
        apply_fg = TEXT_MUT if apply_disabled else GOLD
        _rrect(self.screen, PANEL_BG, self._adv_btn_apply, 8, 1, apply_border)
        a_lbl = self.F_H2.render("Apply", True, apply_fg)
        if apply_disabled:
            # 55% opacity dim
            a_surf = pygame.Surface(self._adv_btn_apply.size, pygame.SRCALPHA)
            _rrect(a_surf, PANEL_BG, pygame.Rect(0, 0, *self._adv_btn_apply.size),
                   8, 1, apply_border)
            a_surf.blit(a_lbl, a_lbl.get_rect(center=(self._adv_btn_apply.w // 2,
                                                       self._adv_btn_apply.h // 2)))
            a_surf.set_alpha(int(255 * 0.55))
            self.screen.blit(a_surf, self._adv_btn_apply.topleft)
        else:
            self.screen.blit(a_lbl, a_lbl.get_rect(center=self._adv_btn_apply.center))

    # ── AI Inspector overlay tile ────────────────────────────────────────────

    def _draw_inspector_tile(self) -> None:
        """Show MCTS introspection (value, top-3, visit heatmap) while AI thinks.

        Spec: ui_kits/desktop/InspectorTile.jsx (380×400). Tile anchors next to
        the active player's panel — arrow on the side that faces the panel.
        Hidden on human turns and when modals are open.
        """
        ins = self.ai_inspection
        player = self.ai_inspection_player
        if ins is None or player is None:
            return

        color = COL_X if player == PLAYER_X else COL_O
        g = self.geo
        # Anchor: place beside the active player's panel
        tw, th = 380, 400
        ty = g["panel_y"] + 4
        if player == PLAYER_X:
            tx = g["l_panel_x"] + g["panel_w"] + 6
            side = "left"   # arrow on left edge points to X panel
        else:
            tx = g["r_panel_x"] - tw - 6
            side = "right"

        # Clamp inside window
        tx = max(8, min(WIN_W - tw - 8, tx))

        rect = pygame.Rect(tx, ty, tw, th)

        # Soft outer glow halo (overlapping alpha rects approximating shadow)
        for layer, a in [(4, 36), (10, 18), (18, 10)]:
            halo = pygame.Surface((tw + 2 * layer, th + 2 * layer), pygame.SRCALPHA)
            pygame.draw.rect(halo, (*color, a), halo.get_rect(), border_radius=14)
            self.screen.blit(halo, (tx - layer, ty - layer))

        _rrect(self.screen, PANEL_BG, rect, 12, 2, color)

        # Side arrow (small diamond pointing toward player panel)
        if side == "left":
            arrow_pts = [(tx - 6, ty + 28),
                         (tx + 2, ty + 22),
                         (tx + 2, ty + 36)]
        else:
            arrow_pts = [(tx + tw + 6, ty + 28),
                         (tx + tw - 2, ty + 22),
                         (tx + tw - 2, ty + 36)]
        pygame.draw.polygon(self.screen, color, arrow_pts)

        # ── Header row: title + sims counter
        title_micro = self.F_MICRO.render("AI INSPECTOR", True, TEXT_MUT)
        self.screen.blit(title_micro, (tx + 18, ty + 16))
        player_lbl = self.F_H1.render(
            f"▶ Player {player} · {ins['label']}", True, color,
        )
        self.screen.blit(player_lbl, (tx + 18, ty + 30))

        sims_lbl = self.F_SMALL.render(f"{ins['sims']} sims", True, TEXT_MUT)
        self.screen.blit(sims_lbl, sims_lbl.get_rect(right=tx + tw - 18, y=ty + 18))

        # ── Value estimate bar (X favoured = LEFT, O favoured = RIGHT)
        ve_y = ty + 78
        ve_micro = self.F_MICRO.render("VALUE ESTIMATE", True, TEXT_MUT)
        self.screen.blit(ve_micro, (tx + 18, ve_y))
        bar_rect = pygame.Rect(tx + 18, ve_y + 14, tw - 36, 18)
        _rrect(self.screen, GRID_C, bar_rect, 4)
        mid_x = bar_rect.x + bar_rect.w // 2
        pygame.draw.line(self.screen, TEXT_MUT, (mid_x, bar_rect.y),
                         (mid_x, bar_rect.y + bar_rect.h), 1)

        # MCTS returns value from CURRENT player's perspective.
        # Convert to X's perspective so the layout is consistent:
        #   x_val > 0 → X favoured → fill LEFT (cyan)
        #   x_val < 0 → O favoured → fill RIGHT (pink)
        val = max(-1.0, min(1.0, ins["value"]))
        x_val = val if player == PLAYER_X else -val
        if x_val >= 0:
            fill_w = int(bar_rect.w / 2 * x_val)
            fill_rect = pygame.Rect(mid_x - fill_w, bar_rect.y + 1,
                                    fill_w, bar_rect.h - 2)
            pygame.draw.rect(self.screen, COL_X, fill_rect, border_radius=4)
        else:
            fill_w = int(bar_rect.w / 2 * (-x_val))
            fill_rect = pygame.Rect(mid_x, bar_rect.y + 1,
                                    fill_w, bar_rect.h - 2)
            pygame.draw.rect(self.screen, COL_O, fill_rect, border_radius=4)

        # Labels under bar — X favoured LEFT, O favoured RIGHT, x_val in middle
        left_lbl  = self.F_SMALL.render("X favoured", True, TEXT_MUT)
        right_lbl = self.F_SMALL.render("O favoured", True, TEXT_MUT)
        mid_color = COL_X if x_val >= 0 else COL_O
        mid_lbl   = self.F_SMALL.render(f"{x_val:+.2f}", True, mid_color)
        self.screen.blit(left_lbl, (bar_rect.x, bar_rect.y + 22))
        self.screen.blit(right_lbl, right_lbl.get_rect(right=bar_rect.right,
                                                        y=bar_rect.y + 22))
        self.screen.blit(mid_lbl, mid_lbl.get_rect(centerx=mid_x,
                                                    y=bar_rect.y + 22))

        # ── Top candidates
        tc_y = ty + 148
        tc_micro = self.F_MICRO.render("TOP CANDIDATE MOVES", True, TEXT_MUT)
        self.screen.blit(tc_micro, (tx + 18, tc_y))

        candidates = ins["candidates"][:3]
        max_n = max((c["n"] for c in candidates), default=1)
        for i, c in enumerate(candidates):
            row_y = tc_y + 16 + i * 26
            if i > 0:
                _soft_hline(self.screen, tx + 18, row_y - 4, tw - 36)
            # Move label (a1, b2, ...)
            m_lbl = self.F_CHIP.render(c["move"], True, color)
            self.screen.blit(m_lbl, (tx + 18, row_y + 3))
            # Visit bar
            bar_left = tx + 18 + 32
            bar_w = tw - 36 - 32 - 100
            n_bar = pygame.Rect(bar_left, row_y + 4, bar_w, 14)
            _rrect(self.screen, GRID_C, n_bar, 3)
            fill_w = int(bar_w * c["n"] / max_n)
            n_fill_surf = pygame.Surface((max(1, fill_w), 14), pygame.SRCALPHA)
            pygame.draw.rect(n_fill_surf, (*color, 115),
                             pygame.Rect(0, 0, fill_w, 14), border_radius=3)
            self.screen.blit(n_fill_surf, (bar_left, row_y + 4))
            # n label inside bar
            n_lbl = self.F_MICRO.render(f"n {c['n']}", True, TEXT_PRI)
            self.screen.blit(n_lbl, (bar_left + 4, row_y + 7))
            # p label
            p_lbl = self.F_MICRO.render(f"p {c['p']:.2f}", True, TEXT_MUT)
            self.screen.blit(p_lbl, p_lbl.get_rect(
                right=tx + tw - 18 - 50, y=row_y + 7,
            ))
            # Q label
            q_color = color if c["q"] >= 0 else (COL_O if player == PLAYER_X else COL_X)
            q_lbl = self.F_MICRO.render(f"Q {c['q']:+.2f}", True, q_color)
            self.screen.blit(q_lbl, q_lbl.get_rect(
                right=tx + tw - 18, y=row_y + 7,
            ))

        # ── Visit-count heatmap (16-18px cells, centered horizontally)
        hm_y = ty + 240
        hm_micro = self.F_MICRO.render("VISIT-COUNT HEATMAP", True, TEXT_MUT)
        self.screen.blit(hm_micro, (tx + 18, hm_y))

        bs = self.bs
        # Cap cell size at 18px; ensure grid fits inside tile padding
        max_grid_w = tw - 36
        max_grid_h = th - (hm_y - ty) - 16 - 18  # below micro label, above tile bottom
        cell_w = min(18, max_grid_w // bs, max_grid_h // bs)
        grid_w = cell_w * bs
        # Center horizontally
        hm_x = tx + (tw - grid_w) // 2
        hm_top = hm_y + 18

        for r in range(bs):
            for c in range(bs):
                cx_ = hm_x + c * cell_w
                cy_ = hm_top + r * cell_w
                cell_rect = pygame.Rect(cx_, cy_, cell_w - 1, cell_w - 1)
                h = ins["heat"].get((r, c), 0.0)
                if h > 0:
                    alpha = int(255 * (0.20 + h * 0.55))
                    cs = pygame.Surface(cell_rect.size, pygame.SRCALPHA)
                    pygame.draw.rect(cs, (*color, alpha),
                                     pygame.Rect(0, 0, *cell_rect.size),
                                     border_radius=2)
                    self.screen.blit(cs, cell_rect.topleft)
                else:
                    _rrect(self.screen, BOARD_BG, cell_rect, 2)
                pygame.draw.rect(self.screen, GRID_C, cell_rect, 1, border_radius=2)

    # ── Tournament viewer ────────────────────────────────────────────────────

    def _load_tournament_data(self, path: str | None = None) -> None:
        """Load and aggregate tournament results from JSON.

        Builds two structures:
          - within: per-board { row: opponent → win_rate }
          - cross:  3×3 matrix of cross-board hard performance
          - top3:   list of top performers
        Stored on self._tour_data for the overlay to render.
        """
        import json
        if path is None:
            path = "results/tournament_final.json"
        self._tour_path = path
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self._tour_data = None
            return

        # Within-board: one matrix per board size, rows=row agent, cols=opponent
        # Use 6×6 (most representative) as the primary view; show others as tabs/labels.
        primary_bs = self.bs if self.bs in (5, 6, 7) else 6
        within_rows = ["Random", "Heuristic", "Minimax", "Medium", "Hard"]
        within_matrix = [[None] * len(within_rows) for _ in range(len(within_rows))]
        for r in data:
            if r["board_size"] != primary_bs:
                continue
            ra, oa = r["x_agent"], r["o_agent"]
            if ra in within_rows and oa in within_rows:
                ri = within_rows.index(ra)
                ci = within_rows.index(oa)
                games = max(1, r["games"])
                wr = r["x_wins"] / games
                within_matrix[ri][ci] = wr
                # symmetric counterpart (oa from oa's perspective)
                if within_matrix[ci][ri] is None:
                    within_matrix[ci][ri] = r["o_wins"] / games

        # Cross-board: rows = trained-on board (5/6/7), cols = play-on board.
        # If cross-board entries exist (agent names containing "→"), use them;
        # otherwise the diagonal is the within-board hard score.
        cross_rows = [5, 6, 7]
        cross_matrix = [[None] * 3 for _ in range(3)]
        for r in data:
            if "→" in r["x_agent"]:
                # e.g. "Hard-5→6"  vs  "Hard-6(native)"
                try:
                    src = int(r["x_agent"].split("-")[1].split("→")[0])
                    dst = int(r["x_agent"].split("→")[1])
                    if src in cross_rows and dst in cross_rows:
                        cross_matrix[cross_rows.index(src)][cross_rows.index(dst)] = (
                            r["x_wins"] / max(1, r["games"])
                        )
                except (ValueError, IndexError):
                    pass
            elif r["x_agent"] == "Hard" and r["o_agent"] == "Medium":
                # diagonal — Hard's win rate on its own board
                bs = r["board_size"]
                if bs in cross_rows:
                    i = cross_rows.index(bs)
                    cross_matrix[i][i] = r["x_wins"] / max(1, r["games"])

        # Top performers — pick top 3 by total wins across all matches
        scores = []
        for r in data:
            scores.append((r["x_agent"], r["board_size"], r["x_wins"], r["games"]))
            scores.append((r["o_agent"], r["board_size"], r["o_wins"], r["games"]))
        agg: dict = {}
        for name, bs, w, g in scores:
            key = f"{name} · {bs}×{bs}"
            agg.setdefault(key, [0, 0])
            agg[key][0] += w
            agg[key][1] += g
        ranked = sorted(agg.items(), key=lambda kv: -kv[1][0])[:3]

        self._tour_data = {
            "primary_bs": primary_bs,
            "within_rows": within_rows,
            "within_matrix": within_matrix,
            "cross_rows": cross_rows,
            "cross_matrix": cross_matrix,
            "top3": [(name, wins, games) for name, (wins, games) in ranked],
            "n_entries": len(data),
        }

    def _tour_tint(self, v: float | None) -> tuple:
        """Heatmap cell color for a win rate."""
        if v is None:
            return PANEL_BG
        if abs(v - 0.5) < 0.01:
            return (158, 180, 215, 90)  # neutral grey
        t = min(1.0, abs(v - 0.5) * 2)
        if v > 0.5:
            alpha = int(255 * (0.18 + t * 0.65))
            return (*COL_X, alpha)
        else:
            alpha = int(255 * (0.18 + t * 0.65))
            return (*COL_O, alpha)

    def _click_tournament(self, pos: tuple) -> None:
        """Click handler for tournament viewer."""
        # Outside the modal → close
        if not self._tour_rect.collidepoint(pos):
            self.show_tournament = False
            return
        # Reload button
        if hasattr(self, "_tour_btn_reload") and self._tour_btn_reload.collidepoint(pos):
            self._load_tournament_data(self._tour_path)
            return
        # Cycle primary board view
        if hasattr(self, "_tour_btn_board") and self._tour_btn_board.collidepoint(pos):
            if self._tour_data:
                cur = self._tour_data["primary_bs"]
                nxt = {5: 6, 6: 7, 7: 5}[cur]
                self._tour_data["primary_bs"] = nxt
                # Re-aggregate primary matrix for new board
                self._load_tournament_data(self._tour_path)
                # carry over the new primary_bs
                if self._tour_data:
                    self._tour_data["primary_bs"] = nxt
                    self._recompute_within(nxt)
            return

    def _recompute_within(self, bs: int) -> None:
        """Re-aggregate the within-board matrix for a different primary board."""
        import json
        try:
            data = json.loads(Path(self._tour_path).read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return
        rows = self._tour_data["within_rows"]
        matrix = [[None] * len(rows) for _ in range(len(rows))]
        for r in data:
            if r["board_size"] != bs:
                continue
            ra, oa = r["x_agent"], r["o_agent"]
            if ra in rows and oa in rows:
                ri = rows.index(ra)
                ci = rows.index(oa)
                games = max(1, r["games"])
                matrix[ri][ci] = r["x_wins"] / games
                if matrix[ci][ri] is None:
                    matrix[ci][ri] = r["o_wins"] / games
        self._tour_data["within_matrix"] = matrix

    def _draw_tournament_overlay(self) -> None:
        """Tournament viewer — pre-computed JSON heatmaps + top performers.

        Spec: ui_kits/desktop/TournamentModal.jsx
          - 760×560 modal, centered, gold border
          - Title left + meta right
          - JSON path strip (path + reload + load button)
          - Two heatmaps side-by-side (within / cross)
          - Gradient legend
          - Top performers list
        """
        # Dim scrim
        dim = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 170))
        self.screen.blit(dim, (0, 0))

        pw, ph = 760, 560
        px = (WIN_W - pw) // 2
        py = (WIN_H - ph) // 2
        self._tour_rect = pygame.Rect(px, py, pw, ph)
        _rrect(self.screen, PANEL_BG, self._tour_rect, 16, 2, GOLD)

        # Title + meta
        title = self.F_H1.render("Tournament Viewer", True, TEXT_PRI)
        self.screen.blit(title, (px + 24, py + 24))
        meta_text = (
            f"{self._tour_data['n_entries']} matches  ·  press R to reload"
            if self._tour_data else "no data — run experiments.tournament first"
        )
        meta = self.F_SMALL.render(meta_text, True, TEXT_MUT)
        self.screen.blit(meta, meta.get_rect(right=px + pw - 24, y=py + 28))

        # JSON path strip
        strip_y = py + 60
        strip_rect = pygame.Rect(px + 24, strip_y, pw - 48, 32)
        _rrect(self.screen, BG, strip_rect, 8, 1, GRID_C)
        json_lbl = self.F_MICRO.render("JSON", True, TEXT_MUT)
        self.screen.blit(json_lbl, (strip_rect.x + 10,
                                     strip_rect.y + (strip_rect.h - json_lbl.get_height()) // 2))
        path_text = getattr(self, "_tour_path", "results/tournament_colab_v2.json")
        path_lbl = self.F_SMALL.render(path_text, True, TEXT_PRI)
        self.screen.blit(path_lbl, (strip_rect.x + 50,
                                      strip_rect.y + (strip_rect.h - path_lbl.get_height()) // 2))
        # Reload button (right end of strip)
        self._tour_btn_reload = pygame.Rect(strip_rect.right - 36, strip_y + 2, 28, 28)
        _rrect(self.screen, PANEL_BG, self._tour_btn_reload, 6, 1, GRID_C)
        rel_lbl = self.F_H2.render("↻", True, TEXT_MUT)
        self.screen.blit(rel_lbl, rel_lbl.get_rect(center=self._tour_btn_reload.center))

        if self._tour_data is None:
            # Empty state
            empty = self.F_BODY.render(
                "Run `python -m experiments.tournament --mode within --boards 5 6 7 "
                "--games 30 --include-baselines` first.",
                True, TEXT_MUT,
            )
            self.screen.blit(empty, empty.get_rect(centerx=WIN_W // 2, y=py + 280))
            return

        # ── Two heatmaps ──
        # Layout: left half = within, right half = cross
        hm_top = py + 110
        hm_height = 240
        hm_w = (pw - 24 * 2 - 16) // 2

        # WITHIN heatmap
        self._draw_heatmap(
            x=px + 24, y=hm_top, w=hm_w, h=hm_height,
            title=f"WITHIN-BOARD · {self._tour_data['primary_bs']}×{self._tour_data['primary_bs']}",
            rows=self._tour_data["within_rows"],
            cols=self._tour_data["within_rows"],
            matrix=self._tour_data["within_matrix"],
            small_labels=True,
        )

        # CROSS heatmap
        self._draw_heatmap(
            x=px + 24 + hm_w + 16, y=hm_top, w=hm_w, h=hm_height,
            title="CROSS-BOARD · HARD MODEL",
            rows=[f"{b}×{b} model" for b in self._tour_data["cross_rows"]],
            cols=[f"plays {b}×{b}" for b in self._tour_data["cross_rows"]],
            matrix=self._tour_data["cross_matrix"],
            small_labels=False,
        )

        # Cycle button under within heatmap
        self._tour_btn_board = pygame.Rect(px + 24, hm_top + hm_height + 6, 120, 24)
        _rrect(self.screen, PANEL_BG, self._tour_btn_board, 6, 1, GRID_C)
        cb_lbl = self.F_SMALL.render(
            f"Cycle board  ({self._tour_data['primary_bs']}×{self._tour_data['primary_bs']})",
            True, TEXT_MUT,
        )
        self.screen.blit(cb_lbl, cb_lbl.get_rect(center=self._tour_btn_board.center))

        # Gradient legend
        leg_top = hm_top + hm_height + 50
        leg_rect = pygame.Rect(px + 100, leg_top, pw - 200, 8)
        for i in range(leg_rect.w):
            t = i / leg_rect.w  # 0 → 1
            # left = pink, middle = neutral, right = cyan
            if t < 0.5:
                fade = (1 - t / 0.5)  # 1 → 0
                color = (
                    int(COL_O[0] * fade + 158 * (1 - fade)),
                    int(COL_O[1] * fade + 180 * (1 - fade)),
                    int(COL_O[2] * fade + 215 * (1 - fade)),
                )
            else:
                fade = (t - 0.5) / 0.5  # 0 → 1
                color = (
                    int(158 * (1 - fade) + COL_X[0] * fade),
                    int(180 * (1 - fade) + COL_X[1] * fade),
                    int(215 * (1 - fade) + COL_X[2] * fade),
                )
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(leg_rect.x + i, leg_rect.y, 1, leg_rect.h))
        # Legend labels
        loss_lbl = self.F_MICRO.render("ROW LOSES", True, COL_O)
        win_lbl = self.F_MICRO.render("ROW WINS", True, COL_X)
        self.screen.blit(loss_lbl, (px + 30, leg_top - 2))
        self.screen.blit(win_lbl,
                         win_lbl.get_rect(right=px + pw - 24, y=leg_top - 2))

        # Top performers list
        tp_top = leg_top + 30
        tp_lbl = self.F_MICRO.render("TOP PERFORMERS · OVERALL", True, TEXT_MUT)
        self.screen.blit(tp_lbl, (px + 24, tp_top))
        for i, (name, wins, games) in enumerate(self._tour_data["top3"]):
            row_y = tp_top + 22 + i * 22
            if i > 0:
                _soft_hline(self.screen, px + 24, row_y - 4, pw - 48)
            colors = [COL_X, COL_O, TEXT_PRI]
            n_lbl = self.F_BODY.render(name, True, colors[i % 3])
            s_lbl = self.F_BODY.render(f"{wins} / {games}", True, TEXT_MUT)
            self.screen.blit(n_lbl, (px + 24, row_y))
            self.screen.blit(s_lbl, s_lbl.get_rect(right=px + pw - 24, y=row_y))

    def _draw_heatmap(
        self, x: int, y: int, w: int, h: int,
        title: str, rows: list, cols: list,
        matrix: list, small_labels: bool,
    ) -> None:
        """Render a single heatmap card."""
        rect = pygame.Rect(x, y, w, h)
        _rrect(self.screen, BG, rect, 10, 1, GRID_C)

        # Title
        t_lbl = self.F_MICRO.render(title, True, TEXT_MUT)
        self.screen.blit(t_lbl, (x + 12, y + 10))

        n = len(rows)
        cell_size = min((w - 90) // n, (h - 60) // n)
        grid_x = x + w - cell_size * n - 10
        grid_y = y + 36

        # Column labels
        col_font = self.F_MICRO if small_labels else self.F_SMALL
        for ci, c in enumerate(cols):
            c_lbl = col_font.render(c, True, TEXT_MUT)
            self.screen.blit(c_lbl, c_lbl.get_rect(
                centerx=grid_x + ci * cell_size + cell_size // 2,
                bottom=grid_y - 4,
            ))

        # Rows + cells
        for ri, r in enumerate(rows):
            row_y = grid_y + ri * cell_size
            r_lbl = col_font.render(r, True, TEXT_PRI)
            self.screen.blit(r_lbl, r_lbl.get_rect(
                right=grid_x - 6,
                centery=row_y + cell_size // 2,
            ))
            for ci in range(n):
                v = matrix[ri][ci]
                cell_rect = pygame.Rect(
                    grid_x + ci * cell_size + 1,
                    row_y + 1,
                    cell_size - 2, cell_size - 2,
                )
                tint = self._tour_tint(v)
                if len(tint) == 4:
                    cell_surf = pygame.Surface(cell_rect.size, pygame.SRCALPHA)
                    pygame.draw.rect(cell_surf, tint,
                                     pygame.Rect(0, 0, *cell_rect.size),
                                     border_radius=5)
                    self.screen.blit(cell_surf, cell_rect.topleft)
                else:
                    _rrect(self.screen, tint, cell_rect, 5)
                # diagonal: dashed-style — soft outline, em-dash content
                if ri == ci:
                    _rrect(self.screen, BG, cell_rect, 5, 1, LINE_SOFT)
                    em = self.F_SMALL.render("—", True, TEXT_MUT)
                    self.screen.blit(em, em.get_rect(center=cell_rect.center))
                elif v is not None:
                    pygame.draw.rect(self.screen, GRID_C, cell_rect, 1, border_radius=5)
                    val_lbl = self.F_SMALL.render(f"{v:.2f}", True, TEXT_PRI)
                    self.screen.blit(val_lbl, val_lbl.get_rect(center=cell_rect.center))
                else:
                    pygame.draw.rect(self.screen, GRID_C, cell_rect, 1, border_radius=5)

    def _draw_rules_overlay(self) -> None:
        dim = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 165))
        self.screen.blit(dim, (0, 0))

        pw, ph = 580, 420
        px = (WIN_W - pw) // 2
        py = (WIN_H - ph) // 2
        _rrect(self.screen, PANEL_BG, pygame.Rect(px, py, pw, ph), 16, 2, VIOLET)

        cx = WIN_W // 2
        y  = py + 26

        title = self.F_H1.render("How to Play", True, TEXT_PRI)
        self.screen.blit(title, title.get_rect(centerx=cx, y=y))
        y += 38

        config = BOARD_CONFIGS.get(self.bs, {})
        stones = config.get("stones_per_player", 10)
        if self.bs % 2 == 1:
            center_desc = "center 1 cell decides"
        else:
            center_desc = "center 4 cells decide"
        rows = [
            ("Board",     f"{self.bs} x {self.bs} grid — two players, X (blue) and O (pink)"),
            ("Stones",    f"Each player places {stones} stones — {stones * 2} moves total"),
            ("Capture",   "A stone is removed when opponent neighbours >= 2"),
            ("",          "AND outnumber its friendly neighbours"),
            ("Bonus",     "Each captured opponent stone = +1 score point"),
            ("Territory", "Empty cells scored by neighbour-stone majority"),
            ("Tiebreak",  f"If tied, {center_desc} — still tied = draw"),
            ("Goal",      "Highest territory + capture total wins"),
        ]

        col_label = px + 28
        col_text  = px + 148

        for heading, text in rows:
            if heading:
                hl = self.F_H2.render(heading + ":", True, GOLD)
                self.screen.blit(hl, (col_label, y))
            tl = self.F_BODY.render(text, True, TEXT_PRI)
            self.screen.blit(tl, (col_text, y))
            y += 30

        y += 10
        hint = self.F_SMALL.render("Click anywhere or press Esc to close", True, TEXT_MUT)
        self.screen.blit(hint, hint.get_rect(centerx=cx, y=y))

    # ── Result overlay ────────────────────────────────────────────────────────

    def _click_result(self, pos: tuple) -> None:
        """Handle clicks on the Result V2 overlay (Play again / Review)."""
        # The two action buttons are computed during draw; if the modal was
        # rendered, they exist.
        play = getattr(self, "_result_btn_play", None)
        review = getattr(self, "_result_btn_review", None)
        if play is not None and play.collidepoint(pos):
            self.show_result = False
            self.review_mode = False
            self._reset()
            return
        if review is not None and review.collidepoint(pos):
            self.show_result = False
            self.review_mode = True  # keeps territory visible, enables history strip
            return
        # Clicks outside the buttons are ignored — Esc still closes.

    def _draw_result_overlay(self) -> None:
        """Result V2 — floating winner badge + denser score table + primary/secondary CTAs.

        Spec: ui_kits/desktop/ResultModal.jsx
          - Card 460 wide, paddingTop 28 so the floating badge can stick out
          - Floating winner badge at the top edge: rounded pill, 2px border in
            player color, dark almost-opaque bg, glow halo
          - Denser score table: "PLAYER X" / "PLAYER O" CAPS headers, Territory
            and Captures rows regular weight, Total bold 28px in gold with a
            1px top divider
          - Buttons at bottom: gold "Play again" primary, outline "Review board"
            secondary
        """
        res = self.result
        assert res is not None

        # Dim scrim @ 170α
        dim = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 170))
        self.screen.blit(dim, (0, 0))

        # Resolve winner color
        if res.winner == PLAYER_X:
            win_col, win_glow = COL_X, (76, 201, 255)
            badge_text = f"PLAYER {res.winner} WINS"
        elif res.winner == PLAYER_O:
            win_col, win_glow = COL_O, (255, 95, 162)
            badge_text = f"PLAYER {res.winner} WINS"
        else:
            win_col, win_glow = GOLD, (255, 215, 80)
            badge_text = "DRAW"

        # Card geometry — 460 wide, ~360 tall, vertically centered
        pw, ph = 460, 360
        px = (WIN_W - pw) // 2
        py = (WIN_H - ph) // 2

        # Card body (border 2px in winner color)
        _rrect(self.screen, PANEL_BG, pygame.Rect(px, py, pw, ph), 16, 2, win_col)

        # ── Floating winner badge ────────────────────────────────────────────
        # Pill: 10px vertical / 22px horizontal padding, border 2px in win_col,
        # bg nearly-opaque dark, sits half-out of the card at top edge.
        badge_label = self.F_H1.render(badge_text, True, win_col)
        star_label = self.F_H1.render("★", True, GOLD)
        badge_inner_w = star_label.get_width() + 10 + badge_label.get_width()
        badge_w = badge_inner_w + 44  # 22px padding each side
        badge_h = badge_label.get_height() + 20  # 10px vertical padding
        badge_x = (WIN_W - badge_w) // 2
        badge_y = py - badge_h // 2 - 2  # sits half-out, top:-2 in spec

        # Glow halo — overlapping alpha fills approximating box-shadow
        for layer, alpha in [(8, 60), (16, 32), (24, 16)]:
            halo = pygame.Surface((badge_w + 2 * layer, badge_h + 2 * layer),
                                  pygame.SRCALPHA)
            pygame.draw.rect(halo, (*win_glow, alpha),
                             halo.get_rect(), border_radius=badge_h)
            self.screen.blit(halo, (badge_x - layer, badge_y - layer))

        # Almost-opaque dark fill
        badge_rect = pygame.Rect(badge_x, badge_y, badge_w, badge_h)
        _rrect(self.screen, (5, 8, 26), badge_rect, badge_h, 2, win_col)
        # Inner content centered
        ix = badge_x + 22
        iy = badge_y + (badge_h - badge_label.get_height()) // 2
        self.screen.blit(star_label, (ix, iy))
        self.screen.blit(badge_label, (ix + star_label.get_width() + 10, iy))

        # ── Score table ──────────────────────────────────────────────────────
        # Two right-aligned numeric columns: O at the far right, X to its left.
        col_o_right = px + pw - 28
        col_x_right = col_o_right - 110   # 110px between the X and O columns
        table_top = py + 56

        # Header row (CAPS small)
        hdr_x = self.F_SMALL.render("PLAYER X", True, COL_X)
        hdr_o = self.F_SMALL.render("PLAYER O", True, COL_O)
        self.screen.blit(hdr_x, hdr_x.get_rect(right=col_x_right, y=table_top))
        self.screen.blit(hdr_o, hdr_o.get_rect(right=col_o_right, y=table_top))

        rows = [
            ("Territory", res.territory_scores[PLAYER_X], res.territory_scores[PLAYER_O], False),
            ("Captures",  res.capture_scores[PLAYER_X],   res.capture_scores[PLAYER_O],  False),
            ("Total",     res.scores[PLAYER_X],           res.scores[PLAYER_O],          True),
        ]
        ry = table_top + 24
        for label, vx, vo, is_total in rows:
            if is_total:
                # 1px top divider in GRID color
                pygame.draw.line(self.screen, GRID_C,
                                 (px + 28, ry - 4),
                                 (px + pw - 28, ry - 4), 1)
                ry += 6
                lbl_font = self.F_H2
                val_font = self.F_SCORE
                lbl_color = GOLD
                val_color_x = GOLD
                val_color_o = GOLD
            else:
                lbl_font = self.F_BODY
                val_font = self.F_H1
                lbl_color = TEXT_MUT
                val_color_x = COL_X
                val_color_o = COL_O

            self.screen.blit(lbl_font.render(label, True, lbl_color),
                             (px + 28, ry + (val_font.get_height() - lbl_font.get_height()) // 2))
            vx_surf = val_font.render(str(vx), True, val_color_x)
            vo_surf = val_font.render(str(vo), True, val_color_o)
            self.screen.blit(vx_surf, vx_surf.get_rect(right=col_x_right, y=ry))
            self.screen.blit(vo_surf, vo_surf.get_rect(right=col_o_right, y=ry))
            ry += val_font.get_height() + (10 if is_total else 8)

        # Decided-by line
        if res.winner:
            margin = abs(res.scores[PLAYER_X] - res.scores[PLAYER_O])
            decided_text = (
                f"Decided by territory · {res.winner} led by "
                f"{margin} point{'' if margin == 1 else 's'}."
            )
        else:
            decided_text = "Decided on the center tiebreak."
        dec_surf = self.F_SMALL.render(decided_text, True, TEXT_MUT)
        self.screen.blit(dec_surf,
                         dec_surf.get_rect(centerx=WIN_W // 2,
                                           y=py + ph - 88))

        # ── Action buttons ────────────────────────────────────────────────────
        btn_h = 40
        btn_gap = 10
        btn_w = (pw - 56 - btn_gap) // 2
        btn_y = py + ph - 58

        # Primary: "Play again" — gold filled, dark text
        self._result_btn_play = pygame.Rect(px + 28, btn_y, btn_w, btn_h)
        _rrect(self.screen, GOLD, self._result_btn_play, 8)
        play_lbl = self.F_H2.render("Play again", True, (58, 42, 0))
        self.screen.blit(play_lbl, play_lbl.get_rect(center=self._result_btn_play.center))

        # Secondary: "Review board" — outline
        self._result_btn_review = pygame.Rect(px + 28 + btn_w + btn_gap, btn_y, btn_w, btn_h)
        _rrect(self.screen, PANEL_BG, self._result_btn_review, 8, 1, GRID_C)
        rev_lbl = self.F_H2.render("Review board", True, TEXT_PRI)
        self.screen.blit(rev_lbl, rev_lbl.get_rect(center=self._result_btn_review.center))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    app = PygameGUI()
    app.run()


if __name__ == "__main__":
    main()
