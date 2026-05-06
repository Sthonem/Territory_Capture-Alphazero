"""Pygame desktop interface for Territory Capture."""

from __future__ import annotations

import sys
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


def _calc_geometry(bs: int) -> dict:
    """Compute dynamic layout values for a given board size."""
    cell = max(60, min(80, 480 // bs))
    board_px = cell * bs
    board_x = (WIN_W - board_px) // 2
    panel_w = board_x - 20
    return {
        "cell": cell,
        "bs": bs,
        "board_px": board_px,
        "board_x": board_x,
        "panel_w": panel_w,
        "l_panel_x": 10,
        "r_panel_x": board_x + board_px + 10,
        "panel_y": BOARD_Y,
        "panel_h": board_px,
        "footer_y": BOARD_Y + board_px + 16,
        "footer_h": WIN_H - (BOARD_Y + board_px + 16),
    }

# ── Palette ───────────────────────────────────────────────────────────────────
BG         = (5,   8,  22)
PANEL_BG   = (13,  22,  48)
BOARD_BG   = (11,  17,  40)
GRID_C     = (33,  52,  93)
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
        bg  = (25, 40, 75) if hov else PANEL_BG
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
        self.ai_waiting  = False
        self.ai_at       = 0

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

    def _init_agents(self) -> None:
        config = BOARD_CONFIGS.get(self.bs, {})
        model_file = config.get("model_file", f"model_{self.bs}x{self.bs}.pth")
        hard_path = f"src/model_hard_{self.bs}x{self.bs}.pth"
        self.agents = {
            DIFF_EASY: RandomAgent(),
            DIFF_MED:  AIAgent(board_size=self.bs),
            DIFF_HARD: AIAgent(board_size=self.bs, model_path=hard_path, num_simulations=100)
                       if __import__("pathlib").Path(hard_path).exists()
                       else AIAgent(board_size=self.bs, num_simulations=100),
        }

    def _init_buttons(self) -> None:
        g = self.geo
        bh = 42
        by = g["footer_y"] + (g["footer_h"] - bh) // 2
        self.btn_new    = _Btn(pygame.Rect( 15, by, 120, bh), "New Game",         COL_X,  COL_X,   self.F_H2)
        self.btn_mode   = _Btn(pygame.Rect(140, by, 160, bh), self._mode_lbl(),   VIOLET, TEXT_PRI, self.F_H2)
        self.btn_board  = _Btn(pygame.Rect(305, by, 110, bh), self._board_lbl(),  GOLD,   TEXT_PRI, self.F_H2)
        self.btn_diff_x = _Btn(pygame.Rect(420, by, 120, bh), self._diff_x_lbl(), COL_X,  TEXT_PRI, self.F_H2)
        self.btn_diff_o = _Btn(pygame.Rect(545, by, 120, bh), self._diff_o_lbl(), COL_O,  TEXT_PRI, self.F_H2)
        self.btn_rules  = _Btn(pygame.Rect(670, by, 130, bh), "How to Play",      VIOLET, TEXT_PRI, self.F_H2)
        self._btns      = [self.btn_new, self.btn_mode, self.btn_board, self.btn_diff_x, self.btn_diff_o, self.btn_rules]

    def _mode_lbl(self) -> str:
        short = {MODE_HVH: "H vs H", MODE_HVAI: "H vs AI", MODE_AIVAI: "AI vs AI"}
        return f"Mode: {short[MODES[self.mode_i]]}"

    def _board_lbl(self) -> str:
        return f"Board: {self.bs}x{self.bs}"

    def _diff_x_lbl(self) -> str:
        return f"X: {DIFFS[self.diff_x]}"

    def _diff_o_lbl(self) -> str:
        return f"O: {DIFFS[self.diff_o]}"

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
                    if self.show_rules or self.show_result:
                        self.show_rules = self.show_result = False

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
        if self.show_rules:
            self.show_rules = False
            return
        if self.show_result:
            self.show_result = False
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

    def _notify_agents_of_move(self, action: Position) -> None:
        for agent in self.agents.values():
            if hasattr(agent, "mcts"):
                agent.mcts.advance_to_action(action)

    def _do_ai_move(self) -> None:
        diff_idx = self.diff_x if self.game.current_player == PLAYER_X else self.diff_o
        agent = self.agents[DIFFS[diff_idx]]
        action = agent.select_action(self.game.clone())
        self._apply_move(action)

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
        self._draw_footer()
        if self.show_rules:
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
        y += 30

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
                (px + PANEL_W - 18, y),
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
        g = self.geo
        fy, fh = g["footer_y"], g["footer_h"]
        pygame.draw.rect(self.screen, (7, 12, 28),
                         pygame.Rect(0, fy, WIN_W, fh))
        pygame.draw.line(self.screen, GRID_C, (0, fy), (WIN_W, fy), 1)

        for btn in self._btns:
            btn.draw(self.screen)

        # Right-side info
        max_moves = self.game.stones_per_player * 2
        info = (
            f"{self.bs}x{self.bs}  Move {self.game.move_count}/{max_moves}     "
            f"Capture bonus  X +{self.game.captured_by[PLAYER_X]}"
            f"   O +{self.game.captured_by[PLAYER_O]}"
        )
        inf_surf = self.F_SMALL.render(info, True, TEXT_MUT)
        self.screen.blit(
            inf_surf,
            inf_surf.get_rect(right=WIN_W - 20, centery=fy + fh // 2),
        )

    # ── Rules overlay ─────────────────────────────────────────────────────────

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

    def _draw_result_overlay(self) -> None:
        res = self.result
        assert res is not None

        dim = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 180))
        self.screen.blit(dim, (0, 0))

        pw, ph = 420, 370
        px = (WIN_W - pw) // 2
        py = (WIN_H - ph) // 2
        border_c = COL_X if res.winner != PLAYER_O else COL_O
        _rrect(self.screen, PANEL_BG, pygame.Rect(px, py, pw, ph), 16, 2, border_c)

        cx = WIN_W // 2
        y  = py + 28

        # Winner line
        if res.winner:
            w_col  = COL_X if res.winner == PLAYER_X else COL_O
            w_text = f"Player {res.winner}  Wins!"
        else:
            w_col  = GOLD
            w_text = "Draw!"

        wl = self.F_TITLE.render(w_text, True, w_col)
        self.screen.blit(wl, wl.get_rect(centerx=cx, y=y))
        y += 52

        # Score table
        headers    = [("", px + 30), ("X", cx - 30), ("O", cx + 60)]
        score_rows = [
            ("Territory", res.territory_scores[PLAYER_X], res.territory_scores[PLAYER_O]),
            ("Captures",  res.capture_scores[PLAYER_X],   res.capture_scores[PLAYER_O]),
            ("Total",     res.scores[PLAYER_X],            res.scores[PLAYER_O]),
        ]

        for label, col_x in headers:
            hc = COL_X if label == "X" else (COL_O if label == "O" else TEXT_MUT)
            hl = self.F_H2.render(label, True, hc)
            self.screen.blit(hl, (col_x, y))
        y += 26

        for row_label, vx, vo in score_rows:
            is_total = row_label == "Total"
            if is_total:
                pygame.draw.line(self.screen, GRID_C,
                                 (px + 20, y - 4), (px + pw - 20, y - 4), 1)
            fn   = self.F_H2 if is_total else self.F_BODY
            lc   = GOLD if is_total else TEXT_MUT
            vc   = GOLD if is_total else TEXT_PRI
            xcol = (GOLD if is_total else COL_X)
            ocol = (GOLD if is_total else COL_O)

            self.screen.blit(fn.render(row_label, True, lc), (px + 30, y))
            self.screen.blit(fn.render(str(vx), True, xcol), (cx - 30, y))
            self.screen.blit(fn.render(str(vo), True, ocol), (cx + 60, y))
            y += 30

        # Tiebreak note
        if res.scores[PLAYER_X] == res.scores[PLAYER_O]:
            tb  = res.tiebreak_scores
            tbl = self.F_SMALL.render(
                f"Center tiebreak — X: {tb[PLAYER_X]}  O: {tb[PLAYER_O]}", True, TEXT_MUT
            )
            self.screen.blit(tbl, tbl.get_rect(centerx=cx, y=y + 8))
            y += 30

        hint = self.F_SMALL.render("Click anywhere or press Esc to close", True, TEXT_MUT)
        self.screen.blit(hint, hint.get_rect(centerx=cx, y=y + 14))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    app = PygameGUI()
    app.run()


if __name__ == "__main__":
    main()
