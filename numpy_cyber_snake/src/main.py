from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import dataclass

import numpy as np


CELL = 26
GRID_W = 32
GRID_H = 20
HUD_H = 88
WIN_W = GRID_W * CELL
WIN_H = GRID_H * CELL + HUD_H

BASE_STEP = 0.12
BOOST_STEP = 0.075
FPS_MS = 16

DIR_UP = np.array([0, -1], dtype=np.int16)
DIR_DOWN = np.array([0, 1], dtype=np.int16)
DIR_LEFT = np.array([-1, 0], dtype=np.int16)
DIR_RIGHT = np.array([1, 0], dtype=np.int16)


def clamp_rgb(v: np.ndarray) -> str:
    r, g, b = np.clip(v.astype(np.int16), 0, 255)
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


NEON_CYAN = np.array([50, 250, 230], dtype=np.float32)
NEON_PURPLE = np.array([190, 95, 255], dtype=np.float32)
NEON_PINK = np.array([255, 85, 165], dtype=np.float32)
BG_TOP = np.array([6, 10, 28], dtype=np.float32)
BG_BOTTOM = np.array([3, 4, 15], dtype=np.float32)


@dataclass
class Particle:
    pos: np.ndarray
    vel: np.ndarray
    life: float
    size: float
    color: np.ndarray


class SnakeCore:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        c = np.array([GRID_W // 2, GRID_H // 2], dtype=np.int16)
        self.snake: list[np.ndarray] = [
            c.copy(),
            c + np.array([-1, 0], dtype=np.int16),
            c + np.array([-2, 0], dtype=np.int16),
        ]
        self.direction = DIR_RIGHT.copy()
        self.next_direction = self.direction.copy()
        self.score = 0
        self.steps = 0
        self.speed_level = 1
        self.combo = 0
        self.combo_ttl = 0.0
        self.alive = True
        self.food = self.spawn_food()
        self.particles: list[Particle] = []
        self.trails: list[np.ndarray] = []

    def set_direction(self, d: np.ndarray) -> None:
        if np.array_equal(d, -self.direction):
            return
        self.next_direction = d

    def spawn_food(self) -> np.ndarray:
        occupied = {tuple(v.tolist()) for v in self.snake}
        while True:
            p = np.array([random.randrange(GRID_W), random.randrange(GRID_H)], dtype=np.int16)
            if tuple(p.tolist()) not in occupied:
                return p

    def step_interval(self, boosting: bool) -> float:
        base = BOOST_STEP if boosting else BASE_STEP
        return max(0.045, base * max(0.52, 1.0 - (self.speed_level - 1) * 0.045))

    def emit_trail(self, pos: np.ndarray) -> None:
        if random.random() > 0.75:
            return
        c = pos.astype(np.float32) + 0.5
        self.particles.append(
            Particle(
                pos=c,
                vel=np.array([random.uniform(-0.9, 0.9), random.uniform(-0.8, 0.8)], dtype=np.float32),
                life=random.uniform(0.18, 0.38),
                size=random.uniform(1.8, 3.8),
                color=random.choice([NEON_CYAN, NEON_PURPLE]).copy(),
            )
        )

    def emit_burst(self, pos: np.ndarray, death: bool = False) -> None:
        c = pos.astype(np.float32) + 0.5
        n = 70 if death else 38
        for _ in range(n):
            self.particles.append(
                Particle(
                    pos=c.copy(),
                    vel=np.array([random.uniform(-5.5, 5.5), random.uniform(-4.8, 4.8)], dtype=np.float32),
                    life=random.uniform(0.25, 1.0 if death else 0.9),
                    size=random.uniform(2.0, 7.2),
                    color=random.choice([NEON_CYAN, NEON_PINK, NEON_PURPLE, np.array([255, 225, 120], dtype=np.float32)]).copy(),
                )
            )

    def update_particles(self, dt: float) -> None:
        keep: list[Particle] = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0:
                continue
            p.pos = p.pos + p.vel * dt
            p.vel *= 0.92
            p.size = max(0.5, p.size * 0.985)
            keep.append(p)
        self.particles = keep

    def step(self) -> None:
        if not self.alive:
            return

        self.direction = self.next_direction.copy()
        head = self.snake[0] + self.direction

        if head[0] < 0 or head[0] >= GRID_W or head[1] < 0 or head[1] >= GRID_H:
            self.alive = False
            self.emit_burst(self.snake[0], death=True)
            return

        body = np.array(self.snake[:-1], dtype=np.int16)
        if np.any(np.all(body == head, axis=1)):
            self.alive = False
            self.emit_burst(self.snake[0], death=True)
            return

        self.snake.insert(0, head.astype(np.int16))
        self.trails.append(head.astype(np.float32) + 0.5)
        if len(self.trails) > 42:
            self.trails.pop(0)

        if np.array_equal(head, self.food):
            self.combo = min(9, self.combo + 1)
            self.combo_ttl = 3.2
            self.score += 10 + self.combo * 2
            self.speed_level = min(14, self.speed_level + 1)
            self.food = self.spawn_food()
            self.emit_burst(head, death=False)
        else:
            self.snake.pop()

        self.steps += 1
        if self.steps % 10 == 0:
            self.speed_level = min(14, self.speed_level + 1)

        self.emit_trail(head)

    def tick(self, dt: float) -> None:
        self.combo_ttl = max(0.0, self.combo_ttl - dt)
        if self.combo_ttl == 0.0:
            self.combo = 0
        self.update_particles(dt)


class TkSnakeApp:
    def __init__(self) -> None:
        import tkinter as tk

        self.tk = tk
        self.root = tk.Tk()
        self.root.title("Numpy Cyber Snake")
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(self.root, width=WIN_W, height=WIN_H, bg="#050812", highlightthickness=0)
        self.canvas.pack()

        self.game = SnakeCore()
        self.boosting = False
        self.running = True
        self.last_frame = time.monotonic()
        self.last_step = self.last_frame
        self.elapsed = 0.0
        self.shake = 0.0

        self.root.bind("<Up>", lambda _: self.game.set_direction(DIR_UP))
        self.root.bind("<Down>", lambda _: self.game.set_direction(DIR_DOWN))
        self.root.bind("<Left>", lambda _: self.game.set_direction(DIR_LEFT))
        self.root.bind("<Right>", lambda _: self.game.set_direction(DIR_RIGHT))
        self.root.bind("w", lambda _: self.game.set_direction(DIR_UP))
        self.root.bind("s", lambda _: self.game.set_direction(DIR_DOWN))
        self.root.bind("a", lambda _: self.game.set_direction(DIR_LEFT))
        self.root.bind("d", lambda _: self.game.set_direction(DIR_RIGHT))
        self.root.bind("<KeyPress-Shift_L>", lambda _: self.set_boost(True))
        self.root.bind("<KeyRelease-Shift_L>", lambda _: self.set_boost(False))
        self.root.bind("<KeyPress-Shift_R>", lambda _: self.set_boost(True))
        self.root.bind("<KeyRelease-Shift_R>", lambda _: self.set_boost(False))
        self.root.bind("r", lambda _: self.restart())
        self.root.bind("<Escape>", lambda _: self.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def set_boost(self, on: bool) -> None:
        self.boosting = on

    def restart(self) -> None:
        if not self.game.alive:
            self.game.reset()
            self.shake = 0.0
            self.last_step = time.monotonic()

    def quit(self) -> None:
        self.running = False
        self.root.destroy()

    def draw_background(self) -> None:
        self.canvas.delete("all")
        game_h = GRID_H * CELL
        for y in range(game_h):
            t = y / max(1, game_h - 1)
            col = BG_TOP * (1 - t) + BG_BOTTOM * t
            self.canvas.create_line(0, y, WIN_W, y, fill=clamp_rgb(col))

        pulse = math.sin(self.elapsed * 1.7) * 0.5 + 0.5
        g = int(95 + pulse * 85)
        grid_col = f"#1f3c{g:02x}"
        for x in range(0, WIN_W, CELL):
            self.canvas.create_line(x, 0, x, game_h, fill=grid_col)
        for y in range(0, game_h, CELL):
            self.canvas.create_line(0, y, WIN_W, y, fill=grid_col)

    def draw_food(self, ox: int, oy: int) -> None:
        c = (self.game.food.astype(np.float32) + 0.5) * CELL + np.array([ox, oy], dtype=np.float32)
        pulse = math.sin(self.elapsed * 8.2) * 0.5 + 0.5
        glow = int(8 + pulse * 13)
        for r in range(glow, 4, -3):
            fade = r / max(1, glow)
            col = NEON_PINK * fade + np.array([20, 10, 30], dtype=np.float32) * (1 - fade)
            self.canvas.create_oval(c[0] - r, c[1] - r, c[0] + r, c[1] + r, fill=clamp_rgb(col), outline="")
        self.canvas.create_oval(c[0] - 4, c[1] - 4, c[0] + 4, c[1] + 4, fill="#fff4b8", outline="")

    def draw_trails(self, ox: int, oy: int) -> None:
        if len(self.game.trails) < 2:
            return
        for i, p in enumerate(self.game.trails):
            fade = (i + 1) / len(self.game.trails)
            r = int(1 + fade * 4)
            c = p * CELL + np.array([ox, oy], dtype=np.float32)
            col = NEON_CYAN * fade + np.array([10, 20, 30], dtype=np.float32) * (1 - fade)
            self.canvas.create_oval(c[0] - r, c[1] - r, c[0] + r, c[1] + r, fill=clamp_rgb(col), outline="")

    def draw_snake(self, ox: int, oy: int) -> None:
        for i, seg in enumerate(self.game.snake):
            t = i / max(1, len(self.game.snake) - 1)
            col = NEON_CYAN * (1 - t) + NEON_PURPLE * t
            x = int(seg[0] * CELL + ox)
            y = int(seg[1] * CELL + oy)
            self.canvas.create_rectangle(x + 3, y + 3, x + CELL - 3, y + CELL - 3, fill=clamp_rgb(col), outline="#dcfaff", width=1)

        h = self.game.snake[0]
        hx, hy = int(h[0] * CELL + ox), int(h[1] * CELL + oy)
        self.canvas.create_oval(hx + CELL // 2 - 7, hy + CELL // 2 - 5, hx + CELL // 2 - 3, hy + CELL // 2 - 1, fill="#f5f5ff", outline="")
        self.canvas.create_oval(hx + CELL // 2 + 3, hy + CELL // 2 - 5, hx + CELL // 2 + 7, hy + CELL // 2 - 1, fill="#f5f5ff", outline="")

    def draw_particles(self, ox: int, oy: int) -> None:
        for p in self.game.particles:
            life = max(0.0, min(1.0, p.life))
            col = p.color * life + np.array([10, 10, 16], dtype=np.float32) * (1 - life)
            r = max(1, int(p.size))
            x, y = p.pos[0] * CELL + ox, p.pos[1] * CELL + oy
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=clamp_rgb(col), outline="")

    def draw_hud(self, ox: int, oy: int) -> None:
        top = GRID_H * CELL + oy
        self.canvas.create_rectangle(0 + ox, top, WIN_W + ox, WIN_H + oy, fill="#040818", outline="")
        self.canvas.create_line(0 + ox, top, WIN_W + ox, top, fill="#469bff", width=2)
        combo_txt = f"x{self.game.combo}" if self.game.combo > 1 else "x1"
        combo_col = "#ffd678" if self.game.combo > 1 else "#788cbe"
        self.canvas.create_text(24 + ox, top + 28, text=f"SCORE {self.game.score}", fill="#e1f0ff", font=("Menlo", 26, "bold"), anchor="w")
        self.canvas.create_text(24 + ox, top + 62, text=f"SPEED LV {self.game.speed_level}", fill="#788cbe", font=("Menlo", 16), anchor="w")
        self.canvas.create_text(280 + ox, top + 62, text=f"COMBO {combo_txt}", fill=combo_col, font=("Menlo", 16), anchor="w")
        self.canvas.create_text(WIN_W - 18 + ox, top + 62, text="WASD/Arrow | Shift加速 | R重开 | Esc退出", fill="#7891cd", font=("Menlo", 13), anchor="e")

    def draw_game_over(self, ox: int, oy: int) -> None:
        cx, cy = WIN_W // 2 + ox, WIN_H // 2 + oy - 20
        self.canvas.create_rectangle(90 + ox, 140 + oy, WIN_W - 90 + ox, WIN_H - 120 + oy, fill="#0a0c19", outline="#ff50a5", width=2)
        self.canvas.create_text(cx, cy - 12, text="SYSTEM FAILURE", fill="#ff55a5", font=("Menlo", 30, "bold"))
        self.canvas.create_text(cx, cy + 24, text="Press R to reboot", fill="#e6f0ff", font=("Menlo", 17))

    def frame(self) -> None:
        if not self.running:
            return
        now = time.monotonic()
        dt = now - self.last_frame
        self.last_frame = now
        self.elapsed += dt

        self.game.tick(dt)
        if self.game.alive and now - self.last_step >= self.game.step_interval(self.boosting):
            self.last_step = now
            self.game.step()
            if not self.game.alive:
                self.shake = 1.2

        self.shake = max(0.0, self.shake - dt * 6.0)

        self.draw_background()
        ox = int(random.uniform(-1, 1) * 8 * self.shake)
        oy = int(random.uniform(-1, 1) * 8 * self.shake)
        self.draw_trails(ox, oy)
        self.draw_food(ox, oy)
        self.draw_snake(ox, oy)
        self.draw_particles(ox, oy)
        self.draw_hud(ox, oy)
        if not self.game.alive:
            self.draw_game_over(ox, oy)

        self.root.after(FPS_MS, self.frame)

    def run(self) -> None:
        self.frame()
        self.root.mainloop()


def run_curses_fallback() -> None:
    import curses

    class CursesMini:
        def __init__(self, stdscr: curses.window) -> None:
            self.s = stdscr
            self.game = SnakeCore()
            curses.curs_set(0)
            self.s.nodelay(True)
            self.last = time.monotonic()
            self.last_step = self.last
            self.boost = False

        def run(self) -> None:
            while True:
                now = time.monotonic()
                dt = now - self.last
                self.last = now
                self.game.tick(dt)

                ch = self.s.getch()
                if ch in (ord("q"), ord("Q"), 27):
                    return
                if ch in (curses.KEY_UP, ord("w"), ord("W")):
                    self.game.set_direction(DIR_UP)
                elif ch in (curses.KEY_DOWN, ord("s"), ord("S")):
                    self.game.set_direction(DIR_DOWN)
                elif ch in (curses.KEY_LEFT, ord("a"), ord("A")):
                    self.game.set_direction(DIR_LEFT)
                elif ch in (curses.KEY_RIGHT, ord("d"), ord("D")):
                    self.game.set_direction(DIR_RIGHT)
                elif ch in (ord("r"), ord("R")) and not self.game.alive:
                    self.game.reset()
                elif ch == ord(" "):
                    self.boost = True
                else:
                    self.boost = False

                if self.game.alive and now - self.last_step >= self.game.step_interval(self.boost):
                    self.last_step = now
                    self.game.step()

                self.s.erase()
                self.s.addstr(0, 0, f"SCORE {self.game.score}  LV {self.game.speed_level}  COMBO x{max(1, self.game.combo)}")
                self.s.addstr(1, 0, "WASD/Arrow move | Space boost | R restart | Q/Esc quit")
                for y in range(GRID_H):
                    row = []
                    for x in range(GRID_W):
                        p = np.array([x, y], dtype=np.int16)
                        if np.array_equal(p, self.game.food):
                            row.append("◆")
                        elif np.any(np.all(np.array(self.game.snake) == p, axis=1)):
                            row.append("█")
                        else:
                            row.append("·")
                    self.s.addstr(3 + y, 0, "".join(row))

                if not self.game.alive:
                    self.s.addstr(3 + GRID_H // 2, max(0, GRID_W // 2 - 12), "SYSTEM FAILURE - Press R")
                self.s.refresh()
                time.sleep(0.016)

    curses.wrapper(lambda stdscr: CursesMini(stdscr).run())


def main() -> None:
    try:
        app = TkSnakeApp()
        app.run()
        return
    except Exception as tk_exc:
        if sys.stdin.isatty() and sys.stdout.isatty():
            try:
                run_curses_fallback()
                return
            except Exception as c_exc:
                print(f"Tk backend error: {tk_exc}")
                print(f"Curses backend error: {c_exc}")
                raise SystemExit(1) from c_exc
        print("无法启动图形窗口（Tk）且当前运行环境不是终端。")
        print("请在系统终端运行该脚本，或使用支持 GUI 的 Python 运行配置。")
        print(f"Tk backend error: {tk_exc}")
        raise SystemExit(1) from tk_exc


if __name__ == "__main__":
    main()
