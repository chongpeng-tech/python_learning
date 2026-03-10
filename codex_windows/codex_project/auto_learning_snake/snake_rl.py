import argparse
import os
import random
import site
import sys
import time
from collections import deque

# Some Conda setups disable user-site in sys.path. Add it back if available.
user_site = site.getusersitepackages()
if user_site and user_site not in sys.path and os.path.isdir(user_site):
    sys.path.append(user_site)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import tkinter as tk
except Exception:
    tk = None


class SnakeEnv:
    ACTION_STRAIGHT = 0
    ACTION_RIGHT = 1
    ACTION_LEFT = 2

    # right, down, left, up
    DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def __init__(self, width=20, height=20, max_steps_no_food=120):
        self.width = width
        self.height = height
        self.max_steps_no_food = max_steps_no_food
        self.reset()

    def reset(self):
        cx, cy = self.width // 2, self.height // 2
        self.direction_idx = 0
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self._spawn_food()
        return self.get_state()

    def _spawn_food(self):
        occupied = set(self.snake)
        empty = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in occupied]
        self.food = random.choice(empty) if empty else self.snake[0]

    def _turn(self, action):
        if action == self.ACTION_STRAIGHT:
            return
        if action == self.ACTION_RIGHT:
            self.direction_idx = (self.direction_idx + 1) % 4
        elif action == self.ACTION_LEFT:
            self.direction_idx = (self.direction_idx - 1) % 4

    def _next_head(self):
        dx, dy = self.DIRS[self.direction_idx]
        hx, hy = self.snake[0]
        return hx + dx, hy + dy

    def _is_collision(self, pt):
        x, y = pt
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return pt in self.snake[:-1]

    def step(self, action):
        self.steps += 1
        self.steps_since_food += 1
        self._turn(action)

        new_head = self._next_head()

        if self._is_collision(new_head):
            return self.get_state(), -10.0, True, {"score": self.score}

        reward = -0.02
        old_dist = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            reward = 10.0
            self.steps_since_food = 0
            self._spawn_food()
        else:
            self.snake.pop()
            new_dist = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
            if new_dist < old_dist:
                reward += 0.12
            else:
                reward -= 0.08

        if self.steps_since_food > self.max_steps_no_food:
            return self.get_state(), -5.0, True, {"score": self.score}

        done = False
        return self.get_state(), reward, done, {"score": self.score}

    def get_state(self):
        head = self.snake[0]
        dir_idx = self.direction_idx

        def danger_at(dir_candidate):
            dx, dy = self.DIRS[dir_candidate]
            pt = (head[0] + dx, head[1] + dy)
            return 1.0 if self._is_collision(pt) else 0.0

        danger_straight = danger_at(dir_idx)
        danger_right = danger_at((dir_idx + 1) % 4)
        danger_left = danger_at((dir_idx - 1) % 4)

        dir_flags = [0.0, 0.0, 0.0, 0.0]  # right, down, left, up
        dir_flags[dir_idx] = 1.0

        food_left = 1.0 if self.food[0] < head[0] else 0.0
        food_right = 1.0 if self.food[0] > head[0] else 0.0
        food_up = 1.0 if self.food[1] < head[1] else 0.0
        food_down = 1.0 if self.food[1] > head[1] else 0.0

        state = np.array(
            [
                danger_straight,
                danger_right,
                danger_left,
                *dir_flags,
                food_left,
                food_right,
                food_up,
                food_down,
            ],
            dtype=np.float32,
        )
        return state


class QNet(nn.Module):
    def __init__(self, in_dim=11, hidden=128, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buf)


class TkRenderer:
    def __init__(self, env, cell=22, fps=18):
        if tk is None:
            raise RuntimeError("Tkinter not available. Disable --render or install tkinter support.")
        self.env = env
        self.cell = cell
        self.fps = fps

        self.root = tk.Tk()
        self.root.title("Snake DQN Training")
        self.canvas = tk.Canvas(
            self.root,
            width=env.width * cell,
            height=env.height * cell + 36,
            bg="#0f1221",
            highlightthickness=0,
        )
        self.canvas.pack()
        self.last_draw = 0.0

    def draw(self, episode, score, epsilon):
        now = time.time()
        if now - self.last_draw < 1.0 / self.fps:
            return
        self.last_draw = now

        c = self.canvas
        c.delete("all")

        for x in range(self.env.width):
            for y in range(self.env.height):
                x0 = x * self.cell
                y0 = y * self.cell
                x1 = x0 + self.cell
                y1 = y0 + self.cell
                c.create_rectangle(x0, y0, x1, y1, fill="#171b2e", outline="#212844")

        fx, fy = self.env.food
        c.create_rectangle(
            fx * self.cell + 3,
            fy * self.cell + 3,
            (fx + 1) * self.cell - 3,
            (fy + 1) * self.cell - 3,
            fill="#ff5a7d",
            outline="",
        )

        for i, (sx, sy) in enumerate(self.env.snake):
            fill = "#4de2c5" if i > 0 else "#b5ffef"
            c.create_rectangle(
                sx * self.cell + 2,
                sy * self.cell + 2,
                (sx + 1) * self.cell - 2,
                (sy + 1) * self.cell - 2,
                fill=fill,
                outline="",
            )

        hud = f"Episode: {episode}   Score: {score}   Epsilon: {epsilon:.3f}"
        c.create_text(8, self.env.height * self.cell + 18, anchor="w", text=hud, fill="#d8e1ff", font=("Consolas", 12, "bold"))
        self.root.update_idletasks()
        self.root.update()

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass


def choose_action(qnet, state, epsilon, device):
    if random.random() < epsilon:
        return random.randint(0, 2)
    with torch.no_grad():
        st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = qnet(st)
        return int(torch.argmax(q, dim=1).item())


def train(
    episodes=600,
    lr=1e-3,
    gamma=0.95,
    batch_size=256,
    memory_size=50000,
    target_sync=300,
    eps_start=1.0,
    eps_end=0.02,
    eps_decay=0.996,
    render=False,
    render_every=1,
    save_path="dqn_snake.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SnakeEnv()
    qnet = QNet().to(device)
    target = QNet().to(device)
    target.load_state_dict(qnet.state_dict())

    optimizer = optim.Adam(qnet.parameters(), lr=lr)
    criterion = nn.MSELoss()
    memory = ReplayBuffer(capacity=memory_size)

    epsilon = eps_start
    best_score = 0
    global_step = 0
    renderer = TkRenderer(env) if render else None

    try:
        for ep in range(1, episodes + 1):
            state = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = choose_action(qnet, state, epsilon, device)
                next_state, reward, done, info = env.step(action)
                memory.push(state, action, reward, next_state, float(done))

                state = next_state
                total_reward += reward
                global_step += 1

                if render and (ep % max(1, render_every) == 0):
                    renderer.draw(ep, info["score"], epsilon)

                if len(memory) >= batch_size:
                    s, a, r, ns, d = memory.sample(batch_size)
                    s = torch.tensor(s, dtype=torch.float32, device=device)
                    a = torch.tensor(a, dtype=torch.long, device=device).unsqueeze(1)
                    r = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
                    ns = torch.tensor(ns, dtype=torch.float32, device=device)
                    d = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)

                    q_values = qnet(s).gather(1, a)
                    with torch.no_grad():
                        next_q = target(ns).max(1, keepdim=True)[0]
                        q_target = r + gamma * next_q * (1 - d)

                    loss = criterion(q_values, q_target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if global_step % target_sync == 0:
                    target.load_state_dict(qnet.state_dict())

            epsilon = max(eps_end, epsilon * eps_decay)
            score = info["score"]
            best_score = max(best_score, score)

            if ep % 10 == 0 or ep == 1:
                print(
                    f"episode={ep:4d} score={score:3d} best={best_score:3d} reward={total_reward:7.2f} eps={epsilon:.3f}",
                    flush=True,
                )

        torch.save(qnet.state_dict(), save_path)
        print(f"Training done. Model saved to {save_path}")
        return qnet
    finally:
        if renderer is not None:
            renderer.close()


def play(model_path="dqn_snake.pt", fps=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SnakeEnv()
    qnet = QNet().to(device)
    qnet.load_state_dict(torch.load(model_path, map_location=device))
    qnet.eval()

    renderer = TkRenderer(env, fps=fps)

    try:
        while True:
            state = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(torch.argmax(qnet(st), dim=1).item())
                state, _, done, info = env.step(action)
                renderer.draw(0, info["score"], 0.0)
                time.sleep(1.0 / max(1, fps))
    except tk.TclError:
        pass
    finally:
        renderer.close()


def main():
    parser = argparse.ArgumentParser(description="Torch + NumPy Snake DQN (with live visualization)")
    sub = parser.add_subparsers(dest="cmd", required=False)

    t = sub.add_parser("train", help="Train DQN")
    t.add_argument("--episodes", type=int, default=600)
    t.add_argument("--render", action="store_true", help="Show live training window")
    t.add_argument("--render-every", type=int, default=1, help="Render every N episodes")
    t.add_argument("--save", type=str, default="dqn_snake.pt")

    p = sub.add_parser("play", help="Run trained model with visualization")
    p.add_argument("--model", type=str, default="dqn_snake.pt")
    p.add_argument("--fps", type=int, default=16)

    args = parser.parse_args()

    if args.cmd in (None, "train"):
        train(
            episodes=args.episodes,
            render=getattr(args, "render", False),
            render_every=getattr(args, "render_every", 1),
            save_path=getattr(args, "save", "dqn_snake.pt"),
        )
    elif args.cmd == "play":
        play(model_path=args.model, fps=args.fps)


if __name__ == "__main__":
    main()
