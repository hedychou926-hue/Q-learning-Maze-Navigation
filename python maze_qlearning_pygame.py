import sys
import time
import numpy as np
import pygame
from dataclasses import dataclass
from collections import deque
import random
import matplotlib.pyplot as plt

# =========================
# 1) Q-learning Maze Env
# =========================

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_TO_DELTA = {
    0: (-1, 0),  # UP
    1: ( 1, 0),  # DOWN
    2: ( 0,-1),  # LEFT
    3: ( 0, 1),  # RIGHT
}

@dataclass
class StepResult:
    next_state: int
    reward: float
    done: bool

class MazeEnv:
    def __init__(self, grid, max_steps=200):
        self.grid = np.array([list(row) for row in grid])
        self.H, self.W = self.grid.shape
        self.max_steps = max_steps
        self.steps = 0

        self.start_pos = self._find('S')
        self.goal_pos = self._find('G')
        self.traps = self._find_all('T')
        self.agent_pos = self.start_pos

    def _find(self, ch):
        pos = np.argwhere(self.grid == ch)
        if len(pos) == 0:
            raise ValueError(f"Grid missing '{ch}'")
        return tuple(pos[0])

    def _find_all(self, ch):
        pos = np.argwhere(self.grid == ch)
        return {tuple(p) for p in pos}

    def _pos_to_state(self, pos):
        r, c = pos
        return r * self.W + c

    def _state_to_pos(self, s):
        return (s // self.W, s % self.W)

    def reset(self):
        self.agent_pos = self.start_pos
        self.steps = 0
        return self._pos_to_state(self.agent_pos)

    def step(self, action):
        self.steps += 1
        r, c = self.agent_pos
        dr, dc = ACTION_TO_DELTA[action]
        nr, nc = r + dr, c + dc

        reward = -1.0   # step penalty
        done = False

        # wall / out-of-bounds
        if not (0 <= nr < self.H and 0 <= nc < self.W) or self.grid[nr, nc] == '#':
            nr, nc = r, c
            reward = -5.0

        self.agent_pos = (nr, nc)

        # terminal checks
        if self.agent_pos == self.goal_pos:
            reward = 100.0
            done = True
        elif self.agent_pos in self.traps:
            reward = -50.0
            done = True
        elif self.steps >= self.max_steps:
            reward = -20.0
            done = True

        return StepResult(self._pos_to_state(self.agent_pos), reward, done)

def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])# explore
    return int(np.argmax(Q[state]))# exploit

def train_q_learning(
    env,
    episodes=2500,
    alpha=0.1,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.995
):
    n_states = env.H * env.W
    n_actions = len(ACTIONS)
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    epsilon = eps_start
    rewards_history = []
    success_history = []

    for ep in range(episodes):
        s = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            a = epsilon_greedy(Q, s, epsilon)
            res = env.step(a)

            best_next = np.max(Q[res.next_state])
            Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (res.reward + gamma * best_next)

            s = res.next_state
            total_reward += res.reward
            done = res.done

        epsilon = max(eps_end, epsilon * eps_decay) #
        rewards_history.append(total_reward)
        success_history.append(1 if env.agent_pos == env.goal_pos else 0)

        if (ep + 1) % 200 == 0:
            print(f"[Train] Episode {ep+1}/{episodes} | eps={epsilon:.3f} | "
                  f"avgR(last200)={np.mean(rewards_history[-200:]):.2f} | "
                  f"success(last200)={np.mean(success_history[-200:]):.2f}")

    return Q, rewards_history, success_history

def plot_learning_curves(rewards_history, success_history, window=200, prefix="qlearning_maze"):
    rewards = np.array(rewards_history, dtype=float)
    success = np.array(success_history, dtype=float)

    # rolling mean（滑动平均）
    if len(rewards) >= window:
        kernel = np.ones(window) / window
        rewards_ma = np.convolve(rewards, kernel, mode="valid")
        success_ma = np.convolve(success, kernel, mode="valid")
        x = np.arange(window, len(rewards) + 1)
    else:
        rewards_ma = rewards
        success_ma = success
        x = np.arange(1, len(rewards) + 1)

    # 1) Reward curve
    plt.figure()
    plt.plot(np.arange(1, len(rewards) + 1), rewards, alpha=0.25, label="Total reward (per episode)")
    plt.plot(x, rewards_ma, linewidth=2, label=f"Rolling mean (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-learning Training: Reward Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_reward_curve.png", dpi=200)
    plt.close()

    # 2) Success rate curve
    plt.figure()
    plt.plot(x, success_ma * 100, linewidth=2, label=f"Success rate (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (%)")
    plt.title("Q-learning Training: Success Rate Curve")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_success_curve.png", dpi=200)
    plt.close()

    print(f"[Saved plots] {prefix}_reward_curve.png, {prefix}_success_curve.png")

def greedy_rollout_trace(env, Q):
    """Generate one greedy episode trace for visualization."""
    s = env.reset()
    trace = []
    total_reward = 0.0
    done = False
    step_i = 0

    # initial frame
    trace.append({
        "step": 0,
        "pos": env.agent_pos,
        "action": "START",
        "reward": 0.0,
        "total_reward": 0.0,
        "done": False,
        "status": "RUNNING"
    })

    while not done and step_i < env.max_steps:
        a = int(np.argmax(Q[s]))
        action_name = ACTIONS[a]
        res = env.step(a)
        s = res.next_state
        total_reward += res.reward
        step_i += 1
        done = res.done

        if done:
            if env.agent_pos == env.goal_pos:
                status = "SUCCESS"
            elif env.agent_pos in env.traps:
                status = "TRAP"
            else:
                status = "TIMEOUT"
        else:
            status = "RUNNING"

        trace.append({
            "step": step_i,
            "pos": env.agent_pos,
            "action": action_name,
            "reward": res.reward,
            "total_reward": total_reward,
            "done": done,
            "status": status
        })

    return trace

def greedy_run_once(env, Q):
    """跑一次 greedy policy，不记录 trace，只返回 success/steps/total_reward"""
    s = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        a = int(np.argmax(Q[s]))
        res = env.step(a)
        s = res.next_state
        total_reward += res.reward
        done = res.done

    success = (env.agent_pos == env.goal_pos)
    steps = env.steps
    return success, steps, total_reward


def evaluate_policy(env_class, grid, Q, runs=30, max_steps=120):
    """
    多次评估学到的策略（greedy policy）
    返回：成功率、平均步数、平均累计奖励
    """
    successes = 0
    steps_list = []
    reward_list = []

    for _ in range(runs):
        env = env_class(grid, max_steps=max_steps)
        success, steps, total_reward = greedy_run_once(env, Q)
        successes += int(success)
        steps_list.append(steps)
        reward_list.append(total_reward)

    return {
        "runs": runs,
        "success_rate": successes / runs,
        "avg_steps": float(np.mean(steps_list)),
        "avg_total_reward": float(np.mean(reward_list)),
        "min_steps": int(np.min(steps_list)),
        "max_steps": int(np.max(steps_list)),
    }

# =========================
# 2) Pygame UI Renderer
# =========================

# Colors
WHITE = (245, 245, 245)
BLACK = (30, 30, 30)
GRID_LINE = (180, 180, 180)
WALL = (70, 70, 70)
EMPTY = (230, 230, 230)
START = (100, 180, 255)
GOAL = (120, 220, 120)
TRAP = (255, 140, 140)
AGENT = (255, 165, 0)
PATH_DOT = (120, 120, 255)
PANEL_BG = (250, 250, 255)

class MazeRenderer:
    def __init__(self, env, cell_size=70, panel_width=320, margin=20):
        self.env = env
        self.cell = cell_size
        self.panel_width = panel_width
        self.margin = margin

        self.grid_w_px = env.W * cell_size
        self.grid_h_px = env.H * cell_size

        self.width = self.grid_w_px + panel_width + margin * 3
        self.height = max(self.grid_h_px + margin * 2, 500)

        pygame.init()
        pygame.display.set_caption("Q-Learning Maze (Pygame UI)")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Arial", 22)
        self.small_font = pygame.font.SysFont("Arial", 18)
        self.big_font = pygame.font.SysFont("Arial", 26)

    def draw_grid(self, agent_pos, visited_positions):
        ox = self.margin
        oy = self.margin

        # cells
        for r in range(self.env.H):
            for c in range(self.env.W):
                ch = self.env.grid[r, c]
                x = ox + c * self.cell
                y = oy + r * self.cell
                rect = pygame.Rect(x, y, self.cell, self.cell)

                if ch == '#':
                    color = WALL
                elif ch == 'S':
                    color = START
                elif ch == 'G':
                    color = GOAL
                elif ch == 'T':
                    color = TRAP
                else:
                    color = EMPTY

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, GRID_LINE, rect, 1)

        # visited path dots
        for (r, c) in visited_positions:
            x = ox + c * self.cell + self.cell // 2
            y = oy + r * self.cell + self.cell // 2
            pygame.draw.circle(self.screen, PATH_DOT, (x, y), max(3, self.cell // 10))

        # agent
        ar, ac = agent_pos
        ax = ox + ac * self.cell + self.cell // 2
        ay = oy + ar * self.cell + self.cell // 2
        pygame.draw.circle(self.screen, AGENT, (ax, ay), self.cell // 3)

    def draw_panel(self, info):
        px = self.margin * 2 + self.grid_w_px
        py = self.margin
        pw = self.panel_width
        ph = self.height - self.margin * 2

        pygame.draw.rect(self.screen, PANEL_BG, (px, py, pw, ph))
        pygame.draw.rect(self.screen, GRID_LINE, (px, py, pw, ph), 1)

        lines = [
            ("Q-Learning Maze UI", self.big_font, BLACK),
            ("", self.font, BLACK),
            (f"Mode: {info.get('mode', 'Playback')}", self.font, BLACK),
            (f"Episode: {info.get('episode', '-')}", self.font, BLACK),
            (f"Epsilon: {info.get('epsilon', '-')}", self.font, BLACK),
            ("", self.font, BLACK),
            (f"Step: {info.get('step', 0)}", self.font, BLACK),
            (f"Action: {info.get('action', 'START')}", self.font, BLACK),
            (f"Reward: {info.get('reward', 0):.1f}", self.font, BLACK),
            (f"Total Reward: {info.get('total_reward', 0):.1f}", self.font, BLACK),
            ("", self.font, BLACK),
            (f"Status: {info.get('status', 'RUNNING')}", self.font, BLACK),
            ("", self.font, BLACK),
            ("Controls:", self.font, BLACK),
            ("Space  : Pause / Resume", self.small_font, BLACK),
            ("R      : Replay same policy", self.small_font, BLACK),
            ("T      : Retrain model", self.small_font, BLACK),
            ("Esc/Q  : Quit", self.small_font, BLACK),
            ("", self.font, BLACK),
            ("Legend:", self.font, BLACK),
            ("Blue cell = Start", self.small_font, BLACK),
            ("Green cell = Goal", self.small_font, BLACK),
            ("Red cell = Trap", self.small_font, BLACK),
            ("Dark cell = Wall", self.small_font, BLACK),
            ("Orange = Agent", self.small_font, BLACK),
            ("", self.font, BLACK),
            ("Eval (Greedy x30):", self.font, BLACK),
            (f"Success rate: {info.get('eval_success_rate', 0) * 100:.1f}%", self.small_font, BLACK),
            (f"Avg steps   : {info.get('eval_avg_steps', 0):.1f}", self.small_font, BLACK),
            (f"Avg reward  : {info.get('eval_avg_reward', 0):.1f}", self.small_font, BLACK),
        ]

        y = py + 15
        for text, font_obj, color in lines:
            surf = font_obj.render(text, True, color)
            self.screen.blit(surf, (px + 12, y))
            y += surf.get_height() + 6

    def render(self, agent_pos, visited_positions, info):
        self.screen.fill(WHITE)
        self.draw_grid(agent_pos, visited_positions)
        self.draw_panel(info)
        pygame.display.flip()


# =========================
# 3) Main app (train + playback)
# =========================
def _has_path_bfs(grid_2d, start, goal):
    H, W = len(grid_2d), len(grid_2d[0])
    sr, sc = start
    gr, gc = goal

    q = deque([(sr, sc)])
    visited = set([(sr, sc)])

    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            return True
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                ch = grid_2d[nr][nc]
                if ch not in ['#', 'T']:
                    visited.add((nr, nc))
                    q.append((nr, nc))
    return False


def generate_random_maze(
    H=10, W=10,
    wall_prob=0.32,
    trap_prob=0.28,
    start=(0, 0),
    goal=None,
    max_tries=200,
    seed=None
):

    if seed is not None:
        random.seed(seed)

    if goal is None:
        goal = (H - 1, W - 1)

    sr, sc = start
    gr, gc = goal

    if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
        raise ValueError("start/goal 越界了")

    for _ in range(max_tries):
        grid = [['.' for _ in range(W)] for _ in range(H)]


        for r in range(H):
            for c in range(W):
                if (r, c) in [start, goal]:
                    continue
                if random.random() < wall_prob:
                    grid[r][c] = '#'


        for r in range(H):
            for c in range(W):
                if (r, c) in [start, goal]:
                    continue
                if grid[r][c] == '.' and random.random() < trap_prob:
                    grid[r][c] = 'T'

        # Put S/G
        grid[sr][sc] = 'S'
        grid[gr][gc] = 'G'


        if _has_path_bfs(grid, start, goal):
            return [''.join(row) for row in grid]


    raise RuntimeError(
        "Random generation failed: A safe path from S to G could not be found."
        "Please lower wall_prob/trap_prob or increase max_tries."
    )

def build_and_train():
    #8x8
    grid = generate_random_maze(
        H=8, W=8,
        wall_prob=0.23,   # wall density
        trap_prob=0.32,   # trap density
        start=(0, 0),
        goal=(7, 7),
        seed=None
    )
    env = MazeEnv(grid, max_steps=120)

    print("\nTraining Q-learning...")
    Q, rewards_history, success_history = train_q_learning(
        env,
        episodes=2500,
        alpha=0.5,
        gamma=0.5,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.995
    )
    plot_learning_curves(rewards_history, success_history, window=200, prefix="maze")

    trace = greedy_rollout_trace(env, Q)

    # ===== Evaluate learned policy =====
    eval_stats = evaluate_policy(MazeEnv, grid, Q, runs=30, max_steps=120)
    print("\n[Evaluation: Greedy Policy over 30 runs]")
    print(f"Success rate: {eval_stats['success_rate']*100:.1f}%")
    print(f"Avg steps   : {eval_stats['avg_steps']:.1f} (min={eval_stats['min_steps']}, max={eval_stats['max_steps']})")
    print(f"Avg reward  : {eval_stats['avg_total_reward']:.1f}")


    train_stats = {
        "last200_avg_reward": float(np.mean(rewards_history[-200:])) if len(rewards_history) >= 200 else float(np.mean(rewards_history)),
        "last200_success": float(np.mean(success_history[-200:])) if len(success_history) >= 200 else float(np.mean(success_history)),
        "episodes": len(rewards_history),
        "final_epsilon": 0.05,
        "eval_success_rate": eval_stats["success_rate"],
        "eval_avg_steps": eval_stats["avg_steps"],
        "eval_avg_reward": eval_stats["avg_total_reward"],
    }
    return env, Q, trace, train_stats

def run_pygame_ui():
    env, Q, trace, train_stats = build_and_train()
    renderer = MazeRenderer(env, cell_size=72, panel_width=340, margin=18)

    paused = False
    frame_idx = 0
    visited_positions = set()
    auto_advance_ms = 450
    last_advance = pygame.time.get_ticks()

    while True:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    sys.exit()

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_r:
                    # replay same learned policy
                    trace = greedy_rollout_trace(env, Q)
                    frame_idx = 0
                    visited_positions = set()
                    paused = False
                    last_advance = pygame.time.get_ticks()

                elif event.key == pygame.K_t:
                    # retrain and replay (nice demo for video)
                    env, Q, trace, train_stats = build_and_train()
                    renderer = MazeRenderer(env, cell_size=72, panel_width=340, margin=18)
                    frame_idx = 0
                    visited_positions = set()
                    paused = False
                    last_advance = pygame.time.get_ticks()

        # auto playback
        if not paused and now - last_advance >= auto_advance_ms:
            if frame_idx < len(trace) - 1:
                frame_idx += 1
                pos = trace[frame_idx]["pos"]
                visited_positions.add(pos)
            last_advance = now

        # current frame info
        current = trace[frame_idx]
        info = {
            "mode": "Greedy Policy Playback",
            "episode": train_stats["episodes"],
            "epsilon": f"{train_stats['final_epsilon']:.2f}",
            "step": current["step"],
            "action": current["action"],
            "reward": current["reward"],
            "total_reward": current["total_reward"],
            "status": current["status"] if current["done"] else ("PAUSED" if paused else "RUNNING")
        }

        renderer.render(
            agent_pos=current["pos"],
            visited_positions=visited_positions,
            info=info
        )

        renderer.clock.tick(60)


if __name__ == "__main__":
    run_pygame_ui()