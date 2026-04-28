"""
GPU 调度策略 RL Demo
====================
用 PPO 训练一个 GPU 任务调度策略，和 FIFO / 贪心基线对比。

场景（设计了真正需要权衡的情况）：
  - 集群有 16 张 GPU
  - 任务有两类：
      普通任务：需要 1/2/4 GPU，运行 10~30 步，完成奖励 +1
      大任务：  需要 8 GPU，运行 5~15 步，完成奖励 +5（高价值但占资源多）
  - 关键权衡：空余 8 GPU 时，是调度 4 个普通任务（各得 1 分），
    还是等待凑够 8 GPU 调度一个大任务（得 5 分）？
  - FIFO 和贪心都无法做这个权衡，RL 可以学到

安装：pip install gymnasium stable-baselines3
运行：python tools/gpu_scheduler_rl.py
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from collections import deque

# ── 超参数 ────────────────────────────────────────────────
TOTAL_GPUS   = 16    # 集群总 GPU 数
MAX_QUEUE    = 8     # 队列最多容纳几个任务
EPISODE_LEN  = 200   # 每个 episode 跑多少步
ARRIVE_PROB  = 0.4   # 每步有新任务到来的概率
SMALL_GPU    = [1, 2, 4]   # 普通任务 GPU 需求
BIG_GPU      = 8           # 大任务 GPU 需求
SMALL_REWARD = 1.0         # 普通任务完成奖励
BIG_REWARD   = 5.0         # 大任务完成奖励（值得等待）
BIG_PROB     = 0.25        # 新任务中大任务的概率
DURATION_MIN = 5
DURATION_MAX = 30

# ── 环境 ──────────────────────────────────────────────────
class GPUSchedulerEnv(gym.Env):
    """
    State（观测）：
      - 当前空闲 GPU 数（1 维）
      - 队列里每个任务的信息：[需要的GPU数, 任务价值, 已等待步数]（MAX_QUEUE × 3 维）
      - 正在运行的任务信息：[占用GPU数, 剩余步数] × 最多8个任务

    Action：
      0 ~ MAX_QUEUE-1：选择队列里第 i 个任务上机
      MAX_QUEUE：等待（不调度任何任务）

    Reward：
      任务完成时获得该任务的价值奖励（普通=1，大任务=5）
      每步有小的时间惩罚（鼓励效率）
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        # 观测空间：[空闲GPU] + [队列任务 × 2] + [运行任务 × 2 × 8]
        obs_dim = 1 + MAX_QUEUE * 3 + 8 * 2  # 队列每个任务多一个"价值"维度
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        # 动作空间：选队列里的任务（0~7）或等待（8）
        self.action_space = spaces.Discrete(MAX_QUEUE + 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.free_gpus   = TOTAL_GPUS
        self.queue       = []   # [(gpu_need, value, wait_steps), ...]
        self.running     = []   # [(gpu_use, remaining_steps, value), ...]
        self.step_count  = 0
        self.total_score = 0.0
        return self._obs(), {}

    def step(self, action):
        reward = -0.01  # 每步时间惩罚，鼓励效率

        # 1. 执行调度动作
        scheduled = False
        if action < MAX_QUEUE and action < len(self.queue):
            gpu_need, value, wait = self.queue[action]
            if gpu_need <= self.free_gpus:
                duration = self.np_random.integers(DURATION_MIN, DURATION_MAX + 1)
                self.running.append([gpu_need, duration, value])
                self.free_gpus -= gpu_need
                self.queue.pop(action)
                scheduled = True

        # 2. 运行中的任务推进一步，完成的给奖励并释放 GPU
        still_running = []
        for gpu_use, remaining, value in self.running:
            remaining -= 1
            if remaining > 0:
                still_running.append([gpu_use, remaining, value])
            else:
                self.free_gpus += gpu_use
                reward += value  # 任务完成，获得价值奖励
                self.total_score += value
        self.running = still_running

        # 3. 新任务随机到来
        if self.np_random.random() < ARRIVE_PROB and len(self.queue) < MAX_QUEUE:
            is_big = self.np_random.random() < BIG_PROB
            gpu_need = BIG_GPU if is_big else int(self.np_random.choice(SMALL_GPU))
            value    = BIG_REWARD if is_big else SMALL_REWARD
            self.queue.append([gpu_need, value, 0])

        # 队列里的任务等待步数 +1
        self.queue = [[g, v, w + 1] for g, v, w in self.queue]

        self.step_count += 1
        done = self.step_count >= EPISODE_LEN

        used_gpus = TOTAL_GPUS - self.free_gpus
        info = {"score": self.total_score,
                "util": used_gpus / TOTAL_GPUS,
                "queue_len": len(self.queue)}
        return self._obs(), reward, done, False, info

    def _obs(self):
        obs = np.zeros(1 + MAX_QUEUE * 3 + 8 * 2, dtype=np.float32)
        obs[0] = self.free_gpus / TOTAL_GPUS
        for i, (gpu_need, value, wait) in enumerate(self.queue[:MAX_QUEUE]):
            obs[1 + i * 3]     = gpu_need / TOTAL_GPUS
            obs[1 + i * 3 + 1] = value / BIG_REWARD      # 任务价值（归一化）
            obs[1 + i * 3 + 2] = min(wait, 50) / 50
        base = 1 + MAX_QUEUE * 3
        for i, (gpu_use, remaining, _) in enumerate(self.running[:8]):
            obs[base + i * 2]     = gpu_use / TOTAL_GPUS
            obs[base + i * 2 + 1] = remaining / DURATION_MAX
        return obs


# ── FIFO 基线 ─────────────────────────────────────────────
def _run_policy(policy_fn, n_episodes=50, seed=42):
    """通用评估函数，policy_fn(env) -> action。"""
    env = GPUSchedulerEnv()
    scores = []
    rng = np.random.default_rng(seed)
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1e6)))
        for _ in range(EPISODE_LEN):
            action = policy_fn(env, obs)
            obs, _, done, _, info = env.step(action)
        scores.append(info["score"])
    return np.mean(scores), np.std(scores)


def run_fifo(n_episodes=50, seed=42):
    """FIFO：调度队列里第一个能运行的任务。"""
    def policy(env, obs):
        for i, (gpu_need, value, wait) in enumerate(env.queue):
            if gpu_need <= env.free_gpus:
                return i
        return MAX_QUEUE
    return _run_policy(policy, n_episodes, seed)


def run_greedy(n_episodes=50, seed=42):
    """价值贪心：优先调度能运行的任务里价值最高的。"""
    def policy(env, obs):
        best_action, best_val = MAX_QUEUE, -1
        for i, (gpu_need, value, wait) in enumerate(env.queue):
            if gpu_need <= env.free_gpus and value > best_val:
                best_action, best_val = i, value
        return best_action
    return _run_policy(policy, n_episodes, seed)


def run_rl(model, n_episodes=50, seed=42):
    env = GPUSchedulerEnv()
    scores = []
    rng = np.random.default_rng(seed)
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1e6)))
        for _ in range(EPISODE_LEN):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(int(action))
        scores.append(info["score"])
    return np.mean(scores), np.std(scores)


# ── 主流程 ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("GPU 调度策略 RL Demo")
    print(f"集群：{TOTAL_GPUS} GPU，队列上限 {MAX_QUEUE}，episode {EPISODE_LEN} 步")
    print("=" * 55)

    # 1. 基线评估
    print("\n评估基线策略（50 局）...")
    fifo_mean,   fifo_std   = run_fifo()
    greedy_mean, greedy_std = run_greedy()
    print(f"  FIFO（先来先到）  得分：{fifo_mean:.1f} ± {fifo_std:.1f}")
    print(f"  贪心（优先高价值）得分：{greedy_mean:.1f} ± {greedy_std:.1f}")

    # 2. 训练 PPO
    print("\n训练 PPO（500k steps，约 3 分钟）...")
    vec_env = make_vec_env(GPUSchedulerEnv, n_envs=8)
    model = PPO("MlpPolicy", vec_env, verbose=0,
                n_steps=512, batch_size=128, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.005,
                learning_rate=3e-4)

    for i in range(10):
        model.learn(total_timesteps=50_000, reset_num_timesteps=(i == 0))
        rl_mean, _ = run_rl(model, n_episodes=20)
        print(f"  {(i+1)*50}k steps  RL 得分：{rl_mean:.1f}"
              f"  vs FIFO {fifo_mean:.1f}  贪心 {greedy_mean:.1f}")

    vec_env.close()

    # 3. 最终对比
    print("\n最终对比（50 局）：")
    rl_mean, rl_std = run_rl(model, n_episodes=50)
    print(f"  FIFO（先来先到）  得分：{fifo_mean:.1f} ± {fifo_std:.1f}")
    print(f"  贪心（优先高价值）得分：{greedy_mean:.1f} ± {greedy_std:.1f}")
    print(f"  PPO RL           得分：{rl_mean:.1f} ± {rl_std:.1f}")

    vs_fifo   = (rl_mean - fifo_mean)   / max(fifo_mean,   1) * 100
    vs_greedy = (rl_mean - greedy_mean) / max(greedy_mean, 1) * 100
    print(f"\n  RL vs FIFO 提升：{vs_fifo:+.1f}%")
    print(f"  RL vs 贪心 提升：{vs_greedy:+.1f}%")

    # 4. 单局可视化
    print("\n单局详细追踪（RL 策略）：")
    env = GPUSchedulerEnv()
    obs, _ = env.reset(seed=0)
    print(f"  {'步':>4}  {'空闲':>4}  {'队列内容':<30}  {'动作'}")
    print("  " + "-" * 60)
    for t in range(40):
        free_before = env.free_gpus
        queue_desc = "  ".join(
            f"{'大' if v==BIG_REWARD else '普'}{g}G" for g, v, w in env.queue
        ) or "（空）"
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, done, _, info = env.step(action)
        if action < MAX_QUEUE and action < len(env.queue) + 1:
            # 任务已被调度走，从调度前的队列描述里读
            action_str = f"→ 调度任务{action}"
        else:
            action_str = "→ 等待"
        if t % 4 == 0:
            print(f"  {t:>4}  {free_before:>3}G  {queue_desc:<30}  {action_str}"
                  f"  累计{info['score']:.0f}分")
