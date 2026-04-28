"""
LunarLander 手写 PPO（教学用）
==============================
纯 PyTorch 实现，可以看到每一个细节：rollout、GAE、clip 更新。
收敛较慢（需要 500k+ steps），用于理解原理。

如需快速看到飞船降落效果，用：python tools/lunarlander_sb3.py

安装：pip install "gymnasium[box2d]" torch
运行：python tools/lunarlander_ppo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from collections import deque

# ── 超参数 ────────────────────────────────────────────────
OBS_DIM     = 8      # 观测维度
ACT_DIM     = 4      # 动作数量
HIDDEN      = 128
LR          = 3e-4
GAMMA       = 0.99
GAE_LAMBDA  = 0.95
EPSILON     = 0.2    # PPO clip 范围
EPOCHS_PPO  = 4      # 每批数据做几次 PPO 更新
BATCH_SIZE  = 64
ROLLOUT_LEN = 2048   # 每次收集多少步再更新
MAX_EPISODES = 3000

# ── Actor-Critic 模型 ────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN),  nn.Tanh(),
        )
        self.actor  = nn.Linear(HIDDEN, ACT_DIM)  # policy head
        self.critic = nn.Linear(HIDDEN, 1)         # value head

    def forward(self, obs):
        feat = self.shared(obs)
        return self.actor(feat), self.critic(feat).squeeze(-1)

    def get_action(self, obs):
        logits, value = self(obs)
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs, actions):
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), value, dist.entropy()

# ── GAE（广义优势估计）────────────────────────────────────
def compute_gae(rewards, values, dones, last_value):
    advantages = []
    gae = 0.0
    values_ext = values + [last_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values_ext[t+1] * (1 - dones[t]) - values_ext[t]
        gae   = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns    = advantages + torch.tensor(values, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns

# ── 训练 ─────────────────────────────────────────────────
def train():
    env   = gym.make("LunarLander-v3")
    model = ActorCritic()
    opt   = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    scores = deque(maxlen=100)
    episode, total_steps = 0, 0
    obs, _ = env.reset()

    print("手写 PPO 训练 LunarLander-v3")
    print(f"{'Episode':>8}  {'得分':>8}  {'近100均分':>10}  {'总步数':>10}")
    print("-" * 50)

    while episode < MAX_EPISODES:
        buf_obs, buf_act, buf_logp, buf_val, buf_rew, buf_done = [], [], [], [], [], []
        ep_reward = 0

        for _ in range(ROLLOUT_LEN):
            obs_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, log_prob, value = model.get_action(obs_t.unsqueeze(0))
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            buf_obs.append(obs); buf_act.append(action.item())
            buf_logp.append(log_prob.item()); buf_val.append(value.item())
            buf_rew.append(reward); buf_done.append(float(done))

            ep_reward += reward; obs = next_obs; total_steps += 1

            if done:
                scores.append(ep_reward); episode += 1
                if episode % 50 == 0:
                    avg = np.mean(scores)
                    print(f"{episode:>8}  {ep_reward:>8.1f}  {avg:>10.1f}  {total_steps:>10}")
                    if avg > 230:
                        print(f"\n✓ 均分 {avg:.1f}，停止训练")
                        env.close(); return model
                ep_reward = 0; obs, _ = env.reset()

        with torch.no_grad():
            _, last_val = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        advantages, returns = compute_gae(buf_rew, buf_val, buf_done, last_val.item())

        obs_t  = torch.tensor(np.array(buf_obs), dtype=torch.float32)
        act_t  = torch.tensor(buf_act, dtype=torch.long)
        logp_t = torch.tensor(buf_logp, dtype=torch.float32)

        indices = np.arange(ROLLOUT_LEN)
        for _ in range(EPOCHS_PPO):
            np.random.shuffle(indices)
            for start in range(0, ROLLOUT_LEN, BATCH_SIZE):
                idx = indices[start:start + BATCH_SIZE]
                new_logp, new_val, entropy = model.evaluate(obs_t[idx], act_t[idx])
                ratio = torch.exp(new_logp - logp_t[idx])
                adv   = advantages[idx]
                policy_loss = -torch.min(ratio * adv,
                                         torch.clamp(ratio, 1-EPSILON, 1+EPSILON) * adv).mean()
                value_loss  = F.mse_loss(new_val, returns[idx])
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

    env.close()
    print(f"\n训练结束，最终近100均分：{np.mean(scores):.1f}")
    return model


if __name__ == "__main__":
    model = train()
    # 取消注释可以看飞船降落动画（需要在自己的终端运行，不能在 Claude Code 里）
    # from lunarlander_sb3 import watch
    # watch(model, use_sb3=False)
