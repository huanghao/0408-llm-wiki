"""
LunarLander SB3 版（实用）
==========================
用 stable-baselines3 训练月球着陆器，5 分钟内收敛到 200+ 分。
支持保存、加载、续训、观看游戏画面。

安装：pip install "gymnasium[box2d]" stable-baselines3
运行：
  python tools/lunarlander_sb3.py           # 训练并观看
  python tools/lunarlander_sb3.py --watch   # 只观看（需已有保存的模型）
  python tools/lunarlander_sb3.py --continue # 在已有模型基础上续训
"""

import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

MODEL_PATH = "data/lunarlander_model.zip"


def train(continue_from=None):
    """训练 500k steps，每 100k 打印一次进度。"""
    vec_env = make_vec_env("LunarLander-v3", n_envs=4)

    if continue_from and os.path.exists(continue_from):
        print(f"从已有模型续训：{continue_from}")
        model = PPO.load(continue_from, env=vec_env)
        reset = False
    else:
        print("从头训练...")
        model = PPO("MlpPolicy", vec_env)  # 全部使用默认参数
        reset = True

    for i in range(5):
        model.learn(total_timesteps=100_000, reset_num_timesteps=(reset and i == 0))
        scores = eval_model(model)
        print(f"  {(i+1)*100}k steps  均分={np.mean(scores):.0f}")

    vec_env.close()
    os.makedirs("data", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"模型已保存：{MODEL_PATH}")
    return model


def eval_model(model, n=10):
    """快速评估，返回 n 局得分列表。"""
    env = gym.make("LunarLander-v3")
    scores = []
    for _ in range(n):
        obs, _ = env.reset()
        total, done = 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        scores.append(total)
    env.close()
    return scores


def watch(model, n_episodes=5):
    """
    打开游戏窗口观看飞船降落。
    注意：必须在你自己的终端运行，不能在 Claude Code 里调用。
    """
    print(f"\n观看 {n_episodes} 局...")
    env = gym.make("LunarLander-v3", render_mode="human")
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total, done = 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        print(f"  Episode {ep+1}: {total:.0f} 分")
    env.close()


if __name__ == "__main__":
    import sys

    only_watch   = "--watch"    in sys.argv
    force_continue = "--continue" in sys.argv

    if only_watch:
        if not os.path.exists(MODEL_PATH):
            print(f"找不到模型文件 {MODEL_PATH}，请先训练。")
            sys.exit(1)
        model = PPO.load(MODEL_PATH)
        scores = eval_model(model, n=20)
        print(f"当前均分：{np.mean(scores):.0f}")
        watch(model)

    else:
        if os.path.exists(MODEL_PATH) and not force_continue:
            # 有模型：评估，够好就直接看，不够好就续训
            model = PPO.load(MODEL_PATH)
            scores = eval_model(model, n=20)
            avg = np.mean(scores)
            print(f"加载已有模型，当前均分：{avg:.0f}")
            if avg < 150:
                print("分数偏低，续训...")
                model = train(continue_from=MODEL_PATH)
        else:
            # 没有模型，或者强制续训
            model = train(continue_from=MODEL_PATH if force_continue else None)

        watch(model)
