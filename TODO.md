# TODO

- 26年值得关注的论文
- 设计一个查看新论文的探索流程


## 文章 C：后训练 RL，用小模型跑通

目标：用一个本地可跑的小模型，跑通完整的 RL 后训练循环，帮助理解 SFT 之后的 RLHF/GRPO 流程。

**方案设计**：

- **基础模型**：GPT-2 small（124M 参数，~500MB），transformers 库直接加载，CPU 可跑
- **算法**：GRPO（Group Relative Policy Optimization，DeepSeek-R1 使用的算法）
  - 选 GRPO 而非 PPO 的原因：GRPO 不需要 critic 网络，只需要 policy 模型本身，内存占用约一半，代码更干净，且更贴近当前后训练实践
- **任务**：两位数加法题（"12 + 34 = ?"），答对得 +1，答错得 0
  - 规则清晰，奖励信号无噪声，容易验证效果
  - 本地不需要奖励模型，直接用字符串匹配判断答案
- **代码结构**：`src/rl_post_training.py`，含数据生成 + GRPO 训练循环 + reward 曲线输出
- **预期效果**：训练前准确率约 30-50%（GPT-2 不擅长算术），训练后应有明显提升

**文章结构**（待写）：
1. 从 SFT 到 RL：为什么 SFT 不够
2. 后训练 RL 的组件：policy / reward / rollout
3. GRPO 算法：组内比较代替 critic
4. 代码跑通：`src/rl_post_training.py`
5. 和 PPO 的区别，以及 o1/R1 路线的联系

**待确认**：
- 124M 模型 CPU 训练速度（估计每步 2-5 秒，100 步约 5-10 分钟，可接受）
- 是否需要加载预训练权重还是从随机初始化（推荐加载预训练，否则收敛太慢）

## rl：GAN、AlphaZero 及其他方向

除了这两个，还值得列入的方向（各一两句）：

- RLHF / DPO：已有 wiki 页，但可以单独一篇"后训练全景"
- Model-based RL：agent 先学环境模型再规划，Dreamer/MuZero 的路线
- Multi-agent RL：多个 agent 互相博弈，OpenAI Five / 星际争霸 AI
- Offline RL：只用历史数据训练，不能和环境交互，适合医疗/自驾场景
