# TODO

- 26年值得关注的论文
- 设计一个查看新论文的探索流程
- 按时间线整理自动驾驶模型的发展
- BEV-LaneDet、MapTR和MapQR
- DINO（Zhang et al., ICLR 2023）：DAB-DETR anchor query + DN-DETR denoising + 对比去噪，DETR 系列精度天花板，COCO 63.3 AP
- Grounding DINO（开放词汇检测）：输入文本→检测对应物体，偏通用视觉/机器人方向，非 AD 主线，优先级低
- RT-DETR（百度，实时 DETR）：hybrid encoder + 解耦 decoder 实现实时推理，工程价值高但无概念突破，关心部署效率时再看

## PNC模型

现在看的wayformer和mtr都是prediction，还没有看过planning模型，也没有看过control的部分


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

## Wiki 文档 Todo（来自 mdv todos）

### wiki/20-concepts/bellman-equation.md

- [ ] **连续 Bellman** — 在「离散 vs 连续：Bellman 方程的适用范围」一节补充连续动作空间的内容（变分法 / Hamilton-Jacobi-Bellman 方程）

### wiki/10-roadmaps/data-engineering-llm.md

- [ ] **Self-Rewarding LMs** — 补充 Self-Rewarding Language Models（Yuan et al., Meta, 2024, arXiv:2401.10020）到路线图相关位置
- [ ] **WizardLM / Evol-Instruct** — 确认路线图中对 WizardLM/Evol-Instruct 的覆盖是否完整
- [ ] **继续补充材料** — 回来看「LLM 数据工程」路线图，补充剩余参考材料

### wiki/20-concepts/mcts.md

- [ ] **AlphaGo 论文** — 回头读 Silver et al. 2016 AlphaGo 原始论文，结合「参考」一节补充内容

### wiki/30-papers/phi-1-2306.11644.md

- [ ] **待验证** — 验证「用强模型注解 → 训练轻量分类器 → 规模化打分成为工业界标准范式」这一说法的来源

### wiki/30-papers/data-mixing-laws-2403.16952.md

- [ ] **PNC 实验** — 用一个更小的 PNC 模型走通 Data Mixing Laws 的方法路子

---

## RL 可把玩的例子

三个候选，按从简单到贴近 LLM 排列：

- **FrozenLake**（Gym）：4×4 冰面，随机滑行，比格子世界多了真实的随机转移 $\sum_{s'} P(s'|s,a)$。`pip install gymnasium` 即可跑。
- **CartPole**（Gym）：推车平衡杆，状态 4 个连续值，动作左/右，视觉最直观——能实时看到 agent 从乱推到学会平衡的过程。用 Actor-Critic 或 PPO，几百局内见效。
- **两位数加法 + GRPO**：即文章 C 的核心，和 LLM 后训练逻辑完全一致，验证信号精确（答案对错一查就知道）。

推荐顺序：先跑 CartPole 看直观效果，再做文章 C 贴近 LLM 实践。

## rl：GAN、AlphaZero 及其他方向

除了这两个，还值得列入的方向（各一两句）：

- RLHF / DPO：已有 wiki 页，但可以单独一篇"后训练全景"
- Model-based RL：agent 先学环境模型再规划，Dreamer/MuZero 的路线
- Multi-agent RL：多个 agent 互相博弈，OpenAI Five / 星际争霸 AI
- Offline RL：只用历史数据训练，不能和环境交互，适合医疗/自驾场景

## 这是哪里看来的，还有其他吗？

BEVFusion：多传感器融合（相机 + LiDAR）框架，性能和效率都很强（Zhijian Liu）
FlashDriveVLA：algorithm + system co-design
做“高效AI + 自动驾驶”的顶级研究者，偏工程落地
3）ParoQuant（W4A8量化） // 工程硬优化

## deer-flow

https://github.com/bytedance/deer-flow
字节跳动做的，42.5k stars，2026 年 2 月底刚发布 2.0
 一个多 agent 编排框架，定位是"能研究、能写代码、能创作内容"的超级 agent  harness。核心思路是主 agent 拆解任务后派发给专用子 agent，每个子 agent  在沙箱里独立执行

## https://github.com/unslothai/unsloth

## adaboost

## smart agent
