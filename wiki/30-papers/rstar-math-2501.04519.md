# rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking

一句话总结：用 MCTS 驱动的自我演化循环，让 7B SLM 通过自己生成高质量推理轨迹和过程偏好模型（PPM），数学推理能力从 58.8% 提升到 90%，不蒸馏大模型即可媲美 o1-preview。

## 基本信息

- 论文：rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking
- 作者：Xinyu Guan, Li Lyna Zhang, Yifei Liu, Ning Shang, Youran Sun, Yi Zhu, Fan Yang, Mao Yang
- 机构：Microsoft Research Asia
- arXiv：2501.04519（2025 年 1 月 8 日）
- 代码：https://github.com/microsoft/rStar
- 本地 PDF：`raw/inbox/2501.04519.pdf`

## 核心问题

**问题**：小语言模型（SLM，1.5B–7B）能否在不依赖 GPT-4 等大模型蒸馏的前提下，自主达到 System 2 级别的数学推理能力？

此前路线的瓶颈：
1. **GPT-4 蒸馏路线（MetaMath/NuminaMath）**：训练数据质量受限于教师模型能力，GPT-4 无法解决奥数级题目，且中间步骤可能出错但最终答案偶然正确（"错误推理碰对答案"）
2. **Process Reward Model（PRM）训练**：逐步打分需要人工标注或 MCTS 生成的精确分数，但精确 Q 值标注天然嘈杂
3. **Best-of-N 采样**：独立采样 N 次不共享信息，效率低

> **概念补充：Q 值是什么？**
> Q 值（Q-value）来自强化学习，表示"从当前状态出发，按某种策略行动，最终能获得的期望总奖励"。在 rStar-Math 里，每个"推理步骤"是一个状态，Q 值衡量的是"这一步对最终得出正确答案的贡献有多大"。直觉：Q 值高 = 这步走对了，能通向正确答案；Q 值低 = 这步是死路，最终会算错。MCTS 通过大量 rollout（模拟走到最终答案）来估计每步的 Q 值——不需要人工打分，靠"最终对不对"自动反推每步的好坏。

> **概念补充：PRM vs PPM vs PPO 的关系？**
> - **PRM（Process Reward Model，过程奖励模型）**：给推理链的每一步打分的模型。区别于 ORM（Outcome Reward Model，只看最终答案对不对），PRM 提供步级反馈，更精细。wiki 里没有单独的 PRM 页，但 [RLHF](../20-concepts/rlhf.md) 里提到了 reward model 的概念。
> - **PPM（Process Preference Model，过程偏好模型）**：rStar-Math 对 PRM 的变体。传统 PRM 直接预测每步的"分数"（需要精确标签），PPM 改为预测"哪步更好"（偏好排序），标签来自 MCTS 的 Q 值比较，更容易训练。
> - **PPO（Proximal Policy Optimization）**：一种强化学习训练算法，用来更新模型参数——[wiki 里有详细介绍](../20-concepts/ppo-explained.md)。PPO 是"怎么用奖励信号更新模型"，PRM/PPM 是"奖励信号从哪里来"，两者在 RL 训练管线里分工不同。rStar-Math 用 SFT（监督微调）而非 PPO 来更新 policy 模型，PPM 的角色是 MCTS 的打分器，不直接参与梯度更新。

> **知识体系角度：读 rStar-Math 前建议补哪些底层？**
> rStar-Math 综合了三条知识线，缺哪条都会感到困难：
> 1. **MCTS 基础**：[MCTS wiki](../20-concepts/mcts.md) — 理解树搜索的四步（Selection/Expansion/Simulation/Backpropagation）和 UCB 公式
> 2. **强化学习基础**：[REINFORCE](../20-concepts/reinforce.md) + [PPO](../20-concepts/ppo-explained.md) — 理解"策略、奖励、Q 值"这套语言
> 3. **指令微调/SFT 基础**：[Instruction Tuning](../20-concepts/instruction-tuning.md) — 理解"用高质量数据 fine-tune 模型"是什么意思
>
> **关于先读 DeepSeek-R1 的问题**：不必要。rStar-Math（2025 年 1 月）和 DeepSeek-R1（2025 年 1 月）是同期工作，两者技术路线不同——rStar-Math 靠 MCTS 显式搜索，DeepSeek-R1 靠 RL（GRPO）内化推理。如果对"LLM 推理能力从哪里来"这个大问题感兴趣，两篇可以并行读，不存在依赖关系。如果想先打基础，建议顺序：MCTS → REINFORCE/PPO → rStar-Math。

rStar-Math 的核心洞察：MCTS 把每一步拆成单步生成，自然产生步级训练数据；反复多轮后，Q 值能可靠区分"对的步骤"和"错的步骤"，即使不能精确打分——这足以构造**偏好对**，从而训练 PPM。

## 方法：三项核心创新

### 1. Code-augmented CoT 数据合成

每步生成时同时产出**自然语言 CoT 注释 + 对应 Python 代码**，Python 代码在沙盒中执行验证。

- 只保留代码执行成功的候选节点作为有效步骤
- 相比纯自然语言 CoT，消除了"中间步骤错误但答案侥幸正确"的噪声
- Q 值通过 MCTS 反向传播自动标注：终局正确答案的节点得到高 Q 值，错误路径节点得到低 Q 值

$$Q(s_i)^k = Q(s_i)^{k-1} + Q(s_d)^k$$

其中 $Q(s_d) = +1$（正确答案终局）或 $-1$（错误终局）。随着 rollout 次数增加，Q 值逐渐稳定到该步骤"对正确答案的贡献"。

> **具体例子**：题目"小明以 3 m/s 走了 4 秒，再以 5 m/s 跑了 6 秒，共走多远？"
>
> 模型在 Step 1 生成两个候选：
> - 候选 A：`# Step 1: 计算第一段距离` + `d1 = 3 * 4`（代码执行成功，得 12）→ **保留**
> - 候选 B：`# Step 1: 总距离 = (3+5) * (4+6)`（代码执行成功，但逻辑错误）→ 保留（代码能跑，但后续会算错）
>
> Step 2：候选 A 继续走向正确答案 42 m；候选 B 走向错误答案 80 m。
>
> 经过 10 次 rollout 后，候选 A 的路径 8 次得到正确答案 → Q 值累积为正（≈ +0.6）；候选 B 的路径 9 次得到错误答案 → Q 值为负（≈ −0.8）。
>
> **结果**：PPM 用这对 (候选 A, 候选 B) 构造偏好对训练——不需要人工告诉模型"候选 A 更好"，MCTS rollout 自动给出了这个判断。

### 2. Process Preference Model（PPM）

**核心想法**：不用 Q 值直接作为奖励标签（嘈杂），而是用 Q 值**选择偏好对**，再用排序损失训练 PPM。

- **正样本**：Q 值最高的 2 个候选步骤（且必须通向正确答案）
- **负样本**：Q 值最低的 2 个候选步骤（且必须通向错误答案）
- 同一中间步骤的正负样本共享相同的前缀上下文

损失函数（Bradley-Terry 模型的排序损失）：

$$\mathcal{L}_{ppm}(\theta) = -\frac{1}{2 \times 2} \mathbb{E}_{(x, y^{pos}, y^{neg})} \left[ \log \sigma \left( r_\theta(x, y^{pos}) - r_\theta(x, y^{neg}) \right) \right]$$

PPM 初始化自 policy SLM 的权重，把 next-token prediction head 替换为线性层 + tanh，输出 [-1, 1] 的步级分数。

**为什么比直接用 Q 值更好**：Q 值在 round 1–2 较嘈杂，区分好坏步骤绰绰有余，但精确打分不够准。构造偏好对只需"这步比那步好"的排序关系，信号更鲁棒。

### 3. 四轮自我演化

从 747k 数学题出发，4 轮迭代：

| 轮次 | Policy 模型 | PPM | 关键变化 |
|------|-------------|-----|----------|
| Round 1 | Bootstrap（DeepSeek-Coder-V2-Instruct 236B）| PPM-r1（不可靠）| 冷启动，终局引导 Q 值，8 rollouts |
| Round 2 | SLM-r1（7B）| PPM-r2（首个可用）| 16 rollouts，Q 值质量显著提升 |
| Round 3 | SLM-r2 + PPM-r2 | PPM-r3 | PPM 增强 MCTS，大幅扩展奥数级题目覆盖 |
| Round 4 | SLM-r3 + PPM-r3 | PPM-r4 | 对难题额外 64–128 rollouts，奥数覆盖从 62% → 80.58% |

每轮：MCTS 生成轨迹 → 筛选 top-2 Q 值轨迹 → SFT 训练新 Policy SLM → 训练新 PPM → 下轮 MCTS 用新模型。

**关键细节**：
- Policy SLM 每轮从基础模型重新 fine-tune，不增量训练前轮模型（防止累积偏差）
- 奥数 AMC/AIME 来自 NuminaMath 竞赛子集；用 GPT-4 补充合成 hard 题目时，只保留 GPT-4 能给出至少 3 个一致答案的题目
- 树最大深度 16，每步 8 候选节点，探索常数 $c=2$

## 关键结果

### MATH benchmark（Pass@1，64 条轨迹）

| 模型 | MATH | AIME 2024 | AMC 2023 |
|------|------|-----------|----------|
| GPT-4o | 76.6 | 9.3 | 47.5 |
| o1-preview | 85.5 | 44.6 | 90.0 |
| o1-mini | **90.0** | **56.7** | **95.0** |
| rStar-Math（Qwen2.5-7B）| **90.0** | 53.3 | 87.5 |
| rStar-Math（Phi3-mini 3.8B）| 86.4 | 43.3 | 80.0 |
| rStar-Math（Qwen2.5-1.5B）| 88.6 | 46.7 | 85.0 |

- Qwen2.5-7B + PPM 在 MATH 上从 58.8% 提升到 90%，匹配 o1-mini
- AIME 2024 平均解出 53.3%（8/15 题），超越 o1-preview 8.7%
- 以同等采样数超越使用 72B ORM 的 Best-of-N 基线（72B ORM vs 7B PPM）

### PPM vs ORM 对比（Table 8）

| Reward Model | Inference | MATH | AIME | Olympiad Bench |
|---|---|---|---|---|
| ORM | Best-of-N | 82.6 | 26.7 | 55.1 |
| PQM（Q值直接当标签）| MCTS | 88.2 | 46.7 | 62.9 |
| **PPM（偏好对）** | **MCTS** | **89.4** | **50.0** | **65.3** |

PPM 一致优于 ORM 和 PQM，说明"偏好对比较"比"精确分数回归"更适合步级奖励。

### 自我演化的进展（Table 6，固定 Qwen2.5-7B 基础）

| 轮次 | MATH | AIME | 相对 GPT-4o |
|------|------|------|------------|
| Base 7B | 58.8 | 0.0 | 远低于 |
| Round 1 | 75.2 | 10.0 | 接近 |
| Round 2 | 86.6 | 43.3 | **超越** |
| Round 3 | 87.0 | 46.7 | 超越 |
| Round 4 | **89.4** | **50.0** | 超越 |

Round 2 开始（PPM-r2 启用）是性能跃升的关键节点。

### Test-time compute scaling

4 条轨迹时已超越 o1-preview（Best-of-N baselines），随轨迹数增加持续提升直至约 64 条时趋于饱和（MATH/AIME），College Math 仍未饱和。

## 重要发现

**PPM 是 System 2 上限的决定因素**：实验显示，在 Policy SLM 达到足够基础能力后，PPM 是系统 2 推理性能的主要瓶颈。不同 policy 大小（1.5B/3.8B/7B）在同一 PPM 下收敛到接近的性能上界（Figure 5）。

**内生自我反思能力涌现**：训练中未包含任何自我纠错数据，但 rStar-Math 能在走错路后主动回头选择更简单的方法（Figure 4）——仅靠 System 2 深度思考就能涌现 self-reflection。

**PPM 偏好定理应用步骤**：PPM 对"调用 Fermat 小定理、AM-GM 不等式、勾股定理"等关键推理步骤给高分，说明 PPM 学会了识别数学证明中的关键跳跃点，而非只看形式正确性。

## 局限性

- **需要可验证的终局信号**：当前依赖 Python 代码执行 + 数值答案匹配。无法直接推广到没有明确答案的开放问题（定理证明、常识推理）；几何题（需要视觉理解）同样无法处理
- **训练成本高**：Round 1 bootstrap 用 236B DeepSeek-Coder-V2-Instruct，10 节点 8×80GB H100，约 2 周。Rounds 2–4 用 7B policy，15 节点 4×40GB A100，每轮约 3 天（Round 4 因扩大 rollout 延长至 1 周）
- **推理延迟大**：AIME 单题平均需要生成约 15,693 tokens（Table 9），适合离线数据生成，不适合实时交互
- **SFT 上限依赖演化数据质量**：Policy SLM 的 Pass@1 仍低于 Qwen2.5-7B-Instruct（SFT 阶段未超越），System 2 推理提升才显现优势

## 现状与影响

一句话定性：**rStar-Math 是"SLM 通过 MCTS 自举数学推理"路线的标志性工作，证明了不依赖 GPT-4 蒸馏也能达到 o1-preview 级别，开放了代码与数据，是 2025 年数学推理开源生态的重要基础设施，但随着 DeepSeek-R1 / Qwen3 等更强基础模型涌现，其作为"最先进"的地位已被超越，核心方法论（PPM + MCTS 自演化）仍被广泛借鉴。**

截至 2026 年初的影响：

- **开放生态**：代码和数据已公开（github.com/microsoft/rStar），多个后续工作在此基础上扩展，包括代码推理、定理证明方向
- **PPM 训练范式**：偏好对替代精确分数的训练方式被后续 PRM 相关工作广泛引用，影响了 reward model 训练的方法论
- **方法被部分取代**：DeepSeek-R1 / o3 / Qwen3 等端到端训练的推理模型直接在模型内部内化了 chain-of-thought，无需推理时 MCTS 搜索，性能进一步超越 rStar-Math；但 rStar-Math 的训练数据生成流程（MCTS 生成高质量轨迹 → SFT/RL）仍是这类模型的数据制备思路之一
- **重要贡献与实现分离**：核心贡献（PPM 偏好训练、Code-augmented CoT、自演化循环）已被工业界广泛借鉴；但 rStar-Math 本身作为推理系统已不是生产前沿选择

## 和 wiki 内其他概念的关联

- [MCTS](../20-concepts/mcts.md)：rStar-Math 的核心搜索框架，把 LLM 推理步骤建模为树节点，PPM 充当节点价值函数，完整实现了 MCTS + 神经网络的 AlphaZero 类循环
- [LIMO](./limo-2502.03387.md)：LIMO 发现 rStar-Math 生成的 MCTS 轨迹是激发强推理能力的关键数据来源，是"少量高质量 MCTS 数据 → SFT → 强推理"路线的实证
- [RLHF / PPO](../20-concepts/rlhf.md)：rStar-Math 的 PPM 用 Bradley-Terry 排序损失，与 RLHF 的 reward model 训练同源；自演化循环与 AlphaZero 的 PPO 迭代结构类似，但 rStar-Math 使用 SFT 而非 on-policy RL
- [Synthetic Data with Verification](../20-concepts/synthetic-data-with-verification.md)：rStar-Math 是"可验证合成数据"方法论的具体实例：Python 执行提供步级验证，MCTS Q 值提供轨迹级筛选
- [UCT（Kocsis & Szepesvári, 2006）](./uct-kocsis-szepesvari-2006.md)：rStar-Math 的 MCTS Selection 步骤直接引用 UCT 公式，$Q(s) + c\sqrt{\ln N_{parent}(s)/N(s)}$

## 值得看的部分 / 相关资料

- **Section 3.2（Code-augmented CoT Generation）**：核心数据合成方法，Figure 2 展示了 NL CoT 与 Python 代码并行生成的具体形式
- **Section 3.3（Process Preference Model）**：PPM 训练的关键创新，Figure 1(b) 展示了偏好对构造逻辑
- **Section 3.4（Self-Evolved Deep Thinking）**：四轮自演化 recipe 的详细描述，Table 2/3 展示每轮改进
- **Section 4.3（Ablation Study）**：Table 7/8 系统对比了 Code-augmented CoT vs GPT-4 蒸馏、PPM vs ORM vs PQM
- **Section 5（Findings）**：自我反思涌现、PPM 偏好定理步骤等关键发现，有 Figure 4 案例
- **Appendix A.1**：训练超参数细节，Table 9 各 benchmark 的 token 消耗，Table 10 Pass@1 对比

## 参考

- Guan et al., 2025: *rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking*（arXiv:2501.04519）
- Kocsis & Szepesvári, 2006: *Bandit Based Monte-Carlo Planning*（UCT，ECML 2006）
- Lightman et al., 2023: *Let's Verify Step by Step*（PRM800k，ICLR 2024）
- Ouyang et al., 2022: *Training Language Models to Follow Instructions with Human Feedback*（InstructGPT/RLHF，NeurIPS 2022）
