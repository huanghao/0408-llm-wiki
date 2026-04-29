# Loss Functions：深度学习 Loss 全景

## 核心问题

面对一个训练任务，用什么 loss？为什么？

判断依据有两个维度：
1. **训练信号的形态**：有正确答案的 token / 有标量标注 / 只有相对排序 / 只有"哪两个样本应该相似"
2. **优化目标是什么**：预测正确输出 / 生成真实样本 / 学到好的表示空间 / 优化人类偏好

这两个维度决定了 loss 的选择。

---

## 一、预测类 Loss

### 回归（连续值预测）

**MSE（Mean Squared Error）**：

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_i (y_i - \hat{y}_i)^2$$

对大误差惩罚更重（平方放大）。适合目标值连续且误差分布接近正态的场景。

**MAE（Mean Absolute Error）**：

$$\mathcal{L}_{\text{MAE}} = \frac{1}{N}\sum_i |y_i - \hat{y}_i|$$

对异常值更鲁棒（线性而不是平方）。但在误差为 0 附近不可微，梯度不连续。

**Huber Loss**：MSE 和 MAE 的折中——误差小时用 MSE（平滑），误差大时用 MAE（鲁棒）。阈值 $\delta$ 是超参数。

| Loss | 对异常值 | 零点可微 | 常见场景 |
|------|---------|---------|---------|
| MSE | 敏感 | 是 | 奖励模型价值头、图像重建 |
| MAE | 鲁棒 | 否 | 深度估计、目标检测边框 |
| Huber | 折中 | 是 | DQN、TD 误差 |

**真实模型**：Diffusion 模型（DDPM）预测噪声用 MSE；RLHF 的 critic（价值函数）用 MSE 拟合 return。

---

### 分类（离散类别预测）

**Cross-Entropy / NLL**（多分类）：

$$\mathcal{L}_{\text{CE}} = -\sum_t \log p_\theta(y_t \mid x)$$

配 Softmax 输出，适合互斥的多类别。语言建模的"下一个 token"就是一个词表大小的多分类。

**Binary Cross-Entropy**（二分类）：

$$\mathcal{L}_{\text{BCE}} = -[y \log \hat{y} + (1-y)\log(1-\hat{y})]$$

配 Sigmoid 输出，适合是/否判断。ELECTRA 的 replaced token detection、RLHF 的偏好判断。

**Focal Loss**（解决类别不平衡）：

$$\mathcal{L}_{\text{Focal}} = -\alpha(1-p_t)^\gamma \log p_t$$

$(1-p_t)^\gamma$ 是调制因子：模型对某个类已经很有把握时（$p_t$ 高），自动降低这个样本的 loss 权重，迫使模型关注难样本。

**真实模型**：RetinaNet 目标检测（正负样本极度不平衡，背景框远多于目标框）；CLIP 的负例太多时也有类似思路。

---

### Masked Prediction（MLM / MAE）

只对被遮住的位置计算 Cross-Entropy，其他位置不参与 loss。

$$\mathcal{L}_{\text{MLM}} = -\sum_{t \in \text{masked}} \log p_\theta(y_t \mid x_{\backslash t})$$

| 模型 | 细节 |
|------|------|
| BERT（2018） | 随机 mask 15%，被 mask 位：80% 换 [MASK]，10% 随机词，10% 不变 |
| RoBERTa（2019） | 动态 mask：每个 epoch 重新随机，而不是固定 |
| ELECTRA（2020） | 改用 replaced token detection（二分类），每个 token 都参与 loss，信号更密集 |
| MAE—图像版（2022） | mask 75% 的 patch，预测被遮像素的 MSE |

---

## 二、生成类 Loss

生成模型的目标是让模型输出的分布逼近真实数据分布，但"真实分布"无法直接写成一个可微函数——这是生成类 loss 需要绕开的核心难题。

### GAN（对抗 Loss）

生成器 G 和判别器 D 互博：

$$\mathcal{L}_D = -\mathbb{E}[\log D(x)] - \mathbb{E}[\log(1 - D(G(z)))]$$
$$\mathcal{L}_G = -\mathbb{E}[\log D(G(z))]$$

D 想区分真假，G 想骗过 D。两者交替训练，理论上 G 最终生成的样本让 D 无法区分真假。

**问题**：训练极不稳定，容易 mode collapse（G 只生成少数几种样本）。后续变体（WGAN、StyleGAN）都在解决这个问题。

**真实模型**：StyleGAN2/3（人脸生成）、Pix2Pix（图像转换）、CycleGAN（无配对图像翻译）。

### Diffusion 去噪 Loss

训练时给真实图像加噪声，让模型预测被加的噪声（或预测原始图像）：

$$\mathcal{L}_{\text{Diffusion}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

本质是一个 MSE——预测噪声 $\epsilon$ 和模型估计 $\epsilon_\theta$ 之间的均方误差。但训练时 $t$（噪声步数）随机采样，模型学会在任意噪声水平下去噪。

推理时从纯噪声出发，反复去噪直到生成图像。

**真实模型**：DALL-E 2、Stable Diffusion、SORA（视频）。

### VAE（ELBO Loss）

同时优化两个目标：重建质量 + 隐空间结构化：

$$\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}[\log p(x|z)]}_{\text{重建 loss（MSE 或 BCE）}} - \underbrace{D_{\text{KL}}(q(z|x) \| p(z))}_{\text{KL 惩罚，让隐空间接近标准正态}}$$

重建 loss 让解码器还原输入；KL 项让隐变量分布有规律（可以从中采样生成新样本）。两者存在张力——KL 太强会牺牲重建质量（posterior collapse）。

**真实模型**：VQ-VAE（离散隐变量，用在 DALL-E 1）；Stable Diffusion 的 latent space 是 VAE 压缩后的表示。

---

## 三、表示学习 Loss

目标不是预测正确输出，而是让 embedding 空间有好的几何结构。

### 对比 Loss（InfoNCE / NT-Xent）

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_j^+)/\tau)}{\sum_{k} \exp(\text{sim}(z_i, z_k)/\tau)}$$

正例对拉近，负例推远。关键问题：**正例怎么构造**？

| 模型 | 正例构造方式 |
|------|------------|
| SimCLR（2020） | 同一图片两次数据增强（裁剪+颜色抖动） |
| MoCo（2020） | 同上，但用 momentum encoder + 负例队列，不需要大 batch |
| CLIP（2021） | 同一图文对（图片 + 描述文字） |
| SimCSE（2021） | 同一句子两次不同 dropout |
| E5 / BGE（文本嵌入） | 人工标注或挖掘的语义相似句对 |

### 非对比 Loss（BarlowTwins / VICReg）

对比 loss 需要大量负例（大 batch 或负例队列）。非对比方法换了思路：不需要负例，直接让两个视角的 embedding **各维度相关系数矩阵接近单位矩阵**（各维度独立、不冗余）。

**BarlowTwins（2021）**：

$$\mathcal{L} = \sum_i (1 - \mathcal{C}_{ii})^2 + \lambda \sum_{i \neq j} \mathcal{C}_{ij}^2$$

$\mathcal{C}$ 是两个视角 embedding 的跨批次相关矩阵：对角线趋近 1（同维度相关），非对角线趋近 0（不同维度独立）。

**适用场景**：资源受限（无法维持大 batch 或负例队列）的自监督学习。

---

## 四、序列与结构预测 Loss

### CTC Loss（Connectionist Temporal Classification）

用于**输入和输出序列长度不对齐**的场景，比如语音识别（音频帧数 ≠ 字符数）。

核心思想：不需要对齐标注，让模型在所有可能的对齐方式上求和，最大化正确输出序列的边缘概率。引入特殊的 blank token，允许重复字符和空白，用动态规划高效计算。

**真实模型**：DeepSpeech（百度）、早期语音识别系统。现在 Whisper 等用 seq2seq + CE 代替了 CTC，但 CTC 在实时流式识别中仍常用。

### CRF Loss（条件随机场）

用于**标签之间有依赖关系**的序列标注，比如 NER（命名实体识别）："B-PER 后面不能接 B-ORG"这类约束。

CRF 在模型输出的 logits 上加一个转移矩阵，联合建模整个序列的标签概率，用 Viterbi 解码找全局最优标签序列。

**真实模型**：早期 BERT + CRF 的 NER 模型。现在大模型直接用 CE 生成标签序列，CRF 使用减少，但在对标注一致性要求高的工业 NER 场景仍有价值。

---

## 五、偏好与强化类 Loss

### Ranking / Preference Loss

**Bradley-Terry**（奖励模型训练）：

$$\mathcal{L} = -\log \sigma(r_w - r_l)$$

**DPO Loss**：

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

| 模型 | 细节 |
|------|------|
| InstructGPT RM（2022） | 人工排序 → Bradley-Terry 训练奖励模型 |
| Llama 2-Chat（2023） | 同上，RLHF 路线 |
| Zephyr（2023） | DPO，AI 生成偏好数据 |
| Claude（Anthropic） | Constitutional AI：AI 自动生成偏好对 |

### REINFORCE / PPO / GRPO

$$\nabla_\theta \mathcal{L} = -\mathbb{E}\left[(R - b) \cdot \nabla_\theta \log p_\theta(\tau)\right]$$

| 模型 | 奖励来源 | 算法 |
|------|---------|------|
| InstructGPT（2022） | 奖励模型打分 | PPO |
| DeepSeek-R1（2025） | 答案对错（0/1） | GRPO |
| Quiet-STaR（2024） | 预测下一词准确率提升 | REINFORCE |
| AlphaGo / AlphaCode | 游戏胜负 / 测试用例通过 | Policy Gradient |

详见 [REINFORCE](./reinforce.md) 和 [RLHF](./rlhf.md)。

---

## 六、输出层与 Loss 的配对关系

### Logits 是什么

**Logits** 是模型最后一层的**原始输出**，即 softmax/sigmoid **之前**的值。数值是任意实数，还没有归一化成概率。

| 术语 | 含义 |
|------|------|
| **Logits** | softmax 之前的原始分数向量 |
| **Probabilities** | softmax 之后，加起来等于 1 |
| **Log-probabilities** | 对概率取 log，NLL loss 直接用这个 |

### 任务 → 输出层 → Loss 完整配对表

| 任务 | 输出维度 | 归一化 | Loss |
|------|---------|--------|------|
| 多分类 / 下一个 token | 词表大小 | Softmax | Cross-Entropy（NLL）|
| Masked prediction（MLM） | 词表大小，只算 mask 位置 | Softmax | Cross-Entropy（masked only）|
| 二分类 | 1 | Sigmoid | Binary Cross-Entropy |
| 回归 | 1 | 无（线性） | MSE / MAE / Huber |
| 多标签分类 | 类别数，各维独立 | Sigmoid（每维独立） | Binary CE per class |
| 句子嵌入（对比学习） | embedding 维度 | L2 归一化 | InfoNCE |
| 知识蒸馏 / 分布对齐 | 词表大小 | Softmax | KL 散度 |
| 偏好对齐（奖励模型） | 1（标量分） | 无（线性） | Bradley-Terry |
| 偏好对齐（DPO） | 词表大小 | Softmax（隐含） | DPO loss |
| RL 策略优化 | 词表大小 + 价值头 | Softmax / 无 | PPO / GRPO + KL 惩罚 |
| 图像生成（Diffusion） | 图像维度（噪声） | 无（线性） | MSE（噪声预测）|
| 图像生成（GAN） | 图像维度 | Tanh（生成器）/ Sigmoid（判别器） | Adversarial loss |
| 语音识别（不对齐） | 字符集大小 | Softmax | CTC loss |
| 序列标注（NER） | 标签集大小 | Softmax + CRF | CE + CRF |

### 为什么输出函数和 Loss 要配对，不能随意组合

主要原因是**数值稳定性**：

```python
# 危险写法：先算 softmax 再取 log
prob = softmax(logits)       # 极小值如 1e-40
loss = -log(prob[target])    # log(1e-40) 精度损失严重

# 正确写法：log-sum-exp trick，PyTorch 内部合并
loss = F.cross_entropy(logits, target)  # 数值稳定
```

PyTorch 规范：**把 logits 直接传给 loss 函数**：
- `nn.CrossEntropyLoss`：内部做 log_softmax，接收 logits
- `nn.BCEWithLogitsLoss`：内部做 sigmoid + log，接收 logits
- 不要自己先加 softmax/sigmoid 再传给对应的 loss

---

## 七、多任务 Loss 加权

一个模型同时优化多个 loss 时，需要决定权重：

$$\mathcal{L} = \sum_k w_k \mathcal{L}_k$$

**常见方法**：

| 方法 | 思路 | 适用 |
|------|------|------|
| 手动加权 | 根据量级对齐（让各 loss 数值相近） | 简单，实践中最常用 |
| Uncertainty Weighting（Kendall 2018） | 把权重视为可学习参数，不确定性大的任务权重自动降低 | 任务差异大时 |
| GradNorm（2018） | 动态调整权重让各任务梯度范数相近 | 梯度冲突严重时 |
| PCGrad（2020） | 梯度冲突时投影掉冲突分量 | 多任务梯度方向相反时 |

**真实场景**：InstructGPT PPO 阶段同时优化奖励 loss + KL 惩罚，$\beta$ 是手动调的权重；CLIP 同时优化 image→text 和 text→image 两个方向的 InfoNCE。

---

## 本文未深入覆盖的方向

| 方向 | 代表 Loss | 备注 |
|------|----------|------|
| 目标检测边框回归 | IoU loss、GIoU、CIoU | 专用于边框坐标的几何 loss |
| 图像分割 | Dice loss、Tversky loss | 处理前景背景极度不平衡 |
| 度量学习 | Triplet loss、ArcFace | 人脸识别、细粒度检索，比 InfoNCE 更早 |
| 图神经网络 | 节点分类 CE、链接预测 BCE | 结构化数据上的标准 loss |
| 时间序列 | DTW loss、Temporal CE | 对齐和预测 |
| 因果推断 | Counterfactual loss | 处理混淆变量 |

这些方向都是在基础 loss 上针对特定几何或分布假设做的变体，理解了本文的基础后，这些都可以按需查阅。

---

## 相关概念

- **[Perplexity](./perplexity.md)**：NLL 的指数形式，语言模型评估指标
- **[REINFORCE](./reinforce.md)**：策略梯度基础算法，奖励/Return/Advantage 的区别
- **[RLHF](./rlhf.md)**：PPO、GRPO、DPO 的完整对比
