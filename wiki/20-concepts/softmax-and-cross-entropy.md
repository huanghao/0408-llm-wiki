# Softmax 与交叉熵

## Softmax 是什么

把一组任意实数（logits）变成概率分布：

$$p_k = \frac{e^{l_k}}{\sum_{j=1}^{K} e^{l_j}}$$

性质：$p_k > 0$，$\sum_k p_k = 1$。

**为什么用指数？** 指数保证所有输出为正。除以总和保证加起来等于 1。另外指数会放大大的值、压缩小的值，让最大 logit 对应的概率更突出。

---

## Softmax 的导数

要对 logits 做梯度下降，需要知道 $\frac{\partial p_k}{\partial l_j}$。

**情况一：$k = j$（对自己求导）**

$$p_k = \frac{e^{l_k}}{S}, \quad S = \sum_j e^{l_j}$$

用商的求导法则 $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$：

$$\frac{\partial p_k}{\partial l_k} = \frac{e^{l_k} \cdot S - e^{l_k} \cdot e^{l_k}}{S^2} = \frac{e^{l_k}}{S} \cdot \frac{S - e^{l_k}}{S} = p_k (1 - p_k)$$

**情况二：$k \neq j$（对其他位置求导）**

$p_k$ 的分子 $e^{l_k}$ 和 $l_j$ 无关，只有分母 $S$ 含 $l_j$：

$$\frac{\partial p_k}{\partial l_j} = \frac{0 \cdot S - e^{l_k} \cdot e^{l_j}}{S^2} = -\frac{e^{l_k}}{S} \cdot \frac{e^{l_j}}{S} = -p_k \cdot p_j$$

**合并**：

$$\frac{\partial p_k}{\partial l_j} = p_k \left(\mathbf{1}[k=j] - p_j\right)$$

写成矩阵（Jacobian）：

$$J = \frac{\partial \mathbf{p}}{\partial \mathbf{l}} = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$$

直觉：提高第 $j$ 个 logit，第 $j$ 个概率上升（系数 $p_j(1-p_j)$），其他所有概率下降（系数 $-p_k p_j$）——概率之和必须保持为 1，一个涨其他必须降。

---

## 交叉熵损失

分类任务的标准损失。真实标签是类别 $y$（one-hot 向量），模型输出概率分布 $\mathbf{p}$：

$$L = -\log p_y$$

只看正确类别的概率——越高越好，取负号变成"越低越好"的损失。

**loss 值和梯度的关系**：你说得对，$p_y$ 接近 0 时 $-\log p_y$ 很大，$p_y$ 接近 1 时接近 0。但 loss 的值本身不直接"乘"到梯度上——反向传播的起点是 $\frac{\partial L}{\partial L} = 1$，然后链式法则一层层往回传，每层用的是该层输出对输入的导数。最终到 logits 的梯度是 $\mathbf{p} - \text{one\_hot}(y)$（下面推导），和 loss 的绝对值无关，只和当前预测概率 $\mathbf{p}$ 与真实标签的差距有关。

**为什么叫"交叉熵"？** 信息论里，分布 $q$ 对分布 $p$ 的交叉熵定义为 $H(p, q) = -\sum_k p_k \log q_k$。当真实标签是 one-hot（$p_y = 1$，其余为 0）时，求和只剩一项：$H = -\log q_y$。这就是我们的损失。名字来自信息论，本质是"用模型分布 $q$ 来描述真实分布 $p$ 需要多少比特"——越小说明 $q$ 越接近 $p$。起了个牛逼的名字，但对单类别标签来说就是一个 $-\log$。

**为什么不用 MSE？** MSE 是 $(p_y - 1)^2$，在 $p_y$ 接近 0 时梯度很小（曲线平坦），梯度消失，训练慢。$-\log p_y$ 在 $p_y \to 0$ 时趋向无穷，梯度大，训练更有力。

---

## 交叉熵接 Softmax：链式法则

实践中总是把两者接在一起：logits → softmax → 交叉熵。对 logits 的梯度用链式法则：

$$\frac{\partial L}{\partial l_j} = \sum_k \frac{\partial L}{\partial p_k} \cdot \frac{\partial p_k}{\partial l_j}$$

先算 $\frac{\partial L}{\partial p_k}$：$L = -\log p_y$，只有 $k = y$ 时非零：

$$\frac{\partial L}{\partial p_k} = \begin{cases} -\frac{1}{p_y} & k = y \\ 0 & k \neq y \end{cases}$$

代入链式法则，求和只剩 $k = y$ 一项：

$$\frac{\partial L}{\partial l_j} = -\frac{1}{p_y} \cdot \frac{\partial p_y}{\partial l_j}$$

再代入 softmax 导数：

- 若 $j = y$：$\frac{\partial p_y}{\partial l_y} = p_y(1 - p_y)$，所以 $\frac{\partial L}{\partial l_y} = -(1 - p_y) = p_y - 1$
- 若 $j \neq y$：$\frac{\partial p_y}{\partial l_j} = -p_y p_j$，所以 $\frac{\partial L}{\partial l_j} = -\frac{1}{p_y} \cdot (-p_y p_j) = p_j$

**合并成一个公式**：

$$\frac{\partial L}{\partial l_j} = p_j - \mathbf{1}[j = y]$$

写成向量：

$$\nabla_\mathbf{l} L = \mathbf{p} - \text{one\_hot}(y)$$

这是整个深度学习里最常用的梯度公式之一。

**是的，最终形式就是"概率减掉 label"**：正确位置的梯度是 $p_y - 1$（概率比 1 少多少），其他位置是 $p_j$（概率是多少就减多少）。写成向量就是 $\mathbf{p} - \text{one\_hot}(y)$，非常干净。

**PyTorch 为什么把 softmax 和交叉熵合并成 `CrossEntropyLoss`？** 两个原因：

1. **数值稳定**：分开算时，softmax 先算 $e^{l_k}$（可能溢出），再算 $\log$（再取 $\log$），等于做了 $\log(e^x)$ 的绕圈。合并后直接用 log-sum-exp 技巧，数值更稳定。
2. **省掉中间层**：合并后反向传播直接得到 $\mathbf{p} - \text{one\_hot}(y)$，不需要分别计算 softmax 的 Jacobian 再和交叉熵的导数相乘，少一步矩阵运算。

---

## 直觉

$$\nabla_\mathbf{l} L = \mathbf{p} - \text{one\_hot}(y)$$

- 对正确类别 $y$：梯度 $= p_y - 1$，是负数。沿梯度反方向更新 → $l_y$ 增大 → $p_y$ 增大。✓
- 对错误类别 $k \neq y$：梯度 $= p_k$，是正数。沿梯度反方向更新 → $l_k$ 减小 → $p_k$ 减小。✓

也就是：每次更新，正确类别的 logit 往上推，所有错误类别的 logit 往下压，推拉的幅度正比于当前的预测概率。

---

## 和 Actor 更新的对比

Actor 更新里的梯度是：

$$\nabla_\mathbf{l} \log \pi(a|s) = \text{one\_hot}(a) - \mathbf{p}$$

和监督学习的梯度 $\mathbf{p} - \text{one\_hot}(y)$ 正好差一个负号，原因：

| | 监督学习 | Actor 更新 |
|---|---|---|
| **目标** | 最小化损失 $-\log p_y$ | 最大化 $\log \pi(a\|s)$ |
| **方向** | 梯度下降（减去梯度） | 梯度上升（加上梯度） |
| **梯度** | $\mathbf{p} - \text{one\_hot}(y)$ | $\text{one\_hot}(a) - \mathbf{p}$ |
| **效果** | 提高正确类别概率 | 提高被选动作的概率（按 $\delta_t$ 加权） |

本质是同一件事：调整 logits，让某个类别/动作的概率变大或变小。

**Actor 是带权重的软监督学习：**

监督学习每步有固定的正确答案（标签），权重是 1。Actor 每步的"标签"是这次采样到的动作，权重是 `td_error`：

- `td_error > 0`（这步比预期好）：把这次动作当正确答案学，权重正比于"好了多少"
- `td_error < 0`（这步比预期差）：反着学，权重正比于"差了多少"
- `td_error = 0`：不更新，这步正好符合预期

这就是为什么说"强化学习是没有标签的监督学习"——标签是 agent 自己通过探索发现的，好坏由环境打分决定。

---

## 和 wiki 内其他概念的关联

- [强化学习基础](./rl-fundamentals.md)：Actor-Critic 附录里用到 softmax 导数推导 Actor 更新
- [Loss Functions](./loss-functions.md)：交叉熵在各类损失函数中的位置
