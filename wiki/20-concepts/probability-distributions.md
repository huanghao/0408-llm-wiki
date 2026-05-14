# 概率分布速查：非常规分布手册

正态、均匀、二项略去。本文聚焦**实际建模中经常碰到但容易混淆的分布**：幂律、指数、泊松、对数正态、混合高斯、Pareto、Weibull、Beta、Dirichlet、负二项、学生 t。

每个分布：密度/质量函数、均值方差、形状直觉、应用场景、和其他分布的关系、判断方法（图形 + 检验）。

---

## 目录

1. [幂律（Power Law）](#1-幂律power-law)
2. [指数分布（Exponential）](#2-指数分布exponential)
3. [泊松分布（Poisson）](#3-泊松分布poisson)
4. [对数正态（Log-Normal）](#4-对数正态log-normal)
5. [混合高斯（Gaussian Mixture）](#5-混合高斯gaussian-mixture)
6. [Pareto 分布](#6-pareto-分布)
7. [Weibull 分布](#7-weibull-分布)
8. [Beta 分布](#8-beta-分布)
9. [Dirichlet 分布](#9-dirichlet-分布)
10. [负二项分布（Negative Binomial）](#10-负二项分布negative-binomial)
11. [学生 t 分布（Student's t）](#11-学生-t-分布students-t)
12. [分布关系总图](#12-分布关系总图)
13. [判断方法汇总](#13-判断方法汇总)

---

## 1. 幂律（Power Law）

### 密度函数

连续形式（定义在 $x \geq x_{\min} > 0$）：

$$p(x) = \frac{\alpha - 1}{x_{\min}} \left(\frac{x}{x_{\min}}\right)^{-\alpha}, \quad \alpha > 1$$

参数：$\alpha$（幂律指数，通常 $2 < \alpha < 3$），$x_{\min}$（分布起点）。

### 均值与方差

$$\mathbb{E}[X] = \frac{\alpha-1}{\alpha-2} x_{\min} \quad (\alpha > 2)$$

$$\text{Var}(X) = \left(\frac{\alpha-1}{\alpha-3}\right)\frac{x_{\min}^2}{(\alpha-2)^2} \quad (\alpha > 3)$$

注意：$\alpha \leq 2$ 时均值无穷大；$\alpha \leq 3$ 时方差无穷大。这是幂律"重尾"的数学含义——极端值没有上界约束。

> **为什么均值会无穷大？** 均值是 $\int x \cdot p(x) \, dx$。幂律的 $p(x) \propto x^{-\alpha}$，所以被积项 $x \cdot p(x) \propto x^{1-\alpha}$。当 $\alpha \leq 2$ 时，$1-\alpha \geq -1$，这个函数衰减得"太慢"——对 $x$ 从 $x_{\min}$ 到 $\infty$ 积分，结果发散（类比 $\int_1^\infty 1/x \, dx = \infty$）。直觉上：尾部还是有足够多的"超大值"，把均值硬是拉到无穷。这不是计算技巧，而是真实的统计性质——如果真实世界里财富分布是 $\alpha \leq 2$ 的幂律，那"平均财富"在数学上没有意义（样本均值会随样本量增大而不断增大，不收敛）。

### 形状直觉

- 大多数值很小，极少数值极大（"80/20 法则"的数学来源）
- 没有特征尺度：把 $x$ 轴整体拉伸 10 倍（即把每个 $x$ 换成 $10x$），分布的形状不变，只是整体上下平移——这叫尺度不变性（scale invariance）。正态分布没有这个性质：把 $x$ 拉伸 10 倍，钟形会变宽变矮，形状改变。幂律之所以尺度不变，是因为 $p(cx) \propto (cx)^{-\alpha} = c^{-\alpha} x^{-\alpha} \propto p(x)$，乘以常数 $c^{-\alpha}$ 不改变幂律形式。
- log-log 坐标下是**直线**，斜率 = $-\alpha$

```
log p(x)
  |  \
  |    \
  |      \  斜率 = -α
  |        \
  +-----------> log x
```

### 应用场景

- 词频（Zipf 定律）：频率第 $k$ 高的词，频率 $\propto 1/k^\alpha$
- 城市人口、财富分布、网页入链数
- LLM scaling laws：loss $\propto N^{-\alpha}$（$\alpha \approx 0.076$）
- 地震震级（Gutenberg-Richter 定律）

### 判断方法

**图形**：画 log-log 散点图，幂律呈直线。若只有尾部直线（中间弯曲），可能是对数正态或截断幂律。

**检验**：Clauset et al.（2009）方法：
1. 用最大似然估计 $\hat{\alpha}$：$\hat{\alpha} = 1 + n\left[\sum_{i=1}^n \ln \frac{x_i}{x_{\min}}\right]^{-1}$
2. 用 Kolmogorov-Smirnov（KS）统计量比较观测数据和拟合幂律的 CDF
3. 用似然比检验区分幂律 vs 对数正态（两者 log-log 图很像）

**常见误判**：对数正态在 log-log 图上中段也接近直线，用眼睛区分不可靠，必须做统计检验。

→ 详见 [幂律与 Scaling](power-law-and-scaling.md)

---

## 2. 指数分布（Exponential）

### 密度函数

$$p(x) = \lambda e^{-\lambda x}, \quad x \geq 0, \quad \lambda > 0$$

参数：$\lambda$（速率参数，rate）。有时用 $\theta = 1/\lambda$（尺度参数，scale）。

### 均值与方差

$$\mathbb{E}[X] = \frac{1}{\lambda}, \quad \text{Var}(X) = \frac{1}{\lambda^2}$$

变异系数（CV = 标准差/均值）= 1，这是指数分布的特征——方差和均值的平方相等。

### 形状直觉

- 从 $x=0$ 单调递减，没有钟形峰
- 半对数图（log $p$ vs $x$）上是**直线**，斜率 $= -\lambda$

```
log p(x)
  |\ 
  |  \
  |    \  斜率 = -λ
  |      \
  +---------> x
```

### 核心性质：无记忆性（Memorylessness）

$$P(X > s + t \mid X > s) = P(X > t)$$

"已经等了 $s$ 分钟，再等 $t$ 分钟的概率，和从头等 $t$ 分钟一样。"这是指数分布的唯一特征——连续分布中只有指数分布有这个性质。

### 应用场景

- 等待时间：泊松过程中两个事件之间的时间间隔服从指数分布
- 硬件寿命（无磨损老化的理想元件）
- 排队论中的服务时间（M/M/1 队列）
- 网络包到达间隔

### 与其他分布的关系

- **泊松的伴侣**：泊松过程中事件数服从泊松，事件间隔服从指数（参数相同的 $\lambda$）
- **Gamma 的特例**：指数分布 = Gamma$(1, \lambda)$，$k$ 个独立指数之和服从 Gamma$(k, \lambda)$
- **Weibull 的特例**：Weibull$(1, \lambda)$ = 指数分布（Weibull 是指数的推广，允许非常数故障率）

### 判断方法

**图形**：半对数图（y 轴取 log）若呈直线，支持指数分布。

**检验**：
- 计算样本均值 $\bar{x}$ 和标准差 $s$，若 $s/\bar{x} \approx 1$（变异系数 ≈ 1），支持指数
- Lilliefors 检验（指数分布版本）
- 过离散检验：若 Var$(X) \gg \mathbb{E}[X]^2$（CV $\gg 1$），更可能是重尾分布（Pareto/Weibull）

---

## 3. 泊松分布（Poisson）

### 质量函数（离散分布）

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

参数：$\lambda > 0$（速率，也等于均值）。

### 均值与方差

$$\mathbb{E}[X] = \lambda, \quad \text{Var}(X) = \lambda$$

均值 = 方差，这是泊松的特征性质。若观测数据中 Var $>$ Mean（过离散，overdispersion），用负二项分布更合适。

### 形状直觉

- $\lambda$ 小时（如 $\lambda < 1$）：右偏，$k=0$ 最可能
- $\lambda$ 大时（如 $\lambda > 10$）：近似对称，趋向正态分布

### 应用场景

- 单位时间内事件计数：每小时到达用户数、每天错误日志条数
- 核衰变计数、放射性粒子计数
- 文本中词语出现次数（低频词近似泊松）
- NLP 中用于语言模型的词频建模（Pitman-Yor 过程的基础）

### 泊松过程

泊松分布是**泊松过程**在单位时间内的事件计数：
- 事件独立发生
- 任意小区间内事件数期望与区间长度成正比（速率 $\lambda$ 恒定）
- 事件间隔服从指数分布（参数 $\lambda$）

### 与其他分布的关系

- $\lambda \to \infty$ 时趋向 $\mathcal{N}(\lambda, \lambda)$
- 二项分布 Bin$(n, p)$ 当 $n \to \infty, p \to 0, np = \lambda$ 时趋向 Poisson$(\lambda)$（稀有事件近似）
- 若 Var $>$ Mean：负二项分布（泊松的过离散版）
- 若事件速率 $\lambda$ 本身是随机变量（服从 Gamma 分布），则计数服从负二项分布

### 判断方法

**检验**：
- 计算样本均值 $\bar{x}$ 和方差 $s^2$，若 $s^2/\bar{x} \approx 1$（离散指数 ≈ 1），支持泊松
- $s^2/\bar{x} > 1$ → 过离散 → 负二项；$s^2/\bar{x} < 1$ → 欠离散 → 二项
- 卡方拟合优度检验（$\chi^2$ goodness-of-fit）

---

## 4. 对数正态（Log-Normal）

### 密度函数

$$p(x) = \frac{1}{x \sigma \sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right), \quad x > 0$$

参数：$\mu$（对数空间中的均值），$\sigma$（对数空间中的标准差）。注意这两个是 $\ln X$ 的均值和标准差，不是 $X$ 本身的。

### 均值与方差

$$\mathbb{E}[X] = e^{\mu + \sigma^2/2}$$

$$\text{Var}(X) = (e^{\sigma^2} - 1) e^{2\mu + \sigma^2}$$

### 形状直觉

- 右偏，只取正值，有单个峰（但峰不在 0 处）
- $\sigma$ 小时接近正态；$\sigma$ 大时右尾很重
- **半对数图**（log $x$ 为 x 轴）上呈**对称钟形**（因为 $\ln X \sim \mathcal{N}(\mu, \sigma^2)$）
- log-log 图上中段也类似直线（容易和幂律混淆）

```
p(x)          log p(x)
  |  /\            |   /\
  | /  \           |  /  \
  |/    \___       | /    \
  +---------> x    +---------> log x
  右偏              对称钟形
```

### 对数正态 vs 幂律的区分

两者都右偏，log-log 图上都"看起来像直线"（中段）。关键区别：
- 对数正态尾部**最终弯曲**（指数衰减更快），幂律尾部永远是直线
- 对数正态在 log-log 图上是**开口向下的抛物线**，幂律是直线
- 统计区分需要似然比检验（对 $x > x_{\min}$ 的样本）

### 应用场景

- **收入分布**：中等收入人群（幂律描述极富裕尾部，对数正态描述中间层）
- 生物体重、细菌大小
- 股票价格变化（短期）
- 网络延迟、文件大小
- 城市规模（和幂律竞争）
- LLM 中的 attention 权重分布

### 来源机制：乘法过程

若 $X$ 是大量独立随机因子的**乘积**（$X = Z_1 \cdot Z_2 \cdots Z_n$），则 $\ln X = \sum \ln Z_i$ 趋向正态（中心极限定理），所以 $X$ 服从对数正态。

这解释了为什么"增长率随机叠加"的过程（财富积累、细胞分裂）产生对数正态。

> **和幂律的来源机制对比**：两者都产生右偏的"大者占多数"现象，但来源不同——
> - **对数正态**来自**乘法叠加**：每步按比例增长（今年收入 = 去年 × 随机增长率），很多步相乘，取对数后变成加法，中心极限定理 → 对数正态。
> - **幂律**来自**优先连接/马太效应**：已经大的更容易变更大（富者愈富、网页已有很多链接更容易被新链接），这种正反馈结构产生幂律，而非乘法叠加。
> 
> 实际数据中两者常常共存：财富分布的中间层是对数正态，最富裕的尾部是幂律。

### 判断方法

**图形**：画 $\ln x$ 的直方图，若呈钟形，支持对数正态；或画 Q-Q 图对 $\ln x$ 用正态参考线。

**检验**：对 $y = \ln x$ 做正态检验（Shapiro-Wilk、K-S、Anderson-Darling）。

---

## 5. 混合高斯（Gaussian Mixture）

### 密度函数

$$p(x) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x \mid \mu_k, \sigma_k^2)$$

参数：混合权重 $\pi_k > 0, \sum \pi_k = 1$；每个成分的 $\mu_k, \sigma_k^2$。

### 均值与方差

$$\mathbb{E}[X] = \sum_k \pi_k \mu_k$$

$$\text{Var}(X) = \sum_k \pi_k(\sigma_k^2 + \mu_k^2) - \left(\sum_k \pi_k \mu_k\right)^2$$

方差 = 组内方差的加权均值 + 组间方差（$\sum \pi_k(\mu_k - \bar{\mu})^2$）。

### 形状直觉

$K$ 个高斯叠加，可以拟合任意多峰分布。两峰之间有谷（若峰间距 $> \sigma$）。

### 应用场景

- 运动预测的轨迹输出：多种可能路线（直行/左转/右转）各对应一个高斯成分
- 聚类（GMM 是软聚类的概率模型，EM 算法求解）
- 语音识别的声学模型（HMM 的发射概率）
- 自然语言处理的语义空间建模

### 拟合方法

EM 算法（Expectation-Maximization）：
- E 步：给定当前参数，计算每个点属于每个成分的后验概率（软分配）
- M 步：用软分配更新 $\pi_k, \mu_k, \sigma_k^2$
- 迭代直到收敛（对数似然不再增长）

### 判断方法

- 直方图有多个峰 → 考虑混合高斯
- 用 BIC（贝叶斯信息准则）选 $K$：BIC $= -2\ln L + p\ln n$，选 BIC 最小的 $K$
- 注意：$K$ 个成分不一定对应 $K$ 个自然聚类，GMM 是密度估计工具，不一定有语义含义

→ 详见 [高斯混合模型](gaussian-mixture-model.md)

---

## 6. Pareto 分布

### 密度函数

$$p(x) = \frac{\alpha x_m^\alpha}{x^{\alpha+1}}, \quad x \geq x_m > 0, \quad \alpha > 0$$

参数：$\alpha$（形状/尾指数），$x_m$（最小值，尺度）。

### 均值与方差

$$\mathbb{E}[X] = \frac{\alpha x_m}{\alpha - 1} \quad (\alpha > 1)$$

$$\text{Var}(X) = \frac{x_m^2 \alpha}{(\alpha-1)^2(\alpha-2)} \quad (\alpha > 2)$$

### Pareto vs 幂律的关系

**Pareto 分布就是幂律分布的标准统计学名称**，两者描述同一形式，只是参数化方式略有不同：
- 幂律习惯写成 $p(x) \propto x^{-\alpha}$（物理学符号）
- Pareto 的 PDF 中指数是 $\alpha+1$（因为归一化后多出来的 1）
- Pareto 的尾指数 $\alpha$ = 幂律的 $\alpha - 1$

**Pareto 法则（80/20）**：$\alpha = \log 5 / \log 4 \approx 1.16$ 时，最富有的 20% 持有 80% 的财富。

### 应用场景

- 财富/收入分布的最富裕尾部
- 保险索赔金额（极端损失）
- 互联网流量（少数用户消耗大多数带宽）

---

## 7. Weibull 分布

### 密度函数

$$p(x) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} \exp\left(-\left(\frac{x}{\lambda}\right)^k\right), \quad x \geq 0$$

参数：$k > 0$（形状），$\lambda > 0$（尺度）。

### 均值与方差

$$\mathbb{E}[X] = \lambda \, \Gamma\!\left(1 + \frac{1}{k}\right)$$

$$\text{Var}(X) = \lambda^2 \left[\Gamma\!\left(1 + \frac{2}{k}\right) - \Gamma\!\left(1 + \frac{1}{k}\right)^2\right]$$

其中 $\Gamma$ 是 Gamma 函数。

### 形状直觉（关键：形状参数 $k$）

| $k$ | 故障率（hazard rate） | 形状 | 类比 |
|---|---|---|---|
| $k < 1$ | 递减（早期失效） | 极右偏，$x=0$ 处概率最高 | 婴儿死亡率、早期 bug |
| $k = 1$ | 常数 | 指数分布（无记忆） | 随机故障 |
| $k > 1$ | 递增（老化失效） | 钟形，右偏 | 老化磨损、疲劳破坏 |
| $k \approx 3.6$ | 近似正态 | 对称钟形 | — |

**故障率（hazard rate）**：$h(x) = p(x) / S(x)$，其中 $S(x) = P(X > x)$ 是生存函数。Weibull 的故障率是 $h(x) = (k/\lambda)(x/\lambda)^{k-1}$，$k$ 控制它是递增还是递减。

### 应用场景

- **可靠性工程**：元件寿命建模（$k > 1$ 描述磨损老化）
- 风速分布（风能评估）
- 材料强度（断裂力学）
- 生存分析（患者存活时间）

### 与其他分布的关系

- $k=1$：指数分布
- $k=2$：Rayleigh 分布（无线信号强度）
- $k \approx 3.6$：近似正态

### 判断方法

**Weibull 概率图**（Weibull plot）：令 $\ln(-\ln(1-F(x)))$ vs $\ln x$，若直线则支持 Weibull，斜率 = $k$。这是工程中最常用的寿命分布拟合工具。

---

## 8. Beta 分布

### 密度函数

$$p(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad x \in [0, 1]$$

其中 $B(\alpha, \beta) = \Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)$ 是归一化常数。参数：$\alpha, \beta > 0$。

### 均值与方差

$$\mathbb{E}[X] = \frac{\alpha}{\alpha + \beta}, \quad \text{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

### 形状直觉（$\alpha, \beta$ 的组合决定一切）

| $\alpha, \beta$ 关系 | 形状 | 举例 |
|---|---|---|
| $\alpha = \beta = 1$ | 均匀分布 | — |
| $\alpha = \beta > 1$ | 对称钟形，集中在 0.5 | 公平硬币的成功率估计 |
| $\alpha > \beta$ | 左偏，集中在高值 | 成功率偏高的任务 |
| $\alpha < \beta$ | 右偏，集中在低值 | — |
| $\alpha < 1, \beta < 1$ | U 形，两端质量大 | 极端偏向 0 或 1 |
| $\alpha = 0.5, \beta = 0.5$ | 弧形 sine 分布 | — |

### 为什么 Beta 分布无处不在：共轭先验

Beta 是二项分布的**共轭先验**：若成功概率 $p \sim \text{Beta}(\alpha, \beta)$，观测到 $k$ 次成功 $n-k$ 次失败后，后验是 $\text{Beta}(\alpha+k, \beta+n-k)$。这使得贝叶斯更新极为简洁。

### 应用场景

- 贝叶斯估计成功率（点击率、转化率、A/B 测试）
- 机器学习中的超参数先验（dropout 率、混合权重）
- 文本建模（LDA 主题模型中文档-主题分布的先验）
- 物理中的 $[0,1]$ 上的任意分布（用 $\alpha, \beta$ 灵活拟合）

---

## 9. Dirichlet 分布

### 密度函数

Beta 的多维推广，定义在 $K$ 维单纯形（$\sum_{k=1}^K x_k = 1, x_k \geq 0$）上：

$$p(\mathbf{x}) = \frac{1}{B(\boldsymbol{\alpha})} \prod_{k=1}^K x_k^{\alpha_k - 1}$$

其中 $B(\boldsymbol{\alpha}) = \prod_k \Gamma(\alpha_k) / \Gamma(\sum_k \alpha_k)$。参数：$\boldsymbol{\alpha} = (\alpha_1, \ldots, \alpha_K)$，$\alpha_k > 0$。

### 均值与方差

$$\mathbb{E}[X_k] = \frac{\alpha_k}{\alpha_0}, \quad \text{Var}(X_k) = \frac{\alpha_k(\alpha_0 - \alpha_k)}{\alpha_0^2(\alpha_0 + 1)}$$

其中 $\alpha_0 = \sum_k \alpha_k$（称为"浓度"参数）。

### 形状直觉（$\alpha_0$ 控制集中程度）

- $\alpha_0$ 小（如每个 $\alpha_k = 0.1$）：稀疏，倾向于集中在某个顶点（近似 one-hot）
- $\alpha_0$ 大（如每个 $\alpha_k = 10$）：集中，分布均匀，接近均匀分布
- $\alpha_k$ 均等：对称，$\mathbb{E}[X_k] = 1/K$

### 应用场景

- **LDA（Latent Dirichlet Allocation）**：文档-主题分布的先验 $\text{Dir}(\alpha)$，主题-词分布的先验 $\text{Dir}(\beta)$
- 多项分布的共轭先验（Dirichlet 之于多项，如同 Beta 之于二项）
- 贝叶斯非参数（Dirichlet Process）
- 混合模型的权重先验

### 与 Beta 的关系

$K=2$ 时 Dirichlet$(\alpha_1, \alpha_2)$ = Beta$(\alpha_1, \alpha_2)$。

---

## 10. 负二项分布（Negative Binomial）

### 质量函数（离散）

$$P(X = k) = \binom{k+r-1}{k} p^r (1-p)^k, \quad k = 0, 1, 2, \ldots$$

参数：$r > 0$（目标成功次数，可以是非整数），$p \in (0,1)$（成功概率）。常见重参数化：$\mu = r(1-p)/p$，$\phi = r$（过离散参数）。

### 均值与方差

$$\mathbb{E}[X] = \frac{r(1-p)}{p} = \mu$$

$$\text{Var}(X) = \frac{r(1-p)}{p^2} = \mu + \frac{\mu^2}{r}$$

关键：方差 = 均值 + 均值²/r，**总是大于均值**（即负二项总是过离散）。$r \to \infty$ 时趋向泊松。

### 过离散（Overdispersion）是什么

泊松要求 Var = Mean。真实数据中计数往往 Var $\gg$ Mean，称为**过离散**。原因通常是：
- 个体间异质性（不同用户的点击率不同，但都用同一个泊松）
- 事件之间有聚集效应（一次交通事故引发连环追尾）

负二项可以理解为：每个个体的 $\lambda$ 服从 Gamma 分布，对 $\lambda$ 边际化后得到负二项。

### 应用场景

- 用户行为计数（购买次数、页面访问次数）：有过离散，比泊松更合适
- 基因组学：RNA-seq 的基因表达量（DESeq2 用负二项）
- 流行病学：感染人数（超级传播者导致过离散）
- 文本中词频建模

### 判断方法

计算离散指数 $D = s^2 / \bar{x}$：
- $D \approx 1$：泊松
- $D > 1$：过离散 → 负二项
- $D < 1$：欠离散 → 二项或其他

---

## 11. 学生 t 分布（Student's t）

### 密度函数

$$p(x) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\!\left(\frac{\nu}{2}\right)} \left(1 + \frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}$$

参数：$\nu > 0$（自由度，degrees of freedom）。位置-尺度版本：$p((x-\mu)/\sigma)/\sigma$。

### 均值与方差

$$\mathbb{E}[X] = 0 \quad (\nu > 1), \quad \text{Var}(X) = \frac{\nu}{\nu - 2} \quad (\nu > 2)$$

$\nu \leq 1$：均值不存在；$\nu \leq 2$：方差无穷大。

### 形状直觉

- 对称，单峰，形如正态但**尾部更重**（heavy-tailed）
- $\nu$ 越小，尾部越重（极端值概率越大）
- $\nu = 1$：Cauchy 分布（均值、方差都不存在）
- $\nu \to \infty$：趋向标准正态 $\mathcal{N}(0,1)$
- 实践中 $\nu > 30$ 基本等同正态

```
p(x)
  |     正态（细尾）
  |    /----\
  |   / 学生t \
  |  /  (重尾)  \
  |-/            \-
  +-----------------> x
```

### 应用场景

- **小样本统计推断**：$n$ 很小时，$(\bar{X} - \mu)/(S/\sqrt{n}) \sim t_{n-1}$，$t$ 检验就是基于此
- 鲁棒回归：正态误差对异常值敏感，换成 $t$ 分布误差可以降低异常值影响（因为 $t$ 的尾部概率更大，不会把异常值视为"几乎不可能"）
- 深度学习的权重先验（比正态更容忍偶尔出现的大权重）
- 金融中建模极端收益（比正态更好地刻画肥尾）

### 与其他分布的关系

- $\nu = 1$：Cauchy 分布
- $\nu \to \infty$：$\mathcal{N}(0,1)$
- $t^2 \sim F(1, \nu)$（F 分布）
- 若 $X_1 \sim \mathcal{N}(0,1)$，$V \sim \chi^2_\nu$，则 $X_1/\sqrt{V/\nu} \sim t_\nu$

---

## 12. 分布关系总图

```
二项(n,p)
  |  n→∞, p→0, np=λ          n→∞, p固定
  ↓                              ↓
泊松(λ)                      正态(np, np(1-p))
  |  λ本身~Gamma               
  ↓                        
负二项(r,p)                  
  |  r→∞                      
  ↓                            
泊松(λ)                      

指数(λ) = Gamma(1,λ)
  |  k个之和
  ↓
Gamma(k,λ)
  | 标准化后的比
  ↓
Beta(α,β)
  | K维推广
  ↓
Dirichlet(α)

Weibull(k=1,λ) = 指数(λ)
Weibull(k≈3.6) ≈ 正态

t(ν→∞) → 正态
t(ν=1)  = Cauchy

Pareto(α,x_m) = 幂律（不同参数化）
对数正态：log-log 图中段似直线，但尾部弯曲（≠幂律）
```

---

## 13. 判断方法汇总

### 图形工具

| 图形 | 画法 | 支持的分布 |
|---|---|---|
| **直方图** | 直接画 | 看形状，多峰→混合高斯 |
| **log-log 图** | x轴log，y轴log（通常画CCDF） | 幂律→直线；对数正态→弯曲抛物线 |
| **半对数图** | x轴线性，y轴log | 指数→直线 |
| **Q-Q 图** | 理论分位数 vs 样本分位数 | 通用，偏离直线说明不拟合 |
| **Weibull 图** | $\ln(-\ln(1-F))$ vs $\ln x$ | Weibull→直线 |
| **$\ln x$ 的直方图** | 对 $\ln x$ 画直方图 | 对数正态→钟形 |

### 统计检验

| 检验 | 用途 | 工具 |
|---|---|---|
| **KS 检验**（Kolmogorov-Smirnov）| 通用拟合优度，比较 ECDF 和理论 CDF | `scipy.stats.kstest` |
| **Anderson-Darling** | 比 KS 对尾部更敏感 | `scipy.stats.anderson` |
| **Shapiro-Wilk** | 正态性检验（对数正态：先取 log 再检验） | `scipy.stats.shapiro` |
| **$\chi^2$ 拟合优度** | 离散分布（泊松/负二项） | `scipy.stats.chisquare` |
| **离散指数** $D = s^2/\bar{x}$ | 区分泊松/负二项/二项 | 手算 |
| **变异系数** CV $= s/\bar{x}$ | CV≈1 支持指数，CV>1 重尾 | 手算 |
| **幂律似然比检验** | 区分幂律 vs 对数正态 | `powerlaw` Python 包 |
| **MLE + BIC** | 选混合高斯的 $K$ | `sklearn.mixture.GaussianMixture` |

### 快速判断流程

```
数据是离散计数？
  ├─ 均值≈方差 → 泊松
  ├─ 方差>均值 → 负二项
  └─ 0/1 结果，有固定 n → 二项

数据是连续正值，右偏？
  ├─ 半对数图直线 → 指数分布
  ├─ log(x) 钟形 → 对数正态
  ├─ log-log 图直线（尤其尾部） → 幂律/Pareto
  ├─ 有已知的最小值、寿命问题 → Weibull
  └─ 需要灵活拟合 [0,1] → Beta

数据有多个峰？
  └─ 混合高斯（用 BIC 选 K）

数据对称、比正态尾部重？
  └─ 学生 t 分布（小样本或鲁棒建模）

数据是概率向量（K个成分之和=1）？
  └─ Dirichlet 分布
```

---

## 和 wiki 内其他概念的关联

- [幂律与 Scaling](power-law-and-scaling.md)：幂律分布的详细推导和 LLM scaling law 应用
- [高斯混合模型](gaussian-mixture-model.md)：GMM 的 EM 算法推导和在运动预测中的用途
- [ccNet / Perplexity](perplexity.md)：语言模型困惑度和词频泊松近似
- [RLHF](rlhf.md)：Beta 分布作为奖励模型的先验（Bradley-Terry 模型的概率基础）
