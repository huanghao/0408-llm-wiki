# Comparison of Waymo Rider-Only Crash Data to Human Benchmarks at 7.1 Million Miles（Kusano et al., 2023）

一句话总结：Waymo 在 7.14M rider-only 里程上，有伤害事故率（any-injury-reported）为人类基准的 1/5，有伤害警察记录事故率为人类基准的 45%，在凤凰城和旧金山均达到统计显著，是 ADS 大规模商业部署后第一批严格方法论的安全性对比研究，发表于 Traffic Injury Prevention（2024）。

## 基本信息

- 论文：Comparison of Waymo Rider-Only Crash Data to Human Benchmarks at 7.1 Million Miles
- 作者：Kristofer D. Kusano, John M. Scanlon, Yin-Hsiu Chen, Timothy L. McMurry, Ruoshu Chen, Tilia Gode, Trent Victor
- 机构：Waymo LLC（Mountain View, CA）
- 发表：Traffic Injury Prevention（Taylor & Francis），2024
- arXiv：2312.12675（首次提交 2023 年 12 月，修订版 2024 年 10 月）
- 数据截止：2023 年 10 月底（7.14M rider-only miles）

> **注意**：本论文是严格同行评审的学术论文，不是 Waymo 每年发布的"Safety Report"公司报告。两者互补：公司报告面向公众，方法较简单；本论文专注统计方法严谨性，但覆盖里程更早（7.1M vs 公司报告截止 2025 年底的 170M+）。

---

## 核心问题

**ADS 安全性如何量化，如何与人类基准做有效比较？**

在 L4 级商业部署（rider-only，无司机无远程监控）之前，只能用前瞻性仿真方法预测安全性。部署后，真实事故数据可以**回溯验证**前期预测。这种验证是安全案例（Safety Case）的关键组成部分。

但简单比较 ADS 和人类事故数据面临三个系统性偏差：

1. **报告门槛不同**：NHTSA SGO（ADS 上报义务）要求报告"任何财产损失或人员伤害"；警察记录的人类事故门槛更高（通常要有伤害或较大财产损失）——直接比较会系统性低估 ADS 事故率相对于人类的优势
2. **漏报率不同**：人类事故中约 60% 的财产损失事故和 32% 的伤害事故未向警察报案；ADS 几乎完整报告（传感器记录 + 法规要求）
3. **选择性偏差**：ADS 运营区域（城市表面道路）和驾驶分布（多高密度区域）与全国平均不同，直接用全国平均基准会引入偏差

**本论文的核心贡献**：开发了一套系统处理上述三个偏差的方法论，用三个不同严格度的结果指标分层对比，得出统计显著的安全改善结论。

---

## 方法

### Waymo 的 ODD

论文明确描述了研究覆盖的 ODD（第 5 页，Methods 节）：

- **地理区域**：凤凰城、旧金山、洛杉矶，在固定区域 7×24 运营
- **道路类型**：非限制进入道路（non-limited access roads），**不含高速公路**，含停车场，速度上限 50 mph
- **天气限制**：**不含**浓雾、大雨、沙暴；**含**小雨和轻雾
- **时间**：全天候，无时段限制
- **ODD 稳定性**：2022 年以来基本不变，与本研究大部分里程吻合

### 数据来源

**ADS 数据**：NHTSA SGO（Standing General Order）2021-01 强制上报的事故数据，公开发布。筛选条件：
- Rider-Only（RO）运营阶段的事故（排除有测试工程师在车内的 TO 阶段）
- 车辆为 in-transport 状态（排除停放）
- ADS 车辆受到碰撞影响（排除 Waymo 车未受影响的事故）

**人类基准**：三个来源，对应三个结果指标层次（Table 1）：

| 结果指标 | 人类基准来源 |
|---|---|
| Any Property Damage or Injury | Scanlon et al. (2023) Blincoe 调整版；Flannagan et al. (2023) NDS；Blanco et al. (2016) NDS |
| Police-Reported | Scanlon et al. (2023) 警察记录（未调整） |
| Any-Injury-Reported | Scanlon et al. (2023) Blincoe 调整版和观测版 |

**里程基准（VMT）**：人类基准数据限制在与 Waymo 运营条件相同的区域（对应郡县的地面道路乘用车 VMT），来自 FHWA。

### 关键方法设计

**三个结果指标分层**（从最宽到最严）：

1. **Any Property Damage or Injury（任意财产损失或人身伤害）**：覆盖最广，ADS 报告最完整，但人类基准需要大量漏报调整，不确定性最大
2. **Police-Reported（警察记录）**：中间层，两侧报告制度差异较小，比较更稳定
3. **Any-Injury-Reported（有人员伤害的事故）**：最严格，漏报率最低，结论最可靠

**IPMM（Incidents Per Million Miles）**：标准化指标，每百万英里事故次数。

**统计方法**：Poisson 精确模型计算 IPMM 置信区间；Nelson (1970) 方法计算 ADS 与基准的比值置信区间；95% 置信区间，双侧检验。

**低 Delta-V 排除分析**：额外做了排除 delta-V < 1 mph 的低速接触（撞击速度极小）的敏感性分析，评估报告门槛的影响。

---

## 关键结果

### 里程数（Table 4）

7.14M RO 里程：凤凰城 5.34M + 旧金山 1.76M + 洛杉矶 0.047M

凤凰城和旧金山事故数量足以做统计推断；洛杉矶里程太少，多数比较不显著（结果列入附录）。

### 三类结果指标下的 ADS vs 人类（Table 5-7）

**Any Property Damage or Injury（Table 5 & 6）**：

| 比较 | 人类基准 IPMM | ADS IPMM | 比值 | 统计显著 |
|---|---|---|---|---|
| 凤凰城（Blincoe-adjusted）| 9.43 | 6.2 | 0.65 | ✓ |
| 旧金山（Blincoe-adjusted）| 10.5 | 16.5 | 1.57 | ✗ |
| 全部（里程加权均值）| 9.67 | 8.8 | 0.91 | ✗ |
| 旧金山（NDS ride-hailing）| 64.9 | 19.4 | **0.30** | ✓ |

旧金山 ADS 数值偏高，部分原因是加州财产损失漏报率与全国不同，导致人类基准被低估。

**Police-Reported（Table 7）**：

| 位置 | 人类 IPMM | ADS IPMM | 降低幅度 | 显著 |
|---|---|---|---|---|
| 凤凰城 | 4.31 | 2.2 | **-48%** | ✓ |
| 旧金山 | 5.86 | 1.7 | **-71%** | ✓ |
| 全部（里程均值）| 4.68 | 2.1 | **-55%** | ✓ |
| 全国平均 | 4.10 | 2.1 | **-49%** | ✓ |

**Any-Injury-Reported（Table 7，Blincoe 调整）**：

| 位置 | 人类 IPMM | ADS IPMM | 降低幅度 | 显著 |
|---|---|---|---|---|
| 旧金山 | 5.82 | 0.6 | **-90%** | ✓ |
| 全部（里程均值）| 2.80 | 0.6 | **-80%** | ✓ |
| 全国平均 | 1.76 | 0.6 | **-68%** | ✓ |
| 凤凰城（观测值）| 1.24 | 0.6 | -55%（方向正确但不显著）| ✗ |

**结论摘要（论文 Conclusions 节原文数据）**：
- any-injury-reported：ADS **0.6 IPMM** vs 人类 **2.80 IPMM**，人类是 ADS 的 **5 倍**（-80%）
- police-reported：ADS **2.1 IPMM** vs 人类 **4.68 IPMM**，人类是 ADS 的 **2.2 倍**（-55%）

### 低 Delta-V 排除分析

排除 delta-V < 1 mph 的低速接触后，结果更强：any property damage or injury 在几乎所有比较中均达到统计显著（除旧金山一个比较外），说明大多数 ADS 碰撞是轻微接触，删除后更能反映"真实"碰撞风险。

---

## 局限性

论文 Discussion → Other Limitations 节明确：

1. **统计功效不足**：7.14M 里程在较严重事故（严重伤害、死亡）上事故数极少，无法对这些结果做有意义的统计比较；需要更多里程积累
2. **地理覆盖有限**：仅三城市，凤凰城和旧金山是主要来源，洛杉矶样本太少；不代表其他城市或道路类型
3. **ODD 随时间变化**：Waymo 运营区域从 2020 年起扩张，不同时期的 ODD 和系统版本混合在一起，同一研究无法精确控制版本差异
4. **人类基准匹配不完美**：ADS 是 ride-hailing 服务（路线更集中于高密度区），而基准数据来自同地区全体驾驶者，驾驶密度和路线选择存在差异
5. **任意财产损失指标不可靠**：加州财产损失报告制度与全国差异大，导致旧金山 ADS 碰撞率"看起来"偏高，不能直接得出结论
6. **无乘客占用情况控制**：ADS 在空载行驶（接乘客途中）时如发生事故，乘客风险为零，但仍计入事故率，略微偏高了乘客实际面临的风险

---

## 现状与影响

一句话定性：**这是截至发表时（2023 年底）基于 L4 商业 rider-only 部署、方法论最严格的 ADS 安全性对比研究——它建立了 ADS 安全性评估的标准方法论（三层结果指标 + 漏报调整 + 地理匹配），后续 Waymo 安全论文和行业安全评估框架都直接引用这套方法，结论"ADS 警察记录事故率和有伤害事故率显著低于人类基准"是第一个有统计支撑的大规模实证。**

- **直接后续**：Kusano et al. 2025（arXiv:2505.01515）将里程扩展到 56.7M，分 11 类事故类型做细分分析，延续本论文方法论框架
- **影响 Waymo 公开数据页面**：`waymo.com/safety/impact` 上的统计数据（截至 2025 年底 170M 里程，减少 82-92% 等数字）正是本论文方法的规模化延伸
- **方法论贡献**：三层结果指标（any property damage / police-reported / any-injury-reported）+ IPMM 比较框架被后续研究广泛采用；漏报调整（Blincoe et al. 2023 方法）成为 ADS 安全研究的标准工具
- **对行业争议的回应**：直接反驳了 Cummings（2024）等因报告门槛混用而得出"Waymo 事故率 4 倍于人类"的错误结论，为正确方法论提供了示范
- **局限明确**：论文明确指出 ADS 安全性"不能仅凭事故率得出无不合理风险的结论"，回溯研究是安全案例的必要补充而非充分条件，保持了学术严谨性

---

## 和 wiki 内其他概念的关联

- [ODD：运行设计域](../00-overview/odd-operational-design-domain.md)：本论文清晰定义了 Waymo RO 服务的 ODD（非限制进入道路、不含高速、含小雨轻雾、速度 ≤50 mph、三城市固定区域），是 ODD 工程定义的具体案例
- [SAE J3016](../00-overview/sae-j3016.md)：Waymo RO 是 SAE L4 级 ADS 的典型代表——无司机、在严格 ODD 内自主、ODD 外自行执行 MRC
- [自动驾驶开放生态](../00-overview/av-open-ecosystem.md)：NHTSA SGO 数据是公开的，是 ADS 事故数据最重要的公开来源；本论文展示了如何利用公开 ADS 和人类数据做比较
- [Tesla Data Engine](../00-overview/tesla-data-engine.md)：特斯拉 FSD 是 L2 系统，其安全性评估方法和适用标准完全不同——特斯拉不受 SGO 强制报告约束，也无法做同样的 rider-only 对比

## 值得看的部分

- **Section Methods → Matching of Benchmark and ADS Data（第 4–6 页）**：完整说明了三类偏差（报告门槛、漏报率、选择性偏差）及对应的数据处理方法，是理解 ADS 安全性评估方法论最关键的部分
- **Table 7 + Figure 3**：police-reported 和 any-injury-reported 的完整结果，是本论文最可靠的核心结论；Figure 3 是百分比降低的可视化对比
- **Section Discussion → Comparison to Previous Studies（第 11 页）**：逐一分析前人研究的方法偏差，特别是对 Cummings（2024）"ADS 是人类 4 倍"错误结论的逐项反驳，方法论教学价值极高
- **Appendix A.4（严重伤害和死亡事故的统计功效分析）**：解释为什么当前里程不足以对严重事故做结论，有助于理解"里程积累与统计能力"的关系
- 相关论文：
  - Kusano et al. 2025（arXiv:2505.01515）——56.7M 里程 + 11 类事故细分，本论文的直接续集
  - Scanlon et al. 2023——人类基准数据的来源论文，提供了 ADS 比较所需的漏报调整方法
  - Favarò et al. 2023a——Waymo 安全确定生命周期（Safety Determination Life Cycle）框架，Figure 1 的来源，是理解"回溯分析在安全案例中的角色"的背景文献
