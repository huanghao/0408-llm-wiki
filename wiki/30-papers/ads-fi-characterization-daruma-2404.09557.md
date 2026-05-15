# Characterization and Mitigation of Insufficiencies in Automated Driving Systems（Fu et al., 2024）

一句话总结：首篇对 ADS 功能不足（FI）做系统性分类的实证研究——分析 2500+ DMV 脱离报告 + 10 小时公路测试视频，提出 16 类输出不足（OI）分类表，并发现 FI 是系统故障的 5 倍；进而提出 Daruma 架构设计模式，通过跨异构 AD 通道分析和仲裁在运行时缓解 FI。

## 基本信息

- 论文：Characterization and Mitigation of Insufficiencies in Automated Driving Systems
- 作者：Yuting Fu¹, Jochen Seemann¹, Caspar Hanselaar², Tim Beurskens¹, Andrei Terechko¹, Emilia Silvas²·³, Maurice Heemels²
- 机构：¹ NXP Semiconductors（中央技术办公室，埃因霍温）；² 埃因霍温理工大学机械工程系；³ TNO（荷兰应用科学研究组织）
- arXiv：2404.09557（2024 年 4 月）
- 资助：荷兰研究委员会（NWO）NEON 项目

---

## 核心问题

**ISO 26262 功能安全覆盖"系统故障"，但 ADS 中大量危险来自"功能不足"（FI）——系统按设计运行，但设计本身存在局限。**

在 ISO 21448 SOTIF 框架中，FI 是传感器、执行器或算法（含神经网络、概率计算）实现上的不足，触发后导致危险行为。然而：

1. 文献中**缺乏对 FI 的系统性分类**：现有研究多聚焦于触发条件（何时激活 FI），而非 FI 本身是什么、分布如何
2. **现有系统级缓解方案不足**：现有多通道冗余架构设计用于处理系统故障（多数表决/TMR），但对 FI 的异构性（不同 AD 通道 FI 分布不同）未加利用

本论文同时解决这两个问题：建立 OI 分类体系 + 提出运行时 FI 缓解的 Daruma 设计模式。

---

## 方法

### 第一部分：OI 分类（Output Insufficiency Characterization）

**为什么聚焦 OI 而非 FI root cause**

SOTIF 把 FI 分为性能不足（Performance Insufficiency）和规格不足（Specification Insufficiency），两者根因多样，且在黑盒 ADS 中难以直接观测。相反，OI（输出不足）是 FI 最终体现的**系统状态错误**（漏检、鬼影、轨迹误判），可直接归属到少数几个 ADS 功能模块（感知/预测/规划），更适合分类。

**数据来源**

- **DMV 脱离报告**：2021 年加利福尼亚 DMV 自动驾驶脱离报告，2500+ 条记录，分类脱离原因
- **公路测试视频**：公开的 9 家 ADS 厂商视频，共 32 段、10 小时 17 分钟；通过对比 ADS 显示屏（内部世界模型）与车外真实视频识别 OI

**OI 判断方法**：视频中将 ADS 显示屏上的感知/规划状态与达到 ground truth（外部摄像头/无人机视角）对比，有差异则为 OI。FI 导致的 OI 不会触发 ADS 故障警告，这与系统故障的关键区别。

### 第二部分：FI 缓解——Daruma 架构设计模式

**核心假设**：在异构多通道 ADS 中（不同厂商的 AD 通道），同一时刻不同通道遭遇相同 FI 的概率较低——即各通道 FI 分布互补。

**验证实验**：在 LG SVL 仿真器中，用同一驾驶场景顺序运行三个真实 AD 通道（Baidu Apollo 5.0、Autoware.Auto AVP、Comma.AI openpilot 0.8.10），ego 车辆由固定 Python 脚本控制（开环），确保三通道感受完全相同的环境。

---

## 关键结果

### OI 分类统计

**DMV 报告**（2500+ 脱离记录）：

| 类别 | 占比 |
|---|---|
| FI（insufficiencies）| **69%** |
| 系统故障（fault）| 9% |
| 原因不明 | 14% |
| 超出范围（ODD exit 等）| 8% |

→ **FI 导致的脱离是系统故障的 5 倍**，是 ADS 可用性的主要瓶颈。

**视频分析**（32 段，71 个含 OI 的驾驶场景）：
- 62/71（**87%**）的 OI 场景若驾驶员不介入将导致碰撞或危险

**OI 类别分布**（DMV 报告 vs 视频）：

| 类别 | DMV 占比 | 视频占比 |
|---|---|---|
| 世界模型（感知/定位）| 50% | 40% |
| 运动规划 | 43% | 38% |
| 交通规则 | 6% | 9% |
| ODD | 0% | 13% |

### 16 类 OI 完整分类表（Table 2）

| ID | 类别 | 名称 | 参考标准 | ADS 模块 | 传感器 | 时序 |
|---|---|---|---|---|---|---|
| 1 | 世界模型 | 自车定位错误 | ground truth | 定位 | GNSS/IMU/camera | 零星、持续 |
| 2 | 世界模型 | 地图错误 | ground truth | 地图 | HD map 文件 | 持续 |
| 3 | 世界模型 | 漏检目标 | ground truth | 感知 | lidar/radar/camera/ultrasonic/mic | 零星、持续 |
| 4 | 世界模型 | 鬼影目标 | ground truth | 感知 | lidar/radar/camera/ultrasonic/mic | 零星、持续 |
| 5 | 世界模型 | 目标位置/朝向/尺寸错误 | ground truth | 感知 | lidar/radar/camera/ultrasonic/mic | 零星、持续 |
| 6 | 世界模型 | 目标分类错误 | ground truth | 感知 | lidar/radar/camera/ultrasonic/mic | 零星、持续 |
| 7 | 世界模型 | 可行驶空间识别错误 | ground truth | 感知 | lidar/radar/camera/ultrasonic | 零星、持续 |
| 8 | 世界模型 | 目标轨迹预测错误 | ground truth | 预测 | — | 未来、零星 |
| 9 | 交通规则 | 交通标志/灯/车道标线识别错误 | 交通规则 | 感知 | camera/V2X | 零星、持续 |
| 10 | 交通规则 | 违反交通规则（如让行）| 交通规则 | 运动规划 | — | 零星、持续 |
| 11 | 运动规划 | 反直觉运动规划 | 人类直觉 | 运动规划 | — | 未来、零星 |
| 12 | 运动规划 | 不确定运动规划 | 人类直觉 | 运动规划 | — | 零星 |
| 13 | 运动规划 | 不安全规划轨迹 | 人类直觉 | 运动规划 | — | 零星、持续 |
| 14 | ODD | 天气分类错误 | ground truth | ODD checker | 雨量/能见度传感器 | 持续 |
| 15 | ODD | 道路类型分类错误 | ground truth | ODD checker | 路面传感器/GNSS | 持续 |
| 16 | ODD | 交通状况分类错误 | ground truth | ODD checker | camera/radar/clock/GNSS/V2X | 持续 |

> **参考标准**含义：ground truth = ADS 与物理现实不符；交通规则 = 违反法规；人类直觉 = 驾驶行为令其他道路参与者困惑或危险。

### 仿真实验：异构 AD 通道 OI 分布（Table 5）

| OI 类型 | Apollo | Autoware.Auto | openpilot | 总计 |
|---|---|---|---|---|
| 自车定位错误 | — | 2 | — | 2 |
| 漏检目标 | 2 | 4 | **17** | 23 |
| 鬼影目标 | — | 1 | 3 | 4 |
| 目标位置/朝向/尺寸错误 | 2 | 1 | — | 3 |
| 目标分类错误 | 1 | 1 | — | 2 |
| 可行驶空间识别错误 | — | — | 4 | 4 |
| 目标轨迹预测错误 | 4 | — | — | 4 |

关键发现：**不同通道 OI 类型和频率差异显著**——Apollo 擅长感知但有轨迹预测问题；openpilot 漏检最多（camera-only，无 lidar）；Autoware.Auto 则有定位问题。互补性验证了 Daruma 假设。

三个具体案例：
1. 自行车快速穿越：Apollo 漏检（#3），Autoware.Auto 提前识别——两者互补
2. 前车尺寸过估：Apollo 过估（#5，导致不必要急刹），Autoware.Auto 正确——两者互补
3. 对向卡车预测：Apollo 错误预测卡车会穿越 ego 路径（#8），导致不必要急刹

### Daruma 架构设计模式

```
AD Channel 1 ──→ 世界模型₁ + ego轨迹₁ + 交通规则₁ ─┐
AD Channel 2 ──→ 世界模型₂ + ego轨迹₂ + 交通规则₂ ─┼→ 跨通道分析 ─→ 安全融合 ─→ 高层仲裁器 ─→ 执行器
AD Channel N ──→ 世界模型ₙ + ego轨迹ₙ + 交通规则ₙ ─┘         ↑
故障监控/ODD checker/预测性维护指标 ──────────────────────────┘
```

**跨通道分析**三种算法选项：
1. **几何叠加（Geometric overlay）**：检测各通道世界模型和轨迹的几何交集，两通道对障碍物方向不同但都安全时不触发脱离（改善可用性）
2. **风险计算（Risk-based）**：用一个通道的世界模型评估另一个通道的轨迹碰撞风险，N 个通道产生 N×N 个风险函数矩阵；结合"最后安全介入时间"确定切换时机
3. **机器学习**：神经网络（如 RiskNet）在跨通道高层状态信息上直接预测碰撞风险

**安全融合**：综合跨通道分析结果和传统故障检测，计算每个通道的聚合安全评分（可以是二值：0/1）。

**高层仲裁器**：选择安全评分最高且考虑舒适度/效率/可用性的通道控制车辆。

**核心优势**：出现 FI 时不触发脱离（手动接管），而是**切换到更安全的通道继续自动驾驶**，同时保持对故障的兼容。

---

## 局限性

1. **视频选择偏差**：部分公路测试视频由 ADS 厂商自己发布，可能倾向展示有利场景，影响 FI 统计代表性
2. **仿真通道功能受限**：三个 AD 通道的仿真版本功能不完整（如 Apollo 仿真版未使用 camera-radar-lidar 融合），OI 数量可能偏高或偏低
3. **触发条件难以追溯**：视频中看到 OI 结果，但无法精确识别触发条件和根因（ADS 内部状态不透明）
4. **开环仿真限制**：三通道顺序而非并行运行，且 ego 车由脚本控制，与真实多通道并行驾驶有差距
5. **硬件资源触发的 FI**：实验中发现 GPU 资源不足时（视频帧丢失）会导致漏检，但这类触发条件未在 OI 分类表中体现

---

## 现状与影响

一句话定性：**这是截至 2024 年对 ADS FI 最系统的实证分类研究——16 类 OI 分类表成为 SOTIF 触发条件分析的具体化工具，Daruma 设计模式为工业界提供了第一个针对 FI（而非仅故障）的系统级架构方案，但仿真验证规模有限，实车验证和算法细节仍是未来工作。**

- **对 SOTIF 实践的贡献**：16 类 OI 把 SOTIF 抽象的"功能不足"概念落实为可观测、可测试、可计数的具体类型，每类都有对应的 ADS 模块和传感器，便于工程团队做触发条件分析
- **对架构设计的贡献**：Daruma 是第一个明确针对 FI（而非故障）的系统级设计模式，与 RSS/SFF（单通道安全包络）互补，提供了一个框架将现有冗余架构扩展到 FI 缓解
- **引用状态**：arXiv 2024 年 4 月发布，期刊/会议正式发表状态待查；被[SOTIF wiki 文档](../00-overview/sotif-iso-21448.md)用作 OI 分类的主要公开来源
- **局限认知**：作者明确指出 Daruma 的有效性高度依赖通道质量——如果通道能力太差，切换反而降低整体性能；设计时需优先选择能力互补且整体高质量的通道

---

## 和 wiki 内其他概念的关联

- [SOTIF：预期功能安全（ISO 21448）](../00-overview/sotif-iso-21448.md)：本论文是 SOTIF FI 概念的实证化——16 类 OI 是 SOTIF 理论的具体工程实现，DMV 报告数据为"FI 比故障更频繁"提供了量化证据
- [ODD：运行设计域](../00-overview/odd-operational-design-domain.md)：OI #14-16（ODD 类不足）正是 ODD 边界检测失效的具体表现；论文也把 ODD 限制作为 FI 缓解策略之一
- [SAE J3016](../00-overview/sae-j3016.md)：论文的 AD 通道架构（Figure 1）和 DDT/OEDR 模块分解与 J3016 术语完全对齐
- [Waymo Rider-Only Safety Study](waymo-safety-rider-only-2312.12675.md)：两篇互补——Kusano 等提供了 FI 导致真实事故率的宏观统计；本论文提供了 FI 的微观分类和缓解架构
- [MotionDiffuser](motiondiffuser-2306.03083.md)：MotionDiffuser 的多模态联合预测正是针对 OI #8（目标轨迹预测错误）的组件级改进方向

## 值得看的部分

- **Table 2（第 8 页）**：16 类 OI 完整分类表，含类别/名称/参考标准/ADS 模块/传感器/时序，是本论文最直接能用于工程实践的产出
- **Figure 3（第 4 页）**：ISO 26262 与 SOTIF 的因果关系图对比，清晰区分 Fault→Error→Failure（26262）和 Triggering Condition→FI→OI→Hazardous Behavior（SOTIF）两条路径
- **Figure 5-6（第 7-9 页）**：DMV 报告和视频的 OI 分布统计，"FI 是系统故障 5 倍"的数据来源
- **Figure 9（第 14 页）**：Venn 图示意多通道 FI 互补性，白色孔洞（shared insufficiencies）是所有通道都无法覆盖的能力盲区，是系统设计的核心目标（最小化白洞）
- **Figure 10（第 15 页）**：Daruma 完整架构图，含跨通道分析/安全融合/高层仲裁器的数据流
