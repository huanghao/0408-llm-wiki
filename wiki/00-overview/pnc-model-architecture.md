# PNC 神经网络模型：架构、规模与训练数据

> 数据来源：`planning_model_architecture.md` 文档 + DolphinFS 实测（`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-pnc/`）

---

## 整体设计：不是一个模型，是一组模型的流水线

```
感知数据（ego / 障碍物 / 路段 / 信号灯）
        ↓
  CommonFeatureExtractor（预处理，上传 GPU）
        ↓
  SharedEncoder / beta / gamma Encoder（场景理解，每帧只跑一次）
        ↓
  SharedEncoderFeature [1, 1082, 1, 256]（被所有 Decoder 共享）
        ↓
  各 Decoder（并行，各自完成不同规划子任务）
        ↓
  候选轨迹 + 置信度
        ↓
  规则层（碰撞检测 / 代价函数 / 轨迹精化）
        ↓
  最终执行轨迹
```

**设计动机**：同一帧场景数据同时支撑多个决策任务（生成轨迹、预测障碍物、变道判断、会车避让……）。Encoder 只跑一次，输出的 `SharedEncoderFeature` 被所有 Decoder 复用，是工程上的核心优化。

---

## Encoder 层

### SharedEncoder（BaseModel）

**输入**（5 个 tensor）：

| Tensor | Shape | 大小（float32） |
|--------|-------|----------------|
| ego_input | [1, 40, 8] | 1.25 KB |
| interact_obstacles | [64, 40, 12] | 120 KB |
| static_obstacle | [32, 1, 60] | 7.5 KB |
| road_graph_segment | [450, 1, 13] | 22.9 KB |
| traffic_light | [10, 40, 10] | 15.6 KB |
| **合计** | | **~167 KB / 帧** |

**输出**：`encoded_feature [1, 1082, 1, 256]`（SharedEncoderFeature，被所有 Decoder 读取）

**模型结构**（Transformer Encoder）：

```
hidden dim (d_model)：384
num_heads：6
num_layers：6
FFN dim：1536
input token 数：1082
```

**参数量估算**：

```
Per layer ≈ 4 × 384² (attention) + 2 × 384 × 1536 (FFN) ≈ 1.77M
6 层 ≈ 10.6M
Input projectors（5 个 modality）≈ 1–2M
Output projector (384→256) ≈ 0.1M
总计：~12–15M 参数
```

### alpha / beta / gamma / omega / delta Encoder

结构和 SharedEncoder 完全一样，权重不同，主要差异在感知范围和障碍物数量上限：

| Encoder | num_static_obstacles | max_distance | 状态 |
|---------|---------------------|--------------|------|
| beta | 32 | 50m | 当前主力，enabled |
| gamma（mapless） | 64 | 100m | 无地图场景 |
| delta | — | — | disabled（测试占位符）|

---

## Decoder 层

### 当前生产高频 Decoder

| Decoder | 额外输入 | 输出 | 用途 |
|---------|---------|------|------|
| query_decoder | query 向量 | 轨迹 + 置信度 | 新一代主力轨迹生成（2026-03 活跃）|
| lane_keep_decoder | 车道中心线特征 | 车道保持轨迹 | 车道保持场景 |
| general_decoder | 参考线路段 | 轨迹 [9, 160] + 置信度 | 通用轨迹生成（老主力）|
| lane_selection_decoder | SEGMENT_SEQUENCE [10,10,60] | 轨迹 [10,80] + 置信度 [10] | 10 条候选参考线各生成一条轨迹 |
| prediction_decoder | 障碍物语义特征 | 障碍物未来轨迹 | 预测周边障碍物运动 |
| lane_change_decoder | 变道目标车道特征 | 变道轨迹 | 变道决策 |
| meeting_decoder | 对向来车特征 | 避让轨迹 | 窄路会车 |
| vad_decoder | 避障特征 | 避障决策 | 主动避障 |

### GeneralDecoder 参数量估算

```
输入：encoded_feature [1, 634, 1, 256] + reference_line [1, 50, 1, 12]
输出：轨迹 [9, 160]（9 条多模态候选，每条 80 点 × 2 维）

cross-attention + FFN × 若干层 ≈ 3–5M
输出头（regression + classification）≈ 0.5M
总计：~4–6M 参数
```

### 场景专用 Decoder（变道 / 会车 / 预测）

参数量更小（~1–3M），训练数据量也更少（场景筛选后数万帧量级）。

**实测**：meeting_decoder 的 onnx 文件 8.4 MB，对应 ~2.1M 参数（8.4 MB ÷ 4 bytes/float32），和估算范围吻合。

---

## 规则层（非神经网络）

规则层不是神经网络，是手写代价函数和约束检查：

- **碰撞检测**：过滤与障碍物碰撞的候选轨迹
- **代价函数打分**：安全（weight=5.0）+ 交通灯（weight=5.0）+ 速度 + 效率 + 边界 + 稳定性
- **轨迹精化**：投影到参考线，速度平滑，碰撞回避微调

**作用**：神经网络负责生成候选，规则层负责安全兜底和多模型仲裁。

---

## 参数量汇总

| 模型 | 参数量 | 说明 |
|------|--------|------|
| SharedEncoder（BaseModel） | ~12–15M | Transformer Encoder，d=384，6 层 |
| GeneralDecoder | ~4–6M | Cross-attention Decoder |
| 场景专用 Decoder | ~1–3M | 变道 / 会车 / 预测等 |
| **单次推理总参数** | **~20–25M** | Encoder × 1 + 活跃 Decoder × N |

对比参考：BERT-base 110M，GPT-2 117M。PNC 模型是**典型的车端轻量级模型**，整体参数量约为 BERT-base 的 1/5，在车载芯片上可以实时推理。

**在 LLM 坐标系里的定位**：12–15M 参数的 Encoder 在 LLM 里属于极小模型——GPT-2 最小版本是 117M，BERT-tiny 是 4M，DistilBERT 是 66M。PNC 的 SharedEncoder 大约相当于 BERT-tiny 的 3 倍，是专门为车载实时推理裁剪的尺寸，不追求通用语言理解能力，只需要对固定格式的驾驶场景特征做有效编码。

**Encoder + Decoder 才是完整参数量**：推理时 SharedEncoder（~12–15M）和各 Decoder（~4–6M 每个）都要加载，单次推理的总参数量是 ~20–25M。但训练时 Encoder 和 Decoder 是**分开独立训练的**（见训练资源一节），不是联合训练，所以训练时的"模型参数量"取决于当前训练的是哪个组件。

---

## 训练数据规模（DolphinFS 实测）

### SharedEncoder（BaseModel）训练集

**最新训练任务实测**（MLP 任务 `psx69ee3d36x6w9cswd5`，2026-04-27）：

数据路径：`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-pnc-zw04/planning/planning_feature_exporter_task_list/manual/liyujing02/shared_encoder/ray_common_262_260303/ssfl/`

| 指标 | 数值 |
|------|------|
| Train clips | **2,013,629** |
| Val clips | 287,660 |
| 平均 clip 时长 | 36.8 秒 |
| 总时长（train） | **20,610 小时** |
| 10Hz 总帧数（train） | **~742M 帧** |
| 存储（train 目录） | 22 GB（ssfl 子集） |
| merged_feature_pool_train.bin | 2.4 GB |

**clip ≠ 训练样本**：一个 clip 是一段连续的驾驶录像（平均 37 秒），包含多帧传感器数据，但并非每帧都会成为训练样本——低速、静止、质量差的帧会被过滤。训练时的一个**样本（sample）是单帧**，但 clip 里只有满足条件的帧才会被导出为 feature。

**实测**：从 `merged_feature_pool_train.bin` 的文件头直接读取：

```
total_samples = 311,884,224（~3.1 亿训练样本）
clips = 2,013,629
每 clip 平均有效帧数 ≈ 155 帧
clip 平均时长 36.8s × 10Hz = 368 帧/clip（理论上限）
实际采样率 ≈ 155 / 368 ≈ 42%（约 58% 的帧被过滤）
```

clip 是数据管理的单位（采集、标注、筛选都以 clip 为粒度），帧是训练的单位，两者之间有一层 feature export 过滤。

**历史版本对比**（来自 `01_train_val_list/` 索引文件）：

| 数据集版本 | Train clips | 备注 |
|-----------|------------|------|
| @67（2024 年） | 154,499 | 早期版本 |
| @80 | 72,011 | 筛选版本 |
| @86 | 288,525 | 2025 年版本 |
| **ray_common_262**（最新） | **2,013,629** | 2026-03 最新，规模扩大 7× |

### 场景专用 Decoder 训练集

| 模型 | 典型 clip 数 | 说明 |
|------|------------|------|
| 变道（lane_change） | 数万 | 场景筛选，只含变道片段 |
| 会车（meeting） | 数万 | 只含窄路会车片段 |
| 预测（prediction） | 数十万 | 障碍物轨迹预测，覆盖更广 |
| neural_planner 实验集 | 2,706–3,264 | DolphinFS 实测，专项实验数据集 |

**DolphinFS 数据格式实测**：`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-pnc/` 下已有 `perception.parquet`、`planning.parquet` 等结构化数据文件，说明 Parquet 格式在 PNC 数据 pipeline 中已有实际使用。

---

## 训练资源（MLP 实测）

**SharedEncoder 是独立训练的 encoder-only 模型**：训练时没有 Decoder 参与，Encoder 自己有一个独立的训练 loss（通常是轨迹回归 loss——给定场景特征，预测 ego 的未来轨迹）。这个 loss 直接作用在 Encoder 的输出上，不需要 Decoder。

训练流程：
```
输入特征（5 个 tensor）
    ↓ SharedEncoder
encoded_feature [1082, 256]
    ↓ 训练专用的轻量 head（不是生产用的 Decoder）
预测轨迹
    ↓
与 GT 轨迹计算 loss（回归 + 分类）
    ↓
反向传播更新 Encoder 权重
```

**为什么不联合训练**：独立训练让 Encoder 的权重更新只依赖自己的 loss，不会被某个 Decoder 的特定任务"带偏"。训练完成后，Encoder 的输出作为固定特征供各 Decoder 使用——Decoder 训练时 Encoder 权重冻结，只更新 Decoder 自己的参数。这和 BERT 预训练 + 下游任务 fine-tune 的逻辑类似，区别在于 PNC 的 Decoder 通常不 fine-tune Encoder，而是完全冻结。

**训练过程看什么**：主要看 loss 曲线下降趋势 + 在 val 集上的轨迹回归误差（ADE/FDE，即平均/最终位移误差）。不像分类任务有明确的 accuracy，轨迹预测的好坏需要结合仿真评估或实车测试才能最终验证。

### MLP 实测数据（两个典型任务，同一份数据集）

| | 任务一（psx69ee3d36x6w9cswd5） | 任务二（psx69eb4181x39hapc9n） |
|--|------|------|
| 任务类型 | openloop 评估 | 完整训练 |
| 完成时间 | 2026-04-27 | 2026-04-26 |
| **耗时** | **~48 分钟** | **~52.6 小时** |
| GPU 配置 | 64× A100 80GB（8 workers × 8） | 64× A100 80GB |
| **GPU 利用率（平均）** | **15%** | **78%** |
| **显存（平均/峰值）** | **4.6 / 13.8 GB** | **78.5 / 87 GB** |
| 数据集 | ray_common_262_260303 ssfl | ray_common_262_260303 ssfl + 场景增强 |

**两个任务的差异解读**：
- 任务一耗时短、GPU 利用率低（15%）——是 openloop 验证任务，不是完整训练，主要做推理而非梯度更新
- 任务二耗时 52 小时、GPU 利用率 78%、显存接近 80GB 上限——是真正的完整训练
- 任务二名称含 `add-800w-Uturn-500w-Turn-600w-Nudge-500w-Meeting`，说明在基础数据上叠加了多个场景的专项数据（U 形转弯 800 万帧、转弯 500 万帧、Nudge/会车各 500 万帧）

**关键观察**：完整训练时显存接近 80GB 上限（78.5 GB 平均），说明当前 batch size 已经接近极限，和之前"I/O 是瓶颈"的结论需要区分任务类型——评估任务 I/O 受限，完整训练任务显存受限。

| 模型 | GPU 数 | 典型训练时长 | 显存使用 |
|------|--------|------------|---------|
| SharedEncoder（完整训练） | 64× A100 | ~50 小时 | ~78 GB（接近上限）|
| SharedEncoder（openloop 评估） | 64× A100 | ~48 分钟 | ~5 GB |
| GeneralDecoder | 4–8× A100 | 1–2 天 | ~1–2 GB |
| 场景专用 Decoder | 2–4× A100 | 数小时~1 天 | <1 GB |

---

## 和业界规模的对比

| | PNC（内部） | nuPlan 公开基准 | Waymo Motion |
|--|------------|---------------|-------------|
| 训练 clip 数 | ~29 万（@86） | ~40 万 | ~10 万 |
| 总时长 | ~4000 小时 | ~1300 小时 | — |
| 模型参数量 | ~20–25M | MTR ~64M | — |
| 定位 | 车端实时推理 | 学术基准 | 学术基准 |

PNC 的训练数据规模已超过 nuPlan 公开数据集，处于**工业界起步到中等规模**之间。模型参数量属于轻量级，适合车载芯片实时部署。

**参数量 × 100 倍样本的估算与存储匹配**：

按经验法则"参数量 × 10–100 倍训练样本"：

```
25M 参数 × 100 = 2.5B（25 亿）训练样本
每个样本 ~167 KB（单帧 feature tensor）
2.5B × 167 KB = ~418 TB（无压缩）
```

但实测 DolphinFS 存储是 106 TB（@86 版本），实测训练帧数是 1.44 亿帧。

**两个数字为什么不匹配**：

1. **实际训练样本远少于"理论需求"**：1.44 亿帧 vs 25 亿的理论需求——这说明 PNC 模型没有达到"充分训练"的数据饱和点，或者说还有提升空间。这和训练失败率高、I/O 是瓶颈的观察一致：当前的制约因素不是模型容量，而是数据 pipeline 的效率。

2. **106 TB vs 1.44 亿帧 × 167 KB = 24 TB 的差距**：106 TB 存的不只是当前版本的 feature，还包含历史版本、多种格式的中间产物、不同 Encoder 版本的输出 feature 等。实际训练消费的数据量约 24 TB，其余是历史积累。

3. **"10–100 倍"是 NLP 经验值，不一定适用于驾驶场景**：驾驶数据的时序相关性强——同一条路上相邻帧的信息高度重叠，有效信息密度远低于独立同分布的 NLP token。实际需要的"有效多样样本数"可能远小于 25 亿。

---

## 相关文档

- [planning_model_architecture.md](../../bots-ws/2026-04-03-pnc_mini_wayformer/planning_model_architecture.md)：完整架构文档，含代码引用和配置文件路径
- [自动驾驶数据 Pipeline 架构](./av-data-pipeline-architecture.md)：数据从采集到训练的完整流程
- [nuScenes 数据集](../30-papers/nuscenes-1903.11027.md)：行业参考数据集，含评估指标设计
