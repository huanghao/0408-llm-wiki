# 自动驾驶数据 Pipeline 架构

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│  车队采集                                                 │
│  传感器录像（.rec / rosbag）                              │
└─────────────────────┬───────────────────────────────────┘
                      │ 上传
                      ▼
┌─────────────────────────────────────────────────────────┐
│  S3 / 对象存储（冷存储）                                   │
│  原始 rec 文件，只读归档                                   │
│  访问频率低，按需取用                                      │
└─────────────────────┬───────────────────────────────────┘
                      │ Ray 分布式处理
                      │（感知/预测/occ 算法离线跑）
                      ▼
┌─────────────────────────────────────────────────────────┐
│  DolphinFS（热存储，训练直接访问）                         │
│                                                          │
│  metadata.parquet   结构化元数据 + 算法结果               │
│  ├── clip_id, scene_id, timestamp                        │
│  ├── ego_pose（位姿）                                     │
│  ├── perception 结果（3D box 等）                         │
│  ├── prediction 结果                                     │
│  └── tags, labels, split 信息                            │
│                                                          │
│  tensors.zarr       训练用 tensor（已处理完的）            │
│  ├── occ[N, 200, 200, 16]   occ 体素                     │
│  ├── bev[N, C, H, W]        BEV 特征                     │
│  └── ...                    其他模型输入 tensor           │
└─────────────────────┬───────────────────────────────────┘
                      │ DataLoader 读取
                      ▼
┌─────────────────────────────────────────────────────────┐
│  模型训练                                                 │
│  DDP（单机多卡）或 Ray Train（多机多卡）                   │
│  DataLoader: DuckDB 查询 Parquet → zarr 随机访问          │
│  Augmentation 在 DataLoader 里做（轻量，每 epoch 随机）    │
└─────────────────────────────────────────────────────────┘
```

---

## 各层详解

### 原始数据层：S3

存放传感器录像原始文件，只做归档，不参与日常训练。

- 访问场景：重跑算法、排查数据问题、新算法需要从原始数据重新处理
- 格式：原始 rec/rosbag，保持采集时的格式不转换
- 成本：冷存储，按实际访问量计费

### 处理层：Ray

**Ray 是什么**：Python 生态里最主流的分布式计算框架，从设计上面向 ML/AI 工作负载，而不是像 Spark 那样从大数据分析场景移植过来。2017 年由 UC Berkeley RISELab 开发，OpenAI、Uber、Shopify、蚂蚁集团等都有生产使用。

Ray 不是一个单一工具，而是一套围绕同一个调度核心（Ray Core）构建的子系统：

| 子系统 | 场景 | 成熟度 |
|--------|------|--------|
| **Ray Core** | 基础任务调度、分布式函数/Actor | 最稳定，大量生产验证 |
| **Ray Data** | 大规模数据预处理（Spark 的 Python 友好替代） | 较新（2021），API 仍在演化 |
| **Ray Train** | 分布式模型训练（封装 DDP/DeepSpeed） | 稳定，被 Anyscale/OpenAI/Uber 使用 |
| **Ray Tune** | 超参数搜索（HPO） | 业界最常用的 HPO 工具之一 |
| **RLlib** | 强化学习 | 最早的成熟 RL 框架之一 |
| **Ray Serve** | 在线推理服务 | 多家公司生产使用 |

**在这个 pipeline 里的角色**：Ray Data 做离线数据预处理（从 S3 读 rec，跑感知算法，写入 DolphinFS），是 Ray 最成熟的使用场景之一——任务之间独立、可以容错重跑、需要水平扩展，完全命中 Ray 的设计目标。

**注意事项**：Ray Data 的 API 在 1.x→2.x 有 breaking change，新项目直接用 2.x，不要混用。调试比单机复杂，需要配合 Ray Dashboard 查看任务状态和错误信息。

从 S3 读取原始数据，跑感知/预测/occ 等算法，输出写入 DolphinFS。

```python
# Ray 处理的典型模式
ds = ray.data.read_binary_files("s3://raw-recs/")
ds.map(run_perception_pipeline)   # 每条 rec 并行跑感知
  .map(generate_occ_tensors)      # 生成 occ tensor
  .write_parquet("dolphinfs://metadata/")
  .write_zarr("dolphinfs://tensors.zarr/")
```

**Ray Data 读写能力说明**：

| 数据格式 | 读 | 写 |
|---------|----|----|
| 原始二进制（rec/rosbag） | `read_binary_files("s3://...")` | — |
| Parquet | `read_parquet("s3://...")` | `.write_parquet("dolphinfs://...")` |
| zarr | 通过 `map` 自定义写入 | `.map(write_zarr_chunk)` |

从 S3 读 Parquet 再写 Parquet 的典型场景（如合并多批算法结果）：

```python
ds = ray.data.read_parquet("s3://intermediate-results/")
ds.map(merge_and_filter)
  .write_parquet("dolphinfs://metadata/")
```

直接读 rec 的场景更常见：rec 是私有二进制格式，用 `read_binary_files` 拿到字节流后在 `map` 里自行解析（调用内部 rec 解析库）。

Ray 解决的是**计算的分布式**，不是存储——大量 rec 并行处理，水平扩展。

### DolphinFS 本地访问

生产环境里 DolphinFS 挂载在集群节点上，本地开发通常通过 `~/bin/ssh-dolphin` 跳转到挂载节点操作。如果想在本机直接用 `dolphinfs://` 协议路径访问，有以下几种方式：

**方式 1：SSHFS 挂载（推荐用于开发调试）**

```bash
# 通过 ssh-dolphin 拿到挂载节点 IP 后，用 sshfs 把 dolphin 目录挂到本地
sshfs user@dolphin-node:/dolphinfs/mount/point ~/mnt/dolphin \
    -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3

# 之后本地直接访问
ls ~/mnt/dolphin/metadata/
```

挂载后代码里的 `dolphinfs://` 路径需要换成本地挂载路径，或者在代码里做路径映射：

```python
DOLPHIN_BASE = os.environ.get("DOLPHIN_BASE", "dolphinfs://")
# 本地开发时 export DOLPHIN_BASE=~/mnt/dolphin/
```

**方式 2：通过 DolphinFS 客户端 SDK（如果内部有提供）**

部分内部 DolphinFS 实现提供 Python SDK，可以直接在本地通过 RPC 读写，不需要挂载：

```python
import dolphinfs
client = dolphinfs.Client(endpoint="dolphin-gateway:port")
data = client.read("dolphinfs://metadata/metadata.parquet")
```

具体接口参考内部 DolphinFS 文档。

**方式 3：只在集群节点上跑**

对于数据量大的处理任务，本地访问意义不大——直接在挂载节点上跑脚本，或者提交 Ray 任务到集群，本地只负责提交和查看结果。

### 结构化数据层：Parquet

存所有**可以用 SQL 查询的数据**。

```
metadata.parquet 的典型列：
  clip_id          场景唯一 ID
  timestamp_start  场景开始时间
  location         采集地点
  weather          天气条件
  ego_pose         自车位姿序列（JSON）
  perception_v2    感知算法版本 v2 的输出（JSON）
  occ_ready        是否已生成 occ tensor（bool）
  split            train / val / test
  tags             标签列表（JSON array）
  snapshot_id      所属数据集快照 ID
```

**为什么用 Parquet 不用 MySQL**：训练时需要按条件批量扫描（"取所有 occ_ready=true 且 split=train 的 clip"），Parquet 的列式存储对这类范围查询效率远高于 MySQL 行存。MySQL 继续用于血缘关系、数据集管理等需要事务的场景。

### Tensor 层：zarr

存**模型直接消费的 tensor**，训练时不再需要原始传感器数据。

```python
import zarr
# schema 示例
store = zarr.open("tensors.zarr", mode='w')
store.create_dataset('occ',   shape=(N, 200, 200, 16), chunks=(1, 200, 200, 16), dtype='float32')
store.create_dataset('bev',   shape=(N, C, H, W),      chunks=(1, C, H, W),     dtype='float32')
store.create_dataset('label', shape=(N,),               chunks=(1000,),          dtype='int64')
```

**分块策略**：按帧（`chunks=(1, ...)`）分块，DataLoader 随机访问单帧时只加载一个 chunk，不扫描其他帧。

**存 augmentation 前的 tensor**：zarr 里存干净的原始 tensor，augmentation（翻转、旋转、噪声）在 DataLoader 里随机做，每个 epoch 用不同随机数——这样用更小的存储换来了更大的数据多样性。只有 augmentation 极重（需要重跑物理仿真）时才考虑预生成多份存到 zarr。

### 训练层：DataLoader

```python
import zarr, duckdb, torch
from torch.utils.data import Dataset

class AVDataset(Dataset):
    def __init__(self, snapshot_id, data_dir):
        # DuckDB 查询这个 snapshot 包含的所有 clip，及其 zarr 索引
        self.meta = duckdb.query(f"""
            SELECT zarr_idx, label 
            FROM '{data_dir}/metadata.parquet'
            WHERE snapshot_id = '{snapshot_id}' AND split = 'train'
        """).df()
        self.zarr_store = None  # 延迟初始化，避免 fork 问题
        self.data_dir = data_dir

    def _get_store(self):
        # 在 worker 进程里第一次访问时才打开
        if self.zarr_store is None:
            self.zarr_store = zarr.open(f"{self.data_dir}/tensors.zarr", mode='r')
        return self.zarr_store

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        store = self._get_store()
        occ = torch.from_numpy(store['occ'][row.zarr_idx])   # 零拷贝
        # augmentation 在这里做
        occ = random_flip(occ)
        return occ, torch.tensor(row.label)
```

**DuckDB 的角色**：查询 Parquet 做数据集筛选，速度极快，单机足够。不需要 Ray.data——Ray.data 是重计算场景用的，这里只是查询。

**多 GPU 训练**：
- 单机多卡：直接用 PyTorch DDP（`torchrun --nproc_per_node=8`），不需要额外框架
- 多机多卡且已用 Ray：Ray Train 包一层 DDP，方便集群调度
- 多机多卡不用 Ray：DDP + Slurm / Kubernetes

---

## 数据版本管理

### 现有系统（MySQL + Parquet）上的最小改造

不需要引入 DVC 等外部工具，在现有系统上加两个能力：

**能力 1：数据集快照（Dataset Snapshot）**

训练开始前固化这次用的数据集，和模型 checkpoint 绑定：

```sql
-- 在现有 MySQL 里加一张表
CREATE TABLE dataset_snapshot (
    snapshot_id   VARCHAR(64) PRIMARY KEY,  -- 和训练 run_id 绑定
    created_at    TIMESTAMP,
    clip_count    INT,
    filters       JSON,    -- 筛选条件（标签、时间范围、split 等）
    clip_ids      LONGTEXT, -- 所有 clip ID 的列表（或存 Parquet 路径）
    parquet_hash  VARCHAR(64)  -- 对应 Parquet 文件的 MD5，验证数据没变
);
```

训练启动脚本：

```python
snapshot_id = create_snapshot(filters={
    "tags": ["highway", "rainy"],
    "split": "train",
    "occ_version": "v2"
})
# snapshot_id 写入训练配置，和 checkpoint 一起保存
train(snapshot_id=snapshot_id)
```

**能力 2：双向血缘**

现有系统已有 clip → 模型方向，补充反向：

```sql
-- 模型 → clips：这个模型用了哪些数据
SELECT clip_id FROM dataset_snapshot 
JOIN snapshot_clips ON snapshot_id 
WHERE model_run_id = 'run_xxx';

-- clips → 模型：某批数据影响了哪些模型（数据发现问题时快速排查）
SELECT DISTINCT model_run_id FROM dataset_snapshot
JOIN snapshot_clips ON snapshot_id
WHERE clip_id IN ('clip_001', 'clip_002', ...);
```

### 未来扩展参考

当系统规模上来、团队协作需求变复杂时，可以参考：

**Amundsen**（Lyft 开源，2019）

核心功能：数据资产搜索与发现——让工程师能像搜 Google 一样搜数据表、字段、Dashboard。
- 搜索：全文搜索表名、字段名、描述，支持 tag 过滤
- 数据目录：展示表的 schema、样例数据、使用频率、最近更新时间
- 社交层：谁是这张表的 owner、谁最近在用、表的"热度"排名
- 血缘：有基础的上下游血缘展示，但不是核心强项

技术栈：Python（Flask 后端）+ React 前端，元数据存 Neo4j（图数据库，用于血缘关系），搜索用 Elasticsearch，元数据抽取通过各数据源的 extractor 插件（支持 Hive、BigQuery、Snowflake、Redshift 等）。

**DataHub**（LinkedIn 开源，2020）

核心功能：更完整的数据治理平台，覆盖发现、血缘、治理、合规。
- 元数据图谱：以图的方式存储数据实体（表、字段、pipeline、模型、用户）及其关系
- 细粒度血缘：支持字段级别血缘（column-level lineage），不只是表级
- 数据质量：集成 DQ 规则，标注数据健康状态
- 访问控制：数据资产的所有权、分类（PII 等）、访问策略管理
- 实时更新：通过 Kafka 接收元数据变更事件，血缘和目录近实时更新

技术栈：Java（Spring Boot 后端）+ React 前端，核心存储用 MySQL（元数据实体）+ Elasticsearch（搜索）+ Kafka（事件流），图查询用自研的 GMS（Generic Metadata Service）层而非直接用图数据库。支持 Python/Java SDK 推送自定义元数据。

**两者对比**：

| | Amundsen | DataHub |
|--|----------|---------|
| 定位 | 数据发现（搜索为主） | 全栈数据治理 |
| 血缘粒度 | 表级 | 字段级 |
| 部署复杂度 | 较低 | 较高（Kafka 依赖）|
| 社区活跃度 | 中等 | 更活跃（LinkedIn 持续投入）|
| 适合场景 | 中小团队，快速搭数据目录 | 大团队，需要完整治理和合规 |

现阶段不建议迁移，先把快照和双向血缘做好。

---

## 各工具的角色总结

| 工具 | 角色 | 不适合做什么 |
|------|------|------------|
| S3 | 原始数据冷归档 | 训练直接读（延迟高）|
| Ray | 重计算分布式处理 | 元数据查询（太重）|
| DolphinFS | 热数据存储，训练直接访问 | 冷数据长期归档 |
| Parquet + DuckDB | 结构化元数据查询和筛选 | 存大体积 tensor |
| zarr | 训练用 tensor 存储，随机分块访问 | 结构化关系查询 |
| MySQL | 数据集管理、血缘、事务操作 | 大规模列式扫描 |
| DDP | 单机多卡分布式训练 | 跨集群调度 |
| Ray Train | 多机多卡 + Ray 生态集成 | 简单单机场景（太重）|

---

## 相关概念

- [zarr 和 Parquet 的区别](../20-concepts/loss-functions.md)（待补充独立概念页）
- [nuScenes 数据集格式](../30-papers/nuscenes-1903.11027.md)：传感器数据标注的行业参考
- [ScenarioNet](../10-roadmaps/llm-learning-roadmap-20260410.md)：把真实数据重建为可交互仿真场景
