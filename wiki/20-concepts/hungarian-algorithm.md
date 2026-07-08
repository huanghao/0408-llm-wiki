# 匈牙利算法（Hungarian Algorithm）

二分图最优匹配算法。给定 N 个 worker 和 M 个 task 的代价矩阵，找到总代价最小的一一对应关系。在 DETR 系列中用于训练时把 N 个预测 slot 和 GT 目标配对。

## 核心问题

**如何在两组元素之间找到总代价最小的一一匹配？**

这是**指派问题（Assignment Problem）**：N 个工人、M 个任务，每个工人做每个任务有不同代价，要求每个工人恰好做一个任务、每个任务恰好被一个工人做，使总代价最小。

```
代价矩阵 C（3 个工人 × 3 个任务）：
         Task A   Task B   Task C
Worker 1:   8       4        2
Worker 2:   4       2        6
Worker 3:   6       8        4

最优匹配：Worker 1→Task C(2), Worker 2→Task B(2), Worker 3→Task A(6)
总代价 = 2 + 2 + 6 = 10（最小）
```

暴力枚举所有排列是 O(N!)，匈牙利算法用 O(N³) 解决。

---

## 算法核心思路

匈牙利算法（Kuhn, 1955；Munkres, 1957 给出多项式实现）的核心思想：**通过行列变换把代价矩阵变成有足够多零元素的形式，使得可以找到 N 个零恰好覆盖每行每列各一个**。

**为什么这能保证最优？** 关键性质：从代价矩阵的某一行（或列）减去一个常数 k，所有可行解的总代价都同时减去 k——因为任何一一匹配在每行（每列）只用一个元素。所以行列减法**不改变最优解，只是平移了所有解的代价**。

```
原矩阵每行减去自己的最小值 → 所有元素 ≥ 0
原矩阵每列再减去自己的最小值 → 所有元素 ≥ 0

任何一一匹配的总代价 ≥ 0
如果能找到一组全是零的匹配 → 它的代价 = 0 → 这就是最优解
（不可能比 0 更小，因为所有元素 ≥ 0）

转换回原矩阵：减去的常数们之和就是真实最优代价。
```

所以算法的目标就是"造出足够多的零，让一一匹配能全用零元素"。

### 步骤（以 3×3 矩阵为例）

```
原始代价矩阵：
  8  4  2
  4  2  6
  6  8  4

Step 1：每行减去该行最小值（行归约）
  (8-2) (4-2) (2-2)     6  2  0
  (4-2) (2-2) (6-2)  =  2  0  4
  (6-4) (8-4) (4-4)     2  4  0

Step 2：每列减去该列最小值（列归约）
  (6-2) (2-0) (0-0)     4  2  0
  (2-2) (0-0) (4-0)  =  0  0  4
  (2-2) (4-0) (0-0)     0  4  0

Step 3：用最少的行/列线覆盖所有零
  转换后矩阵：
    4  2  0
    0  0  4
    0  4  0
  零位置：(0,2), (1,0), (1,1), (2,0), (2,2)
  最少需要几条横线/竖线才能覆盖所有零？
    选 column 0（覆盖 (1,0)(2,0)） + row 1（覆盖 (1,1)） + column 2（覆盖 (0,2)(2,2)）
    = 3 条线
  线数 = 3 = 矩阵大小 N → 满足条件，可以找到全零匹配！

  König 定理：最少覆盖线数 = 最大零匹配数。
  线数 < N 意味着还不够零，需要继续制造零（额外调整步骤）。
  线数 = N 意味着零数量充足，存在一一匹配全用零。

Step 4：从零元素中选出一一匹配（每行每列各取一个零）
  Worker 1→Task C (位置 [0,2])    ← row 0 只有一个零
  Worker 2→Task B (位置 [1,1])    ← row 1 有两个零，但 (1,0) 选不了（见下）
  Worker 3→Task A (位置 [2,0])    ← row 2 只剩 (2,0) 可选
```

**为什么不能选 (1,0)？** 如果 Worker 2 选 (1,0)，则 column 0 被占用。Worker 3 在 row 2 的两个零是 (2,0) 和 (2,2)，但 column 0 被占了、column 2 又被 Worker 1 的 (0,2) 占了，**Worker 3 没有可用的零**。所以 (1,0) 这条路走不通。本例的最优匹配是**唯一的**：(0,2)+(1,1)+(2,0)，原代价 2+2+6 = 10。

如果 Step 3 中覆盖线数 < N，需要额外调整（找未覆盖区域最小值，从未覆盖行减去，未覆盖列加上），重复直到线数 = N。

### 不等大小矩阵

当 N ≠ M 时（如 DETR 中 100 个预测 vs 3 个 GT），用虚拟行/列填充为方阵，虚拟元素代价设为 0 或某个常数。

---

## Python 实现

实际工程中直接调用 `scipy.optimize.linear_sum_assignment`：

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

# 代价矩阵：3 个工人 × 3 个任务
cost_matrix = np.array([
    [8, 4, 2],
    [4, 2, 6],
    [6, 8, 4],
])

# 匈牙利算法求解
row_indices, col_indices = linear_sum_assignment(cost_matrix)
# row_indices = [0, 1, 2]     ← 工人编号
# col_indices = [2, 1, 0]     ← 分配到的任务编号

total_cost = cost_matrix[row_indices, col_indices].sum()  # 10

for r, c in zip(row_indices, col_indices):
    print(f"Worker {r} → Task {c}, cost = {cost_matrix[r, c]}")
# Worker 0 → Task 2, cost = 2
# Worker 1 → Task 1, cost = 2
# Worker 2 → Task 0, cost = 6
```

复杂度：O(N³)，对于 DETR 的 N=100/300 完全可行（~毫秒级）。

---

## 在 DETR 中的应用

DETR 的核心创新就是用匈牙利算法替代了 anchor 匹配 + NMS。

### 问题设定

- **预测集合**：N=100 个 slot，每个预测 (类别概率, bounding box)
- **GT 集合**：n_gt 个目标（通常 n_gt << 100）
- **目标**：找到预测和 GT 的最优一一对应，使得匹配到的 slot 预测尽可能接近 GT

### 代价矩阵构造（100 × n_gt）

**类比上面的 Worker/Task 例子**：
- "Worker" → 100 个预测 slot（每行一个）
- "Task" → n_gt 个 GT 目标（每列一个）
- 代价 → 让 slot i 负责 GT j 有多"亏"（cls 错得越多、box 偏得越远，代价越大）

**3 预测 × 2 GT 的小例子**（实际 DETR 是 100 × n_gt）：

```
假设图中有 2 个 GT：GT_狗 和 GT_猫
模型输出 3 个 slot：slot_0, slot_1, slot_2

代价矩阵（行=slot, 列=GT）：
              GT_狗   GT_猫
  slot_0       5.2     1.1   ← slot_0 的预测和猫匹配最好
  slot_1       0.8     6.3   ← slot_1 的预测和狗匹配最好
  slot_2       4.5     5.0   ← slot_2 和两个都不匹配

匈牙利匹配（注意：实际 100 个 slot，n_gt=2 时用虚拟列补齐为方阵）：
  slot_0 → GT_猫（代价 1.1）
  slot_1 → GT_狗（代价 0.8）
  slot_2 → ∅（no object）

每个 slot 的 GT label：
  slot_0: cat（计算 cls + box loss）
  slot_1: dog（计算 cls + box loss）
  slot_2: ∅（只计算 cls loss，box loss 不参与）
```

DETR 中的实际情况：100 行 × n_gt 列，n_gt 通常很小（图中目标数）。匈牙利算法挑出 n_gt 个 slot 负责 n_gt 个 GT，剩下的 100 - n_gt 个 slot 分配给 ∅。



```python
def build_cost_matrix(cls_pred, box_pred, gt_labels, gt_boxes):
    """
    cls_pred:  [100, num_classes+1]   每个 slot 的类别概率
    box_pred:  [100, 4]              每个 slot 的预测框 (cx,cy,w,h)
    gt_labels: [n_gt]                GT 类别
    gt_boxes:  [n_gt, 4]             GT 框
    返回：     [100, n_gt] 代价矩阵
    """
    # 分类代价：预测 GT 类别的概率越高 → 代价越低
    cost_cls = -cls_pred[:, gt_labels]                    # [100, n_gt]

    # L1 代价：预测框和 GT 框的 L1 距离
    cost_l1 = torch.cdist(box_pred, gt_boxes, p=1)       # [100, n_gt]

    # GIoU 代价：预测框和 GT 框的 GIoU（越大越好，取负号）
    cost_giou = -generalized_iou(box_pred, gt_boxes)      # [100, n_gt]

    # 总代价（三者加权组合）
    cost = cost_cls + 5 * cost_l1 + 2 * cost_giou        # [100, n_gt]
    return cost

# 调用匈牙利算法
cost = build_cost_matrix(cls_pred, box_pred, gt_labels, gt_boxes)
pred_indices, gt_indices = linear_sum_assignment(cost.cpu().numpy())
# pred_indices: 被选中的 slot 编号（长度 = n_gt）
# gt_indices:   对应的 GT 编号
# 未被选中的 slot 的 GT label 设为 ∅（no object）
```

### 完整训练流程

```
每次前向传播：
  1. 模型输出 100 个 (cls, box) 预测
  2. 构造 100×n_gt 代价矩阵
  3. 匈牙利算法求最优匹配（O(N³)，N=100 → ~0.1ms）
  4. 匹配到的 slot 计算 cls + L1 + GIoU loss
  5. 未匹配的 slot 只计算 ∅ 分类 loss
  6. 反向传播更新模型（匹配过程本身不需要梯度）
```

**关键点**：匈牙利匹配在前向传播中执行（不是可微的），但匹配结果决定了 loss 的计算方式，间接影响梯度。

---

## 与其他匹配方法的对比

| 方法 | 用在哪里 | 匹配方式 | 是否允许重复 |
|------|---------|---------|------------|
| **匈牙利匹配** | DETR/Deformable DETR | 全局最优一一匹配 | 不允许（每个 GT 只配一个 slot）|
| **Anchor IoU 匹配** | Faster R-CNN/YOLO | 按 IoU 阈值（>0.7=正, <0.3=负）| 允许（多个 anchor 匹配同一 GT）|
| **SimOTA** | YOLOX | 动态正样本数，近似最优匹配 | 允许（每个 GT 匹配多个 anchor）|

匈牙利匹配的一一对应性质是 DETR 不需要 NMS 的根本原因——每个目标只有一个预测 slot 负责，不会产生重复。

---

## 也被用在哪些方法中

wiki 内引用匈牙利匹配的方法：

- **[DETR](../30-papers/detr-2005.12872.md)**：首次把匈牙利匹配引入目标检测，替代 anchor 匹配 + NMS
- **[Deformable DETR](../30-papers/deformable-detr-2010.04159.md)**：继承 DETR 的匈牙利匹配框架
- **[MapTR](../30-papers/maptr-2208.14437.md)**：用匈牙利匹配处理车道线点集的置换等价性
- **[BEVFormer](../30-papers/bevformer-2203.17270.md)**：检测头（Deformable DETR 风格）使用匈牙利匹配

---

## 算法背景

- 提出：Harold Kuhn（1955），命名来自他引用的匈牙利数学家 Koenig 和 Egerváry 的工作
- 多项式实现：James Munkres（1957），因此也叫 **Kuhn-Munkres 算法**
- 复杂度：O(N³)，最优的指派问题算法
- Python：`scipy.optimize.linear_sum_assignment` 是标准实现（内部用 Jonker-Volgenant 算法的改进版本，也是 O(N³)）
