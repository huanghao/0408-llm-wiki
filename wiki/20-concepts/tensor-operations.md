# Tensor 操作参考：reshape、expand、einsum、permute

PyTorch/NumPy 里有几个非直觉的 tensor 操作，在 Attention 和 Transformer 代码里反复出现。这篇文档从"操作改变了什么"的角度逐一解释。

---

## 基础：tensor 是什么

Tensor 是一个多维数组，有两个核心属性：
- **数据（storage）**：一块连续的内存，存着所有元素的数值
- **shape（形状）**：描述这块数据应该被解读为几行几列几层…的元数据

关键认知：**shape 只是对同一块数据的解读方式**，改变 shape 不一定移动数据。

---

## reshape：改变解读方式，不移动数据

```python
x = torch.arange(24)          # [24]，值是 0,1,2,...,23

a = x.reshape(4, 6)           # [4, 6]，同一块内存
b = x.reshape(2, 3, 4)        # [2, 3, 4]，同一块内存
c = x.reshape(6, 4)           # [6, 4]，同一块内存
```

所有操作的内存里都是 `0,1,2,...,23`，只是"几行几列"的解读不同。

**限制**：reshape 要求元素总数不变（$4 \times 6 = 2 \times 3 \times 4 = 6 \times 4 = 24$）。

**为什么非相邻维度也能乘到一起**：从 `[B, T, S, D]` reshape 到 `[B*T, S, D]` 时，T 和 S 不相邻，为什么能直接把 B 和 T 合并？

这需要理解内存布局（row-major / C-contiguous）：

```
tensor [2, 3, 4] 在内存里的顺序：
  [0,0,0] [0,0,1] [0,0,2] [0,0,3]   # 第0行第0列的4个元素
  [0,1,0] [0,1,1] [0,1,2] [0,1,3]   # 第0行第1列
  [0,2,0] [0,2,1] [0,2,2] [0,2,3]
  [1,0,0] [1,0,1] ...
```

**最后一个维度在内存里是连续的，越往前的维度跨度越大。**

所以 `[B, T, S, D]` 在内存里是：先把所有 D 排完，再换 S，再换 T，再换 B。把前两维合并成 `[B*T, S, D]`，只是改变"多少步换一层"的解读——内存里的数据顺序没变，数学上完全等价。

**不相邻的维度合并要先 permute**：如果想把 B 和 S（不相邻）合并，需要先用 permute 把它们挪到相邻位置，再 reshape。

---

## permute / transpose：改变维度顺序（以及为什么不能只靠 reshape）

```python
x: [B, T, S, D]

# transpose：交换两个维度
x.transpose(1, 2)   # [B, S, T, D]，T 和 S 互换

# permute：任意重排维度顺序
x.permute(0, 2, 1, 3)   # [B, S, T, D]，等同于上面
```

**permute 在物理上做了什么**：

permute **不移动数据**，只修改 tensor 的"步长（stride）"元数据。stride 告诉 PyTorch"在第 k 维前进 1 步，在内存里跳多少个位置"。

```
原始 [B, T, S, D]，stride = (T*S*D, S*D, D, 1)
  → 在 B 维移动 1 步 = 在内存里跳 T*S*D 个位置
  → 在 T 维移动 1 步 = 跳 S*D 个位置
  → ...

permute 后 [B, S, T, D]，stride 变成 (T*S*D, D, S*D, 1)
  → 现在在 S 维移动 1 步 = 跳 D 个位置（原来 S 的步长）
  → 在 T 维移动 1 步 = 跳 S*D 个位置（原来 T 的步长）
内存里的数据顺序没变，只是"怎么索引"改变了
```

**为什么 permute 后不能直接 reshape**：

reshape 要求内存是连续的（stride 是递减的标准顺序）。permute 后 stride 变得"乱序"，数据在内存里不连续（从逻辑顺序看），直接 reshape 会出错或得到错误结果。`.contiguous()` 会真正移动数据，把内存重新排成标准连续顺序，然后才能 reshape。

```python
x = x.permute(0, 2, 1, 3)        # stride 变为非标准顺序，逻辑上是 [B, S, T, D]
x.is_contiguous()                 # False，内存不连续
x = x.contiguous()               # 真正移动数据，内存变连续，shape 仍是 [B, S, T, D]
x = x.reshape(B*S, T, D)         # 现在可以安全 reshape
```

**Axial Attention 里为什么必须 permute（不能只靠 reshape 的参数）**：

这是一个关键的坑。假设 T=S=10，输入 `[A, T=10, S=10, D]`：

```python
# 错误做法：只靠 reshape 参数区分，当 T=S 时会混淆
x.reshape(A*S, T, D)   # [A*10, 10, D]——到底是对 T 还是 S 做 attention？
x.reshape(A*T, S, D)   # [A*10, 10, D]——参数完全一样！

# 正确做法：先 permute 把目标序列维度移到倒数第二位，再 reshape
# 想对 T 做 attention（T 是序列，S 和 A 是 batch）：
x_t = x.permute(0, 2, 1, 3)          # [A, S, T, D]，T 在倒数第二位
x_t = x_t.contiguous().reshape(A*S, T, D)  # batch=A*S，序列=T，特征=D
x_t = self_attention(x_t)             # 对 T 维做 attention
x_t = x_t.reshape(A, S, T, D)
x = x_t.permute(0, 2, 1, 3)          # 还原 [A, T, S, D]

# 想对 S 做 attention（S 是序列，T 和 A 是 batch）：
x_s = x                               # [A, T, S, D]，S 已经在倒数第二位
x_s = x_s.reshape(A*T, S, D)         # batch=A*T，序列=S，特征=D
x_s = self_attention(x_s)             # 对 S 维做 attention
x = x_s.reshape(A, T, S, D)
```

关键：**self_attention 总是对输入的第二维（序列维度）做 attention**。所以要对哪个维度做 attention，就要先把那个维度移到倒数第二位（序列位置），其他维度合并进 batch（第一维）。

**attention 到底在对哪一维计算**：

```python
# 输入 [B, T, D]，标准 self-attention
Q, K, V = W_Q(x), W_K(x), W_V(x)    # [B, T, D]
attn = softmax(Q @ K.T / sqrt(D))    # [B, T, T]——T × T 的 attention 矩阵
output = attn @ V                     # [B, T, D]
```

attention 计算的是**第二维（T）里每对 token 之间的相关性**：`attn[b, i, j]` 表示第 b 个 batch，第 i 个 token 对第 j 个 token 的关注度。D 维度是每个 token 的特征，用来计算相似度，但不参与"token 之间的配对"——所以说"对 T 维做 attention"，意思是在 T 个 token 之间建立关联，shape 输入 [B,T,D] 输出 [B,T,D] 不变，但每个 token 的特征已经融合了其他 T-1 个 token 的信息。

---

## expand：广播视图，不复制数据

```python
seeds: [K, D]   # K 个可学习向量

# 想得到 [A, K, D]，让每个 agent 各自有一份 seeds
seeds_expanded = seeds.expand(A, K, D)   # [A, K, D]
# 内存里仍然只有 K*D 个数，通过 stride 技巧复用同一块内存
```

**expand 和 repeat 的区别**：

```python
a = torch.ones(1, 3)         # [1, 3]

b = a.expand(4, 3)           # [4, 3]，不分配新内存，4 行指向同一块数据
c = a.repeat(4, 1)           # [4, 3]，分配新内存，真实复制 4 份

b.data_ptr() == a.data_ptr()  # True，b 和 a 共享内存
c.data_ptr() == a.data_ptr()  # False，c 是独立拷贝
```

**使用时的陷阱**：expand 得到的 tensor 不可原地修改（`b[0] = 0` 会影响所有行）。如果需要修改，要先 `.clone()` 复制一份真实数据。

**expand 要求被扩展的维度原来是 1**（或者添加新维度用 `unsqueeze`）：

```python
seeds: [K, D]
seeds.unsqueeze(0).expand(A, K, D)   # 先在第0维添加 1 维，再 expand
```

---

## einsum：爱因斯坦求和，描述任意张量缩并

einsum 是一种简洁的方式描述"多个张量之间的乘法+求和"操作，比手写矩阵乘法更灵活。

**基本语法**：`torch.einsum("ij,jk->ik", A, B)`

- 字母代表维度名
- `->` 左边是输入，右边是输出
- 出现在输入但不在输出的维度会被求和（缩并）

**常见例子**：

```python
# 矩阵乘法：[I, J] @ [J, K] → [I, K]
torch.einsum("ij,jk->ik", A, B)
# 等价于：A @ B

# Batch 矩阵乘法：[B, I, J] @ [B, J, K] → [B, I, K]
torch.einsum("bij,bjk->bik", A, B)
# 等价于：torch.bmm(A, B)
# 理解：B 维度在输入和输出都有，代表"独立的 B 个矩阵乘法"
# i 和 k 在输出里有，代表行/列索引；j 只在输入有，代表被求和的维度
```

**Attention 分数计算的 einsum**（#4 批注的问题）：

```python
# Q: [B, H, L, D]，K: [B, H, L, D]
# 想计算每个 (B, H) 的 L×L attention 矩阵
torch.einsum("bhld,bhmd->bhlm", Q, K)
```

字母含义：
- `b`：batch，在输入和输出都有 → 独立处理
- `h`：head，在输入和输出都有 → 独立处理
- `l`：Q 的序列位置（第 l 个 query），在输出有 → 保留
- `m`：K 的序列位置（第 m 个 key），在输出有 → 保留
- `d`：特征维度，**只在输入有，不在输出** → 求和（点积）

结果 `bhlm`：第 b 个 batch、第 h 个 head，第 l 个 query 对第 m 个 key 的 attention 分数。`l` 和 `m` 都是序列位置，只是用不同字母区分 Q 和 K 的位置——输出是 L×L 的矩阵（L 个 query × L 个 key）。

```python
# 等价于：
Q @ K.transpose(-1, -2)   # 转置后 K 变成 [B, H, D, L]，Q @ K = [B, H, L, L]
```

**einsum 更难还是更简单**（#5 批注的问题）：

确实不是所有人都觉得 einsum 更直观。einsum 对"习惯矩阵数学符号"的人直观——看字母直接知道哪些维度对齐、哪些被求和。但对"习惯 shape 流动"的人，`Q @ K.T` 更自然，因为可以跟着 shape 想"输入多少行多少列，输出多少行多少列"。

**实际建议**：在 Axial Attention 里，用 reshape+permute 更透明（每步都看得到 shape 变化），用 einsum 更简洁但需要对字母命名有感觉。两种写法在数值上完全等价，选你更容易 debug 的那种。einsum 在**需要描述复杂的跨维度操作**（如张量缩并）时有不可替代的优势，简单的矩阵乘法用 `@` 就够了。

**在 Axial Attention 里用 einsum 的示例**（仅供参考，reshape+permute 版更易读）：

```python
# 沿 S 轴做 attention（T 和 B 是独立的 batch）
# Q: [B, T, S, D_head], K: [B, T, S, D_head]
# 字母：b=batch, t=时间步（保持独立）, s=空间位置 Q 侧, k=空间位置 K 侧（求和后 S×S）, d=特征（求和）
attn = torch.einsum("btsd,btkd->btsk", Q, K)   # [B, T, S, S]，每个 (B,T) 位置的 S×S attention
# 等价于 reshape [B*T, S, D] 后做 @ 操作，然后 reshape 回来
```

---

## 为什么 reshape 和 for 循环本质一样

这是理解 Axial Attention 的核心问题。

**直觉**：`self_attention([B*S, T, D])` 和 "对每个 b,s 组合各自做 `self_attention([T, D])`" 是完全等价的——因为 attention 操作在 batch 维度上是**完全独立的**，没有任何 batch 之间的信息交流。

**数学上**：Attention 的公式是：

$$\text{output}[b] = \text{softmax}\left(\frac{Q[b] \cdot K[b]^T}{\sqrt{d}}\right) \cdot V[b]$$

每个 batch 只用自己的 Q/K/V，不涉及其他 batch。所以：

```python
# 这两种写法完全等价（假设 attention 实现里 batch 维度独立）

# 写法 1：reshape 合并 batch
output = self_attention(x.reshape(B*S, T, D))
output = output.reshape(B, S, T, D)

# 写法 2：for 循环（概念上等价，实际上慢）
for b in range(B):
    for s in range(S):
        output[b, s] = self_attention(x[b, s])   # x[b,s]: [T, D]
```

**为什么 reshape 能工作**（理解这个需要的基础知识）：

1. **内存布局**（上面讲了）：`[B, S, T, D]` 里 B 维度的相邻元素在内存里间隔 `S*T*D` 个位置，S 的相邻元素间隔 `T*D` 个位置，T 的相邻元素间隔 D 个位置，D 的相邻元素是连续的。

2. **矩阵运算对 batch 的语义**：PyTorch 的矩阵乘法（`@` 和 `bmm`）在 batch 维度上**广播**——对 batch 里每个独立的"切片"分别做运算。

3. **等价性**：当我们把 `[B, S, T, D]` reshape 成 `[B*S, T, D]` 时，每个"独立切片"是 `[T, D]`，正好是我们想沿 T 轴做 attention 的序列。batch 里的 `B*S` 个切片之间完全独立，和 for 循环逐个处理等价。

**什么时候 reshape 不等价**：如果 attention 的实现里有跨 batch 的信息交流（比如 batch normalization），那 reshape 就不等价了。但标准 self-attention 没有跨 batch 的交互，所以安全。

---

## 总结：三个操作的用途

| 操作 | 改变什么 | 不改变什么 | 典型用途 |
|------|---------|----------|---------|
| reshape | 数据的维度解读方式 | 内存中的数据顺序 | 合并/拆分维度，批量处理 |
| permute/transpose | 维度的顺序 | 数据内容 | 调整维度顺序，为 reshape 做准备 |
| expand | 维度的大小（广播） | 内存数据（共享） | 广播共享参数，不占额外内存 |
| einsum | 描述任意张量缩并 | — | 精确描述复杂的批量运算 |

---

## 和 wiki 内其他概念的关联

- [Attention 优化技术](./attention-optimization.md)：Axial Attention 用到 reshape+permute，Latent Queries 用到 expand
- [Wayformer](../30-papers/wayformer-2207.05844.md)：附录「expand 与 broadcast」，完整的 seeds.expand 使用说明
- [强化学习基础](./rl-fundamentals.md)：Actor-Critic 里的 tensor 操作
