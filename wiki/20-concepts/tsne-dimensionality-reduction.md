# t-SNE and Embedding Visualization

t-SNE is a nonlinear dimensionality-reduction method often used to project high-dimensional embeddings into 2D or 3D for visual inspection; it is useful for seeing local neighborhoods, but it is not a quantitative proof of global data quality.

Source references:

- van der Maaten and Hinton, 2008, *Visualizing Data using t-SNE*.
- McInnes, Healy, and Melville, 2018, *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*.
- Used in [MagPie](../30-papers/magpie-2406.08464.md) to visualize instruction embedding coverage across synthetic instruction datasets.

## Core Idea

In LLM data analysis, t-SNE is usually applied after each text sample has already been converted into a vector by an embedding model. For example, MagPie embeds instructions with a sentence embedding model, then projects those vectors onto a 2D plane to compare MagPie-Pro against Alpaca, Evol-Instruct, and UltraChat.

t-SNE does not understand text directly. It only sees vectors and tries to preserve local neighbor structure:

- If two high-dimensional points are close, t-SNE tries to keep them close in the 2D map.
- If a point has a cluster of similar neighbors, t-SNE often makes that cluster visible.
- The axes, absolute coordinates, cluster area, and long-range distances are usually not semantically meaningful.

This makes t-SNE good for exploratory visualization: "Does this dataset appear to cover many local regions of the embedding space?" It is weak evidence for statements like "this dataset is better" or "this cluster is a real task category."

## How To Read A t-SNE Plot

t-SNE 图的核心读法：图上的每个点是一条数据（比如一条 instruction），点的位置由 embedding 距离决定——embedding 相似的点被拉到一起，embedding 差异大的点被推开。

**Useful readings（这些解读相对可靠）：**

- **Dense local groups（密集的局部点群）** 表示这批样本的 embedding 彼此接近，通常意味着语义相似。例如在 MagPie 图中，如果 Alpaca 的点大量聚成几个紧密团，说明 Alpaca 的 instruction 类型比较集中，缺乏多样性。

- **Wider visual spread（图上点分布更分散）** 如果用同一个 embedding 模型和相同参数，分布更广通常暗示数据覆盖更多不同方向。例如 MagPie-Pro 的点散布在整张图的大部分区域，支持其"覆盖广泛任务类型"的主张。注意：只有在其他条件完全一致时才能做跨数据集比较。

- **Strong overlap between two datasets（两个数据集的点高度重叠）** 说明它们覆盖相似的 embedding 区域，但不证明是重复样本（因为 embedding 只是近似）。

- **Isolated islands（孤立的小点群）** 值得手工检查。可能是真实的稀有任务类型，也可能是噪声样本、特殊格式的数据，或者 embedding 模型对某类输入的奇异响应（embedding-model quirks：有些 embedding 模型对特定格式如代码、数学公式处理方式不同，会导致这类样本在 embedding 空间中偏离大多数点）。

Unsafe readings:

- Do not interpret the x-axis or y-axis as named semantic dimensions.
- Do not treat distance between far-away clusters as reliable.
- Do not infer dataset quality from cluster size or plot area alone.
- Do not compare two t-SNE plots if they were produced with different sampling, embeddings, seeds, or hyperparameters.

## t-SNE vs PCA

两者都把高维向量压缩到 2D/3D，但机制和适用场景不同：

| | PCA | t-SNE |
|---|---|---|
| 方法 | 线性投影（找方差最大的方向） | 非线性，把高维邻居关系映射到低维 |
| 速度 | 快，确定性 | 慢，有随机性（需固定 seed） |
| 保留结构 | 全局结构（整体方差分布） | 局部结构（邻居关系） |
| 可解释性 | 坐标轴有意义（主成分） | 坐标轴无意义 |
| 适用场景 | 快速基线检查，降噪预处理 | 探索性可视化，发现局部聚类 |

实践中常先跑 PCA 做快速检查，再用 t-SNE 看细粒度的局部结构。两者都不能用来做量化比较，只是视觉辅助工具。

## Related Techniques

**PCA** is a linear projection method. It is fast, deterministic, and easier to interpret than t-SNE, but often misses nonlinear local structure. Use it as a baseline sanity check before more complex visualizations.

**UMAP** is another nonlinear projection method. It is often faster than t-SNE, can preserve more global structure in some settings, and is widely used for large embedding visualization. Its output is still a visualization, not a ground-truth map.

UMAP 和 t-SNE 的主要区别：
- **速度**：UMAP 通常比 t-SNE 快 5–10 倍，对百万级样本更实用。
- **全局结构**：UMAP 在保留类间距离方面表现更好，不同 cluster 之间的相对位置更可靠（但仍然不是精确距离）。
- **参数**：t-SNE 核心参数是 perplexity（控制邻居数，通常 5–50），UMAP 核心参数是 n_neighbors（控制局部邻域大小）和 min_dist（控制 cluster 紧凑度）。
- **确定性**：两者都有随机性，都需要固定 seed 才能复现。
- **用途**：在 LLM 数据分析中，两者都被用来可视化 instruction embedding 分布，UMAP 在大数据集（>10 万样本）中更常见。

**MDS** tries to preserve pairwise distances more directly. It is conceptually simple but can become expensive and less practical for large embedding datasets.

**Clustering** methods such as k-means or HDBSCAN assign points to groups instead of only drawing a map. They can make visualization more actionable, but the labels are only as good as the embedding space and cluster assumptions.

聚类方法在 t-SNE 可视化中的典型用途：先用 k-means 或 HDBSCAN 对高维 embedding 分组，再在 t-SNE 图上用颜色标注每个点所属的 cluster，让视觉结构更清晰。例如 DEITA 用 k-means 对 65k 条 instruction 做多样性过滤，可以在 t-SNE 上直观看到覆盖的 cluster 是否均匀。局限在于：cluster 标签的质量取决于 embedding 本身是否区分任务，而 embedding 模型不是为此设计的。

**Nearest-neighbor analysis** is often more reliable than reading plot geometry. For data curation, inspecting nearest neighbors can reveal duplicates, templated samples, topic over-concentration, or low-diversity synthetic data.

## Practical Use In Data Papers

For instruction-tuning datasets, a good analysis pipeline usually combines visualization and quantitative checks:

- Embed every instruction with the same embedding model.
- Use t-SNE or UMAP only as a visual sanity check.
- Fix the random seed and report key parameters such as perplexity or number of neighbors.（t-SNE 有随机性，每次运行结果不同。固定 seed 保证复现；报告 perplexity（通常 5–50）和 n_iter 让读者能理解图的性质。如果两个图用了不同 perplexity，它们的 cluster 形状就没有可比性。）
- Pair the plot with category distributions, quality scores, difficulty scores, duplicate checks, and downstream evaluation.（t-SNE 图只是视觉线索，不能作为数据质量的证明。配套的量化指标才是真正的证据：任务类别分布（instruction 属于哪些任务类型，分布是否均匀）；quality/difficulty scores（奖励模型评分、IFEval 等自动质量分）；duplicate checks（embedding 近邻去重，或精确字符串去重）；downstream evaluation（在目标 benchmark 上微调后的实际性能）。MagPie 正是这样做的：t-SNE 图是一张插图，真正支撑其结论的是 reward model scores、task-category statistics 和 MT-bench 评分。）
- Manually inspect representative samples from dense clusters and outliers.

MagPie uses t-SNE in this role: the figure supports the claim that MagPie-Pro covers a broad instruction space compared with several public datasets, but the stronger evidence still comes from filtering analysis, reward-model scores, task-category statistics, and downstream model evaluation.

## Common Confusions

- "t-SNE shows clusters, so the clusters are real categories." Not necessarily. The clusters may reflect embedding bias, projection artifacts, prompt templates, or sampling imbalance.
- "A larger visual area means better diversity." Not by itself. Area changes with parameters and sampling.
- "Two points are far apart, so they are semantically unrelated." Far distances in t-SNE are not reliable.
- "The plot proves this dataset is high quality." It only gives a visual clue about coverage or concentration. Quality needs separate evidence.

## Related Pages

- [Word Embedding](./word-embedding.md)
- [Instruction Tuning](./instruction-tuning.md)
- [Synthetic Data with Verification](./synthetic-data-with-verification.md)
- [MagPie](../30-papers/magpie-2406.08464.md)
- [Deita](../30-papers/deita-2312.15685.md)
