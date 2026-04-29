# LLM 数据工程：过滤只是起点

你已经读过 ccNet / fastText / MinHash / DCLM，建立了"过滤和去重"这条线。但数据工程还有三个维度没有涉及，而它们在 2024 年之后变得和过滤同等重要：

## 数据混合配方（Data Mixture）

各来源比例怎么定，比单个来源的质量更关键。Llama 3 在预训练最后阶段把代码/数学比例拉高，就是这个逻辑。

1. **DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining**（Xie et al., 2023，Stanford）  
   链接：https://arxiv.org/abs/2305.10429  
   看点：用小代理模型自动搜索最优数据混合比例，把人工猜比例变成可优化的问题。是 Llama 3 数据配方思路的方法论来源。

2. **Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance**（2024）  
   链接：https://arxiv.org/abs/2403.16952  
   看点：把 scaling law 的思路用到数据混合上，小规模实验预测大规模混合效果。

## 合成数据（Synthetic Data）

真实数据的天花板越来越低，用强模型生成训练数据成为主要补充。

3. **Textbooks Are All You Need / Phi-1**（Gunasekar et al., 2023，Microsoft）  
   链接：https://arxiv.org/abs/2306.11644  
   看点：用 GPT-4 生成"教科书质量"的代码训练数据，1.3B 模型在代码上超越大得多的模型。直接挑战"数据量越大越好"的直觉。

4. **Phi-2 / Phi-3**（Microsoft，2023–2024）  
   链接：https://arxiv.org/abs/2404.14219（Phi-3）  
   看点：把合成数据思路从代码扩展到通用能力，3.8B 模型在多项任务上接近 Llama 3 70B。核心洞察：数据质量对小模型的影响比对大模型更显著。

## 数据质量提升（Data Quality）

过滤是"去掉坏数据"，质量提升是"让好数据更好"——重写、精选、标注细化。这是近两年从合成数据实践中总结出来的独立方向。

5. **Instruction Tuning with GPT-4**（Peng et al., 2023，Microsoft）  
   链接：https://arxiv.org/abs/2304.03277  
   看点：用 GPT-4 重写低质量指令数据，而不是只过滤掉它。质量提升的核心动作：改写 > 删除。

6. **AlpaGasus: Training a Better Alpaca with Fewer Data**（Chen et al., 2023）  
   链接：https://arxiv.org/abs/2307.08701  
   看点：用 ChatGPT 对 52K 条 Alpaca 数据逐条打分，保留高分 9K 条，效果反而更好。核心洞察：质量分布比数量更重要，噪声数据不只是"没用"，还会主动拉低模型质量。

7. **Deita: What Makes Good Data for Alignment?**（Liu et al., 2024，清华）  
   链接：https://arxiv.org/abs/2312.15685  
   看点：系统研究"什么是高质量 SFT 数据"——复杂度（instruction complexity）和质量（response quality）是两个独立维度，同时高才有用。提出了可自动化的数据评分方法。

8. **MagPie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing**（2024）  
   链接：https://arxiv.org/abs/2406.08464  
   看点：直接让对齐好的模型自发生成高质量指令数据，不需要 seed prompt，质量显著高于 Alpaca/Self-Instruct 类方法。

## 数据课程（Data Curriculum）

数据不是一次性全喂，先易后难、先通用后专业会显著影响最终效果。

9. **LIMA: Less Is More for Alignment**（Zhou et al., 2023，Meta）  
   链接：https://arxiv.org/abs/2305.11206  
   看点：仅 1000 条精心挑选的 SFT 数据就能达到强指令跟随能力。和 LIMO 的逻辑在 SFT 阶段的对应版本——质量 >> 数量。

10. **Scaling Laws for Data Constrained Language Models**（Muennighoff et al., 2024）  
    链接：https://arxiv.org/abs/2305.16264  
    看点：数据重复几次的影响——结论是高质量数据重复 4 次比低质量数据不重复更好，为"精选小数据集多轮训练"提供了理论依据。

## 核心认知

数据工程的完整图景是：**过滤/去重**（ccNet/DCLM）→ **合成**（Phi）→ **质量提升**（AlpaGasus/Deita）→ **混合配方**（DoReMi）→ **课程安排**（Annealing/LIMA）。质量提升这一步之前常被跳过，但它往往是小数据集训练效果的关键杠杆。

---

相关：[自动驾驶数据工程](data-engineering-av.md) — 同一套思维在不同模态的应用
