# LLM 推理框架概览

一句话总结：LLM 推理框架沿"服务端高吞吐 vs 本地低资源"两条轴分化——vLLM/SGLang/TensorRT-LLM 面向 GPU 集群多并发，llama.cpp/Ollama/MLX 面向个人设备和边缘场景，核心差异在量化格式、KV cache 管理策略和硬件绑定程度。

---

## 背景：为什么需要专门的推理框架

直接用 HuggingFace Transformers 推理有两个根本问题：

1. **内存碎片**：朴素 KV cache 预分配最大长度的显存，实际使用率低（见 [KV Cache](kv-cache.md)）
2. **批处理低效**：等一批请求全部完成再开始下一批（static batching），GPU 利用率低

专用推理框架的核心价值：**PagedAttention + continuous batching** 的组合让 GPU 利用率从 30–40% 提升到 80%+，吞吐量翻倍。

---

## 服务端 / 高吞吐框架

### vLLM

**名称来源**：v = "virtual"，LLM 的 virtual memory management——核心创新 PagedAttention 把 KV cache 的物理显存管理类比操作系统的虚拟内存分页，"v" 暗指这个设计哲学。

**维护组织**：UC Berkeley Sky Lab（Ion Stoica 组）孵化，2023 年开源后成立 vLLM Project，现由社区主导，Sky Computing Lab 持续贡献，2024 年获 a16z 投资成立商业公司。

**核心技术**：
- PagedAttention：KV cache 按固定大小 block 分页，消除内部/外部碎片
- Continuous batching：每步 decode 后动态加入新请求，不等批次完成
- OpenAI 兼容 API：`/v1/completions`、`/v1/chat/completions` 直接替换

**适用场景**：开放权重模型的服务端部署事实标准；Llama/Qwen/Mistral 等主流模型开箱即用。

---

### SGLang

**名称来源**：Structured Generation Language——不只是推理引擎，更是一套描述 LLM 程序（带控制流、并行调用、结构化输出约束）的 DSL。名字强调"结构化生成"而非单纯"快速推理"。

**维护组织**：UC Berkeley Sky Lab（和 vLLM 同一个组），Lianmin Zheng 主导，2024 年初开源。

**核心技术**：
- RadixAttention：把 prefix cache 组织为 radix tree，自动识别和复用公共前缀（system prompt、few-shot examples）
- 比 vLLM 更激进的 speculative decoding 支持
- 原生支持 JSON schema 约束生成（不需要后处理）

**适用场景**：agent pipeline（长且固定的 system prompt）、RAG（公共上下文复用）、结构化输出（JSON、函数调用）。在这些场景下吞吐超过 vLLM，纯文本生成差距不大。

---

### TensorRT-LLM

**名称来源**：TensorRT 是 NVIDIA 的深度学习推理优化库（Tensor + Real Time），TensorRT-LLM 是其针对 LLM 的专项扩展。

**维护组织**：NVIDIA 官方，2023 年开源，持续由 NVIDIA 工程团队维护。

**核心技术**：
- 计算图优化：算子融合、kernel autotuning（针对 A100/H100 特定硬件）
- In-flight batching（类似 continuous batching 的 NVIDIA 版实现）
- FP8 量化（H100 专属，精度损失极小）
- Tensor parallelism 原生支持

**适用场景**：A100/H100 集群上的极致吞吐；速度最快但只支持 NVIDIA GPU，部署复杂度高（需要 engine build 步骤）。

---

### Text Generation Inference（TGI）

**名称来源**：直白的功能描述，HuggingFace 的文本生成推理服务。

**维护组织**：HuggingFace，2022 年开源，持续官方维护。

**定位**：介于 Transformers（研究）和 vLLM（极致吞吐）之间——比 Transformers 快，比 vLLM 易集成 HuggingFace 生态（模型卡、量化格式、tokenizer 自动加载）。HuggingFace Inference Endpoints 后端即 TGI。

---

## 本地 / 边缘框架

### llama.cpp

**名称来源**：最初就是专门运行 Meta 的 LLaMA 模型的 C++ 实现，所以直接叫 llama.cpp。后来支持范围大幅扩展（Mistral、Qwen、Gemma 等几乎所有主流模型），但名字沿用。

**维护组织**：Georgi Gerganov（保加利亚独立开发者）个人创建，2023 年 3 月发布后迅速成为社区项目，现由 ggerganov/llama.cpp 仓库维护，贡献者数百人。Gerganov 同时是 whisper.cpp（语音识别本地化）的作者。

**核心技术**：
- GGUF 格式（Georgi Gerganov Unified Format，前身 GGML）：自描述的量化模型文件格式，包含权重、分词器、模型配置
- Q2\_K 到 Q8\_0 等多种量化精度，支持 CPU 运行
- CPU + GPU 混合推理：部分层卸载到 GPU（`-ngl` 参数），显存不足时利用内存扩展容量

**适用场景**：无 GPU 或显存不足时运行大模型；嵌入式设备；作为 Ollama 等上层框架的后端。

---

### Ollama

**名称来源**：造词，没有明确词源——创始人解释过灵感来自"llama"（骆驼，LLaMA 模型的吉祥物），前缀 "Ol'" 暗指"好伙伴/老朋友"的随意感，整体定位就是让本地 LLM "像老朋友一样好用"。

**维护组织**：Ollama Inc.，Matt Williams（前 Apple）和 Michael Chiang 联合创立，2023 年成立，已获融资，团队约 20 人。

**定位**：llama.cpp 的用户友好封装，不是独立推理引擎。核心价值：
- 模型仓库（`ollama pull llama3`，自动下载 GGUF）
- 进程管理（后台常驻，自动加载/卸载模型）
- OpenAI 兼容 REST API（`localhost:11434`）
- Modelfile：类似 Dockerfile 的模型配置文件（system prompt、参数、基础模型）

**适用场景**：个人开发者本地快速上手；不需要了解 GGUF 格式和 llama.cpp 参数。

---

### MLX

**名称来源**：Apple Machine Learning eXchange 的缩写——eXchange 暗指统一内存架构下 CPU/GPU 数据无需拷贝的特性。

**维护组织**：Apple 官方，Apple Machine Learning Research 团队，2023 年 12 月开源，持续维护。

**核心技术**：
- 专为 Apple Silicon unified memory 设计：CPU 和 GPU 共享同一块物理内存，零拷贝
- 懒惰求值（lazy evaluation）：不立即执行算子，积累后合并为高效计算图
- MLX-LM：专门针对语言模型的封装库

**适用场景**：Mac M2/M3/M4 用户；在 unified memory（最大 192 GB on M3 Ultra）上跑超大模型；速度通常优于同配置的 llama.cpp。

---

### MLC LLM

**名称来源**：Machine Learning Compilation for LLM——TVM 编译框架的 LLM 专项应用，强调"编译"而非"推理"，针对目标硬件生成优化代码。

**维护组织**：MLC AI 社区，陈天奇（CMU/UW）的 TVM 团队衍生项目，2023 年开源。

**适用场景**：移动端（iOS/Android）、浏览器（WebGPU/WebAssembly）——唯一能把 7B 模型跑在手机上并保持合理速度的主流方案。

---

## 横向对比

| 框架 | 硬件要求 | 并发优化 | 量化格式 | 部署复杂度 | 适用场景 |
|---|---|---|---|---|---|
| vLLM | NVIDIA/AMD GPU | ✅ PagedAttention + continuous batching | GPTQ/AWQ/FP8 | 中 | 服务端主力 |
| SGLang | NVIDIA/AMD GPU | ✅ RadixAttention | GPTQ/AWQ | 中 | agent/RAG/结构化输出 |
| TensorRT-LLM | NVIDIA GPU only | ✅ in-flight batching | FP8/INT8 | 高（需 build engine）| 极致吞吐 |
| TGI | GPU（AMD 支持有限）| ✅ continuous batching | GPTQ/AWQ | 低（HF 生态）| HF 生态部署 |
| llama.cpp | CPU + 可选 GPU | ❌ 单请求为主 | GGUF（Q2~Q8）| 低 | 个人/边缘 |
| Ollama | CPU + 可选 GPU | ❌ | GGUF | 极低 | 个人开发者 |
| MLX | Apple Silicon only | ❌ | MLX 格式 | 低 | Mac 用户 |
| MLC LLM | 手机/WebGPU | ❌ | 编译时优化 | 高 | 移动端/浏览器 |

---

## 选择决策树

```
需要服务多用户（并发）？
├── 是 → GPU 是 NVIDIA/AMD？
│         ├── 是 → 有结构化输出 or 长公共前缀？
│         │         ├── 是 → SGLang
│         │         └── 否 → vLLM（默认）
│         └── NVIDIA 且追求极致吞吐 → TensorRT-LLM
└── 否（个人使用）→
          ├── Mac（Apple Silicon）→ MLX 或 Ollama
          ├── 想要最简单 → Ollama
          ├── 需要低级控制 → llama.cpp
          └── 手机/浏览器 → MLC LLM
```

---

## 和 wiki 内其他概念的关联

- [KV Cache 与 Prompt Cache](kv-cache.md)：vLLM 的 PagedAttention 和 SGLang 的 RadixAttention 都是 KV cache 管理的工程实现；Prompt Cache 对应 SGLang 的 prefix cache 功能
- [量化（Quantization）](quantization.md)：llama.cpp 的 GGUF 和 vLLM 的 GPTQ/AWQ 是两套量化体系；本地框架倾向更激进的 4-bit 量化
- [自回归生成](autoregressive-generation.md)：continuous batching 的本质是在自回归 decode 循环中动态插入新请求，而非等待整批完成
- [LoRA / QLoRA](lora-qlora.md)：QLoRA 微调产出的 adapter 需要在推理框架中合并或动态加载，vLLM/TGI 均支持 LoRA adapter 热加载（多个 adapter 共享基础模型权重）
