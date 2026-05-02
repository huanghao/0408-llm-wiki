# ONNX

⚠️ 内容基于官方文档与公开资料，非单篇论文总结。

一句话总结：ONNX 是机器学习模型的"中间语言"，把训练框架和推理运行时解耦，让同一个模型可以在不同硬件和平台上部署。

---

## 核心问题

ML 工程里有一个长期存在的摩擦：

- **训练**通常在 PyTorch / TensorFlow 里完成
- **推理**需要在 GPU 服务器、手机、浏览器、嵌入式设备等不同环境下运行
- 这些环境不一定装得了 PyTorch，或者装了也很慢

ONNX 的解决方案是：定义一套与框架无关的标准中间表示（IR），任何框架都可以把模型导出成这个格式，任何支持 ONNX 的运行时都可以加载并执行。

```
PyTorch 模型
TensorFlow 模型   →  导出  →  .onnx 文件  →  部署  →  ONNX Runtime (CPU/GPU/NPU)
scikit-learn 模型                                         TensorRT
                                                          CoreML
                                                          浏览器 (onnxruntime-web)
```

---

## ONNX 格式：核心概念

### 1. 计算图（Graph）

一个 ONNX 模型本质上是一个有向无环图（DAG）：

- **Node（节点）**：一个计算操作，比如 `MatMul`、`Conv`、`Add`、`Relu`
- **Input / Output**：数据流入流出
- **Initializer（初始化器）**：存储在图里的常量，通常是训练好的权重
- **Attribute（属性）**：算子的固定参数，比如卷积的 `kernel_size`，不参与运行时计算

### 2. Opset（算子集版本）

ONNX 用 opset 做版本管理。每个算子都有自己的版本历史，opset 号越高，支持的算子越新。

| opset | 大致时间 | 主要变化 |
|-------|---------|---------|
| 11    | 2019    | 动态 shape 改进 |
| 13    | 2020    | 更多算子，支持 bfloat16 |
| 17    | 2022    | LayerNorm、GELU 等成为标准算子 |
| 20+   | 2023+   | 持续扩展 |

**实践原则**：选 opset 要看目标运行时支持哪个版本。PyTorch 默认导出 opset 17（2.x），ONNX Runtime 会说明支持的 opset 范围。

### 3. 数据类型

ONNX 支持：`float32`、`float16`、`bfloat16`、`int8`、`uint8`、`int32`、`int64`、`bool`、`string` 等。

注意：不是所有运行时都支持所有类型。`bfloat16` 在部分旧版 ONNX Runtime 上可能有问题。

---

## 从 PyTorch 导出 ONNX

### 两种 Exporter

PyTorch 提供两种导出路径：

| | Dynamo Exporter（推荐） | TorchScript Exporter（旧） |
|---|---|---|
| 启用方式 | `torch.onnx.export(...)`，PyTorch 2.6+ 默认 | `dynamo=False` |
| 底层机制 | `torch.export.ExportedProgram` | TorchScript trace/script |
| 控制流处理 | 消除 Python 控制流，转换为图操作 | 记录 trace，循环需特殊处理 |
| 推荐版本 | PyTorch 2.6+ | 旧项目兼容 |

### 基本用法

```python
import torch
import torch.onnx

model = MyModel()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,          # 目标 opset，按运行时支持选
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={             # 指定哪些维度是动态的（旧 API）
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }
)
```

**动态 shape**：如果 batch size 或序列长度在推理时会变，必须通过 `dynamic_axes`（旧）或 `dynamic_shapes`（新，Dynamo）声明，否则导出的模型只支持固定 shape。

### 常见问题

- **超过 2GB 的模型**：需要 `external_data=True`，权重单独存文件
- **含循环的模型**：需要用 `torch.cond` 或重写逻辑
- **自定义算子**：需要为算子写 ONNX 导出逻辑

---

## ONNX Runtime

ONNX Runtime（ORT）是微软开源的推理引擎，是目前最成熟的 ONNX 执行后端。

### Execution Provider（执行后端）

ORT 通过 EP 机制支持不同硬件加速：

| EP | 硬件 | 说明 |
|----|------|------|
| CPUExecutionProvider | CPU | 默认，随处可用 |
| CUDAExecutionProvider | NVIDIA GPU | 需要 CUDA |
| TensorRTExecutionProvider | NVIDIA GPU | 更激进的 kernel 融合，延迟更低 |
| CoreMLExecutionProvider | Apple Silicon / iOS | macOS/iOS 部署 |
| QNNExecutionProvider | Qualcomm NPU | 移动端 NPU |
| OpenVINOExecutionProvider | Intel CPU/GPU | Intel 平台优化 |
| DirectMLExecutionProvider | Windows GPU | DirectX 12 |
| NNAPIExecutionProvider | Android | Android 神经网络 API |

ORT 会自动把图中支持加速的部分分发给对应 EP，其余回退到 CPU。

### 基本推理用法

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: np.random.randn(1, 3, 224, 224).astype(np.float32)})
```

---

## 图优化（Graph Optimization）

ORT 在加载模型时自动做图级优化，分三个层级：

### Basic（默认开启）

- **Constant folding**：静态计算提前算好，不留到推理时
- **Node elimination**：去掉 Identity、冗余的 Slice/Unsqueeze/Dropout
- **算子融合**：把 `Conv + Add`、`Conv + BatchNorm` 合并成单个内核

### Extended

- **Transformer 专项融合**：把 Multi-Head Attention 的多个算子合并成单个高效 kernel
- **GELU、LayerNorm、SkipLayerNorm 融合**
- **BERT Embedding 层优化**

### Layout Optimization（CPU）

- 把卷积的内存布局从 NCHW 转成 NCHWc，提升 CPU 缓存局部性

实践上：ORT 默认开启所有优化，通常无需手动干预。可以用 `session_options.graph_optimization_level` 调整。

---

## 量化

ORT 的量化工具把 FP32 模型压缩成 INT8，大幅减少内存和延迟（见 [量化](./quantization.md) 概念页）。

### 动态量化 vs 静态量化

| | 动态量化 | 静态量化 |
|---|---|---|
| 量化时机 | 推理时实时计算 scale/zero_point | 提前用校准数据计算 |
| 精度 | 通常更高 | 通常稍低 |
| 适用场景 | Transformer、RNN | CNN |
| 是否需要校准数据集 | 否 | 是 |

### 量化格式

- **QOperator**：直接使用 `QLinearConv`、`MatMulInteger` 等量化算子
- **QDQ（Quantize-DeQuantize）**：在标准算子前后插入 Q/DQ 节点，更适合 TensorRT 等工具链

### INT4 / FP16

ORT 还支持：
- **INT4 weight-only 量化**：仅量化权重（不量化激活），对 MatMul 和 Gather 算子压缩效果好，适合 LLM 部署
- **FP16**：用 `onnxconverter_common.float16.convert_float_to_float16()` 转换，需要 GPU 支持

---

## 其他框架的导出工具

| 来源框架 | 工具 |
|---------|------|
| TensorFlow / Keras | `tf2onnx` |
| scikit-learn | `sklearn-onnx`（重写预测函数为 ONNX 算子） |
| LightGBM / XGBoost | `onnxmltools` |
| PyTorch | `torch.onnx.export`（内置）|
| HuggingFace Transformers | `optimum` 库，封装 ORT 导出+推理 |

---

## 验证和调试工具

```bash
# 检查模型结构是否合法
python -c "import onnx; onnx.checker.check_model('model.onnx')"

# 可视化计算图
pip install netron
netron model.onnx
```

**onnxruntime-tools / ort_optimizer**：可以离线保存优化后的图，方便调试和对比优化前后差异。

---

## 局限性

**不是所有操作都能导出**：ONNX opset 里没有的算子（自定义 CUDA kernel、某些 Python 动态操作）导出会失败或需要手写注册。

**控制流支持有限**：复杂的动态图（含 Python 级 if/for）在 TorchScript 路径下需要改写；Dynamo 路径改善了这一点，但仍有边界。

**版本兼容性管理成本高**：PyTorch 升级、ONNX opset 升级、ORT 版本升级，三者需要对齐，经常在框架交界处出现兼容性问题。

**不适合所有 LLM 场景**：超大模型（70B+）的 ONNX 导出和推理有较高工程难度；LLM 推理更多走 vLLM / TensorRT-LLM / llama.cpp 等专用路径，ONNX 主要用于 7B 以下或端侧部署。

**量化工具链碎片化**：ONNX 量化 + TensorRT 量化 + llama.cpp 量化（GGUF）不互通，格式需要独立维护。

---

## 现状与影响

**一句话定性**：ONNX 是目前跨框架模型部署最通用的中间格式，在 CV/NLP 中等规模模型和端侧部署场景仍是主流，但对超大 LLM 推理的覆盖有限。

### 还在普遍使用吗？

是，尤其在这些场景：

- **端侧部署**：手机 APP（CoreML/NNAPI via ORT）、浏览器（onnxruntime-web）、嵌入式
- **工业 CV**：目标检测、分类、OCR，PyTorch → ONNX → TensorRT 是标准三段式
- **HuggingFace Transformers**：`optimum` 库用 ORT 做推理加速，≤7B 模型场景覆盖广

### 被什么限制或取代？

- **超大 LLM**（13B+）：vLLM、TensorRT-LLM、llama.cpp 更专用，更高效
- **Apple Silicon 全栈优化**：MLX 框架直接针对 Apple Silicon，有时比经过 ONNX 转换更快
- **NVIDIA 全链路**：PyTorch → TorchScript → TensorRT，有时绕过 ONNX 更直接

### 核心价值

ONNX 的核心价值不是性能最优，而是**互操作性**：一次导出，多端部署，减少重复实现成本。

---

## 和 wiki 内其他概念的关联

- [量化（Quantization）](./quantization.md)：ORT 的量化工具是 ONNX 生态的重要组成，INT8/INT4 量化减小模型体积
- [开放权重模型全景](../30-papers/open-weight-models-landscape.md)：端侧部署场景中，ONNX 是 7B 以下模型最常用的导出格式之一
- [Phi-1 / Phi-2 / Phi-3](../30-papers/phi-2-phi-3.md)：Microsoft 自家模型，与 ONNX Runtime 有深度整合，Phi-3 有官方 ONNX 版本

---

## 值得看的部分 / 相关资料

- [ONNX 官方文档 - Core Concepts](https://onnx.ai/onnx/intro/concepts.html)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)
- [torch.onnx.export 文档](https://docs.pytorch.org/docs/stable/onnx.html)
- [ORT Graph Optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
- [HuggingFace Optimum](https://huggingface.co/docs/optimum)：封装 ORT 的高层库，适合 Transformers 模型快速转换
- [Netron](https://netron.app)：ONNX 模型可视化工具，浏览器可用
