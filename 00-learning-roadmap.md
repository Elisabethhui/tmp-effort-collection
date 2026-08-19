# 00｜快速学习路线：算法 / LLM 面试基础与代码能力

> 状态：draft
> 目标：用最短路径建立算法与大模型面试的基础、代码和工程解释能力。
> 适用对象：中国算法岗 / 大模型算法岗 / LLM Engineer / Agent Engineer。

## 1. 学习目标

本路线不追求“读完所有资料”，而追求每个核心问题都能完成五层回答：

```text
What → Why → How → Trade-off → Production
```

每个模块必须留下三类产出：

1. 一页自己的概念/公式笔记；
2. 一个可以运行或验证的最小实验；
3. 一组 30 秒、3 分钟、15 分钟面试回答。

## 2. 官方学习资源地图

| 能力 | 仓库章节 | 首选资源 | 学习产出 |
|---|---|---|---|
| Python / Tensor / Autograd | `08`、`11` | [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro) | tensor、梯度、训练循环 |
| Transformer / Attention | `01` | [PyTorch Transformer Building Blocks](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html) | 手写 MHA，对齐 SDPA |
| 从零理解语言模型 | `01`、`02`、`03` | [Stanford CS336](https://cs336.stanford.edu/) | tokenizer、Transformer、训练、评测、部署链路 |
| Transformers / LLM 工具 | `01`、`02` | [Hugging Face Learn](https://huggingface.co/learn) | tokenizer、model、dataset、训练脚本 |
| SFT / DPO / GRPO | `02` | [Hugging Face TRL](https://huggingface.co/docs/trl/) | 最小 SFT、DPO、GRPO 实验 |
| 推理与 Serving | `03` | [vLLM Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/) | TTFT、TPOT、吞吐、并发对比 |
| DDP / FSDP / TP | `12` | [PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html) | DDP 与 FSDP 通信/显存账本 |
| GPU 集合通信 | `12` | [NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/index.html) | AllReduce、AllGather、ReduceScatter |
| RAG / 检索评测 | `04`、`13` | [Ragas Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) | Recall/MRR/NDCG/faithfulness 对比 |
| Agent / 状态 / 恢复 | `05`、`06` | [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview) 与 [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) | checkpoint、resume、失败恢复 |

优先使用官方课程、论文、文档和代码；面经只用于判断“最近在问什么”，不用于单独证明技术结论。

## 3. 八周学习顺序

### 第 1 周：PyTorch 与张量基础

- PyTorch tensor、autograd、module、optimizer、save/load。
- 对应：`08`、`11`。
- 验收：不用教程复制，独立写出一个可 overfit 的 toy classifier。

### 第 2 周：Transformer / Attention

- Q/K/V、scale、mask、shape、MHA、GQA、RoPE、Norm、FFN。
- 对应：`01`、`08`。
- 验收：手写 MHA，并和官方 scaled dot-product attention 对齐。

### 第 3 周：SFT 与参数高效微调

- causal loss、answer-only loss、LoRA、checkpoint、混合精度。
- 对应：`02`、`10`。
- 验收：20 条 toy data 过拟合；证明 prompt/pad 不计 loss。

### 第 4 周：DPO / PPO / GRPO

- preference data、policy ratio、clip、reference、group advantage、reward failure。
- 对应：`02`、`06`。
- 验收：打印 reward/advantage/ratio，构造一次 reward hacking。

### 第 5 周：推理与 Serving

- prefill/decode、KV Cache、PagedAttention、prefix cache、speculative decoding。
- 对应：`03`。
- 验收：至少完成 KV cache 显存估算；有 Linux/CUDA 环境再做 vLLM benchmark。

### 第 6 周：RAG 与搜索推荐

- BM25、dense retrieval、hybrid/RRF、rerank、chunk、Recall/MRR/NDCG、权限。
- 对应：`04`、`13`。
- 验收：dense-only / sparse-only / hybrid 做一次 ablation。

### 第 7 周：Agent、Memory 与 Evaluation

- tool schema、state、memory/context、checkpoint、retry、idempotency、trajectory eval。
- 对应：`05`、`06`。
- 验收：实现一个带 checkpoint、重试上限和失败 trace 的 toy agent。

### 第 8 周：分布式、项目深挖与模拟面试

- DDP/FSDP/NCCL、系统设计、项目 baseline、bad case、成本和可观测性。
- 对应：`07`、`12`、`09`。
- 验收：完成 10 道真实面经的 30 秒/3 分钟/15 分钟回答。

## 4. 每日节奏

默认每天 60～90 分钟：

1. 20 分钟：读一个知识节点；
2. 25～35 分钟：写公式、代码或实验；
3. 15 分钟：做一道真实面经题；
4. 10 分钟：口头回答并录下不会的追问；
5. 5 分钟：更新错题和下一次实验。

每周至少形成：一个概念卡、一个实验结果、五道完整回答和一条复盘记录。

## 5. 代码能力分级

| 等级 | 必须能做什么 | 对应验证 |
|---|---|---|
| L0 | 看懂 tensor shape、loss、optimizer | 手写训练循环 |
| L1 | 手写 MHA / Softmax / CE / InfoNCE | 单元测试与数值对齐 |
| L2 | 写 SFT / LoRA / DPO toy trainer | toy data、checkpoint 恢复 |
| L3 | 解释并测 KV Cache / Serving | 显存账本、延迟/吞吐指标 |
| L4 | 写 Hybrid RAG 和评测 | ablation、Recall/NDCG、失败分类 |
| L5 | 写可恢复 Agent | state、checkpoint、retry、trace |
| L6 | 解释 DDP/FSDP/NCCL | 通信、显存和故障恢复账本 |

## 6. 当前环境门槛

当前仓库没有 Python 依赖或 Lab 代码；本机检测到 Python 3.14.3，但 `torch`、`transformers`、`datasets`、`trl`、`vllm`、`langgraph`、`ragas` 都未安装。

先建立隔离环境，再安装依赖。PyTorch 官方当前 macOS 指南将 Python 3.10～3.14 列为推荐范围；vLLM 的 Serving 实验应放在有合适 Linux/CUDA 环境的机器或云实例中。

## 7. 学习与仓库更新门禁

- 面经新增：先进入 `09-real-interviews/` 或待审核区；
- 真实重复题：只增加证据，不复制答案；
- 新追问：更新对应主教材；
- 新机制：补教材、代码和来源；
- 没有实验结果的“会写”：只能标为未验证；
- 每次更新都记录来源、日期、Question ID 和验证状态。

## 8. 第一轮七天行动

1. 建立 Python 隔离环境并跑通 PyTorch 基础例子；
2. 学完 `01` 的 Q/K/V、scale、mask、shape；
3. 写出纯 PyTorch MHA；
4. 和官方 SDPA 做小样本数值对齐；
5. 用 DeepSeek / 美团真实题做 30 秒和 3 分钟回答；
6. 记录失败案例和代码 bug；
7. 提交第一份学习记录，再进入 SFT。
