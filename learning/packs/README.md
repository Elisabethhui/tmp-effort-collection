# R4–R11 教学包总入口

R2–R3 已经完成基础与 Transformer 主干。本目录把后续路线拆成可独立审核、学习和验收的教学包；本轮先完成“教案和知识地图”，不提前假设后续代码或 GPU 性能实验已经完成。

## 状态与学习车道

| 标签 | 含义 |
| --- | --- |
| `DRAFT` | 教学包已写成，可审核；尚未代表学习完成 |
| `RUNNABLE_CPU` | 可在当前 Mac 上用小数据或 CPU 验证正确性 |
| `MPS_OPTIONAL` | 可尝试 Apple Silicon MPS，但不作为必过门槛 |
| `SOURCE_READ` | 通过官方源码、调用链、公式和 CPU reference 学习 |
| `REMOTE_GPU` | 只有在远端 GPU 上才做真正的吞吐、显存或多卡实验 |
| `GATED` | 只有通过检索题、代码/源码任务和面试口述后，才可进入 knowledge |

## 包清单

| 阶段 | 教学包 | 当前状态 | 主交付物 |
| --- | --- | --- | --- |
| R4 | [训练闭环与 Decoder LM](./r4-training-loop-decoder-lm.md) | `DRAFT` | tiny LM、label shift、checkpoint、训练诊断 |
| R5 | [后训练：SFT / LoRA / DPO / GRPO](./r5-posttraining-sft-lora-dpo-grpo.md) | `DRAFT` | 数据契约、参数高效微调、偏好优化、奖励账本 |
| R6 | [推理与 Serving 源码阅读](./r6-inference-serving-source-reading.md) | `DRAFT` | generation、KV Cache、Paged/Prefix Cache、服务指标 |
| R7 | [RAG 与搜索](./r7-rag-search.md) | `DRAFT` | 检索管线、混合召回、重排、可追溯回答、评估 |
| R8 | [Agent Runtime / MCP / Memory](./r8-agent-runtime-mcp-memory.md) | `DRAFT` | 工具契约、状态图、重试幂等、记忆与人工介入 |
| R9 | [评测与分布式训练](./r9-evaluation-distributed.md) | `DRAFT` | 评测集、轨迹评分、回归门禁、通信/显存账本 |
| R10 | [面试治理与综合项目](./r10-interview-capstone.md) | `DRAFT` | 题库、三档回答、项目证据、模拟面试 |
| R11（可选） | [多模态扩展](./r11-multimodal-optional.md) | `DRAFT / BACKLOG` | ViT、视觉语言对齐、图像 RAG 与多模态 Agent |

R11 不阻塞 R4–R10；如果面试岗位明确要求视觉、多模态或语音，再从 R10 的项目证据切入。

## 每个包的固定结构

每个包都按相同的闭环编排：

1. `Mission`：它为什么服务于面试目标；
2. `Prerequisites`：哪些 R2–R3 知识必须能回忆；
3. `Knowledge units`：按 What → Why → Mechanism 组织，不堆名词；
4. `Mac validation lane`：明确 `RUNNABLE_CPU`、`MPS_OPTIONAL`、`SOURCE_READ` 或 `REMOTE_GPU`；
5. `Planned labs`：只描述下一步代码验收，不把计划当成已完成；
6. `Interview rehearsal`：同一主题准备 30 秒、3 分钟、15 分钟版本；
7. `Acceptance gate`：通过后才允许写入长期 `knowledge/`；
8. `Primary sources`：论文负责原理，官方文档负责 API，源码负责调用链和边界。

## 建议学习顺序

```text
R2/R3 回忆
   ↓
R4 训练闭环与 Decoder LM
   ↓
R5 SFT / LoRA / DPO / GRPO
   ↓
R6 推理、KV Cache、Serving
   ├──────────────┐
   ↓              ↓
R7 RAG        R8 Agent Runtime
   └──────┬───────┘
          ↓
R9 评测与分布式
          ↓
R10 面试治理与综合项目
          ↓（可选）
R11 多模态
```

R6 可以和 R5 并行阅读，但真实 Serving 优化要等推理的 shape、缓存和指标都能解释后再做。R7/R8 以小型、可追溯的 CPU 项目为主；R9 先做评测和账本，再阅读多卡实现。

## 单包学习循环

```text
读一个知识单元（≤30 分钟）
  → 不看资料写公式/shape/伪代码
  → 跑 CPU tiny lab 或完成源码调用链
  → 做 3 道 retrieval questions
  → 录一遍 30 秒 + 3 分钟面试回答
  → 记录错误与证据
  → 通过 gate 后才晋级到 knowledge
```

学习记录不预先生成。只有用户实际完成检索、代码或口述并得到反馈后，才在 `learning/learning-records/` 创建记录；这避免把“看过教案”误记成“已经掌握”。

## Mac 约束的统一解释

- CPU 是默认正确性环境：数据集小、张量小、优先观察 shape、数值和失败模式。
- MPS 只做可选烟测；先检查 `torch.backends.mps.is_available()`，不把 CUDA kernel、吞吐或显存结论迁移到 Mac。
- GPU 专题先走 `SOURCE_READ`：阅读官方源码，画调用链，写 CPU reference，再把远端 GPU benchmark 标为 `REMOTE_GPU`。
- 每个实验都记录硬件、PyTorch 版本、数据规模、耗时和是否使用近似实现。

## knowledge 晋级门槛

一个结论只有同时满足以下条件，才从教学包提炼到长期知识库：

- 能不看资料解释定义、动机、核心机制和一个反例；
- 能在 CPU tiny lab 中复现关键行为，或能定位官方源码调用链；
- 能回答一个“为什么这样设计”和一个“如果失败怎么办”；
- 能给出至少一条官方来源和验证日期；
- 明确标注 `verified`、`source_read`、`unverified` 或 `remote_gpu_only`。
