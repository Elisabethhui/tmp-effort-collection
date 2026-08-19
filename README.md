# 2026 算法与大模型面试系统学习库 V2

> 更新时间：2026-08-19  
> 面向岗位：算法工程师 / 大模型算法工程师 / LLM Engineer / Agent Engineer / RAG Engineer / AI Infrastructure 相关岗位  
> 定位：**系统教材 + 面试题库 + 工程实践 + 2026 真实大厂面经索引**

> 快速开始：先看 [`00-learning-roadmap.md`](./00-learning-roadmap.md)，按“概念 → 代码 → 面经 → 模拟回答”推进。

> 官方资源研究：[`sources/learning-resources-research.md`](./sources/learning-resources-research.md)。

---

# 1. 这个仓库解决什么问题

市面上很多“大模型八股”存在三个问题：

1. **只给结论，不讲为什么**：知道名词，但一追问数学、代码、工程就断。
2. **知识和真实面试脱节**：学习了一堆内容，却不知道 2026 大厂到底在问什么。
3. **只会背，不会做**：能说 Attention、GRPO、RAG、Agent，却写不出 MHA、SFT loop，也不会估算 KV Cache 或设计失败恢复。

这个仓库因此按五层能力组织：

```text
L1 基础概念
 ↓
L2 数学 / 机制
 ↓
L3 代码实现
 ↓
L4 工程 Trade-off
 ↓
L5 真实面试追问 / 系统设计 / 项目证据
```

只有五层都能回答，才算真正“掌握”。

---

# 2. 学习地图

```text
                    ┌─────────────────────────┐
                    │  01 Transformer/Attention│
                    └────────────┬────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │  02 Post-training       │
                    │ SFT/PPO/DPO/GRPO/...    │
                    └────────────┬────────────┘
                                 ↓
          ┌──────────────────────┴──────────────────────┐
          ↓                                             ↓
┌──────────────────────┐                    ┌──────────────────────┐
│03 Inference/Serving  │                    │04 RAG/Retrieval      │
└──────────┬───────────┘                    └──────────┬───────────┘
           └──────────────────┬────────────────────────┘
                              ↓
                   ┌──────────────────────┐
                   │05 Agent Runtime      │
                   │Memory/Context/State  │
                   └──────────┬───────────┘
                              ↓
                   ┌──────────────────────┐
                   │06 Agent Data & Eval  │
                   └──────────┬───────────┘
                              ↓
                   ┌──────────────────────┐
                   │07 Project Deep Dive │
                   └──────────┬───────────┘
                              ↓
                   ┌──────────────────────┐
                   │08 Coding / 手撕       │
                   └──────────────────────┘

贯穿所有专题：
09 真实大厂面试专栏 ←→ 10 工程实践 Labs

补齐算法岗底座：11 ML/DL ← 12 分布式训练 ← 13 搜索推荐 ← 14 多模态
```

---

# 3. 文件说明

## 核心教材

1. [`01-transformer-attention.md`](./01-transformer-attention.md)  
   Decoder-only、Self-Attention、MHA/MQA/GQA、mask、RoPE、Norm、SwiGLU、MoE、复杂度、手写 PyTorch、真实面试。

2. [`02-post-training.md`](./02-post-training.md)  
   SFT、causal loss、LoRA、RLHF、PPO、DPO、GRPO、DAPO/GSPO、reward、on/off-policy、TRL/verl、代码与系统问题。

3. [`03-inference-serving.md`](./03-inference-serving.md)  
   Prefill/Decode、KV Cache、FlashAttention、PagedAttention、vLLM、Continuous Batching、Prefix Cache、Speculative Decoding、量化、性能指标。

4. [`04-rag-retrieval.md`](./04-rag-retrieval.md)  
   BM25、Embedding、Bi/Cross Encoder、InfoNCE、Chunking、Hybrid Retrieval、Rerank、RRF、热更新、Recall/NDCG/MRR、Ragas、权限治理。

5. [`05-agent-memory-context.md`](./05-agent-memory-context.md)  
   Workflow vs Agent、ReAct、Function Calling/MCP、State、Memory、Context Engineering、Checkpoint、Retry、Idempotency、Multi-Agent、Eval。

6. [`06-agent-data-eval.md`](./06-agent-data-eval.md)  
   Trajectory、Synthetic Data、User Simulator、Environment、Verifier、Reward、Process/Outcome supervision、Data Flywheel、评测体系。

7. [`07-project-deep-dive.md`](./07-project-deep-dive.md)  
   项目介绍、Baseline、指标、A/B、Bad Case、系统设计、分布式训练、可观测性、项目拷打。

8. [`08-algorithm-coding.md`](./08-algorithm-coding.md)  
   LeetCode 基础 + LLM 岗手写：MHA、Softmax、CE、SFT、InfoNCE、LoRA，以及 2026 AI Coding / 排序题信号。

## 扩展基础与专项

11. [`11-ml-dl-foundations.md`](./11-ml-dl-foundations.md)  
    经典 ML/DL 底座：LR、SVM、GBDT/XGBoost、K-Means/PCA、Loss、Optimizer、Norm、AUC、数据泄漏与评估。

12. [`12-distributed-training.md`](./12-distributed-training.md)  
    DDP、NCCL、ZeRO/FSDP、TP/PP/CP/EP、显存账本、Mixed Precision、Checkpoint/故障恢复。

13. [`13-search-recommendation-ranking.md`](./13-search-recommendation-ranking.md)  
    搜索推荐工业链路、Two-Tower、InfoNCE、负样本、ANN、AUC/NDCG/MRR、LLM 检索模型、索引热更新。

14. [`14-multimodal-vlm.md`](./14-multimodal-vlm.md)  
    ViT、Swin、CLIP、BLIP/BLIP2、Q-Former、LLaVA、Qwen-VL 通用训练逻辑、多模态 SFT/RL、幻觉与评测。

## 面试与实践

9. [`09-2026-real-interviews.md`](./09-2026-real-interviews.md)  
   按公司/岗位/日期整理 2026 真实题目：美团、DeepSeek、百度、阿里、小红书、腾讯、字节等；每道题解释为什么问、怎么答、跳到哪个专题。

10. [`10-engineering-labs.md`](./10-engineering-labs.md)  
   10 个工程实验，把“会背”变成“会写、会测、会诊断”。

   已落地的可运行实验入口：[`labs/README.md`](./labs/README.md)。

## 证据与来源

- [`sources/2026-08-19-sources.md`](./sources/2026-08-19-sources.md)

---

# 4. 高频等级怎么理解

本库不用“看到过一次 = 高频”这种粗糙判断。

| 等级 | 含义 | 学习标准 |
|---|---|---|
| S | 跨公司反复出现 + 核心基础 | 会推导、会手写、会追问 |
| A | 高频工程核心 | 会讲 trade-off + 做系统设计 |
| B | 2026 升温 / 岗位高价值 | 理解机制 + 至少做一个实验 |
| C | 岗位/JD 相关 | 面试前定向补齐 |

---

# 5. 每个知识点的标准学习模板

以后新增内容都遵守：

## 5.1 What

是什么？解决什么问题？

## 5.2 Why

为什么会出现？旧方案哪里不够？

## 5.3 Mechanism

公式、数据流、张量 shape、状态变化是什么？

## 5.4 Code

最小代码怎么写？常见 bug 在哪里？

## 5.5 Engineering

显存、延迟、吞吐、稳定性、并发、成本、安全怎么权衡？

## 5.6 Interview

- 高频问法；
- 中低频追问；
- 陷阱问题；
- 2026 哪些公司问过。

## 5.7 Validation

怎样用最小实验证明自己真的理解？

---

# 6. 建议学习方式

不要从第 1 页一直背到最后一页。采用“专题 + 实验 + 面经”三线循环：

```text
第一遍：读专题，建立 Mental Model
第二遍：自己写公式 / 画数据流
第三遍：跑对应 Lab
第四遍：打开 09 面试专栏做真实题
第五遍：只看标题，口头回答 30 秒 / 3 分钟 / 15 分钟
第六遍：把答不好的问题回写到错题记录
```

---

# 7. 2026 当前最值得优先投入的方向

## 第一优先级

- Self-Attention / MHA 手写
- Transformer block / Norm / RoPE / GQA
- SFT / LoRA
- PPO / DPO / GRPO
- KV Cache / vLLM / PagedAttention / FlashAttention
- RAG pipeline / retrieval metrics
- Agent tool calling / state / memory / context
- LeetCode 常见基础题
- 经典 ML/DL：LR / GBDT / AUC / Optimizer / 数据泄漏
- 分布式训练：DDP / ZeRO / FSDP / NCCL

## 第二优先级：明显升温

- GRPO variants / DAPO / GSPO
- Speculative Decoding
- Agent trajectory / synthetic data
- Agentic RL
- Agent checkpoint / resume / idempotency
- long-context engineering
- Multi-Agent 并发与冲突
- AI System Design / cost / observability

---

# 8. 面试回答的统一结构

遇到陌生问题，不要一上来散讲。使用：

```text
1. 一句话定义
2. 它解决什么问题
3. 核心机制 / 公式 / 数据流
4. 与相邻方案比较
5. 工程实现与瓶颈
6. 一个失败案例或边界条件
7. 如何验证
```

这套结构尤其适合 Transformer、RL、RAG、Agent 和系统设计。

---

# 9. 项目类回答的统一结构

```text
Problem
 → Baseline
 → Change
 → Metric
 → Evidence
 → Failure / Bad Case
 → Trade-off
 → Next Step
```

如果项目只能讲“我用了 LangGraph + RAG + Redis”，但说不清 baseline、提升、失败与证据，2026 的大模型面试很容易被连续追问击穿。

---

# 10. 更新原则

1. 新面经先进入 `09-2026-real-interviews.md`；
2. 同题不复制，在原知识点增加公司/日期/新追问；
3. 新机制才扩充专题；
4. 网友答案不作为最终 Authority；
5. 数学/工程结论优先回到论文、官方文档、官方代码验证；
6. 真实面经和“整理型题库”分级；
7. 每次更新同时检查是否需要新增 Lab。

---

# 11. 当前版本和旧版的本质区别

旧版更像目录摘要；V2 的目标是成为真正能用于：

- 系统学习；
- 面试复习；
- 手写代码；
- 工程设计；
- 大厂面经追踪；
- 长期增量维护

的一套学习仓库。
