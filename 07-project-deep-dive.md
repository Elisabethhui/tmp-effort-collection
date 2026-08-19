# 07｜项目深挖与 AI System Design：把“我做过”变成可验证的工程能力

> 目标：解决 2026 面试最容易挂的地方——项目不是讲技术名词，而是证明你理解问题、选择、指标、失败和 trade-off。

---

## 0. 项目面试的真实结构

越来越多面经会追：

- 数据规模多少？
- 为什么选这个方案？
- baseline 是什么？
- 提升来自哪里？
- 线上 QPS/P99？
- bad case？
- 为什么不用另一种方法？
- 指标怎么定义？
- 如果重新做会改什么？

所以每个项目都应准备：

```text
Problem
→ Constraints
→ Baseline
→ Design
→ Implementation
→ Measurement
→ Failure
→ Trade-off
→ Next iteration
```

---

# Part A｜项目开场：90 秒必须讲清

## 1. 四句话框架 ★★★★★

1. **业务目标**：解决谁的什么问题。
2. **规模/约束**：数据、QPS、延迟、成本、安全。
3. **核心方案**：只说最关键 2-3 个设计。
4. **结果**：离线/线上指标 + 你的贡献。

错误示例：

> “我们用了 LangChain、Milvus、Qwen、RAG、Agent、Redis……”

这是技术清单，不是项目。

---

# Part B｜Baseline

## 2. 为什么没有 Baseline 的“提升 20%”没有意义？ ★★★★★

必须说明：

- 相对什么；
- 指标定义；
- dataset/time window；
- statistical variance；
- 是否相同成本预算。

例如 RAG：

```text
Baseline A: keyword retrieval
Baseline B: dense retrieval
New: hybrid + rerank
```

然后比较 Recall@20、answer correctness、P95 latency、token cost。

---

# Part C｜指标体系

## 3. 离线与线上指标 ★★★★★

### Offline

- accuracy/F1/AUC/NDCG；
- retrieval recall；
- LLM judge；
- human eval。

### Online

- CTR/CVR/task success；
- retention；
- latency；
- error；
- cost；
- escalation rate。

### Guardrail

- safety violation；
- hallucination；
- privacy；
- failure rate。

不能只报一个模型分数。

---

# Part D｜A/B Test

## 4. A/B Test 面试至少要会什么 ★★★★☆

- treatment/control；
- randomization unit；
- sample size；
- duration；
- primary metric；
- guardrail metric；
- statistical significance；
- novelty/seasonality；
- rollback condition。

如果没有线上实验，可以明确说“项目未上线”，使用 offline benchmark + shadow/canary 作为证据，不要编造 A/B。

---

# Part E｜Failure / Bad Case

## 5. “你项目最大失败是什么？” ★★★★★

这是加分题，不是陷阱。

推荐：

```text
Symptom
→ Evidence
→ Hypotheses
→ Experiment
→ Root cause
→ Fix
→ Regression test
```

例如 RAG accuracy 下降：

- 先拆 retrieval recall；
- 再拆 rerank；
- 再拆 generation faithfulness；
- 找到是 chunk parser 破坏表格，而不是直接“换大模型”。

---

# Part F｜System Design 统一框架

## 6. 设计题怎么开口 ★★★★★

1. clarify requirements；
2. define scale；
3. define SLO；
4. draw high-level architecture；
5. identify bottleneck；
6. data model/state；
7. failure/safety；
8. observability；
9. evaluation；
10. trade-off。

---

# Part G｜典型设计题 1：企业 RAG

## 7. Multi-tenant RAG ★★★★★

组件：

```text
Ingestion
→ Parser
→ Chunk
→ Embedding
→ Sparse/Dense Index
→ ACL Metadata

Query
→ Auth
→ Rewrite
→ Hybrid Retrieval + ACL
→ Rerank
→ Context Builder
→ LLM
→ Citation
```

必须谈：

- tenant isolation；
- index update；
- deletion；
- prompt injection；
- cache；
- eval；
- observability。

---

# Part H｜典型设计题 2：高风险 Agent

## 8. 财务/发券/订单 Agent ★★★★★

不要让 LLM 直接做最终高风险 side effect。

```text
LLM proposes action
→ policy validation
→ business validation
→ approval if needed
→ idempotent executor
→ audit log
→ verify result
```

失败设计：

- timeout；
- duplicate；
- partial success；
- rollback/compensation；
- permission change。

---

# Part I｜典型设计题 3：LLM Serving

## 9. 高并发 Chat API ★★★★☆

需要谈：

- gateway/auth/rate limit；
- request queue；
- model router；
- vLLM/SGLang；
- continuous batching；
- KV/prefix cache；
- streaming；
- timeout/cancel；
- observability；
- fallback。

指标：TTFT/TPOT/P99/tokens per sec/GPU cost。

---

# Part J｜数据工程深挖

## 10. 训练数据 pipeline ★★★★★

美团真实面经已问：原始获取 → 清洗 → 规整 → 入库，以及领域/难度 mixture。

你需要准备：

- source；
- license/privacy；
- dedup；
- quality classifier；
- contamination；
- mixture；
- version；
- lineage；
- reproducibility。

“我们收了 100 万条”远远不够。

---

# Part K｜分布式训练深挖

## 11. ZeRO-1/2/3 ★★★★☆

粗略记忆：

- ZeRO-1：shard optimizer states；
- ZeRO-2：再 shard gradients；
- ZeRO-3：再 shard parameters。

但面试应该继续说：

- communication increases；
- parameter gather；
- CPU/NVMe offload；
- microbatch；
- activation checkpointing。

美团北斗 2026 已明确问 ZeRO-2 vs ZeRO-3。

---

# Part L｜AI 项目如何做 Observability

## 12. 三类 trace ★★★★★

### Model

- prompt/version；
- tokens；
- latency；
- model/version。

### Retrieval/Tool

- query；
- retrieved IDs；
- scores；
- tool args/result/error。

### Agent

- state transitions；
- actions；
- checkpoint；
- retries；
- final outcome。

Without trace，bad case 只能猜。

---

# Part M｜项目最常见 20 个追问

1. 为什么做这个项目？
2. 用户是谁？
3. baseline？
4. 数据规模？
5. 数据怎么来的？
6. 数据质量？
7. 为什么这个模型？
8. 为什么不是更大模型？
9. 为什么 RAG/SFT/RL？
10. 线上延迟？
11. GPU/成本？
12. 最大瓶颈？
13. 最大 bad case？
14. 怎么 debug？
15. 指标？
16. A/B？
17. 你个人做了什么？
18. 如果数据扩大 100 倍？
19. 如果流量扩大 100 倍？
20. 如果重做？

每个项目都应写答案。

---

# Part N｜2026 真实题趋势

## 美团

非常重：数据 pipeline、模型/RL 选择、MoE、分布式、评测、Agentic RL、harness。

## 小红书

很重：项目指标、检索训练、loss、数值 debug、搜索推荐建模。

## 阿里/腾讯 Agent

很重：架构、容错、RAG、Memory、工具、安全、并发。

## 百度后训练

很重：算法 dataflow + 框架 + coding。

---

# 本章实验

## Lab 1：为自己的一个项目写 Project Card

必须一页内包含 Problem/Scale/Baseline/Design/Metric/Failure。

## Lab 2：Metric Tree

从业务 KPI 向下拆到 model/retrieval/system metrics。

## Lab 3：Failure Table

至少 10 个 failure mode + detection + mitigation。

## Lab 4：Architecture defense

给每个核心组件写：

```text
Why this?
Why not alternative A/B?
Evidence?
```

---

# 本章验收

- [ ] 90 秒讲清项目。
- [ ] 所有提升有 baseline。
- [ ] 所有架构选择能说 trade-off。
- [ ] 至少准备 3 个 bad case。
- [ ] 能画线上架构和 dataflow。
- [ ] 不编造不存在的线上指标。
