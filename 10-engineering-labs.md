# 10｜工程实践训练：从“会背”到“真的能写”

> 目标：把八股知识转成可验证的代码和工程能力。  
> 原则：每个实验都必须有输入、实现、测试、指标、失败样例和复盘。

---

# Lab 01：从零手写 Multi-Head Attention

## 目标

不用 `nn.MultiheadAttention`，只允许使用基础 tensor op / Linear，实现：

- self-attention；
- causal mask；
- padding mask；
- multi-head reshape；
- dropout；
- output projection。

## 验收

- 与 PyTorch 官方实现对齐一个小样本输出；
- 用 shape assertion 覆盖不同 B/T/H；
- 对 causal mask 做可视化/单元测试；
- FP16/BF16 下没有 NaN；
- 解释 `sqrt(d_head)`。

## 面试能力

对应 DeepSeek 2026 的完整 MHA 手写题。

---

# Lab 02：手写一个最小 Qwen2 SFT Trainer

## 不允许一上来用 Trainer 一键跑

先手写：

1. tokenizer/chat template；
2. prompt/answer 拼接；
3. label mask；
4. padding；
5. forward；
6. shifted CE；
7. optimizer；
8. backward；
9. grad accumulation；
10. checkpoint。

然后再用 Hugging Face Trainer/SFTTrainer 复现，比较两者。

## 验收

- 证明 prompt token 的 loss 被 mask；
- 证明 pad 不计 loss；
- overfit 20 条 toy data；
- 恢复 checkpoint 后 loss 连续；
- 记录显存峰值。

---

# Lab 03：Toy GRPO / Policy Optimization

不追求大模型规模，而是把数学链条写通。

```text
prompt
 → sample G candidates
 → reward
 → normalized group advantage
 → policy ratio
 → clipping
 → KL/reference
 → update
```

## 验收

- 打印每个 sample 的 reward/advantage；
- 构造 policy lag；
- 比较 clip 前后 gradient；
- 故意设计一个可被 reward hacking 的 reward，观察失败；
- 加一个约束修复。

---

# Lab 04：vLLM Serving Benchmark

## 变量

- batch/concurrency；
- prompt length；
- output length；
- prefix reuse；
- quantization；
- model size。

## 指标

- TTFT；
- TPOT/ITL；
- request latency P50/P95/P99；
- tokens/s；
- GPU memory。

## 目标

亲眼看到“吞吐更高”和“单请求更快”不是同一件事。

---

# Lab 05：Hybrid RAG

实现：

```text
query
 ├─ BM25
 └─ Dense Retriever
       ↓
       Fusion (RRF)
       ↓
     Reranker
       ↓
  Context Packing
       ↓
      LLM
```

## 验收

- Recall@K；
- MRR/NDCG；
- answer faithfulness；
- 失败 query 分类；
- dense-only / sparse-only / hybrid ablation。

---

# Lab 06：RAG 热更新

实现 versioned index，而不是直接修改线上索引。

要求：

- build v1/v2；
- validation gate；
- atomic alias switch；
- rollback；
- 删除文档不出现 ghost chunks；
- embedding model version 写进 metadata。

---

# Lab 07：Durable Agent Runtime

实现一个至少包含以下状态的 Agent：

```text
PLANNING
TOOL_PENDING
TOOL_RUNNING
WAITING_RETRY
WAITING_HUMAN
COMPLETED
FAILED
```

## 必须实现

- checkpoint；
- resume；
- idempotency key；
- retry/backoff；
- max step；
- tool schema validation；
- trace。

## 故障注入

- tool timeout；
- API 429；
- tool 成功但模型未收到 response；
- 进程 crash；
- 重复 resume。

---

# Lab 08：Memory / Context Governance

构造 50～100 轮长会话，分别测试：

- full transcript；
- sliding window；
- summary；
- retrieval memory；
- structured state + pointer。

比较：

- token cost；
- factual recovery；
- stale memory；
- contradictory memory；
- latency。

---

# Lab 09：Agent Evaluation Harness

至少覆盖：

- task success；
- tool selection accuracy；
- argument correctness；
- loop rate；
- retry rate；
- latency；
- token cost；
- side-effect safety；
- resume success。

要求对每个失败保存完整 trace，并能自动归类。

---

# Lab 10：模拟真实面试

每个专题用三种模式练：

## 模式 A：30 秒

只回答定义 + 核心结论。

## 模式 B：3 分钟

补公式/结构/取舍。

## 模式 C：15 分钟追问

面试官连续追：

```text
为什么？
怎么算？
怎么写？
哪里会坏？
为什么不用 X？
线上怎么验证？
```

如果只能回答模式 A，说明仍然停留在“背八股”。
