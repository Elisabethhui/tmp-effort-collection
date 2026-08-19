# 04｜RAG 与检索系统：从 BM25、Embedding、Hybrid Retrieval 到 Rerank、评测与线上工程

> 目标：能设计、实现、评测和 debug 一个真实 RAG 系统。面试时不再只回答“切块→向量化→向量库→LLM”。

---

## 0. 2026 RAG 面试的变化

公开面经中的问法已经明显工程化：

- 美团北斗（2026-08-06）：为什么引入 RAG？什么场景比纯 SFT 更有效？
- 腾讯 Agent（2026-04）：为什么关键词+向量混合检索？行级向量化 vs metadata？RAG 知识库如何不停服更新？
- 阿里 Agent：一万个长文档如何构建 RAG 知识库？
- 阿里国际：BM25 数学原理、Ragas Faithfulness/Answer Relevance。
- 小红书：检索训练、InfoNCE、假负样本、cosine vs logits、线上向量检索数值问题。

所以 RAG 应拆成六层：

```text
数据摄取
  ↓
解析与切块
  ↓
索引（Sparse + Dense）
  ↓
召回
  ↓
Rerank / Filtering / Context Packing
  ↓
Generation + Citation + Evaluation
```

---

# Part A｜什么时候该用 RAG，什么时候该 SFT？

## 1. RAG 解决的核心问题 ★★★★★

RAG 的目的不是“让模型更聪明”，而是把**外部、可更新、可追溯的知识**在推理时提供给模型。

典型适合 RAG 的情况：

- 企业文档频繁更新；
- 知识量大，不适合全部固化进参数；
- 要求回答能追溯原文；
- 多租户、权限隔离；
- 需要按用户/时间/业务动态检索。

### SFT 更适合什么？

- 固定行为风格；
- 输出格式；
- 领域任务模式；
- 工具调用习惯；
- 某些稳定能力迁移。

### 高频题：RAG vs SFT

不要回答成二选一。

```text
知识更新 / 可追溯 → RAG
行为模式 / 能力迁移 → SFT
二者经常组合
```

---

# Part B｜Sparse Retrieval：BM25

## 2. TF-IDF 为什么不够？ ★★★★☆

TF-IDF 用词频和逆文档频率衡量关键词重要性，但朴素 TF 对高频词可能过度增长，也没有自然处理文档长度。

BM25 加入两个关键修正：

1. **term frequency saturation**：一个词出现 20 次不应该比出现 10 次重要两倍；
2. **document length normalization**：长文天然包含更多词，需要校正。

常见形式：

```text
score(q,d) = Σ IDF(q_i) · [ f(q_i,d)(k1+1) / (f(q_i,d)+k1(1-b+b·|d|/avgdl)) ]
```

### 为什么 2026 仍然问 BM25？

因为 dense embedding 并没有让 sparse retrieval 失效。精确产品名、ID、错误码、法规编号、代码符号等场景，关键词检索往往很强。

---

# Part C｜Dense Retrieval 与 Embedding

## 3. Dense Retrieval 的基本流程 ★★★★★

```text
query -> encoder -> q vector
chunks/items -> encoder -> d vectors
similarity(q,d)
-> ANN top-k
```

常见 similarity：

- cosine similarity；
- dot product；
- L2 distance。

### Cosine 与 Dot Product 的区别

若向量已 L2 normalize：

```text
cos(q,d) == q · d
```

未 normalize 时，dot product 同时受方向与 norm 影响。

面试官问“为什么用 cosine 不用 logits”，通常在验证你是否理解**检索 embedding 空间的 scoring objective**，而不是直接使用 language-model token logits。

---

## 4. Bi-Encoder vs Cross-Encoder ★★★★★

### Bi-Encoder

Query 和 Document 独立编码，可提前建立向量索引。

优点：快、可大规模召回。

缺点：query-document 交互只通过最终向量相似度，表达上受限。

### Cross-Encoder

把 query 和 document 一起送入模型，直接输出 relevance score。

优点：交互充分、精度高。

缺点：每个候选都要前向，无法直接对百万文档全量算。

因此常见：

```text
Bi-Encoder top-100
    ↓
Cross-Encoder rerank top-20
    ↓
LLM context
```

---

# Part D｜InfoNCE 与检索训练

## 5. InfoNCE 为什么高频 ★★★★★

一个 batch 内，假设每个 query 有对应正样本 document，其余 document 作为 negatives。

矩阵：

```text
Q: [B,D]
D: [B,D]
S = Q Dᵀ / τ  -> [B,B]
```

理想情况下对角线为正样本。

loss 相当于让每个 query 在 batch documents 中分类出正确 document。

### False Negative 问题

batch 中其他 document 不一定真的不相关。如果一个“负样本”其实也能回答 query，强行推远会伤害表示。

缓解：

- curated negatives；
- teacher filtering；
- duplicate/semantic similarity filtering；
- multi-positive loss；
- hard negative mining 时加入 false-negative guard。

小红书 2026 面经已出现这类追问。

---

# Part E｜Chunking：RAG 最容易被低估的环节

## 6. Chunk 越小越好吗？ ★★★★★｜经典坑

不是。

### 太小

- 语义被切碎；
- 缺乏上下文；
- answer evidence 跨 chunk。

### 太大

- embedding 表示被多个主题稀释；
- top-k context token 成本高；
- 精确证据被无关信息淹没。

### 设计维度

- token size；
- overlap；
- semantic boundaries；
- heading hierarchy；
- table/code preservation；
- parent-child retrieval；
- document type specific parser。

### 面试回答模板

不要说“我用 512 token + 50 overlap”。

要说：

> 我用 retrieval recall / answer faithfulness / context precision 做 chunk-size ablation，并针对 FAQ、PDF、表格、代码采用不同策略。

---

# Part F｜Hybrid Retrieval

## 7. 为什么“关键词 + 向量”经常比单路更稳？ ★★★★★

Sparse 强：

- 精确实体；
- 编号；
- 稀有词；
- 代码符号。

Dense 强：

- 同义改写；
- 语义相似；
- 用户口语 query。

Hybrid 需要解决 score calibration。

常见策略：

- weighted score fusion；
- rank fusion；
- Reciprocal Rank Fusion (RRF)。

RRF 直接融合排名，不要求两路 score 在同一数值尺度，工程上常用。

---

# Part G｜Query Understanding / Rewrite

## 8. 为什么不能把用户原 query 原样拿去检索？ ★★★★☆

多轮对话中用户可能说：

> “那它第二个版本呢？”

脱离历史无法检索。

常见步骤：

- intent classification；
- conversational query rewrite；
- entity normalization；
- expansion；
- decomposition；
- HyDE 类生成式 query representation。

### 风险

LLM rewrite 可能改错用户意图，因此要：

- 保留原 query；
- 记录 rewrite；
- 可同时检索 original + rewritten；
- eval query rewrite accuracy。

---

# Part H｜Rerank 与 Context Packing

## 9. 召回到了为什么还要 Rerank？ ★★★★★

ANN top-k 优化的是 embedding similarity，不等于最终 answer relevance。

Reranker 可以做更精细 query-doc interaction。

Context packing 还要处理：

- 去重；
- diversity；
- source authority；
- time freshness；
- token budget；
- contradictory documents；
- document permission。

真正的 RAG system 不是“top-k 全塞 prompt”。

---

# Part I｜Index 与线上更新

## 10. 一万个长文档怎么建库？ ★★★★★

回答时拆 offline pipeline：

```text
Object storage
→ parser workers
→ chunk workers
→ embedding workers
→ metadata store
→ vector index / inverted index
→ validation
→ publish new index version
```

### 关键工程点

- idempotent job；
- document/chunk stable ID；
- content hash；
- incremental update；
- deleted document tombstone；
- retry/dead-letter；
- versioned index；
- observability；
- permissions metadata。

---

## 11. RAG 知识库更新如何不停服？ ★★★★★｜腾讯真实题

思路：版本化 + 双写/双索引切换。

例如：

1. 线上读 `index_v1`；
2. 后台构建/增量更新 `index_v2`；
3. 完成 consistency + recall validation；
4. atomic alias 切换到 `v2`；
5. 保留 `v1` 一段时间便于 rollback。

若实时性要求更高，可用 base index + delta index，再周期 compaction。

不要回答“直接重新 embedding 一遍”。

---

# Part J｜Evaluation：RAG 不能只看最终答案

## 12. 检索层指标 ★★★★★

有 gold relevant docs 时：

- Recall@K；
- Precision@K；
- MRR；
- NDCG；
- hit rate。

### 为什么 Recall@K 特别重要？

如果正确证据没被召回，后面的 LLM 再强也难以 grounded answer。

---

## 13. Generation / RAG 指标 ★★★★★

要区分：

- **answer correctness**：答案对不对；
- **faithfulness / groundedness**：答案声明是否被 context 支持；
- **answer relevance**：有没有回答用户问题；
- **context precision**：召回内容是否很多噪声；
- **context recall**：需要的 evidence 是否覆盖。

Ragas 的 Faithfulness 核心思想是将 response 拆为 claims，并判断多少 claims 能从 retrieved context 支持。

### LLM-as-judge 能不能直接当真值？

不能。需要：

- judge calibration；
- human audit；
- deterministic checks；
- 多 judge / bias analysis；
- 固定 benchmark version。

---

# Part K｜RAG 为什么会幻觉？

## 14. “用了 RAG 就不会幻觉”是错的 ★★★★★

失败可能发生在：

1. 文档根本没有答案；
2. 召回失败；
3. rerank 失败；
4. context 冲突；
5. prompt 未要求 grounded；
6. LLM 忽略 evidence；
7. citation 错配；
8. knowledge freshness；
9. query rewrite 改错。

因此要对 pipeline 分层测，而不是只怪模型。

---

# Part L｜权限、安全与多租户

## 15. 企业 RAG 最危险的 bug：检索到不该看的数据 ★★★★★

必须把 authorization 放在检索链路中。

原则：

- tenant/user ACL metadata；
- retrieval filter；
- post-retrieval permission recheck；
- source audit；
- cache isolation；
- prompt injection detection；
- 不允许文档内容提升自身权限。

如果先全库检索，再让 LLM “不要泄漏”，安全设计是错误的。

---

# Part M｜真实面试拆解

## 16. 腾讯 Agent｜为什么混合检索？ ★★★★★

面试官看的是你能不能说明：

- sparse/dense 各自 failure mode；
- fusion；
- rerank；
- 用线上/离线指标证明收益。

## 17. 美团北斗｜RAG vs SFT ★★★★★

答题核心：知识时效/追溯 vs 参数化行为/能力，二者可组合。

## 18. 阿里 Agent｜一万长文档 ★★★★★

这是 distributed data pipeline + retrieval architecture 题，不是“选 Chroma 还是 Milvus”的产品题。

## 19. 小红书｜InfoNCE / false negatives ★★★★☆

这是检索模型训练题，证明 RAG 岗也可能深入到 representation learning。

---

# Part N｜代码实验

## Lab 1：BM25 vs Dense

构造：

- 精确错误码 query；
- 同义改写 query；

观察两路差异。

## Lab 2：Hybrid + RRF

实现两路 top-k rank fusion，并比较 Recall@10。

## Lab 3：InfoNCE

实现 `[B,B]` similarity matrix 与 cross entropy，加入 false negative case。

## Lab 4：Chunk ablation

比较 128/256/512/1024 tokens，记录：

- Recall@K；
- context precision；
- answer correctness；
- token cost。

## Lab 5：Index hot swap

实现 `v1 → v2` alias 切换与 rollback。

---

# 高频题库

## S 级 ★★★★★

- RAG vs SFT
- BM25
- Dense retrieval
- Bi-encoder vs cross-encoder
- Chunking
- Hybrid retrieval
- Rerank
- Recall@K / NDCG / MRR
- RAG hallucination
- knowledge hot update

## A 级 ★★★★☆

- InfoNCE
- hard negative / false negative
- query rewrite
- metadata filter
- context packing
- Ragas metrics
- multi-tenant permission

## B 级 ★★★☆☆

- parent-child retrieval
- multi-vector retrieval
- GraphRAG
- delta index / compaction
- learned sparse retrieval

---

# 本章验收

- [ ] 能从零画 RAG data plane / query plane。
- [ ] 能解释 BM25 和 dense 的互补。
- [ ] 能实现 InfoNCE。
- [ ] 能说明 chunk size 怎么实验选择。
- [ ] 能设计不停服 index update。
- [ ] 能分 retrieval 和 generation 指标。
- [ ] 能回答企业数据权限问题。
