# 13｜搜索 / 推荐 / 排序：传统算法与大模型正在融合的高频方向

> 2026 信号：小红书大模型算法面经把 Query-Item 检索、AUC、BCE+InfoNCE、NTP 模型改检索模型、vLLM embedding 数值问题一起问；美团也出现 AI Search / NDCG Coding。  
> 这类岗位不能只懂 LLM，也要懂工业检索与排序闭环。

---

# 1. Search vs Recommendation `S`

## Search

用户显式输入 Query，目标是满足当前意图。

## Recommendation

系统根据 User / Context 主动发现内容，意图更隐式。

### 建模差异

```text
Search: Query × Item (+ User/Context)
Rec:    User × Item × Context (+ Query optional)
```

### 标签差异

搜索更强调 query-item relevance；推荐更强调点击、停留、完播、长期留存、多样性等。

---

# 2. 工业 pipeline `S`

典型：

```text
Request
 → Query/User Understanding
 → Multi-channel Recall
 → Pre-rank
 → Rank
 → Re-rank
 → Rule/Safety/Diversity
 → Impression
 → Feedback Logging
```

### 为什么多阶段

全库直接用最强模型打分成本太高。通过逐层缩小 candidate set，在 latency 与 quality 之间权衡。

---

# 3. Candidate Generation / Recall `S`

常见：

- inverted index/BM25；
- collaborative filtering；
- two-tower embedding；
- graph recall；
- popularity/freshness；
- LLM query rewrite / semantic retrieval。

目标：高 recall、可承受 latency。

---

# 4. Two-Tower `S`

Query/User 和 Item 独立编码：

```text
q = f(query/user)
i = g(item)
score = q · i  or cosine(q, i)
```

优势：Item embedding 可离线预计算并进 ANN。

缺点：Query-Item 交互发生得晚，表达力弱于 cross-encoder。

---

# 5. Cross-Encoder / Reranker `S/A`

Query 和 Item 一起输入，让 token-level 深交互。

优点：效果强。

缺点：无法对全库预计算 item representation，在线成本高。

所以常用：two-tower recall + cross-encoder rerank。

---

# 6. BM25 `A`

要理解：

- term frequency 饱和；
- inverse document frequency；
- document length normalization。

面试常问为什么 BM25 在 semantic era 还存在：

**精确词、ID、专有名词、罕见实体、可解释性、低成本**仍很有价值。

---

# 7. ANN / Vector Index `A`

常见思想：

- HNSW；
- IVF；
- PQ / quantization。

需要理解 recall-latency-memory trade-off。

### 典型工程问题

- index build 多久？
- incremental update？
- delete？
- shard？
- embedding version migration？

---

# 8. Negative Sampling `S`

负样本决定 retrieval model 的边界。

类型：

- random negatives；
- in-batch negatives；
- hard negatives；
- impression negatives；
- teacher-mined negatives。

### False Negative `🔥`

小红书 2026 直接问。

如果 batch 里另一个 item 其实也是正相关，却被当负例，会把真实相关样本推远。

处理：dedup、多正例 loss、teacher/规则过滤、soft negative、debiased loss。

---

# 9. InfoNCE `S`

对于 normalized embedding：

```python
q = F.normalize(q, dim=-1)
k = F.normalize(k, dim=-1)
logits = q @ k.T / tau
labels = torch.arange(q.size(0), device=q.device)
loss = F.cross_entropy(logits, labels)
```

## Temperature

小 tau → sharper distribution、harder gradients；但更敏感于 noise/false negatives。

---

# 10. BCE + InfoNCE `A`

2026 小红书直接问联合损失。

- BCE：pointwise absolute relevance / calibration signal；
- InfoNCE：relative contrastive separation。

不能说一定提升。必须通过 ablation：BCE-only / InfoNCE-only / joint。

---

# 11. Pointwise / Pairwise / Listwise `S/A`

## Pointwise

每个 item 独立预测 label/probability。

## Pairwise

学习 `positive > negative`。

## Listwise

直接围绕整个 ranking list 优化。

面试要能说明：训练目标如何与线上排序指标对齐。

---

# 12. AUC / Recall@K / MRR / NDCG `S`

## Recall@K

目标相关 item 有多少在 Top-K 被召回。

## MRR

第一个 relevant result 的 reciprocal rank。

## DCG/NDCG

越靠前的相关结果权重越高；NDCG 用理想排序归一化。

### 2026 美团 AI Coding

出现了以 NDCG@10 为目标的搜索排序问题，说明算法岗正在出现更贴近业务的 AI coding/task-style assessment。

---

# 13. Offline vs Online Metrics `S`

离线：

- Recall/NDCG/AUC；
- label quality；
- latency benchmark。

线上：

- CTR/CVR；
- dwell time；
- retention；
- negative feedback；
- revenue；
- diversity/ecosystem metrics。

**离线涨不等于线上涨。**

---

# 14. Exposure / Position Bias `A`

点击不是纯 relevance。

受到：

- rank position；
- thumbnail/title；
- prior recommendation policy；
- popularity；
- user selection bias。

需要：counterfactual thinking、IPS/propensity、randomized bucket 等概念。

---

# 15. Cold Start `A`

## New User

依赖 context、热门、探索、内容特征。

## New Item

依赖 content embedding、metadata、exploration。

LLM/VLM 的语义表示能改善 content-based cold start，但仍不替代 online feedback。

---

# 16. Query Understanding `A`

搜索 Query 常见：

- typo correction；
- normalization；
- intent classification；
- NER；
- rewrite；
- expansion；
- structured filter extraction。

LLM 可用于 rewrite/intent，但要控制 latency、hallucination 和 deterministic constraints。

---

# 17. LLM 如何进入搜索推荐 `A/↗`

1. embedding backbone；
2. reranker；
3. query rewrite；
4. user/item summarization；
5. synthetic labels；
6. feature extraction；
7. conversational recommendation；
8. generative retrieval / recommendation。

### 关键坑

不要因为用了 LLM 就把成熟检索系统全部换掉。成本、可控性、增量更新和反馈闭环仍是核心。

---

# 18. 从 NTP Qwen 改成 Retrieval Model `🔥`

小红书 2026 直接问。

基本做法：

```text
Qwen backbone
 → choose pooling representation
 → projection head
 → normalize embedding
 → contrastive / retrieval objective
 → ANN serving
```

关键：

- pooling 不能读 padding；
- train/serve tokenizer 必须一致；
- cosine/dot 与 ANN metric 一致；
- item offline embedding version；
- hard negative quality。

---

# 19. Mean Pooling FP16 NaN `🏗`

长序列 FP16 sum 可能 overflow；全 padding 可能 0/0。

安全写法：

```python
x = hidden.float()
mask = attention_mask[..., None].float()
s = (x * mask).sum(dim=1)
n = mask.sum(dim=1).clamp_min(1.0)
emb = s / n
emb = F.normalize(emb, dim=-1)
```

要先定位第一处 non-finite，而不是直接 clamp 掩盖问题。

---

# 20. Index Hot Update `A`

建议：

```text
source v2
 → build embedding/index v2
 → validate
 → atomic alias switch
 → drain v1
 → rollback available
```

metadata / vector / source version 要可追踪。

---

# 21. 2026 小红书真实题拆解

题组：

- Query/Item 业务量级；
- 测试集怎么搭；
- AUC；
- MSE 降但排序差；
- BCE + InfoNCE；
- Qwen NTP → retrieval；
- cosine vs logits；
- FP16 pooling NaN；
- LoRA；
- false negatives；
- 手写 InfoNCE；
- quicksort。

这是一套非常完整的“模型 + 检索 + 指标 + 数值 + coding”面试模板。

---

# 22. 高频等级

## S

- search vs rec
- recall/rank pipeline
- two-tower
- InfoNCE
- negative sampling
- AUC/Recall/NDCG/MRR
- online/offline metrics

## A

- BM25 + dense hybrid
- reranker
- ANN
- bias
- cold start
- query understanding
- index hot update

## B

- generative retrieval
- counterfactual learning details
- LLM search agent

---

# 23. Labs

1. Two-tower + in-batch negative；
2. 加 hard negative 与 false-negative mask；
3. BM25 + dense + RRF；
4. Cross-encoder rerank；
5. 比较 AUC vs NDCG；
6. versioned ANN index hot swap；
7. FP16 pooling 故障注入。

---

# 24. 自测

- [ ] 能解释 search 与 recommendation 标签为什么不同
- [ ] 能手写 InfoNCE
- [ ] 能解释 false negative
- [ ] 能解释 NDCG
- [ ] 能从 NTP model 设计 retrieval fine-tuning
- [ ] 能定位 MSE 降而 NDCG 差
- [ ] 能设计 index hot update
