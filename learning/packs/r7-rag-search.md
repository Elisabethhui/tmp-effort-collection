# R7：RAG 与搜索

> 状态：`DRAFT`
> 默认车道：`RUNNABLE_CPU` / `SOURCE_READ`；小语料优先，线上索引和大规模向量库不作为本地门槛。
> 目标：从“把文档塞进 prompt”升级为能解释召回、重排、引用、更新和评测的检索系统。

## 1. Mission

RAG 面试不只问“什么是 embedding”。真正的追问是：切多大、怎么召回、为什么漏召回、如何避免幻觉、文档更新后怎样生效、权限怎么隔离。R7 用端到端链路组织知识：

```text
source documents → parse/normalize → chunk + metadata
                 → sparse/dense/hybrid retrieve
                 → rerank/filter → context budget
                 → grounded generation + citations → evaluate/update
```

每个回答都要能回到“证据来自哪一块文档、为什么它被选中、如果没有证据怎么办”。

## 2. Prerequisites

- R2：概率、相似度、数据切分和评测基本概念；
- R3：Transformer 表示、token 长度和上下文窗口；
- R6：推理延迟、上下文预算和缓存的基本概念；
- 不要求先部署 Milvus/Elasticsearch；本地小语料可用纯 Python/NumPy 或轻量向量索引。

## 3. Learning outcomes

完成本包后能够：

1. 设计含 `doc_id/version/tenant/acl/source_span` 的 chunk schema；
2. 解释 sparse、dense、hybrid、RRF、rerank 的作用和失败边界；
3. 把 chunk size、overlap、top-k、score threshold 看成可测量的检索参数，而不是经验常数；
4. 设计带引用和“证据不足”分支的生成 prompt/contract；
5. 用 recall@k、MRR/nDCG、answer faithfulness、citation precision/recall 和 latency 分开评估。

## 4. Knowledge units

| 单元 | What | Why | Mechanism | 验证方式 |
| --- | --- | --- | --- | --- |
| U1 文档管道 | parse、normalize、section、table、version | 垃圾输入会污染所有后续指标 | 保留原文 span 和结构化 metadata | 给 PDF/Markdown 样例找丢失信息 |
| U2 Chunking | fixed、sentence、semantic、parent-child | chunk 决定可检索粒度与上下文噪声 | overlap、标题继承、边界和 token budget | 对 3 种策略画召回差异 |
| U3 Representation | BM25/TF-IDF、embedding、distance | sparse 擅长词面，dense 擅长语义 | normalize、cosine/dot、embedding version | 构造同义词与精确 ID 查询 |
| U4 Retrieval | top-k、filter、hybrid、RRF | 召回优先于生成质量 | 候选合并、去重、权限和版本过滤 | 手算两个 rank list 的 RRF |
| U5 Rerank/context | cross-encoder、MMR、budget、ordering | 高召回候选仍可能不相关 | 相关性分数、冗余惩罚、窗口拼接 | 比较 top-k 与 rerank 后证据 |
| U6 Grounding | citation、abstain、quote/span | 防止“有上下文但无证据” | 证据 ID 传递、引用格式、拒答分支 | 设计无法回答的样例 |
| U7 Eval/update | retrieval vs answer metrics、hot update、ACL | 线上质量需要可回归和可追溯 | gold set、版本、增量索引、权限边界 | 写一个小型评测表和更新流程 |

### 一条重要边界

`Retrieval hit` 不等于 `answer supported`：召回命中文档只说明候选集合里有证据，生成模型仍可能忽略、拼接或篡改它。因此至少分开记录检索指标、引用指标、答案正确性、拒答正确性和延迟。

## 5. Mac validation lane

- `RUNNABLE_CPU`：用 20–100 篇本地 Markdown/HTML，完成 chunk、TF-IDF/BM25-like、向量相似度、hybrid 和引用拼接。
- `SOURCE_READ`：阅读所选索引库/embedding 模型官方 API，记录版本和向量维度；不要把默认参数当成普适结论。
- `MPS_OPTIONAL`：embedding 小批量可以尝试 MPS；不比较不同硬件下的绝对吞吐。
- `REMOTE_GPU`：大规模 embedding、cross-encoder rerank、在线索引压测另做远端实验。

## 6. Planned labs（本包后续实现）

1. `labs/rag/chunking_reference.py`：保留标题、span、版本和 token 预算的三种切分策略；
2. `labs/rag/hybrid_retrieval.py`：实现 sparse/dense rank merge、RRF、去重和 ACL filter；
3. `labs/rag/rerank_budget.py`：用简单相关性函数模拟 rerank、MMR 和 context budget；
4. `labs/rag/grounded_answer.py`：只允许引用候选证据，证据不足时返回 abstain；
5. `labs/rag/eval_set.py`：分别计算 recall@k、MRR、citation precision/recall、answer correctness 和延迟。

## 7. Failure modes

1. **召回不到答案**：切分边界、embedding 版本、查询改写、权限过滤或 top-k 不合适；
2. **召回很多但上下文变差**：top-k 过大、重复 chunk、没有 rerank/MMR 或标题丢失；
3. **引用看似正确但答非所问**：引用 span 与生成 claim 没有逐条绑定；
4. **文档更新不生效**：version、删除 tombstone、缓存和索引刷新策略不一致；
5. **越权泄漏**：检索前未做 ACL/tenant filter，不能依赖生成模型“自己不说”；
6. **指标互相矛盾**：只优化 retrieval recall 可能增加延迟和上下文噪声，必须同时看质量、忠实性和成本。

## 8. Interview rehearsal

- **30 秒**：描述一条从文档到带引用回答的 RAG 链路。
- **3 分钟**：比较 sparse/dense/hybrid，解释 chunk、top-k、rerank 和 citation 的作用。
- **15 分钟白板/代码**：设计可热更新、有 ACL、有 abstain、可评测的 RAG 服务，并排查一次召回下降。

推荐 retrieval questions：

1. 为什么“embedding 相似度最高”不一定是最好的上下文？
2. RRF 如何合并两个排序列表，为什么不直接平均分数？
3. 如何证明回答里的每个关键 claim 都被检索证据支持？

## 9. Acceptance gate

- [ ] 在本地小语料上复现一次“召回命中但回答不被支持”的反例；
- [ ] 能手算/实现 hybrid rank merge，并保留 doc/version/span/ACL 元数据；
- [ ] 完成带引用与证据不足分支的最小回答器；
- [ ] 评测表至少分开 retrieval、grounding、correctness、latency；
- [ ] 解释一次更新、删除或越权故障的诊断顺序；
- [ ] 通过三档面试回答后，再把稳定结论晋级到 `knowledge/`。

## 10. Primary sources

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [FAISS documentation](https://faiss.ai/)
- [Sentence Transformers documentation](https://www.sbert.net/)
- [Ragas documentation](https://docs.ragas.io/)
- [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)
