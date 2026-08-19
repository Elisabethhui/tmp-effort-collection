# Lab｜检索指标与 RRF 融合

## 目标

不用向量数据库或框架，先把检索系统的评测契约写清楚：

- Recall@K；
- MRR@K；
- NDCG@K；
- 多路排序的 Reciprocal Rank Fusion（RRF）。

之后再把这些纯 Python 指标接到 BM25、dense retriever 和 reranker 实验。

## 运行

```bash
.venv/bin/python -m unittest labs/rag/test_metrics.py -v
```

## 验收标准

- 指标在空结果、重复结果和无相关文档时有明确行为；
- RRF 对不同排序列表可重复、可解释；
- 每次检索实验同时报告指标和失败 query，而不是只报告最终答案质量。
