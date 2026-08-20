# Lab｜算法、数学与 PyTorch 基础

## 目标

把 R2 的高频基础变成 CPU 上可测试的最小实现：

- `algorithms.py`：top-k、BFS、LRU；
- `math_ops.py`：stable softmax、logits cross-entropy、KL、AdamW-style update；
- 后续可在这里加入 tiny train loop 和 checkpoint。

## 运行

```bash
.venv/bin/python -m unittest labs/foundations/test_algorithms.py labs/foundations/test_math_ops.py -v
```

## 面试验收

- 能说出每个算法的复杂度和不变量；
- 能解释 `logsumexp`、`ignore_index`、KL 和 Adam bias correction；
- 能指出 CPU correctness、MPS smoke 和 GPU benchmark 是不同证据。
