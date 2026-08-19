# Lab｜KV Cache 显存账本

## 目标

把“7B 模型推理需要多少显存”从背诵题变成可计算的假设题。

先计算 KV Cache，不涉及权重、workspace、allocator reserve 或 batching 的额外开销；然后在面试中明确补充这些边界。

## 运行

```bash
.venv/bin/python -m unittest labs/inference/test_memory.py -v
```

## 验收标准

- 能写出 K/V 两份 cache 的 bytes 公式；
- 能解释 GQA/MQA 如何改变 `num_kv_heads`；
- 能区分参数显存、KV Cache、激活和 runtime workspace；
- 计算前明确 batch、序列长度、层数、KV heads、head dim 和 dtype。

