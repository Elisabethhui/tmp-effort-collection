# 学习使命

## Why

在算法、大模型和 Agent 面试中，建立从数学机制到代码实现、再到工程追问的连续能力，避免只会背名词。

## Learner constraints

- 当前主要使用 Apple Silicon Mac；GPU/CUDA 性能实验不可作为本地必需门槛；
- 优先 CPU/tiny tensor 正确性、源码阅读、显存/通信账本和口述验证；
- 学习语言为中文，代码和公式保留英文术语；
- 先完成 R2 基础和 R3 Transformer，再进入训练、后训练、推理、RAG 和 Agent。

## Success criteria

完成 R2–R3 后，能够：

1. 独立解释常见算法、数学、ML/DL 基础并写出最小实现；
2. 从 `[B, T]` 追踪 Decoder-only Transformer 到 `[B, T, V]` 的每个 shape；
3. 手写并测试 MHA、mask、RoPE、RMSNorm、SwiGLU、GQA 和 causal loss；
4. 对 KV Cache、Flash/Paged Attention 等 GPU 主题完成源码调用链和 CPU reference，不假装完成 GPU 性能验证；
5. 对高频问题给出 30 秒、3 分钟和 15 分钟回答。

## Out of scope for this stage

- 真实 CUDA kernel 性能、vLLM/FSDP benchmark；
- 全量迁移 54 个 Agent Notebook；
- 在没有验证门禁时创建长期 `knowledge/` 副本；
- 直接追逐最新框架 API，而不先掌握原理和 shape。
